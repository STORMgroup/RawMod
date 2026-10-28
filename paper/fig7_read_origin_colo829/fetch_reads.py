#!/usr/bin/env python3
"""Select the reads that cover each window in each sample (forward strand, full coverage with a margin, up to --per-group per window) from the published alignments, and fetch only their raw signal from the remote POD5 files with HTTP range requests."""
from __future__ import annotations

import argparse
import collections
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pysam
import requests

sys.path.insert(0, str(Path(__file__).parent))
import remote_pod5 as RP  # noqa: E402

S3_HTTP = 'https://ont-open-data.s3.amazonaws.com/colo829_2024.03'
# sample -> bucket sub-directory holding that sample's flowcells
SAMPLE_DIR = {'tumor': 'colo829', 'normal': 'colo829bl'}


def flowcell_dirs() -> dict:
    """(flowcell_id, run_id8) -> full s3 prefix of that run's directory."""
    import subprocess
    out = {}
    for sample, sub in SAMPLE_DIR.items():
        r = subprocess.run(
            ['aws', 's3', 'ls', '--no-sign-request',
             f's3://ont-open-data/colo829_2024.03/flowcells/{sub}/'],
            capture_output=True, text=True, check=True)
        for line in r.stdout.splitlines():
            name = line.split()[-1].rstrip('/')
            m = re.match(r'\d+_\d+_\w+_(PAU\d+)_([0-9a-f]+)$', name)
            if m:
                out[(m.group(1), m.group(2))] = (
                    f'{S3_HTTP}/flowcells/{sub}/{name}', sample)
    return out


def pod5_url(fn: str, dirs: dict) -> str | None:
    """Map a BAM `fn` tag (source pod5 filename) to its full https URL."""
    m = re.match(r'(PAU\d+)_(pass|fail|skip)_([0-9a-f]+)_', fn)
    if not m:
        return None
    key = (m.group(1), m.group(3))
    if key not in dirs:
        return None
    base, _ = dirs[key]
    return f'{base}/pod5_{m.group(2)}/{fn}'


def collect_candidates(bam_path: Path, sites: pd.DataFrame,
                       min_mapq: int, margin: int,
                       strand: str = '+') -> tuple[dict, dict]:
    """(site index -> usable read ids, read id -> source pod5 filename).

    strand '+' keeps the forward-strand-only convention an earlier run was trained on.
    'both' also collects reverse-strand reads, which roughly triples the depth
    available per window (median 28 -> 57 per group) — only safe if strand turns
    out not to be recoverable from the embeddings, since the checkpoint has never
    seen a reverse row. Check before pooling.
    """
    bam = pysam.AlignmentFile(str(bam_path), 'rb')
    per_site = collections.defaultdict(list)
    read_fn = {}
    for i, row in enumerate(sites.itertuples()):
        lo, hi = int(row.start) - margin, int(row.start) + margin
        for br in bam.fetch(row.chrom, max(0, lo), hi + 1):
            if br.is_secondary or br.is_supplementary or br.is_unmapped:
                continue
            if strand == '+' and br.is_reverse:
                continue
            if strand == '-' and not br.is_reverse:
                continue
            if br.mapping_quality < min_mapq:
                continue
            if br.reference_start > lo or br.reference_end < hi:
                continue
            try:
                fn = br.get_tag('fn')
            except KeyError:
                continue
            read_fn[br.query_name] = fn
            per_site[i].append(br.query_name)
    bam.close()
    return per_site, read_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sites', required=True)
    ap.add_argument('--bams', required=True,
                    help='comma-separated group:flowcell:path triples')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--per-group', type=int, default=18)
    ap.add_argument('--min-mapq', type=int, default=20)
    ap.add_argument('--strand', default='+', choices=['+', '-', 'both'],
                    help="'both' collects reverse-strand reads too — see "
                         "collect_candidates")
    ap.add_argument('--margin', type=int, default=15,
                    help='bases each side of the site a read must fully cover')
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    out = Path(a.out_dir)
    (out / 'pod5').mkdir(parents=True, exist_ok=True)
    (out / 'mirrors').mkdir(parents=True, exist_ok=True)
    sites = pd.read_csv(a.sites, sep='\t')

    specs = []
    for tok in a.bams.split(','):
        g, fc, p = tok.split(':')
        specs.append((g, fc, Path(p)))

  # ── candidate reads per (site, group) ────────────────────────────────────
    cand = collections.defaultdict(lambda: collections.defaultdict(list))
    read_fn, read_group, read_fc = {}, {}, {}
    for g, fc, p in specs:
        print(f'scanning {p.name} ({g}/{fc}) ...', flush=True)
        per_site, fns = collect_candidates(p, sites, a.min_mapq, a.margin,
                                          a.strand)
        for rid, fn in fns.items():
            read_fn[rid] = fn
            read_group[rid] = g
            read_fc[rid] = fc
        for si, rids in per_site.items():
            cand[si][g].extend(rids)

  # how many target sites each read covers -> prefer multi-site reads
    cover = collections.Counter()
    for si, bygrp in cand.items():
        for g, rids in bygrp.items():
            for rid in set(rids):
                cover[rid] += 1

    selected = set()
    site_rows = []
    for si in range(len(sites)):
        row = {'site_i': si, 'site_id': sites.iloc[si].site_id}
        for g in sorted({s[0] for s in specs}):
            rids = sorted(set(cand[si].get(g, [])),
                          key=lambda r: (-cover[r], r))
            already = [r for r in rids if r in selected]
            need = max(0, a.per_group - len(already))
            take = already + [r for r in rids if r not in selected][:need]
            selected.update(take)
            row[f'n_{g}'] = len(take)
            row[f'n_avail_{g}'] = len(rids)
        site_rows.append(row)
    cov_df = pd.DataFrame(site_rows)
    cov_df.to_csv(out / 'site_read_counts.tsv', sep='\t', index=False)
    print(f'\n{len(selected):,} distinct reads selected across {len(sites)} sites')
    print(cov_df.describe().to_string())

    meta = pd.DataFrame({
        'read_id': sorted(selected),
        'group': [read_group[r] for r in sorted(selected)],
        'flowcell': [read_fc[r] for r in sorted(selected)],
        'pod5': [read_fn[r] for r in sorted(selected)],
    })
    meta.to_csv(out / 'reads.tsv', sep='\t', index=False)

    by_file = collections.defaultdict(list)
    for r in meta.itertuples():
        by_file[r.pod5].append(r.read_id)
    print(f'{len(by_file):,} distinct source pod5 files '
          f'({len(selected) / max(1, len(by_file)):.1f} reads/file)')
    if a.dry_run:
        return

    session_dirs = flowcell_dirs()
    print(f'{len(session_dirs)} flowcell run directories resolved')

  # one pod5 file was probed already to learn the signal batch height; it is a
  # writer-level constant across a run, so reuse it and skip a 6 MB probe/file
    batch_rows_hint = 100

    def work(fn_reads):
        import pod5
        fn, rids = fn_reads
        url = pod5_url(fn, session_dirs)
        if url is None:
            return fn, 0, f'no url for {fn}'
        dest = out / 'pod5' / fn
        if dest.exists():
            return fn, -1, 'cached'
        tmp = dest.with_suffix('.tmp.pod5')
        try:
            with pod5.Writer(str(tmp)) as w:
                got, _ = RP.extract_reads(
                    url, rids, w, out / 'mirrors',
                    session=requests.Session(),
                    signal_batch_rows=batch_rows_hint, verbose=False)
            if not got:
  # A pod5 holding zero reads still has a valid header, and
  # `pod5 merge` aborts on it with "Batch number 0 out of range" —
  # so never leave one behind. (Happens for the ~1% of reads whose
  # `fn` tag does not resolve to a read in that file.)
                tmp.unlink(missing_ok=True)
                return fn, 0, None
            tmp.rename(dest)
            return fn, len(got), None
        except Exception as e:
            tmp.unlink(missing_ok=True)
            return fn, 0, repr(e)

    done = nread = 0
    errors = []
    with ThreadPoolExecutor(max_workers=a.threads) as ex:
        futs = [ex.submit(work, item) for item in by_file.items()]
        for f in as_completed(futs):
            fn, k, err = f.result()
            done += 1
            if err and err != 'cached':
                errors.append((fn, err))
            elif k > 0:
                nread += k
            if done % 25 == 0:
                print(f'  {done}/{len(by_file)} pod5 files, {nread} reads',
                      flush=True)
    print(f'\nfetched {nread} reads from {len(by_file)} remote pod5 files')
    if errors:
        print(f'{len(errors)} failures, first few:')
        for fn, e in errors[:5]:
            print('  ', fn, e)


if __name__ == '__main__':
    main()
