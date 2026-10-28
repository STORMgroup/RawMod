#!/usr/bin/env python3
"""Per-read embeddings at every site of a set of windows (CpG, non-CpG C, A or T sites), built from the raw signal with no label information. Writes an npz with one 96-dimensional embedding per (read, site)."""

from __future__ import annotations
import os as _os
from pathlib import Path as _Path
REPO = str(_Path(__file__).resolve().parents[2])
RAWMOD_RESULTS = _os.environ.get('RAWMOD_RESULTS') or f'{REPO}/paper/data/runs'  # training runs; shipped metrics by default
RAWMOD_DATA = _os.environ.get('RAWMOD_DATA', 'data')  # datasets (not shipped: set this to score raw signal)
RAWMOD_WORK = _os.environ.get('RAWMOD_WORK') or f'{REPO}/paper/data/work'  # outputs of the paper scripts; shipped by default

import argparse
import collections
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import torch
from tqdm import tqdm

sys.path.insert(0, f'{REPO}/rawmod')
import featurization as FZ  # noqa: E402
try:  # fast path: bitwise identical, ~17x
    from featurize_fast import extract_base_segments_fast as _EXTRACT
except Exception:  # pragma: no cover
    _EXTRACT = FZ.extract_base_segments
if os.environ.get('RAWMOD_LEGACY_FEATURIZE'):
    _EXTRACT = FZ.extract_base_segments

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sys as _sys  # noqa: E402
_REPO = _Path(__file__).resolve().parents[2]
for _d in (_REPO / 'rawmod', _REPO / 'scripts' / 'evaluate', _REPO / 'scripts' / 'train'):
    _sys.path.insert(0, str(_d))
from model import add_delta_channels as _add_delta  # noqa: E402
from score_genome import load_model  # noqa: E402


def add_delta_channels(x, half_window, L):
    """(1, 9, H, W*L) -> (1, 11, H, W*L): the deviation channels of rawmod.model, centre base = half_window."""
    return _add_delta(x[0], half_window, L)[None]


def sites_in_window(fa: pysam.FastaFile, chrom: str, lo: int, hi: int,
                    site_type: str, min_cpg_dist: int, max_sites: int,
                    seed: int) -> list:
    """Candidate positions, chosen from the REFERENCE only — never from calls.

    site_type 'cpg' reproduces the original behaviour. 'noncpg_c' and 'a' exist
    to use the one thing RawMod has that dorado's 5mCG_5hmCG model does not: a
    reading at positions that model cannot score at all. Those two are the whole
    point of the comparison, so they carry two guards:

      min_cpg_dist  a non-CpG C sitting 3 bp from a CpG is not independent
                    evidence — RawMod's +-10 base window would see that CpG's
                    signal and the "novel" site would just be a shifted view of
                    it. Requiring a gap wider than the window makes contamination
                    impossible rather than merely unlikely.
      max_sites     the CpG arm gets ~2.8k sites, so handing the non-CpG arm 30k
                    would confound "different sites" with "more evidence". Match
                    the count and the only difference left is which positions.
    """
    seq = fa.fetch(chrom, lo, hi).upper()
    cpg = [i for i in range(len(seq) - 1) if seq[i:i + 2] == 'CG']
    if site_type == 'cpg':
        return [lo + i for i in cpg]
    if site_type == 'noncpg_c':
        cand = [i for i in range(len(seq)) if seq[i] == 'C']
  # only this type needs the CpG-distance guard: it exists to prove a
  # "novel" cytosine is not just a shifted view of a nearby CpG
        cset = set(cpg)
        if cpg:
            arr = np.asarray(cpg)
            cand = [i for i in cand
                    if i not in cset
                    and np.min(np.abs(arr - i)) > min_cpg_dist]
    elif site_type == 'a':
        cand = [i for i in range(len(seq)) if seq[i] == 'A']
    elif site_type == 't':
  # 5hmU. SPO1 hypermodifies essentially every T, and dorado has no 5hmU
  # model at all, so these are positions no basecaller-based method can
  # score. No CpG guard: CpGs are irrelevant to a T-borne mark.
        cand = [i for i in range(len(seq)) if seq[i] == 'T']
    else:
        raise ValueError(site_type)
    if max_sites and len(cand) > max_sites:
        rng = np.random.default_rng(seed)
        cand = sorted(rng.choice(cand, size=max_sites, replace=False).tolist())
    return [lo + i for i in cand]


def encode_and_write(a, model, device, rows, out_read, out_site, out_grp,
                     out_win, st):
    """read_encoder over every assembled row, then the standard npz."""
    print(f'\nrunning read_encoder on {rows.shape[0]:,} read-rows '
          f'({len(st):,} sites)', file=sys.stderr)
    embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(rows), a.batch), file=sys.stderr):
            xb = torch.from_numpy(rows[i:i + a.batch]).to(device)
            if a.mask_bases:  # training blanks ch 2-5 on load
                xb = xb.clone(); xb[:, 2:6] = 0.0
            embs.append(model.read_encoder(xb).float().cpu().numpy())
    emb = np.concatenate(embs)
  # write-then-rename, same reason as the cache
    tmp = str(a.output) + '.part.npz'
    np.savez_compressed(
        tmp, emb=emb, read_id=np.array(out_read), site_i=np.array(out_site),
        group=np.array(out_grp), window=np.array(out_win),
        site_window=st.window.values, site_chrom=st.chrom.values,
        site_pos=st.pos.values, site_nreads=st.n_reads.values)
    Path(tmp).replace(a.output)
    print(f'wrote {a.output}: {emb.shape[0]:,} (read, site) rows, '
          f'{emb.shape[1]}-d, {len(st):,} sites', file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pod5', required=True)
    ap.add_argument('--bam', required=True, help='coordinate-sorted + indexed')
    ap.add_argument('--peaks', required=True)
    ap.add_argument('--windows', required=True,
                    help='TSV: name<TAB>chrom<TAB>start<TAB>end')
    ap.add_argument('--read-groups', required=True,
                    help='TSV read_id<TAB>group — evaluation metadata only')
    ap.add_argument('--ref', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--level-table', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--half-window', type=int, default=10)
    ap.add_argument('--L', type=int, default=10)
    ap.add_argument('--min-reads', type=int, default=20)
    ap.add_argument('--min-mapq', type=int, default=0)
    ap.add_argument('--strand', default='+', choices=['+', '-', 'both'])
    ap.add_argument('--site-type', default='cpg',
                    choices=['cpg', 'noncpg_c', 'a', 't'],
                    help="which reference positions to score; noncpg_c / a / t are "
                         "positions dorado's CpG model cannot call — see "
                         "sites_in_window")
    ap.add_argument('--min-cpg-dist', type=int, default=25,
                    help='non-CpG sites must be further than this from any CpG, '
                         'so the +-10 base window cannot see one')
    ap.add_argument('--max-sites', type=int, default=0,
                    help='per window; 0 = no cap. Match the CpG arm to keep the '
                         'comparison about site identity, not site count')
    ap.add_argument('--site-seed', type=int, default=0)
    ap.add_argument('--sites-bed', default=None,
                    help='BED of DESIGNED sites (chrom, start, ...). When given '
                         'it overrides --site-type entirely: sites are taken '
                         'from the file rather than found by sequence motif. '
                         'Needed for the all-5mers oligos, where the modified '
                         'positions are a property of how the construct was '
                         'synthesised and cannot be recovered from the '
                         'reference sequence. Pass several comma-separated to '
                         'take their union (e.g. the C and A site lists).')
    ap.add_argument('--normalize', action='store_true',
                    help='MAD-normalize each read signal and z-score the level '
                         'table, as featurization.py --normalize does. REQUIRED '
                         'for checkpoints trained on the strand-resolved MAD '
                         'features (the final models); earlier runs used zscore.')
    ap.add_argument('--mask-bases', action='store_true',
                    help='blank the base-identity channels (2-5) before '
                         'encoding, as PileupDataset(mask_all_bases=True) does '
                         'in training under PILEUP_MASK_BASES=1 (earlier runs). '
                         'the final model_vis_* were trained with bases VISIBLE.')
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--rows-cache', default=None,
                    help='npz of the assembled read-rows. Everything upstream of '
                         'read_encoder is model-independent — the same rows feed '
                         'any checkpoint with these 11 channels — so caching them '
                         'turns a second model into a few minutes of matmul '
                         'instead of re-reading every pod5 signal. Written if '
                         'absent, loaded if present.')
    a = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = load_model(a.ckpt, device)
    model.eval()

    cache = Path(a.rows_cache) if a.rows_cache else None
    if cache is not None and cache.exists():
        c = np.load(cache, allow_pickle=True)
        cached_norm = str(c['normalization']) if 'normalization' in c.files else 'zscore'
        want_norm = 'MAD' if a.normalize else 'zscore'
        if cached_norm != want_norm:
            raise SystemExit(f'{cache} holds {cached_norm} rows but {want_norm} '
                             f'was requested; use a separate --rows-cache')
        rows = c['rows']
        out_read, out_site = list(c['read_id']), list(c['site_i'])
        out_grp, out_win = list(c['group']), list(c['window'])
        st = pd.DataFrame(dict(window=c['site_window'], chrom=c['site_chrom'],
                               pos=c['site_pos'], n_reads=c['site_nreads']))
        print(f'loaded {rows.shape[0]:,} cached read-rows from {cache}',
              file=sys.stderr)
        encode_and_write(a, model, device, rows, out_read, out_site, out_grp,
                         out_win, st)
        return

    fa = pysam.FastaFile(a.ref)
    wins = pd.read_csv(a.windows, sep='\t')
    rg = pd.read_csv(a.read_groups, sep='\t')
    read_group = dict(zip(rg['read_id'], rg['group']))

    kmer_levels = FZ.load_level_table(a.level_table)  # no --normalize,
    if a.normalize:  # ...unless asked
        kmer_levels, _, _ = FZ.normalize_level_table(kmer_levels)
    kmer_size = len(next(iter(kmer_levels)))  # matching an earlier run
    seg_borders = FZ.load_peaks_file(a.peaks)
    readers = FZ.get_pod5_readers(a.pod5)

  # centre index of the k-mer, detected exactly as the stock featurizer does
    center_idx = None

    bed_sites = None
    if a.sites_bed:
        bed_sites = collections.defaultdict(set)
        for path in a.sites_bed.split(','):
            with open(path.strip()) as fh:
                for line in fh:
                    if not line.strip() or line.startswith(('#', 'track')):
                        continue
                    f = line.split('\t')
                    bed_sites[f[0]].add(int(f[1]))
        bed_sites = {k: sorted(v) for k, v in bed_sites.items()}
        print(f'sites-bed: {sum(len(v) for v in bed_sites.values())} designed '
              f'sites over {len(bed_sites)} contigs')

    out_emb, out_read, out_site, out_grp, out_win = [], [], [], [], []
    site_rows = []

    for w in wins.itertuples():
        if bed_sites is not None:
            cpg = [p for p in bed_sites.get(w.chrom, [])
                   if int(w.start) <= p < int(w.end)]
        else:
            cpg = sites_in_window(fa, w.chrom, int(w.start), int(w.end),
                                  a.site_type, a.min_cpg_dist, a.max_sites,
                                  a.site_seed)
        want = {p: i for i, p in enumerate(cpg)}
        margin = a.half_window + 12

  # ── pass 1: per-read segments, restricted to this window ────────────
        bam = pysam.AlignmentFile(a.bam, 'rb')
        site_reads = collections.defaultdict(list)
        ref_context = {}
        sample = []
        for br in bam.fetch(w.chrom, max(0, int(w.start) - margin),
                            int(w.end) + margin):
            if br.is_supplementary or br.is_secondary or br.is_unmapped:
                continue
            if br.mapping_quality < a.min_mapq:
                continue
            rid = br.query_name
            if rid not in seg_borders or rid not in readers:
                continue
            strand = -1 if br.is_reverse else +1
            if a.strand != 'both' and strand != (+1 if a.strand == '+' else -1):
                continue
            try:
                p5 = next(readers[rid].reads(selection=[rid]))
                raw = p5.signal.astype(np.float64)
                sig = FZ.mad_normalize(raw) if a.normalize else FZ.zscore(raw)
                ref_seq, ref_pos = FZ.get_ref_info_from_bam(br)
                if ref_seq is None:
                    continue
                segs = _EXTRACT(sig, seg_borders[rid], ref_seq,
                                                ref_pos, a.L,
                                                br.mapping_quality, strand)
                if center_idx is None and len(sample) < 50:
                    sample.append((sig, seg_borders[rid], ref_seq, ref_pos))
                for rpos, entry in segs.items():
                    ref_context[rpos] = entry['base'].upper()
                for rpos in segs:
                    si = want.get(rpos)
                    if si is None:
                        continue
                    local = FZ.trim_segments_to_window(segs, rpos, a.half_window)
                    site_reads[si].append((rid, FZ.make_read_record(local, br)))
            except Exception:
                continue
        bam.close()

        if center_idx is None:
            center_idx, best_r = FZ.auto_detect_center_idx(kmer_size, sample,
                                                           kmer_levels)
            print(f'center_idx={center_idx} (mean r={best_r:.4f})',
                  file=sys.stderr)

  # ── pass 2: one tensor per site holding EVERY read, then read_encoder ─
        for si in sorted(site_reads):
            recs = site_reads[si]
            if len(recs) < a.min_reads:
                continue
            pos = cpg[si]
            ref_row = FZ.build_reference_row(
                center_pos=pos, half_window=a.half_window, L=a.L,
                ref_base_map=ref_context, kmer_levels=kmer_levels,
                kmer_size=kmer_size, center_idx=center_idx)
            n = len(recs)
            tensor = FZ.build_pileup_tensor([r for _, r in recs], ref_row, pos,
                                            a.half_window, a.L, n)
            x = np.transpose(tensor[None], (0, 3, 1, 2)).astype(np.float32)
            x = add_delta_channels(x, a.half_window, a.L)  # (1,11,n+1,W)
            site_rows.append(dict(window=w.name, chrom=w.chrom, pos=pos,
                                  n_reads=n))
            k = len(site_rows) - 1
            for j, (rid, _) in enumerate(recs):
                out_read.append(rid)
                out_site.append(k)
                out_win.append(w.name)
                out_grp.append(read_group.get(rid, '?'))
  # rows 1: are the reads; row 0 is the reference row
            out_emb.append(x[0, :, 1:, :].transpose(1, 0, 2))  # (n,11,W)

        print(f'{w.name}: {len(cpg)} {a.site_type} sites, {len(site_reads)} covered, '
              f'{len(site_rows)} kept so far', file=sys.stderr, flush=True)

    rows = np.concatenate(out_emb, axis=0)
    st = pd.DataFrame(site_rows)
    if cache is not None:
  # uncompressed: these are multi-GB float32 arrays and the point of the
  # cache is to be faster to read than to rebuild
        cache.parent.mkdir(parents=True, exist_ok=True)
  # write-then-rename: a killed job must never leave a cache that
  # looks complete (np.savez appends .npz unless the name ends in it)
        ctmp = cache.with_name(cache.name + '.part.npz')
        np.savez(ctmp, rows=rows, read_id=np.array(out_read),
                 site_i=np.array(out_site), group=np.array(out_grp),
                 window=np.array(out_win), site_window=st.window.values,
                 site_chrom=st.chrom.values, site_pos=st.pos.values,
                 site_nreads=st.n_reads.values,
                 normalization='MAD' if a.normalize else 'zscore')
        ctmp.replace(cache)
        print(f'cached read-rows -> {cache}', file=sys.stderr)
    encode_and_write(a, model, device, rows, out_read, out_site, out_grp,
                     out_win, st)


if __name__ == '__main__':
    main()
