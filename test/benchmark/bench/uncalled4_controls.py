#!/usr/bin/env python3
"""Train one Uncalled4 control pore model per unmodified control library (bench.uncalled4_sites).

For each control library: take reads of the FULL library that are in no benchmark read set of that library
(test/data/read_ids/<sample>.txt.gz of its positions and region read sets), primary, mapped, mean qscore >= 10, the
N with the lowest read_rank; subset their pod5 and BAM; `uncalled4 train` from the released R10.4.1 model
(-i iterations). The last iteration's model is written to $UNCALLED4_MODELS/<lib>/model.npz. 'bundled' copies the
released model unchanged (the control of the one-sample rows).

  python -m bench.uncalled4_controls --control syn_control ecoli_dm hp26695_wga spo1_bc01 bundled
"""
import argparse
import gzip
import os
import shutil
import subprocess
from pathlib import Path

import pysam

from . import common as C
from . import settings as ST

POOL_MIN_QS = 7  # comparison reads; training reads need qs >= 10
SD = os.environ.get('BENCH_DATA', 'data')  # benchmark data root
V = os.environ.get('BENCH_CONTROL_DATA', SD)  # unmodified control libraries
UP = os.environ.get('BENCH_SPO1_POD5', os.path.join(SD, 'spo1/pod5'))
# control library: (full BAM with move tables, its pod5, reference, benchmark read sets to exclude)
SOURCES = {
    'syn_control': (f'{V}/syn_control/moves.bam', f'{SD}/synthetic_all5mers_r10.4.1_ont_open_data/pod5_files/control_rep1.pod5',
                    'syn_control', ['syn_control']),
    'ecoli_dm': (f'{SD}/ecoli_k12_damdcm_r10.4.1_ontbasemod_2024/basecalled/reads.bam',
                 f'{SD}/ecoli_k12_damdcm_r10.4.1_ontbasemod_2024/pod5_files/Ecoli_DM_5kHz.pod5',
                 'ecoli_dm_wreg', ['ecoli_dm_w1m', 'ecoli_dm_wreg']),
    'hp26695_wga': (f'{SD}/hpylori_26695_wga_r10.4.1_ontbasemod_2024/basecalled/reads.bam',
                    f'{SD}/hpylori_26695_wga_r10.4.1_ontbasemod_2024/pod5_files',
                    'hp26695_wga_reg', ['hp26695_wga', 'hp26695_wga_reg']),
    'spo1_bc01': (f'{V}/spo1_bc01/reads.sorted.bam', f'{UP}/barcode01.pod5', 'spo1_bc01_reg', ['spo1_bc01', 'spo1_bc01_reg']),
}


def bundled_npz():
    u4 = Path(ST.need('UNCALLED4', 'Uncalled4')).resolve()
    hits = list((u4.parent.parent / 'lib').glob('python3*/site-packages/uncalled4/models/dna_r10.4.1_400bps_9mer.npz'))
    if not hits:
        raise SystemExit(f'released R10.4.1 model not found next to {u4}')
    return hits[0]


def train(lib, n_reads, iters):
    out = Path(ST.need('UNCALLED4_MODELS', 'Uncalled4 control models')) / lib
    out.mkdir(parents=True, exist_ok=True)
    if lib == 'bundled':
        if (out / 'model.npz').exists():
            print(f'{lib}: exists'); return
        shutil.copy(bundled_npz(), out / 'model.npz.tmp'); os.replace(out / 'model.npz.tmp', out / 'model.npz')
        (out / 'PROVENANCE.txt').write_text(f'copy of {bundled_npz()}\n'); print(f'{lib}: copied'); return
    bam, pod5, ref_sample, excl_sets = SOURCES[lib]
    excl = set()
    for s in excl_sets:
        excl |= set(gzip.open(ST.DATA_DIR / 'read_ids' / f'{s}.txt.gz', 'rt').read().split())
    from .holdout import _region_for
    reg = _region_for(ref_sample) or {}  # the held-out region the region benchmark scores (none for the oligos)
    ok, inreg = [], []  # inreg: (read, qs) of unused reads over the held-out region, qs >= POOL_MIN_QS
    with pysam.AlignmentFile(bam, check_sq=False) as f:
        for r in f.fetch(until_eof=True):
            if r.is_unmapped or r.is_secondary or r.is_supplementary or r.query_name in excl or not r.has_tag('mv'):
                continue
            q = float(r.get_tag('qs')) if r.has_tag('qs') else 99.0
            over = any(a < r.reference_end and r.reference_start < b for a, b in reg.get(r.reference_name, ()))
            if over and q >= POOL_MIN_QS:
                inreg.append((r.query_name, q))
            if q >= 10 and not over:
                ok.append(r.query_name)
  # the held-out rule: the pore model learns only from reads outside the held-out region (like UniMeth-FT); the
  # unused reads over the region are kept apart as the comparison sample at the test sites (pool.*), never trained
  # on -- qs >= 10 first, then qs 7-10 to fill (SPO1 has too few unused qs >= 10 reads over its window)
    inq = {}
    for n, q in inreg:
        inq[n] = max(q, inq.get(n, 0))
    train_ids = set(sorted(set(ok), key=C.read_rank)[:n_reads])
    pool_ids = set(sorted(inq, key=lambda n: (inq[n] < 10, C.read_rank(n)))[:n_reads])
    print(f'{lib}: {len(train_ids):,} training reads outside the held-out region, {len(pool_ids):,} comparison reads over it '
          f'({sum(inq[n] < 10 for n in pool_ids):,} with qs 7-10)', flush=True)
    rec = '--recursive ' if Path(pod5).is_dir() else ''
    have_model = (out / 'model.npz').exists()
    for name, keep in (('reads', train_ids), ('pool', pool_ids)):
        if name == 'reads' and have_model:
            continue
        txt = '\n'.join(sorted(keep)) + '\n'
        if not (out / f'{name}_ids.txt').exists() or (out / f'{name}_ids.txt').read_text() != txt:
            (out / f'{name}.pod5').unlink(missing_ok=True)
            for a in out.glob(f'{name}_align_m*.tsv*'):  # alignments of the old read set
                a.unlink()
        (out / f'{name}_ids.txt').write_text(txt)
        with pysam.AlignmentFile(bam, check_sq=False) as f, pysam.AlignmentFile(str(out / f'{name}.unsorted.bam'), 'wb', template=f) as g:
            for r in f.fetch(until_eof=True):
                if r.query_name in keep and not (r.is_secondary or r.is_supplementary or r.is_unmapped):
                    g.write(r)
        pysam.sort('-@', '4', '-o', str(out / f'{name}.bam'), str(out / f'{name}.unsorted.bam')); os.remove(out / f'{name}.unsorted.bam')
        pysam.index(str(out / f'{name}.bam'))
        if not (out / f'{name}.pod5').exists():
            subprocess.run(f"{ST.get('POD5', 'pod5')} filter {pod5} {rec}--ids {out}/{name}_ids.txt --missing-ok "
                           f"--output {out}/{name}.pod5.tmp && mv {out}/{name}.pod5.tmp {out}/{name}.pod5", shell=True, check=True)
    (out / 'read_ids.txt').unlink(missing_ok=True)
    if have_model:
        print(f'{lib}: {out / "model.npz"} exists (comparison reads rebuilt)'); return
    ref = C.samples()[ref_sample]['ref']
    cmd = (f"{ST.need('UNCALLED4', 'Uncalled4')} train -m {bundled_npz()} -k 9 -i {iters} --ref {ref} --reads {out}/reads.pod5 "
           f"--bam-in {out}/reads.bam --out-dir {out}/train --min-aln-length 50 -p {min(8, len(os.sched_getaffinity(0)))}")
    shutil.rmtree(out / 'train', ignore_errors=True)  # a requeued job restarts training cleanly
    print(cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)
    last = sorted((out / 'train').glob('it*.model.npz'), key=lambda p: int(p.name[2:].split('.')[0]))
    if not last:
        raise SystemExit(f'{lib}: no it*.model.npz in {out / "train"}: {sorted(p.name for p in (out / "train").iterdir())}')
    shutil.copy(last[-1], out / 'model.npz.tmp'); os.replace(out / 'model.npz.tmp', out / 'model.npz')
    (out / 'PROVENANCE.txt').write_text(f'{cmd}\n{len(train_ids):,} training reads of {bam} outside the held-out region '
                                        f'{reg}; comparison sample pool.bam: {len(pool_ids):,} reads over it; excluding benchmark read sets '
                                        f'{", ".join(excl_sets)}; primary, qs>=10, lowest read_rank); model = {last[-1].name}\n')
    print(f'{lib}: {len(train_ids):,} training reads -> {out / "model.npz"}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--control', nargs='+', required=True, choices=sorted(SOURCES) + ['bundled'])
    ap.add_argument('--reads', type=int, default=20000)
    ap.add_argument('--iterations', type=int, default=2)
    a = ap.parse_args()
    for lib in a.control:
        train(lib, a.reads, a.iterations)


if __name__ == '__main__':
    main()
