#!/usr/bin/env python3
"""Library invariance with and without the curriculum (Supplementary Section S4).

For every two-library benchmark row, the library fingerprint is the AUROC of native-library background positions
against the same positions in the unmodified control (positions with no known modification); invariance is
1 - 2|AUROC - 0.5|. This scores the benchmark's library-control images with any set of checkpoints, e.g. the final
recipe and the no-curriculum ablation over three seeds each (ablation checkpoints are available on request).

Run from test/benchmark with the benchmark settings, after `python -m bench.library_control cands` and
`python -m bench.library_control rawmod` (which write the background positions and their pileup images):

  cd test/benchmark
  BENCH_HOLDOUT=region python ../../paper/supp_ablations/curriculum_library_invariance.py \
      --checkpoint curriculum_s42=PATH ... --checkpoint no_curriculum_s42=PATH ... --out-dir OUT
  python ../../paper/supp_ablations/curriculum_library_invariance.py --summary [FINGERPRINT_TSV]

The shipped result is paper/data/work/curriculum_libinv/fingerprint.tsv.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
SHIPPED = REPO / 'paper' / 'data' / 'work' / 'curriculum_libinv' / 'fingerprint.tsv'


def load_scores(d, smp):
    parts = []
    for s, sg in (('plus', '+'), ('minus', '-')):
        f = d / f'{smp}_{s}_scores.tsv.gz'
        if f.exists():
            x = pd.read_csv(f, sep='\t', dtype={'contig': str})
            x = x.groupby(['contig', 'pos']).score.max().reset_index()
            parts.append(x.assign(strand=sg))
    return pd.concat(parts, ignore_index=True)


def run(checkpoints, out_dir, images_from):
    sys.path.insert(0, str(REPO / 'test' / 'benchmark'))
    from bench import common as C, models
    from bench import library_control as LC
    from bench.paper_table import RAWMOD_COLS
    lc_dir = f'{models.subtool_name(images_from or RAWMOD_COLS[0])}_lc'  # default: the Table 1 column (BENCH_RAWMOD)
    rows = LC.two_library_rows()
    samples = sorted({r[k] for r in rows for k in ('pos_sample', 'neg_sample')})
    out = []
    for spec in checkpoints:
        name, ck = spec.split('=', 1)
        d = Path(out_dir) / name
        d.mkdir(parents=True, exist_ok=True)
        for smp in samples:
            for s in ('plus', 'minus'):
                h5 = C.WORK / smp / lc_dir / f'features_{s}.h5'
                if h5.exists() and not (d / f'{smp}_{s}_scores.tsv.gz').exists():
                    subprocess.run([sys.executable, str(REPO / 'scripts/evaluate/score_genome.py'), '--h5', str(h5),
                                    '--dataset', f'{smp}_{s}', '--checkpoint', ck, '--out-dir', str(d)], check=True,
                                   env={**os.environ, 'PILEUP_MASK_BASES': '0'})
        for r in rows:
            p = LC.OUT / f"bg_{r['row']}.tsv"
            if not p.exists():  # no background positions (SPO1: every thymine of the native phage carries 5hmU)
                continue
            bg = pd.read_csv(p, sep='\t', dtype={'contig': str})
            m = (bg.merge(load_scores(d, r['pos_sample']), on=['contig', 'pos', 'strand'])
                 .merge(load_scores(d, r['neg_sample']), on=['contig', 'pos', 'strand'], suffixes=('_n', '_c')))
            auc = roc_auc_score(np.r_[np.ones(len(m)), np.zeros(len(m))], np.r_[m.score_n, m.score_c])
            out.append(dict(model=name, row=r['row'], n=len(m), fingerprint_auroc=auc, invariance=1 - 2 * abs(auc - 0.5)))
    t = pd.DataFrame(out)
    p = Path(out_dir) / 'fingerprint.tsv'
    t.to_csv(p, sep='\t', index=False, float_format='%.4f')
    print(f'-> {p}')
    summary(p)


def summary(path):
    t = pd.read_csv(path, sep='\t')
    s = t.groupby('model').agg(mean_fingerprint=('fingerprint_auroc', 'mean'), mean_invariance=('invariance', 'mean'),
                               min_invariance=('invariance', 'min'))
    print(s.round(3).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', action='append', default=[], help='NAME=PATH, repeatable')
    ap.add_argument('--out-dir', default='curriculum_libinv')
    ap.add_argument('--images-from', default='', help='config/rawmod_models.tsv entry whose library-control images to score '
                                                       '(default: the RawMod column, BENCH_RAWMOD)')
    ap.add_argument('--summary', nargs='?', const=str(SHIPPED), default=None, help='print the summary of a fingerprint table')
    a = ap.parse_args()
    if a.summary is not None:
        summary(a.summary)
    elif a.checkpoint:
        run(a.checkpoint, a.out_dir, a.images_from)
    else:
        ap.print_help()
