#!/usr/bin/env python3
"""Supplementary Table S10: SPO1 native-against-PCR typing by models that did not train on these reads, against Dorado.

  python spo1_heldout_table.py collect RUN_DIR   per-window AUROC of every model from spo1_heldout_models.sh
                                                 -> paper/data/work/read_origin/spo1_heldout.tsv
  python spo1_heldout_table.py                   the table from the shipped per-window AUROCs

Per window the AUROC is the median over the coverage-matched repeats of the oriented AUROC, max(AUC, 1 - AUC), as for
the paper's SPO1 panel. Dorado (the cytosine model at 200 random C sites per window) comes from read_origin/auroc.tsv.
The three windows bc02_q00-q02 lie inside the region that the region-holdout model never saw.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

DATA = Path(__file__).resolve().parents[1] / 'data' / 'work' / 'read_origin'
MODELS = [('released', 'Released (all data)'), ('region', 'Region holdout'), ('loco5hmU', 'Without 5hmU'),
          ('noSPO1', 'Without SPO1')]
IN_REGION = ['bc02_q00', 'bc02_q01', 'bc02_q02']


def collect(run_dir):
    rows = []
    for m, _ in MODELS:
        for f in sorted((Path(run_dir) / m).glob('bc*/covmatched.tsv')):
            t = pd.read_csv(f, sep='\t')
            t['a'] = np.maximum(t.auroc_vs_sample, 1 - t.auroc_vs_sample)
            g = t.groupby('window').agg(auroc=('a', 'median'), n_reads=('n_reads', 'median'))
            rows += [dict(model=m, group=f'{f.parent.name}_{w}', n_reads=x.n_reads, auroc=x.auroc) for w, x in g.iterrows()]
    out = DATA / 'spo1_heldout.tsv'
    pd.DataFrame(rows).to_csv(out, sep='\t', index=False, float_format='%.6f')
    print(f'-> {out}')


def table():
    t = pd.read_csv(DATA / 'spo1_heldout.tsv', sep='\t').pivot(index='group', columns='model', values='auroc')
    a = pd.read_csv(DATA / 'auroc.tsv', sep='\t')
    dor = a[(a.dataset == 'SPO1') & (a.method == 'dorado')].set_index('group').auroc
    rows = []
    for m, label in MODELS:
        d = (t[m] - dor).dropna()
        rows.append(dict(model=label, median=t[m].median(), ahead=f'{(d > 0).sum()}/{len(d)}',
                         p=wilcoxon(d[d != 0]).pvalue, **{w: t.loc[w, m] for w in IN_REGION}))
    rows.append(dict(model='Dorado (cytosine model)', median=dor.median(), ahead='', p=np.nan, **{w: dor[w] for w in IN_REGION}))
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == 'collect':
        collect(sys.argv[2])
    elif len(sys.argv) == 1:
        table()
    else:
        sys.exit(__doc__)
