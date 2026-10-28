#!/usr/bin/env python3
"""Direction check of the label-free partition against methylation measured on the other flow-cell pair of the same lines."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

# modkit pileup bedMethyl: chrom start end code score strand ... then
# Nvalid Npct Nmod Ncanon ...  (col 10 = percent modified, col 11 = Nmod)
BED_COLS = dict(chrom=0, start=1, strand=5, nvalid=9, pct=10)


def load_bed(path, min_cov):
    """chrom,pos -> percent modified, forward strand only."""
    d = pd.read_csv(path, sep='\t', header=None,
                    usecols=[BED_COLS['chrom'], BED_COLS['start'],
                             BED_COLS['strand'], BED_COLS['nvalid'],
                             BED_COLS['pct']],
                    names=['chrom', 'start', 'strand', 'nvalid', 'pct'])
    d = d[(d.strand == '+') & (d.nvalid >= min_cov)]
    return dict(zip(d.start.to_numpy(), d.pct.to_numpy()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reads', required=True, help='discovered_reads.tsv')
    ap.add_argument('--meth-npz', required=True,
                    help='dorado_feats.py --mode single npz: P(5mC) per (read, CpG)')
    ap.add_argument('--read-meta', required=True, help='reads.tsv with flowcell')
    ap.add_argument('--anchor-tumor', required=True, help='pair A tumour bedMethyl')
    ap.add_argument('--anchor-normal', required=True, help='pair A normal bedMethyl')
    ap.add_argument('--test-flowcells', nargs='+', required=True,
                    help='the pair B flowcells whose reads are scored')
    ap.add_argument('--min-cov', type=int, default=10)
    ap.add_argument('--min-delta', type=float, default=25.0)
    ap.add_argument('--min-reads', type=int, default=12)
    ap.add_argument('--min-anchors', type=int, default=5)
    ap.add_argument('--shuffle-signs', type=int, default=None,
                    help='PERMUTATION CONTROL. Randomise each anchor\'s sign with '
                         'this seed, keeping every other quantity identical. The '
                         'anchors then encode no biological direction, so the '
                         'orientation must degrade to a coin flip. If it does '
                         'not, the test is leaking the label somewhere and the '
                         'real result means nothing.')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    tum = load_bed(a.anchor_tumor, a.min_cov)
    nor = load_bed(a.anchor_normal, a.min_cov)
    delta = {p: tum[p] - nor[p] for p in tum.keys() & nor.keys()}
    if a.shuffle_signs is not None:
        rng = np.random.default_rng(a.shuffle_signs)
        flip = rng.choice([-1.0, 1.0], size=len(delta))
        delta = {p: v * f for (p, v), f in zip(delta.items(), flip)}
        print(f'PERMUTATION CONTROL: anchor signs shuffled with seed '
              f'{a.shuffle_signs}; magnitudes untouched')
    print(f'anchors available: {len(delta):,} CpGs covered in both pair-A arms')

    d = np.load(a.meth_npz, allow_pickle=True)
  # emb column 0 is P(5mC); site_i indexes site_pos
    p5 = d['emb'][:, 0].astype(np.float64)
    m_read = d['read_id']
    m_pos = d['site_pos'][d['site_i']]
    m_win = d['window']

    reads = pd.read_csv(a.reads, sep='\t')
    rm = pd.read_csv(a.read_meta, sep='\t')
    fc = dict(zip(rm.read_id, rm.flowcell))
    test_fc = set(a.test_flowcells)

    rows = []
    for w, sub in reads.groupby('group'):
        sel = m_win == w
        if not sel.any():
            continue
        pos_w = m_pos[sel]
        anch = {p: delta[p] for p in np.unique(pos_w)
                if p in delta and abs(delta[p]) >= a.min_delta}
        if len(anch) < a.min_anchors:
            rows.append(dict(window=w, n_anchor=len(anch), n_reads=np.nan,
                             corr=np.nan, directed_auroc=np.nan))
            continue

  # b(read): mean over covered anchors of sign(delta) * P(5mC)
        keep = sel.copy()
        keep[sel] = np.isin(pos_w, list(anch))
        sgn = np.array([np.sign(anch[p]) for p in m_pos[keep]])
        b = pd.DataFrame(dict(read_id=m_read[keep],
                              v=sgn * p5[keep])).groupby('read_id').v.mean()

        sub = sub[sub.read_id.map(lambda r: fc.get(r) in test_fc)]
        sub = sub[sub.read_id.isin(b.index)]
        if len(sub) < a.min_reads or sub.is_pos.nunique() < 2:
            rows.append(dict(window=w, n_anchor=len(anch), n_reads=len(sub),
                             corr=np.nan, directed_auroc=np.nan))
            continue

        s = sub.score.to_numpy()
        bb = b.loc[sub.read_id].to_numpy()
        c = np.corrcoef(s, bb)[0, 1] if np.std(s) > 0 and np.std(bb) > 0 else np.nan
  # orientation comes ONLY from the held-out anchors, never from is_pos
        oriented = s * (1.0 if (np.isnan(c) or c >= 0) else -1.0)
        auc = roc_auc_score(sub.is_pos.to_numpy(), oriented)
        rows.append(dict(window=w, n_anchor=len(anch), n_reads=len(sub),
                         corr=c, directed_auroc=auc))

    res = pd.DataFrame(rows)
    res.to_csv(a.out, sep='\t', index=False)
    ok = res.dropna(subset=['directed_auroc'])
    if len(ok):
        k = int((ok.directed_auroc > 0.5).sum())
        p = binomtest(k, len(ok), 0.5, alternative='greater').pvalue
        print(res.round(3).to_string(index=False))
        print(f'\nwindows scored: {len(ok)}   correctly oriented: {k}   '
              f'binomial p={p:.3g}')
        print(f'median directed AUROC: {ok.directed_auroc.median():.3f}  '
              f'(0.5 = orientation is a coin flip)')
    else:
        print('no window had enough held-out anchors and reads')


if __name__ == '__main__':
    main()
