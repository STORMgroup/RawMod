#!/usr/bin/env python3
"""Rerun of the consensus with the two samples matched on how many sites each read covers, for sample pairs with different read-length distributions (SPO1 native against PCR)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, roc_auc_score

from discover import consensus, site_split


def matched_reads(ns: pd.DataFrame, bin_w: int, seed: int) -> list:
    """Coverage-stratified 1:1 match. ns has columns read_id, group, n_sites."""
    rng = np.random.default_rng(seed)
    ns = ns.assign(b=(ns.n_sites // bin_w).astype(int))
    keep = []
    for _, sub in ns.groupby('b'):
        by_g = {g: s.read_id.to_numpy() for g, s in sub.groupby('group')}
        if len(by_g) != 2:
            continue  # bin occupied by one arm only
        k = min(len(v) for v in by_g.values())
        for v in by_g.values():
            keep.extend(rng.choice(v, size=k, replace=False).tolist())
    return keep


def run(emb, rid, si, grp, reads, min_reads, method, wmask):
  # wmask is essential: site_i is indexed globally across windows, and a read
  # long enough to span two windows would otherwise pull the neighbouring
  # window's sites into this window's consensus.
    sel = np.isin(rid, list(reads)) & wmask
    splits = {}
    for s in np.unique(si[sel]):
        m = sel & (si == s)
        if m.sum() < min_reads:
            continue
        sp = site_split(emb[m], method=method)
        if sp is not None:
            splits[int(s)] = (list(rid[m]), sp)
    if len(splits) < 3:
        return None
    r = sorted({x for v in splits.values() for x in v[0]})
    score, part, load, lam = consensus(splits, r)
    return r, score, part, len(splits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pos-group', required=True)
    ap.add_argument('--read-lengths', default=None,
                    help='TSV read_id<TAB>len<TAB>mapq — reported, never matched on')
    ap.add_argument('--bin-width', type=int, default=25)
    ap.add_argument('--min-reads', type=int, default=20)
    ap.add_argument('--split-method', default='pc1', choices=['gmm', 'pc1'])
    ap.add_argument('--n-rep', type=int, default=5,
                    help='independent matched draws; the match is random, so a '
                         'single draw could be lucky')
    a = ap.parse_args()

    d = np.load(a.npz, allow_pickle=True)
    emb, rid, si, grp, win = (d['emb'], d['read_id'], d['site_i'],
                              d['group'], d['window'])
    L = {}
    if a.read_lengths:
        rl = pd.read_csv(a.read_lengths, sep='\t', header=None,
                         names=['read_id', 'len', 'mapq'])
        L = dict(zip(rl.read_id, rl.len))

    rows = []
    for g in sorted(set(win)):
        k = win == g
        ns = (pd.DataFrame({'read_id': rid[k], 'group': grp[k]})
              .groupby(['read_id', 'group']).size()
              .reset_index(name='n_sites'))
        for rep in range(a.n_rep):
            reads = matched_reads(ns, a.bin_width, seed=rep)
            out = run(emb, rid, si, grp, set(reads), a.min_reads,
                      a.split_method, k)
            if out is None:
                continue
            r, score, part, nsites = out
            r2g = dict(zip(rid, grp))
            y = np.array([1 if str(r2g[x]) == a.pos_group else 0 for x in r])
            if len(set(y)) < 2:
                continue
            auc = roc_auc_score(y, score)
            rec = dict(window=g, rep=rep, n_reads=len(r), n_sites=nsites,
                       frac_pos=float(y.mean()),
                       ari_vs_sample=adjusted_rand_score(y, part),
                       auroc_vs_sample=max(auc, 1 - auc))
  # the two diagnostics that failed before matching
            cnt = pd.Series([x for v in [r] for x in v]).map(
                dict(zip(ns.read_id, ns.n_sites)))
            nsv = cnt.to_numpy(float)
            acov = roc_auc_score((nsv >= np.median(nsv)).astype(int), score)
            rec['auroc_vs_coverage'] = max(acov, 1 - acov)
            if L:
                ln = np.array([L.get(x, np.nan) for x in r], float)
                ok = ~np.isnan(ln)
                alen = roc_auc_score(
                    (ln[ok] >= np.median(ln[ok])).astype(int), score[ok])
                rec['auroc_vs_length'] = max(alen, 1 - alen)
  # absolute value: the eigenvector's sign is arbitrary, so the
  # raw correlation flips between reps and its sign means nothing
                neg = ok & (y == 0)
                rec['within_neg_abscorr_len'] = float(abs(
                    np.corrcoef(score[neg], ln[neg])[0, 1])) if neg.sum() > 8 else np.nan
            rows.append(rec)

    res = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out, sep='\t', index=False)
    print(res.round(3).to_string(index=False))
    if len(res):
        print('\nmedian over reps, by window:')
        print(res.groupby('window')[[c for c in res.columns
                                     if c.startswith(('ari', 'auroc', 'within'))
                                     ]].median().round(3).to_string())
    print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
