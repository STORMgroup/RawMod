#!/usr/bin/env python3
"""Rebuild region test sets from dumped per-site scores (dump_region_scores.py) and compare definitions.

Test-set definitions (all negatives, as in the paper's 'all_neg' numbers):
  orig      core members (not BENCH), every site in the region-only files     == score_holdout_region.py
  inreg     orig, restricted to sites inside the withheld region               (drops oligo training constructs)
  inreg+ext inreg plus the extra organisms that have both labels in the region (E. coli, A. thaliana, human);
            for loco_X only organisms that carry X (they are removed from loco_X training entirely)
Fold rules follow region_test_idx: loco_X = chem-X positives + negatives on X's base from CHEM_ORGS[X].

  python analyze.py SCORES_DIR OUT_PREFIX
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CHEM_BASES = {'5hmU': 'T', '6mA': 'A', '4mC': 'C', '5mC': 'C', '5hmC': 'C'}
CHEM_ORGS = {'5hmU': ('SPO1::',), '4mC': ('HP::',), '6mA': ('ONT::', 'SPO1::', 'HP::'),
             '5mC': ('ONT::', 'SPO1::'), '5hmC': ('ONT::', 'SPO1::')}
EXT_GROUP = {'Ecoli_WT_5kHz': 'E. coli', 'Ecoli_DM_5kHz': 'E. coli', 'Ecoli_DM_MSssI_5kHz': 'E. coli',
             'arabidopsis': 'A. thaliana', 'hg001': 'human', 'hg002': 'human'}
EXT_CHEMS = {'E. coli': {'6mA', '5mC'}, 'A. thaliana': {'5mC'}, 'human': {'5mC'}}


def dataset(member):
    return member.split('::', 1)[1].split('|')[0]


def test_mask(s, fold, variant):
    core = ~s.member.str.startswith('BENCH::')
    pos = s.label == 1
    if fold == 'mixed':
        m = core.copy()
        if variant != 'orig':
            m &= s.in_region == 1
        if variant == 'inreg+ext':
            m |= s.ext.notna() & (s.in_region == 1)
        return m
    x = fold[len('loco_'):]
    on_base = s.refbase == CHEM_BASES[x]
    in_org = s.org.isin(CHEM_ORGS[x])
    m = core & ((pos & (s.chem == x)) | (~pos & in_org & on_base))
    if variant != 'orig':
        m &= s.in_region == 1
    if variant == 'inreg+ext':
        carries = s.ext.map(lambda g: isinstance(g, str) and x in EXT_CHEMS[g])
        m |= carries & (s.in_region == 1) & ((pos & (s.chem == x)) | (~pos & on_base))
    return m


def main(scores_dir, out_prefix):
    d = Path(scores_dir)
    s = pd.read_csv(d / 'sites.tsv.gz', sep='\t', keep_default_na=False, dtype={'chem': str})
    s['dataset'] = s.member.map(dataset)
    s['ext'] = s.dataset.map(EXT_GROUP)
    rows, per = [], []
    for f in sorted(d.glob('*.tsv.gz')):
        tag = f.name[:-len('.tsv.gz')]
        if tag == 'sites':
            continue
        fold, seed = tag.rsplit('_s', 1)
        sc = pd.read_csv(f, sep='\t').set_index('site').score.reindex(s.site).to_numpy()
        for v in ('orig', 'inreg', 'inreg+ext'):
            m = test_mask(s, fold, v).to_numpy()
            y = s.label.to_numpy()[m]
            rows.append(dict(fold=fold, seed=seed, variant=v, n_pos=int(y.sum()), n_neg=int((1 - y).sum()),
                             auroc=roc_auc_score(y, sc[m]) if 0 < y.sum() < len(y) else np.nan))
            if v == 'inreg+ext':
                t = s[m].assign(score=sc[m])
                for ds, g in t.groupby('dataset'):
                    yy = g.label.to_numpy()
                    per.append(dict(fold=fold, seed=seed, dataset=ds, n_pos=int(yy.sum()), n_neg=int((1 - yy).sum()),
                                    auroc=roc_auc_score(yy, g.score) if 0 < yy.sum() < len(yy) else np.nan))
    r = pd.DataFrame(rows)
    r.to_csv(f'{out_prefix}_auroc.tsv', sep='\t', index=False)
    pd.DataFrame(per).to_csv(f'{out_prefix}_per_dataset.tsv', sep='\t', index=False)
    piv = r.pivot_table(index=['fold', 'variant'], columns='seed', values='auroc')
    piv['mean'] = piv.mean(axis=1)
    cnt = r[r.seed == r.seed.min()].set_index(['fold', 'variant'])[['n_pos', 'n_neg']]
    print(piv.join(cnt).round(3).to_string())


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
