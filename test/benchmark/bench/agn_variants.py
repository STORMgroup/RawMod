#!/usr/bin/env python3
"""The two UniMeth-Agn variants side by side (Supplementary Section S9.4): UniMeth's own context tokens (CpG, CHG, CHH at
C, m6A at A, a new 5hmU token at T; subtool unimeth_agn_reg) against one new [MOD] token at every C, A and T
(unimeth_agn_mod_reg, the Table 1 column), next to the RawMod column. Per-row AUROC, mean over rows, mean over the five
modifications, worst modification, and the pooled AUROC with one threshold (1,000 sites of each class per row, as in
bench.paper_table).

  BENCH_HOLDOUT=region python -m bench.agn_variants
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import common as C
from . import score as Sc
from .models import subtool_name
from .paper_table import GROUPS, RAWMOD_COLS, family_score

VARIANTS = ['unimeth_agn_reg', 'unimeth_agn_mod_reg']


def main():
    rawmod = RAWMOD_COLS[0]  # the Table 1 column unless BENCH_RAWMOD chooses another
    tools = VARIANTS + [subtool_name(rawmod)]
    rows = {r['row']: r for r in Sc.rows_cfg()}
    S, cache = C.samples(), {}
    rng = np.random.default_rng(0)
    per, pool_y, pool_s = [], [], {t: [] for t in tools}
    for g, members in GROUPS:
        for row, _ in members:
            blocks = Sc.row_sites(rows[row], S, cache)
            y = np.concatenate([b[2] for b in blocks])
            take = np.concatenate([rng.permutation(np.flatnonzero(y == k))[:1000] for k in (0, 1)])
            pool_y.append(y[take])
            for t in tools:
                sc = family_score(blocks, [t])
                per.append((g.split(',')[0], row, t, roc_auc_score(y, sc)))
                pool_s[t].append(sc[take])
    d = pd.DataFrame(per, columns=['mark', 'row', 'tool', 'auroc'])
    print(d.pivot(index='row', columns='tool', values='auroc').round(4).to_string())
    yy = np.concatenate(pool_y)
    summ = []
    for t in tools:
        x = d[d.tool == t]
        pm = x.groupby('mark').auroc.mean()
        summ.append(dict(tool=t, mean_rows=x.auroc.mean(), per_mod=pm.mean(), worst=pm.min(),
                         pooled=roc_auc_score(yy, np.concatenate(pool_s[t]))))
    s = pd.DataFrame(summ)
    print(s.round(4).to_string(index=False))
    out = C.WORK / f'agn_variants_{C.PAPER_TAG}.tsv'
    d.to_csv(out, sep='\t', index=False, float_format='%.4f')
    print(f'-> {out}')


if __name__ == '__main__':
    main()
