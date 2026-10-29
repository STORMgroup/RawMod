#!/usr/bin/env python3
"""Supplementary threshold table: operating points at one shared threshold, and the pooled AUROC with a bootstrap CI.

Each tool scores a site with its modification-agnostic score (the maximum over its models), as in the agnostic table.
One threshold per tool is set at 1% false positives on the pooled negatives of the agnostic table (at most 1,000
positives and 1,000 negatives per row, seed 0). Per row it reports the true-positive rate at that threshold, and the
true- and false-positive rates at 0.5. The summary rows average over the 15 rows and over the five modification types
(the rows of each type averaged first), and give the worst type. The pooled AUROC has a 95% CI from a cluster bootstrap
over reference positions (500 resamples).

  cd test/benchmark && BENCH_HOLDOUT=region python ../../paper/supp_threshold/fixed_threshold.py compute
        scores the benchmark work tree -> paper/data/work/threshold/{fixed_threshold,pooled_ci}.tsv
  python paper/supp_threshold/fixed_threshold.py
        the LaTeX table from the shipped results -> paper/tables/supp_threshold.tex
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get('RAWMOD_WORK') or REPO / 'paper' / 'data' / 'work') / 'threshold'
TABLES = Path(os.environ.get('RAWMOD_TABLES') or REPO / 'paper' / 'tables')
REPS = int(os.environ.get('CI_REPS', '500'))
COLS = [('rawmod', r'\sysname'), ('dorado', 'Dorado'), ('unimeth_agn_mod_reg', r'UniMeth-Agn$^{\dagger}$')]
ROWS = [('arabidopsis_cpg_chr4', r'\textit{A.~thaliana} CpG'), ('hg002_cpg', 'HG002'), ('rice_cpg', r'\textit{O.~sativa}'),
        ('mouse_cpg', 'Mouse brain'), ('ecoli_mssi_5mC_reg', r'\textit{E.~coli} M.SssI'),
        ('ecoli_dcm_5mC_reg', r'\textit{E.~coli} Dcm'), ('arabidopsis_noncpg_chr4', r'\textit{A.~thaliana} non-CpG'),
        ('syn_5mC', 'ONT 5mC'), ('syn_5hmC', 'ONT 5hmC'), ('ecoli_dam_6mA_reg', r'\textit{E.~coli} Dam'),
        ('anabaena_6mA_reg', r'\textit{Anabaena}'), ('hp26695_6mA_reg', r'\textit{H.~pylori} 6mA'), ('syn_6mA', 'ONT 6mA'),
        ('hp26695_4mC_reg', r'\textit{H.~pylori} 4mC'), ('spo1_5hmU_reg', r'\phage 5hmU')]


def compute():
    sys.path.insert(0, str(REPO / 'test' / 'benchmark'))
    from sklearn.metrics import roc_auc_score, roc_curve
    from bench import common as C, score as Sc
    from bench.paper_table import GROUPS, RAWMOD_COLS, families, family_score

    S, T = C.samples(), C.subtools()
    fam = families(T)
    rows = {r['row']: r for r in Sc.rows_cfg()}
    tools = [RAWMOD_COLS[0], 'dorado', 'unimeth', 'unimeth_agn_mod_reg']
    rng = np.random.default_rng(0)
    cache, data = {}, {}
    for g, members in GROUPS:
        for row, _ in members:
            blocks = Sc.row_sites(rows[row], S, cache)
            y = np.concatenate([b[2] for b in blocks])
            keys = np.concatenate([(row + ':' + b[1]['contig'].astype(str) + ':' + b[1]['pos'].astype(str)).to_numpy()
                                   for b in blocks])
            take = np.concatenate([rng.permutation(np.flatnonzero(y == k))[:1000] for k in (0, 1)])
            sc = {f: s for f in tools if (s := family_score(blocks, fam.get(f) or [f])) is not None}
            data[row] = (y, keys, sc, take, g.split(',')[0])
            print(row, len(y), int(y.sum()), {f: round(roc_auc_score(y, s), 3) for f, s in sc.items()}, flush=True)

    def boot_auroc(y, s, keys):
        uk, inv = np.unique(keys, return_inverse=True)
        order = np.argsort(inv, kind='mergesort')
        starts = np.searchsorted(inv[order], np.arange(len(uk)))
        lens = np.r_[starts[1:], len(order)] - starts
        vals = []
        for _ in range(REPS):
            pick = rng.integers(0, len(uk), len(uk))
            L = lens[pick]
            off = np.repeat(np.r_[0, np.cumsum(L)[:-1]], L)
            idx = order[np.repeat(starts[pick], L) + (np.arange(int(L.sum())) - off)]
            if 0 < y[idx].sum() < len(idx):
                vals.append(roc_auc_score(y[idx], s[idx]))
        return np.percentile(vals, [2.5, 97.5])

    pooled, thr = [], {}
    for f in tools:
        ys, ss, ks = [], [], []
        for y, keys, sc, take, _ in data.values():
            if f in sc:
                ys.append(y[take]); ss.append(sc[f][take]); ks.append(keys[take])
        y, s, k = map(np.concatenate, (ys, ss, ks))
        lo, hi = boot_auroc(y, s, k)
        thr[f] = float(np.quantile(s[y == 0], 0.99))
        pooled.append(dict(tool=f, n_rows=len(ys), n_sites=len(y), auroc=roc_auc_score(y, s), lo=lo, hi=hi,
                           thr_1pct_fpr=thr[f]))

    def at(y, s, t):
        pred = s >= t
        tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        tpr, fpr = tp / max(tp + fn, 1), fp / max(int((y == 0).sum()), 1)
        prec = tp / max(tp + fp, 1)
        return tpr, fpr, prec, 2 * prec * tpr / max(prec + tpr, 1e-9)

    out = []
    for row, (y, keys, sc, take, mod) in data.items():
        for f, s in sc.items():
            rec = dict(row=row, mod=mod, tool=f, n_pos=int(y.sum()), n_neg=int((y == 0).sum()), auroc=roc_auc_score(y, s))
            for name, t in (('t05', 0.5), ('tpool', thr[f])):
                tpr, fpr, prec, f1 = at(y, s, t)
                rec.update({f'{name}_tpr': tpr, f'{name}_fpr': fpr, f'{name}_prec': prec, f'{name}_f1': f1})
            fpr_c, tpr_c, _ = roc_curve(y, s)
            rec['tpr_at_1pct_fpr_row'] = float(tpr_c[fpr_c <= 0.01].max()) if (fpr_c <= 0.01).any() else np.nan
            out.append(rec)
    DATA.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pooled).to_csv(DATA / 'pooled_ci.tsv', sep='\t', index=False, float_format='%.4f')
    pd.DataFrame(out).to_csv(DATA / 'fixed_threshold.tsv', sep='\t', index=False, float_format='%.4f')
    print(f'-> {DATA}')


def f2(x):
    return f'{float(x):.2f}'


def table():
    D = pd.read_csv(DATA / 'fixed_threshold.tsv', sep='\t')
    thr = pd.read_csv(DATA / 'pooled_ci.tsv', sep='\t').set_index('tool').thr_1pct_fpr
    tools = [t for t, _ in COLS]
    cols = ('tpool_tpr', 't05_tpr', 't05_fpr')
    V = {c: D.pivot(index='row', columns='tool', values=c)[tools] for c in cols}
    M = {c: D.groupby(['mod', 'tool'])[c].mean().unstack()[tools] for c in cols}
    worst = {'tpool_tpr': 'min', 't05_tpr': 'min', 't05_fpr': 'max'}  # the worst type: lowest TPR, highest FPR
    line = lambda lab, vals: lab + ' & ' + ' & '.join(vals) + r' \\'
    body = [line(lab, [f2(V[c].loc[row, t]) for c in cols for t in tools]) for row, lab in ROWS]
    summary = [line(r'\textbf{Mean over rows}', [f2(V[c][t].mean()) for c in cols for t in tools]),
               line('Mean over the five types', [f2(M[c][t].mean()) for c in cols for t in tools]),
               line('Worst type', [f2(getattr(M[c][t], worst[c])()) for c in cols for t in tools]),
               r'Threshold (1\% FPR) & ' + ' & '.join(f'{thr[t]:.3f}' for t in tools) + ' & & & & & & \\\\']
    head = ' & '.join(n for _, n in COLS)
    tex = '\n'.join([
        r'\setlength{\tabcolsep}{3pt}%', r'\begin{tabular}{@{}l ccc ccc ccc@{}}', r'\toprule',
        r'& \multicolumn{3}{c}{\textbf{TPR at the shared 1\% FPR threshold}} & \multicolumn{3}{c}{\textbf{TPR at 0.5}} '
        r'& \multicolumn{3}{c}{\textbf{FPR at 0.5}} \\',
        r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}',
        r'\textbf{Row} & ' + ' & '.join([head] * 3) + r' \\', r'\midrule', *body, r'\midrule', *summary,
        r'\bottomrule', r'\end{tabular}', ''])
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / 'supp_threshold.tex').write_text(tex)
    print(tex)
    print(f'-> {TABLES / "supp_threshold.tex"}')


if __name__ == '__main__':
    compute() if sys.argv[1:] == ['compute'] else table()
