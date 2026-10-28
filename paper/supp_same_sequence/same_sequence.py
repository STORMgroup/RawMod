"""Supplementary table: different modifications on the same sequence under one threshold per tool (E. coli wild type with 6mA and 5mC, plus the M.SssI library; ONT oligonucleotides with 5mC, 5hmC and 6mA), from the benchmark per-site scores. Run from test/benchmark with BENCH_WORK and BENCH_HOLDOUT set."""
import os as _os
from pathlib import Path as _Path
REPO = str(_Path(__file__).resolve().parents[2])
RAWMOD_RESULTS = _os.environ.get('RAWMOD_RESULTS') or f'{REPO}/paper/data/runs'  # training runs; shipped metrics by default
RAWMOD_DATA = _os.environ.get('RAWMOD_DATA', 'data')  # datasets (not shipped: set this to score raw signal)
RAWMOD_WORK = _os.environ.get('RAWMOD_WORK') or f'{REPO}/paper/data/work'  # outputs of the paper scripts; shipped by default

import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, f'{REPO}/test/benchmark')
from bench import common as C, score as Sc
from sklearn.metrics import roc_auc_score
OUT = f'{RAWMOD_WORK}/same_sequence'
RAWMOD = os.environ.get('RAWMOD_SUBTOOL', 'rawmod')
FAM = {'RawMod': [RAWMOD],
       'Dorado 6mA': ['dorado_6mA'], 'Dorado 4mC+5mC': ['dorado_4mC_5mC'], 'Dorado 5mC+5hmC': ['dorado_5mC_5hmC'],
       'Dorado 5mCG+5hmCG': ['dorado_5mCG_5hmCG'],
       'Dorado, best single model': None, 'Dorado, all models (max)': ['dorado_6mA', 'dorado_4mC_5mC', 'dorado_5mC_5hmC', 'dorado_5mCG_5hmCG'],
       'UniMeth 5mC': ['unimeth_5mC'], 'UniMeth 6mA': ['unimeth_6mA'], 'UniMeth, both models (max)': ['unimeth_5mC', 'unimeth_6mA'],
       'UniMeth-FT (3 fine-tuned, max)': ['unimeth_reg_5hmC', 'unimeth_reg_4mC', 'unimeth_reg_5hmU'],
       'UniMeth-Agn': ['unimeth_agn_mod_reg'], 'Rockfish': ['rockfish'], 'DeepMod2': ['deepmod2']}
SETS = {'A: E. coli WT library, 6mA (GATC) + 5mC (CCWGG) vs knockout': ['ecoli_dam_6mA_reg', 'ecoli_dcm_5mC_reg'],
        'B: E. coli genome, Dam 6mA + Dcm 5mC + M.SssI 5mC vs knockouts': ['ecoli_dam_6mA_reg', 'ecoli_dcm_5mC_reg', 'ecoli_mssi_5mC_reg'],
        'C: ONT oligonucleotides, 5mC + 5hmC + 6mA vs controls': ['syn_5mC', 'syn_5hmC', 'syn_6mA']}
S = C.samples(); rows = {r['row']: r for r in Sc.rows_cfg()}; cache = {}
rng = np.random.default_rng(0)

def scores(blocks, subtools):
    if any(not C.sites_path(smp, t).exists() for smp, _, _ in blocks for t in subtools): return None
    return np.concatenate([np.max([Sc.site_scores(smp, t, sites)[0] for t in subtools], axis=0) for smp, sites, _ in blocks])

def tpr_at_fpr(y, s, fpr=0.01):
    neg = np.sort(s[y == 0])[::-1]; thr = neg[int(np.floor(fpr * len(neg)))] if len(neg) else np.inf
    return float((s[y == 1] > thr).mean())

def boot(y, s_a, s_b, n=1000):
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i): d.append(roc_auc_score(y[i], s_a[i]) - roc_auc_score(y[i], s_b[i]))
    d = np.array(d); return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

report = {}
for name, rlist in SETS.items():
    blocks, y, rowid, coords = [], [], [], []
    for r in rlist:
        b = Sc.row_sites(rows[r], S, cache); blocks += b
        for smp, sites, lab in b:
            y.append(np.full(len(sites), lab) if np.isscalar(lab) else lab); rowid += [r] * len(sites)
            coords += list(zip([r] * len(sites), sites['contig'], sites['pos'], sites['strand']))
    y = np.concatenate(y).astype(int); rowid = np.array(rowid)
    print(f'\n== {name}: {len(y):,} sites, {int(y.sum()):,} modified')
    res = {}; sc_all = {}
    for fam, subs in FAM.items():
        if subs is None: continue
        s = scores(blocks, subs)
        if s is None: print(f'   {fam}: missing'); continue
        sc_all[fam] = s
        per = {r: round(float(roc_auc_score(y[rowid == r], s[rowid == r])), 3) for r in rlist}
        res[fam] = dict(pooled=round(float(roc_auc_score(y, s)), 3), tpr1=round(tpr_at_fpr(y, s), 3), per_row=per)
    best = max((f for f in sc_all if f.startswith('Dorado ') and 'max' not in f), key=lambda f: res[f]['pooled'], default=None)
    if best: res['Dorado, best single model'] = dict(res[best], which=best)
    for fam, r in res.items(): print(f"   {fam:32s} pooled {r['pooled']:.3f}  TPR@1%FPR {r['tpr1']:.3f}  per row {r['per_row']}" + (f"  [{r['which']}]" if 'which' in r else ''))
    for other in ('Dorado, all models (max)', 'UniMeth, both models (max)', 'UniMeth-Agn'):
        if other in sc_all:
            lo, hi = boot(y, sc_all['RawMod'], sc_all[other]); res.setdefault('ci', {})[other] = (lo, hi)
            print(f"   RawMod - {other}: pooled AUROC difference {res['RawMod']['pooled'] - res[other]['pooled']:+.3f} (95% bootstrap CI {lo:+.3f} to {hi:+.3f})")
    report[name] = res
  # same-position check for the oligo set: do the three libraries share coordinates?
    if name.startswith('C'):
        cs = pd.DataFrame(coords, columns=['row', 'contig', 'pos', 'strand']); cs['y'] = y
        pos = cs[cs.y == 1].groupby(['contig', 'pos', 'strand']).row.nunique()
        print(f'   modified positions shared by 2+ oligo libraries: {(pos >= 2).sum()} of {len(pos)}')
json.dump(report, open(f'{OUT}/same_sequence.json', 'w'), indent=1, default=str); print('DONE')
