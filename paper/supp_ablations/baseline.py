#!/usr/bin/env python3
"""No-learning baseline: AUROC of fixed pileup statistics (mean deviation of the current from the expected level at the queried base, its absolute value, the window-wide deviation and the fraction of reads above the expected level) on the same test sets as the models."""
import os as _os
from pathlib import Path as _Path
REPO = str(_Path(__file__).resolve().parents[2])
RAWMOD_RESULTS = _os.environ.get('RAWMOD_RESULTS') or f'{REPO}/paper/data/runs'  # training runs; shipped metrics by default
RAWMOD_DATA = _os.environ.get('RAWMOD_DATA', 'data')  # datasets (not shipped: set this to score raw signal)
RAWMOD_WORK = _os.environ.get('RAWMOD_WORK') or f'{REPO}/paper/data/work'  # outputs of the paper scripts; shipped by default

import sys, numpy as np, pandas as pd, h5py
from pathlib import Path
from sklearn.metrics import roc_auc_score
REPO = Path(f'{REPO}')
sys.path.insert(0, str(REPO / 'scripts' / 'train')); sys.path.insert(0, str(REPO / 'scripts' / 'evaluate')); sys.path.insert(0, str(REPO / 'rawmod'))
import run_matched_loco as M
from score_holdout_region import region_test_idx
OUT = Path(__file__).resolve().parent
RES = Path(f'{RAWMOD_RESULTS}')
FOLDS = ['mixed', 'loco_5mC', 'loco_5hmC', 'loco_6mA', 'loco_4mC', 'loco_5hmU']

members = M.build_members(); pool = M.R.Group(list(members), members)
print(f'region pool: {pool.N:,} images across {len(members)} files', flush=True)
mod_map = M.build_umces_mod_map(M.R.UMCES_PILEUPS, M.R.UMCES_REF)
refbase = M.ref_base_center(pool); chem = M.chem_array(pool, mod_map, refbase)
is_pos = pool.labels > 0
is_bench = np.array([pool.names[int(pool.file_of[i])].startswith('BENCH::') for i in range(pool.N)])
neg_masks = {}
for vname, allneg in (('subsampled', False), ('all_neg', True)):
    m = np.zeros(pool.N, bool)
    if allneg: m[~is_pos] = True
    else: m[M.subsample_negatives(pool, seed=M.SPLIT_SEED)] = True
    neg_masks[vname] = m
need = np.zeros(pool.N, bool); tests = {}
for fold in FOLDS:
    for vname in neg_masks:
        idx, held = region_test_idx(pool, chem, refbase, is_pos, neg_masks[vname], is_bench, fold)
        if len(idx) and is_pos[idx].sum():
            tests[(fold, vname)] = idx; need[idx] = True
print('images to read:', need.sum(), flush=True)

# statistics per image, read file by file in index order (chunked h5: one image per chunk)
stats = {k: np.full(pool.N, np.nan, np.float32) for k in ('dctr_abs', 'dctr', 'dwin_abs', 'dwin_max', 'ks_ctr')}
start = np.r_[0, np.cumsum(pool.file_sizes)]
for fi, p in enumerate(pool.paths):
    loc = np.nonzero(need[start[fi]:start[fi + 1]])[0]
    if not len(loc): continue
    with h5py.File(p, 'r') as hf:
        T = hf['tensors']; n, H, WL, C = T.shape
        Lb = 10; Wb = WL // Lb; ci = Wb // 2; cs, ce = ci * Lb, ci * Lb + Lb
        for j0 in range(0, len(loc), 256):
            sel = loc[j0:j0 + 256]
            x = T[sel, :, :, 0]  # (b, H, WL) signal channel
            ref = x[:, 0, :]  # expected level per sample
            reads = x[:, 1:, :]
            present = np.abs(reads).sum(axis=2) > 0  # (b, H-1) non-empty rows
            d = reads - ref[:, None, :]  # observed - expected
            dctr = d[:, :, cs:ce].mean(axis=2)  # (b, H-1)
            nr = np.maximum(present.sum(axis=1), 1)
            g = start[fi] + sel
            stats['dctr_abs'][g] = (np.abs(dctr) * present).sum(1) / nr
            stats['dctr'][g] = (dctr * present).sum(1) / nr
            dabs = np.abs(d).reshape(len(sel), H - 1, Wb, Lb).mean(axis=3)  # (b, H-1, Wb)
            per_base = (dabs * present[:, :, None]).sum(1) / nr[:, None]  # (b, Wb)
            stats['dwin_abs'][g] = per_base.mean(1); stats['dwin_max'][g] = per_base.max(1)
  # one-sample KS of the reads' centre currents (raw, not differenced) vs the expected level: fraction of
  # reads above the expected level, |F(c) - 0.5| scaled -> use max(|mean sign|) as a cheap proxy
            s = np.sign(dctr) * present
            stats['ks_ctr'][g] = np.abs(s.sum(1)) / nr
    print(f'  {pool.names[fi]}: {len(loc)} images', flush=True)

np.savez_compressed(OUT / 'stats.npz', need=need, **stats)
rows = []
for (fold, vname), idx in tests.items():
    keys = pool.source_keys(idx)
    held = fold[5:] if fold != 'mixed' else ''
    mp = (RES / ('region_mixed_region_score' if fold == 'mixed' else 'region_loco_region_score') / 'metrics' / f'{fold}.tsv')
    model_auc = np.nan
    if mp.exists():
        d = pd.read_csv(mp, sep='\t'); d = d[d.test_set.str.endswith(vname)]
        if len(d): model_auc = float(d.auroc.iloc[0])
    r = dict(fold=fold, test_set=vname, model=model_auc)
    for k, v in stats.items():
        t, pv, _ = M.R.aggregate_by_position(is_pos[idx].astype(float), v[idx].astype(float), keys)
        t = np.asarray(t); pv = np.asarray(pv)
        r['n_pos'] = int((t > 0.5).sum()); r['n_test'] = len(t)
        r[k] = roc_auc_score((t > 0.5).astype(int), pv)
    rows.append(r); print(r, flush=True)
pd.DataFrame(rows).to_csv(OUT / 'baseline.tsv', sep='\t', index=False, float_format='%.4f')
print('DONE')
