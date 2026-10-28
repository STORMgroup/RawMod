#!/usr/bin/env python3
"""Position-resampling bootstrap of the zero-shot AUROC differences between objectives (final minus classification only, final minus no Deep SAD)."""
import json, numpy as np, sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import auxloss_metrics as MT
from sklearn.metrics import roc_auc_score
out = {}
rng = np.random.default_rng(0)
for chem in ('5mC', '5hmC', '4mC', '6mA', '5hmU'):
    fold = f'loco_{chem}'
    d = np.load(MT.OUT / f'emb_{fold}.npz', allow_pickle=True)
    Y, OG = d['Y'], d['OG']
    ref = (Y == 0) & np.array([MT.h01(c, q, 'auxref') < 0.5 for c, q in zip(d['contig'], d['ref_pos'])])
    ev = ~ref
    S = {}
    for a in MT.ARMS:
        sc = MT.anomaly_scores(d[f'{a}__z'], Y, OG, ref, ev)
        S[(a, 'logit')] = d[f'{a}__p'][ev]
        S[(a, 'centroid')] = sc['centroid_per_dataset'][ev]
        S[(a, 'knn10')] = sc['knn10'][ev]
    S[('final', 'sad')] = d['final__sad_dist'][ev]
    y = Y[ev]; n = len(y)
    B = [rng.integers(0, n, n) for _ in range(1000)]
    fo = {}
    for k in ('logit', 'centroid', 'knn10'):
        for o in ('bce', 'supcon'):
            diffs = np.array([roc_auc_score(y[b], S[('final', k)][b]) - roc_auc_score(y[b], S[(o, k)][b]) for b in B])
            fo[f'{k}_final_minus_{o}'] = dict(point=float(roc_auc_score(y, S[('final', k)]) - roc_auc_score(y, S[(o, k)])),
                                             lo=float(np.percentile(diffs, 2.5)), hi=float(np.percentile(diffs, 97.5)))
  # SAD score vs BCE's best label-free score (centroid) and vs BCE logit
    for k in ('centroid', 'knn10', 'logit'):
        diffs = np.array([roc_auc_score(y[b], S[('final', 'sad')][b]) - roc_auc_score(y[b], S[('bce', k)][b]) for b in B])
        fo[f'sad_minus_bce_{k}'] = dict(point=float(roc_auc_score(y, S[('final', 'sad')]) - roc_auc_score(y, S[('bce', k)])),
                                       lo=float(np.percentile(diffs, 2.5)), hi=float(np.percentile(diffs, 97.5)))
    out[fold] = fo
    print(fold, {k: (round(v['point'], 3), round(v['lo'], 3), round(v['hi'], 3)) for k, v in fo.items()}, flush=True)
json.dump(out, open(MT.OUT / 'loco_boot.json', 'w'), indent=1)
