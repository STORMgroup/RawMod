#!/usr/bin/env python3
"""Embedding metrics per objective: k-means ARI against the label within each dataset and reference base, silhouette, nearest-neighbour purity, and the classifier AUROC per withheld modification."""
import os as _os
from pathlib import Path as _Path
REPO = str(_Path(__file__).resolve().parents[2])
RAWMOD_RESULTS = _os.environ.get('RAWMOD_RESULTS') or f'{REPO}/paper/data/runs'  # training runs; shipped metrics by default
RAWMOD_DATA = _os.environ.get('RAWMOD_DATA', 'data')  # datasets (not shipped: set this to score raw signal)
RAWMOD_WORK = _os.environ.get('RAWMOD_WORK') or f'{REPO}/paper/data/work'  # outputs of the paper scripts; shipped by default

import json, sys, zlib, collections
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, roc_auc_score, silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.neighbors import NearestNeighbors

import os
OUT = Path(os.environ.get('AUX_OUT', str(Path(__file__).resolve().parent / 'out')))
ARMS = ['final', 'bce', 'supcon']
NBOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 50
MINC = 30


def ari(z, lab, seed=0):
    k = len(set(lab))
    if k < 2:
        return None
    pred = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(z)
    return float(adjusted_rand_score(lab, pred))


def probe(z, lab, seed=0):
    vals, cnt = np.unique(lab, return_counts=True)
    keep = np.isin(lab, vals[cnt >= 5])
    z, lab = z[keep], lab[keep]
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    acc = cross_val_score(LogisticRegression(max_iter=2000), z, lab, cv=cv).mean()
    return float(acc), float(cnt[cnt >= 5].max() / cnt[cnt >= 5].sum())


def probe2(X, y):
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    pr = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)), X, y, cv=cv)
    return round(accuracy_score(y, pr), 3), round(balanced_accuracy_score(y, pr), 3)


def lookup(keys, y):
    d = collections.defaultdict(collections.Counter)
    for k, t in zip(keys, y):
        d[k][t] += 1
    pr = np.array([d[k].most_common(1)[0][0] for k in keys])
    return round(accuracy_score(y, pr), 3), round(balanced_accuracy_score(y, pr), 3)


# ---------- new unsupervised separation metrics ----------
def km2_ari(Z, y, seed=0):
    pred = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(Z)
    return float(adjusted_rand_score(y, pred))


def sil(Z, y, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.choice(len(y), min(n, len(y)), replace=False)
    return float(silhouette_score(Z[s], y[s]))


def knn_purity(Z, y, k=10):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    _, ix = nn.kneighbors(Z)
    same = (y[ix[:, 1:]] == y[:, None]).mean(1)
    return float(np.mean([same[y == c].mean() for c in np.unique(y)]))


def group_sep(Z, Y, G, boot=0, seed=0):
    """per-group k-means ARI / silhouette / kNN purity; optional bootstrap."""
    rows = []
    for g in sorted(set(G)):
        m = G == g
        n1, n0 = int(Y[m].sum()), int((~Y[m].astype(bool)).sum())
        if n1 < MINC or n0 < MINC:
            continue
        Zg, yg = Z[m], Y[m]
        rows.append(dict(group=g, n_pos=n1, n_neg=n0, km2_ari=km2_ari(Zg, yg),
                         silhouette=sil(Zg, yg), knn_purity=knn_purity(Zg, yg)))
    return rows


def boot_group(Zs, Y, G, nboot, seed=0):
    """Paired bootstrap (resample positions within group) of km2_ari and
    knn_purity for several embeddings Zs (dict arm->Z). Returns per-arm arrays
    of the n-weighted mean over groups, shape (nboot,)."""
    rng = np.random.default_rng(seed)
    groups = [g for g in sorted(set(G)) if (Y[G == g] == 1).sum() >= MINC and (Y[G == g] == 0).sum() >= MINC]
    res = {a: {'km2_ari': [], 'knn_purity': []} for a in Zs}
    for b in range(nboot):
        acc = {a: {'km2_ari': 0.0, 'knn_purity': 0.0, 'w': 0} for a in Zs}
        for g in groups:
            ii = np.flatnonzero(G == g)
            s = rng.choice(ii, len(ii), replace=True)
            yg = Y[s]
            for a, Z in Zs.items():
                Zg = Z[s] + 1e-6 * rng.standard_normal((len(s), Z.shape[1]))  # break duplicate ties for kNN
                acc[a]['km2_ari'] += len(s) * km2_ari(Zg, yg, seed=b)
                acc[a]['knn_purity'] += len(s) * knn_purity(Zg, yg)
                acc[a]['w'] += len(s)
        for a in Zs:
            for k in ('km2_ari', 'knn_purity'):
                res[a][k].append(acc[a][k] / acc[a]['w'])
    return {a: {k: np.array(v) for k, v in d.items()} for a, d in res.items()}


def wmean(rows, key):
    w = np.array([r['n_pos'] + r['n_neg'] for r in rows], float)
    v = np.array([r[key] for r in rows], float)
    return float((w * v).sum() / w.sum())


def mixed():
    d = np.load(OUT / 'emb_mixed.npz', allow_pickle=True)
    Y, OG, RB = d['Y'], d['OG'], d['RB']
    CH = d['CH'].astype(object).copy(); CH[Y == 0] = 'unmodified'; CH = CH.astype(str)
    out = {'n_positions': int(len(Y)), 'n_pos': int(Y.sum())}
    G = np.array([f'{o}{b}' for o, b in zip(OG, RB)])
    spaces = {a: d[f'{a}__z'] for a in ARMS}
    spaces['final_sad16'] = d['final__sad']
    spaces['final_proj128'] = d['final__proj']
    spaces['supcon_proj128'] = d['supcon__proj']
    for name, Z in spaces.items():
        r = {}
        pos = Y == 1
        if name in ARMS:
            r['ari_mod_pos'] = ari(Z[pos], CH[pos])
            r['ari_org_all'] = ari(Z, OG)
            r['ari_org_neg'] = ari(Z[~pos], OG[~pos])
            r['ari_label_all'] = ari(Z, Y.astype(str))
            r['probe_mod_pos'], r['probe_mod_pos_majority'] = probe(Z[pos], CH[pos])
            r['probe_org_all'], r['probe_org_all_majority'] = probe(Z, OG)
            r['probe_org_neg'], r['probe_org_neg_majority'] = probe(Z[~pos], OG[~pos])
            r['auroc_check'] = float(roc_auc_score(Y, d[f'{name}__p']))
  # confound script
            typed = (Y == 1) & (CH != 'untyped')
            r['cf_probe_mod_typed'] = probe2(Z[typed], CH[typed])
            r['cf_lookup_org_base_typed'] = lookup(list(zip(OG[typed], RB[typed])), CH[typed])
            r['cf_lookup_base_typed'] = lookup(list(RB[typed]), CH[typed])
            for org, base in (('HP::', 'C'), ('ONT::', 'C')):
                m = typed & (OG == org) & (RB == base)
                r[f'cf_probe_{org[:-2]}_{base}'] = probe2(Z[m], CH[m])
            neg = Y == 0
            r['cf_probe_org_neg'] = probe2(Z[neg], OG[neg])
        else:
            r['ari_label_all'] = ari(Z, Y.astype(str))
        rows = group_sep(Z, Y, G)
        r['within_group'] = rows
        r['within_group_wmean'] = {k: wmean(rows, k) for k in ('km2_ari', 'silhouette', 'knn_purity')}
        r['within_group_mean'] = {k: float(np.mean([x[k] for x in rows])) for k in ('km2_ari', 'silhouette', 'knn_purity')}
        out[name] = r
        print('mixed', name, json.dumps(r['within_group_wmean']), r.get('ari_label_all'), flush=True)
    if NBOOT:
        bs = boot_group({a: spaces[a] for a in ARMS}, Y, G, NBOOT)
        out['bootstrap'] = {}
        for k in ('km2_ari', 'knn_purity'):
            for a in ('bce', 'supcon'):
                diff = bs['final'][k] - bs[a][k]
                out['bootstrap'][f'{k}_final_minus_{a}'] = dict(
                    mean=float(diff.mean()), lo=float(np.percentile(diff, 2.5)),
                    hi=float(np.percentile(diff, 97.5)), frac_gt0=float((diff > 0).mean()))
    json.dump(out, open(OUT / 'metrics_mixed.json', 'w'), indent=1, default=str)
  # flat tsv
    rows = []
    for name in spaces:
        for g in out[name]['within_group']:
            rows.append(dict(space=name, **g))
    pd.DataFrame(rows).to_csv(OUT / 'mixed_within_group.tsv', sep='\t', index=False)


def h01(c, p, tag):
    return (zlib.crc32(f'{tag}:{c}:{p}'.encode()) & 0xffffffff) / 0xffffffff


def anomaly_scores(Z, Y, OG, ref, ev):
    """per-dataset centroid distance, pooled centroid distance, 10-NN distance."""
    s_org = np.full(len(Y), np.nan); s_knn = np.full(len(Y), np.nan)
    c_all = Z[ref].mean(0)
    s_all = np.linalg.norm(Z - c_all, axis=1)
    for o in np.unique(OG[ev]):
        r = ref & (OG == o); e = ev & (OG == o)
        if r.sum() < 10:
            r = ref  # no same-dataset reference: fall back to pooled
        c = Z[r].mean(0)
        s_org[e] = np.linalg.norm(Z[e] - c, axis=1)
        nn = NearestNeighbors(n_neighbors=10).fit(Z[r])
        dist, _ = nn.kneighbors(Z[e])
        s_knn[e] = dist.mean(1)
    return {'centroid_per_dataset': s_org, 'centroid_pooled': s_all, 'knn10': s_knn}


def loco():
    tsv = {}
    for a in ARMS:
        for f in Path(f'{RAWMOD_RESULTS}').glob(
                {'final': 'region_loco_region_score', 'bce': 'ablation_bce_region_score',
                 'supcon': 'ablation_nosad_region_score'}[a] + '/metrics/loco_*.tsv'):
            t = pd.read_csv(f, sep='\t'); t = t[t.test_set.str.endswith('all_neg')].iloc[0]
            tsv[(a, t['fold'])] = (float(t['auroc']), t.get('auroc_sad'))
    out = {}; flat = []
    for chem in ('5mC', '5hmC', '4mC', '6mA', '5hmU'):
        fold = f'loco_{chem}'
        p = OUT / f'emb_{fold}.npz'
        if not p.exists():
            print('missing', p); continue
        d = np.load(p, allow_pickle=True)
        Y, OG = d['Y'], d['OG']
        ref = (Y == 0) & np.array([h01(c, q, 'auxref') < 0.5 for c, q in zip(d['contig'], d['ref_pos'])])
        ev = ~ref
        fo = {'n_pos': int(Y.sum()), 'n_neg': int((Y == 0).sum()), 'n_ref_neg': int(ref.sum()),
              'n_eval_neg': int((ev & (Y == 0)).sum()),
              'datasets_pos': dict(collections.Counter(OG[Y == 1])),
              'datasets_neg': dict(collections.Counter(OG[Y == 0]))}
        spaces = {a: d[f'{a}__z'] for a in ARMS}
        spaces['final_sad16'] = d['final__sad']
        for name, Z in spaces.items():
            r = {}
            arm = name.split('_')[0]
            if name in ARMS:
                r['logit_auroc'] = float(roc_auc_score(Y, d[f'{name}__p']))
                r['logit_auroc_tsv'] = tsv.get((name, fold), (None,))[0]
                r['logit_auroc_eval'] = float(roc_auc_score(Y[ev], d[f'{name}__p'][ev]))
            sc = anomaly_scores(Z, Y, OG, ref, ev)
            for k, s in sc.items():
                r[f'auroc_{k}'] = float(roc_auc_score(Y[ev], s[ev]))
            r['km2_ari_pooled'] = km2_ari(Z[ev], Y[ev])
            wr = []
            for o in np.unique(OG[ev]):
                m = ev & (OG == o)
                if (Y[m] == 1).sum() >= MINC and (Y[m] == 0).sum() >= MINC:
                    wr.append(dict(dataset=o, n_pos=int(Y[m].sum()), n_neg=int((Y[m] == 0).sum()),
                                   km2_ari=km2_ari(Z[m], Y[m]),
                                   auroc_centroid=float(roc_auc_score(Y[m], sc['centroid_per_dataset'][m])),
                                   auroc_knn10=float(roc_auc_score(Y[m], sc['knn10'][m])),
                                   logit_auroc=float(roc_auc_score(Y[m], d[f'{arm}__p'][m]))))
            r['within_dataset'] = wr
            if wr:
                r['km2_ari_within_wmean'] = wmean(wr, 'km2_ari')
            if name == 'final':
                sd = d['final__sad_dist']
                r['sad_score_auroc_all'] = float(roc_auc_score(Y, sd))
                r['sad_score_auroc_eval'] = float(roc_auc_score(Y[ev], sd[ev]))
                r['sad_auroc_tsv_image_level'] = tsv.get(('final', fold), (None, None))[1]
            fo[name] = r
            flat.append(dict(fold=fold, space=name, **{k: v for k, v in r.items() if not isinstance(v, list)}))
            print(fold, name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, list)}, flush=True)
        out[fold] = fo
    json.dump(out, open(OUT / 'metrics_loco.json', 'w'), indent=1, default=str)
    pd.DataFrame(flat).to_csv(OUT / 'loco_summary.tsv', sep='\t', index=False)


if __name__ == '__main__':
    which = sys.argv[2] if len(sys.argv) > 2 else 'both'
    if which in ('both', 'loco'):
        loco()
    if which in ('both', 'mixed'):
        mixed()
