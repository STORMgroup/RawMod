#!/usr/bin/env python3
"""Embedding statistics on the mixed test set inside the withheld regions: k-means ARI by label and by modification, linear probes for modification, reference base and dataset, and nearest-neighbour label and organism agreement."""
import os as _os
from pathlib import Path as _Path
REPO = str(_Path(__file__).resolve().parents[2])
RAWMOD_RESULTS = _os.environ.get('RAWMOD_RESULTS') or f'{REPO}/paper/data/runs'  # training runs; shipped metrics by default
RAWMOD_DATA = _os.environ.get('RAWMOD_DATA', 'data')  # datasets (not shipped: set this to score raw signal)
RAWMOD_WORK = _os.environ.get('RAWMOD_WORK') or f'{REPO}/paper/data/work'  # outputs of the paper scripts; shipped by default

import os, sys, json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

REPO = Path(f'{REPO}')
sys.path.insert(0, str(REPO / 'scripts/train'))
import run_matched_loco as M  # noqa: E402
R = M.R
from run_convformer_v2 import ConvFormerV2  # noqa: E402

CKPT = Path(f'{RAWMOD_RESULTS}/region_mixed/models/mixed/mixed/best_model.pt')
sys.path.insert(0, f'{REPO}/scripts/evaluate')
from score_holdout_region import region_test_idx  # noqa: E402
OUT = Path(f'{RAWMOD_WORK}/figures')
NSUB = int(os.environ.get('NSUB', '1500'))  # UMAP points per class


@torch.no_grad()
def embed(model, loader, dev):
    feats = []
    h = model.norm.register_forward_hook(lambda m, i, o: feats.append(o.detach().cpu()))
    ys, ps = [], []
    for x, y in loader:
        lo = model(x.to(dev, non_blocking=True)).squeeze(1)
        ps.append(torch.sigmoid(lo).cpu().numpy()); ys.append(y.numpy())
    h.remove()
    return np.concatenate(ys), np.concatenate(ps), torch.cat(feats).numpy()


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


def main():
    hp = R.HP(); R.set_seed(hp.seed)
    dev = torch.device('cuda')
    members = M.build_members(); names = list(members)
    pool = R.Group(names, members)
    mod_map = M.build_umces_mod_map(R.UMCES_PILEUPS, R.UMCES_REF)
    refbase = M.ref_base_center(pool)
    chem = M.chem_array(pool, mod_map, refbase)
    is_pos = pool.labels > 0
    assert 'region_only' in os.environ.get('RAWMOD_STRANDRES_ROOT', ''), 'point RAWMOD_STRANDRES_ROOT at the region-only tree'
    is_bench = np.array([pool.names[int(pool.file_of[i])].startswith('BENCH::') for i in range(pool.N)])
    test_idx, _ = region_test_idx(pool, chem, refbase, is_pos, ~is_pos, is_bench, 'mixed')  # all_neg test set of the region table
    org = np.array([M.org_of(pool.names[int(pool.file_of[i])]) for i in test_idx])
    print('test images', len(test_idx), flush=True)

    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    model = ConvFormerV2(in_ch=ck.get('in_channels', 11), dropout=hp.dropout,
                         supcon_dim=int(os.environ.get('SUPCON_DIM', '128')),
                         sad_dim=int(os.environ.get('SAD_DIM', '0')), h=M.HEIGHT,
                         layers=int(os.environ.get('TF_LAYERS', '2')))
    model.load_state_dict(ck['model_state']); model.to(dev).eval()

    ds = R.make_ds(pool, test_idx, False, hp)
    keys = pool.source_keys(test_idx)
    loader = DataLoader(ds, shuffle=False, **R.make_loader_kwargs(hp.batch, hp.num_workers, dev, R._wif))
    yt, yp, z = embed(model, loader, dev)

  # per-position mean of embedding and probability
    uk, inv = np.unique(np.array([str(k) for k in keys]), return_inverse=True)
    n = len(uk)
    Z = np.zeros((n, z.shape[1]), np.float32); np.add.at(Z, inv, z)
    cnt = np.bincount(inv, minlength=n).astype(np.float32); Z /= cnt[:, None]
    P = np.bincount(inv, weights=yp, minlength=n) / cnt
    Y = np.zeros(n, np.int64); np.maximum.at(Y, inv, (yt > 0).astype(np.int64))
    first = np.full(n, -1); first[inv[::-1]] = np.arange(len(inv))[::-1]
    CH = np.array([str(chem[test_idx[j]]) for j in first])
    OG = np.array([str(org[j]) for j in first])
    RB = np.array([refbase[test_idx[j]].decode() if isinstance(refbase[test_idx[j]], bytes)
                   else str(refbase[test_idx[j]]) for j in first])
    CH[Y == 0] = 'unmodified'
    print('positions', n, 'pos', int(Y.sum()), flush=True)

    res = {'n_positions': int(n), 'n_pos': int(Y.sum())}
    pos = Y == 1
    res['ari_mod_pos'] = ari(Z[pos], CH[pos])
    res['ari_org_all'] = ari(Z, OG)
    res['ari_org_neg'] = ari(Z[~pos], OG[~pos])
    res['ari_label_all'] = ari(Z, Y.astype(str))
    res['probe_mod_pos'], res['probe_mod_pos_majority'] = probe(Z[pos], CH[pos])
    res['probe_org_all'], res['probe_org_all_majority'] = probe(Z, OG)
    res['probe_org_neg'], res['probe_org_neg_majority'] = probe(Z[~pos], OG[~pos])
    from sklearn.metrics import roc_auc_score
    res['auroc_check'] = float(roc_auc_score(Y, P))
    print(json.dumps(res, indent=1), flush=True)

  # class-balanced subsample for UMAP: up to NSUB per modification class
    rng = np.random.default_rng(0)
    sub = np.concatenate([rng.choice(np.flatnonzero(CH == c), min(NSUB, int((CH == c).sum())), replace=False)
                          for c in np.unique(CH)])
    import umap
    U = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(Z[sub])
    np.savez_compressed(OUT / 'embedding_region.npz', Z=Z, P=P, Y=Y, CH=CH, OG=OG, RB=RB, sub=sub, U=U)
    json.dump(res, open(OUT / 'embedding_region.json', 'w'), indent=1)
    print('wrote', OUT / 'embedding_region.npz', flush=True)


if __name__ == '__main__':
    main()
