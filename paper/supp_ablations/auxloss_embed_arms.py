#!/usr/bin/env python3
"""Site embeddings of the test positions inside the withheld regions for three training objectives (final, classification only, no Deep SAD), for the mixed model and every withheld-modification model."""
import os as _os
from pathlib import Path as _Path
REPO = str(_Path(__file__).resolve().parents[2])
RAWMOD_RESULTS = _os.environ.get('RAWMOD_RESULTS') or f'{REPO}/paper/data/runs'  # training runs; shipped metrics by default
RAWMOD_DATA = _os.environ.get('RAWMOD_DATA', 'data')  # datasets (not shipped: set this to score raw signal)
RAWMOD_WORK = _os.environ.get('RAWMOD_WORK') or f'{REPO}/paper/data/work'  # outputs of the paper scripts; shipped by default

import os, sys, collections
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader

REPO = Path(f'{REPO}')
sys.path.insert(0, str(REPO / 'scripts/train'))
import run_matched_loco as M  # noqa: E402
R = M.R
from run_convformer_v2 import ConvFormerV2  # noqa: E402

RES = Path(f'{RAWMOD_RESULTS}')
ARMS = {'final': 'region_loco_{fold}', 'bce': 'ablation_bce',
        'supcon': 'ablation_nosad'}
OUT = Path(os.environ.get('AUX_OUT', f'{RAWMOD_WORK}/auxloss'))


def ckpt_path(arm, fold):
    sub = 'mixed/mixed' if fold == 'mixed' else f'{fold}/loco'
    if arm == 'final' and fold == 'mixed':
        return RES / 'region_mixed' / 'models' / sub / 'best_model.pt'
    return RES / ARMS[arm].format(fold=fold) / 'models' / sub / 'best_model.pt'


def load(path, hp, dev):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck['model_state']
    sad_dim = int(sd['sad_head.weight'].shape[0]) if 'sad_head.weight' in sd else 0
    m = ConvFormerV2(in_ch=ck.get('in_channels', 11), dropout=hp.dropout,
                     supcon_dim=int(sd['proj.net.2.weight'].shape[0]) if 'proj.net.2.weight' in sd else 0,
                     sad_dim=sad_dim, h=M.HEIGHT, layers=int(os.environ.get('TF_LAYERS', '2')))
    m.load_state_dict(sd); m.to(dev).eval()
    return m


@torch.no_grad()
def embed_multi(models, loader, dev):
    """models: dict arm -> model. Returns per-image arrays per arm."""
    store = {a: collections.defaultdict(list) for a in models}
    hooks = []
    cur = {}
    for a, m in models.items():
        hooks.append(m.norm.register_forward_hook(
            lambda mod, i, o, a=a: cur.__setitem__(a, o.detach())))
    ys = []
    for x, y in loader:
        x = x.to(dev, non_blocking=True)
        ys.append(y.numpy())
        for a, m in models.items():
            lo = m(x).squeeze(1)
            rep = cur[a]
            s = store[a]
            s['p'].append(torch.sigmoid(lo).cpu().numpy())
            s['z'].append(rep.cpu().numpy())
            if m.proj is not None:
                s['proj'].append(m.proj(rep).cpu().numpy())
            if m.sad_head is not None:
                sd_ = m._sad
                s['sad'].append(sd_.cpu().numpy())
                s['sad_dist'].append(((sd_ - m.sad_center) ** 2).sum(1).sqrt().cpu().numpy())
    for h in hooks:
        h.remove()
    return np.concatenate(ys), {a: {k: np.concatenate(v) for k, v in s.items()} for a, s in store.items()}


def per_position(keys, y, arrs):
    uk, inv = np.unique(np.array([str(k) for k in keys]), return_inverse=True)
    n = len(uk)
    cnt = np.bincount(inv, minlength=n).astype(np.float64)
    Y = np.zeros(n, np.int64); np.maximum.at(Y, inv, (y > 0).astype(np.int64))
    first = np.full(n, -1); first[inv[::-1]] = np.arange(len(inv))[::-1]
    out = {}
    for k, v in arrs.items():
        if v.ndim == 1:
            out[k] = (np.bincount(inv, weights=v, minlength=n) / cnt).astype(np.float32)
        else:
            Z = np.zeros((n, v.shape[1]), np.float64); np.add.at(Z, inv, v)
            out[k] = (Z / cnt[:, None]).astype(np.float32)
    return Y, first, out, cnt


def run_split(name, pool, idx, folds_for_arm, hp, dev, meta_fn, extra=None):
    out_f = OUT / f'emb_{name}.npz'
    if os.environ.get('DRY') == '1':
        keys = pool.source_keys(np.asarray(idx, np.int64))
        uk = {str(k) for k in keys}
        lab = {}
        for k, i in zip(keys, idx):
            lab[str(k)] = max(lab.get(str(k), 0), int(pool.labels[i] > 0))
        print(f'DRY {name}: images {len(idx)} positions {len(uk)} pos_positions {sum(lab.values())}', flush=True)
        return
    if out_f.exists():
        print('skip existing', out_f, flush=True); return
    idx = np.asarray(idx, np.int64)
    models = {a: load(ckpt_path(a, folds_for_arm), hp, dev) for a in ARMS}
    ds = R.make_ds(pool, idx, False, hp)
    loader = DataLoader(ds, shuffle=False, **R.make_loader_kwargs(hp.batch, hp.num_workers, dev, R._wif))
    y, per = embed_multi(models, loader, dev)
    keys = pool.source_keys(idx)
    save = {}
    Y = None
    for a, arrs in per.items():
        Y, first, agg, cnt = per_position(keys, y, arrs)
        for k, v in agg.items():
            save[f'{a}__{k}'] = v
    meta = meta_fn(idx[first])
    save.update(meta); save['Y'] = Y; save['n_img'] = cnt
    if extra is not None:
        save.update(extra(idx[first]))
    np.savez_compressed(out_f, **save)
    from sklearn.metrics import roc_auc_score
    print(name, 'positions', len(Y), 'pos', int(Y.sum()), 'images', len(idx),
          {a: round(float(roc_auc_score(Y, save[f'{a}__p'])), 6) for a in ARMS}, flush=True)
    del models; torch.cuda.empty_cache()


def main():
    which = sys.argv[1:] or ['mixed', 'loco_5mC', 'loco_5hmC', 'loco_4mC', 'loco_6mA', 'loco_5hmU']
    OUT.mkdir(parents=True, exist_ok=True)
    hp = R.HP(); R.set_seed(hp.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    members = M.build_members(); names = list(members)
    pool = R.Group(names, members)
    mod_map = M.build_umces_mod_map(R.UMCES_PILEUPS, R.UMCES_REF)
    refbase = M.ref_base_center(pool)
    chem = M.chem_array(pool, mod_map, refbase)
    is_pos = pool.labels > 0
    kept_neg = M.subsample_negatives(pool, seed=M.SPLIT_SEED)
    neg_mask = np.zeros(pool.N, bool); neg_mask[kept_neg] = True
    is_bench = np.array([pool.names[int(pool.file_of[i])].startswith('BENCH::') for i in range(pool.N)])

    def meta_fn(ii):
        nm = [pool.names[int(pool.file_of[i])] for i in ii]
        return {'CH': np.array([str(chem[i]) for i in ii]),
                'OG': np.array([M.org_of(n) for n in nm]),
                'DS': np.array([n for n in nm]),
                'RB': np.array([refbase[i].decode() if isinstance(refbase[i], bytes) else str(refbase[i]) for i in ii]),
                'contig': np.array([str(pool.contig[i]) for i in ii]),
                'ref_pos': np.array([int(pool.ref_pos[i]) for i in ii])}

    sys.path.insert(0, str(REPO / 'scripts/evaluate'))
    from score_holdout_region import region_test_idx
    for fold in which:
  # Region holdout: the exact all-negatives, in-region test set of score_holdout_region.py
        test_idx, _ = region_test_idx(pool, chem, refbase, is_pos, ~is_pos, is_bench, fold)
        print(f'{fold}: region test images {len(test_idx)} positives {int(is_pos[test_idx].sum())}', flush=True)
        run_split(fold, pool, test_idx, fold, hp, dev, meta_fn)

if __name__ == '__main__':
    main()
