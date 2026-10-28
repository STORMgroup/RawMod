#!/usr/bin/env python3
"""Figure 4b: AUROC of the region-holdout mixed model inside the withheld regions against the number of reads kept per pileup. Scores every test image with 1, 3, 6, ... 15 reads."""
import os as _os
from pathlib import Path as _Path
REPO = str(_Path(__file__).resolve().parents[2])
RAWMOD_RESULTS = _os.environ.get('RAWMOD_RESULTS') or f'{REPO}/paper/data/runs'  # training runs; shipped metrics by default
RAWMOD_DATA = _os.environ.get('RAWMOD_DATA', 'data')  # datasets (not shipped: set this to score raw signal)
RAWMOD_WORK = _os.environ.get('RAWMOD_WORK') or f'{REPO}/paper/data/work'  # outputs of the paper scripts; shipped by default

import os, sys, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

REPO = Path(f'{REPO}')
sys.path.insert(0, str(REPO / 'scripts/train')); sys.path.insert(0, str(REPO / 'scripts/evaluate'))
import run_matched_loco as M  # noqa: E402
R = M.R
from run_convformer_v2 import ConvFormerV2  # noqa: E402
from score_holdout_region import region_test_idx  # noqa: E402

CKPT = Path(os.environ.get('CKPT', f'{RAWMOD_RESULTS}/region_mixed/models/mixed/mixed/best_model.pt'))
OUT = Path(f'{RAWMOD_WORK}/figures')
KS = [1, 2, 3, 4, 6, 8, 10, 12, 15]
NREP = int(os.environ.get('NREP', '3'))


class KeepK(nn.Module):
    def __init__(self, model, k, seed):
        super().__init__(); self.m = model; self.k = k
        self.g = torch.Generator(device='cuda'); self.g.manual_seed(seed)

    def forward(self, x):
        B, C, H, W = x.shape
        real = x[:, 0, 1:].abs().sum(-1) > 1e-6  # (B, H-1)
        r = torch.rand(B, H - 1, device=x.device, generator=self.g)
        r = torch.where(real, r, torch.full_like(r, -1.0))
        order = r.argsort(dim=1, descending=True).argsort(dim=1)  # rank of each row
        keep = (order < self.k) & real
        x = x.clone()
        x[:, :, 1:, :] *= keep[:, None, :, None].float()
        return self.m(x)


def main():
    if 'region_only' not in os.environ.get('RAWMOD_STRANDRES_ROOT', ''):
        raise SystemExit('RAWMOD_STRANDRES_ROOT must point at the region-only tree')
    hp = R.HP(); R.set_seed(hp.seed)
    dev = torch.device('cuda')
    members = M.build_members(); names = list(members)
    pool = R.Group(names, members)
    mod_map = M.build_umces_mod_map(R.UMCES_PILEUPS, R.UMCES_REF)
    refbase = M.ref_base_center(pool)
    chem = M.chem_array(pool, mod_map, refbase)
    is_pos = pool.labels > 0
    is_bench = np.array([pool.names[int(pool.file_of[i])].startswith('BENCH::') for i in range(pool.N)])
    neg_mask = ~is_pos  # all_neg variant of the region table
    test_idx, _ = region_test_idx(pool, chem, refbase, is_pos, neg_mask, is_bench, 'mixed')
    org = np.array([M.org_of(pool.names[int(pool.file_of[i])]) for i in test_idx])
    print('region test images', len(test_idx), 'positives', int(is_pos[test_idx].sum()), flush=True)

    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    model = ConvFormerV2(in_ch=ck.get('in_channels', 11), dropout=hp.dropout,
                         supcon_dim=int(os.environ.get('SUPCON_DIM', '128')),
                         sad_dim=int(os.environ.get('SAD_DIM', '0')), h=M.HEIGHT,
                         layers=int(os.environ.get('TF_LAYERS', '2')))
    model.load_state_dict(ck['model_state']); model.to(dev).eval()

    ds = R.make_ds(pool, test_idx, False, hp)
    keys = pool.source_keys(test_idx)
    attr = {}
    for j, kk in enumerate(keys):
        if kk not in attr:
            attr[kk] = (chem[test_idx[j]], org[j], refbase[test_idx[j]])

    rows = []
    for k in KS:
        for rep in range(NREP if k < 15 else 1):
            loader = DataLoader(ds, shuffle=False,
                                **R.make_loader_kwargs(hp.batch, hp.num_workers, dev, R._wif))
            yt, yp = R.run_inference(KeepK(model, k, 1000 * k + rep), loader, dev)
            t, p, ks = R.aggregate_by_position(yt, yp, keys)
            a = [attr[x] for x in ks]
            ch = np.array([x[0] for x in a]); og = np.array([x[1] for x in a])
            rb = np.array([x[2] for x in a])
            row = dict(k=k, rep=rep, auroc=float(roc_auc_score(t, p)),
                       auprc=float(average_precision_score(t, p)), n=int(len(t)))
            for c in M.CHEMS:
                pos = (t == 1) & (ch == c)
                neg = (t == 0) & np.isin(og, M.CHEM_ORGS[c]) & np.isin(rb, M.CHEM_BASES[c])
                sel = pos | neg
                if pos.sum() > 20 and neg.sum() > 20:
                    row[f'auroc_{c}'] = float(roc_auc_score(t[sel], p[sel]))
                    row[f'npos_{c}'] = int(pos.sum()); row[f'nneg_{c}'] = int(neg.sum())
            print(json.dumps(row), flush=True)
            rows.append(row)
            json.dump(rows, open(OUT / 'read_depth_results_region.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
