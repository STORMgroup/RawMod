#!/usr/bin/env python3
"""Per-position probabilities of the region-holdout mixed model inside the withheld regions (the all_neg test set
of the region table), for calibration (reliability diagram) and threshold analyses. Needs the training feature tree
(RAWMOD_STRANDRES_ROOT and the training environment of scripts/train/run_matched_loco.sh); the result is shipped in
paper/data/work/calibration/region_sites.npz. Writes region_sites.npz with
label, probability, chemistry, organism and reference base per position. Same loading as eval_read_depth_region.py."""
import os as _os
from pathlib import Path as _Path
REPO = str(_Path(__file__).resolve().parents[2])
RAWMOD_RESULTS = _os.environ.get('RAWMOD_RESULTS') or f'{REPO}/paper/data/runs'  # training runs; shipped metrics by default
RAWMOD_DATA = _os.environ.get('RAWMOD_DATA', 'data')  # datasets (not shipped: set this to score raw signal)
RAWMOD_WORK = _os.environ.get('RAWMOD_WORK') or f'{REPO}/paper/data/work'  # outputs of the paper scripts; shipped by default
import os, sys, json, numpy as np, torch
from pathlib import Path
from torch.utils.data import DataLoader
sys.path.insert(0, f'{REPO}/scripts/train'); sys.path.insert(0, f'{REPO}/scripts/evaluate')
import run_matched_loco as M
R = M.R
from run_convformer_v2 import ConvFormerV2
from score_holdout_region import region_test_idx
CKPT = Path(os.environ.get('CKPT', f'{REPO}/checkpoints/region_holdout/mixed.pt'))
OUT = Path(f'{RAWMOD_WORK}/calibration'); OUT.mkdir(parents=True, exist_ok=True)
hp = R.HP(); R.set_seed(hp.seed)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
members = M.build_members(); pool = R.Group(list(members), members)
mod_map = M.build_umces_mod_map(R.UMCES_PILEUPS, R.UMCES_REF)
refbase = M.ref_base_center(pool); chem = M.chem_array(pool, mod_map, refbase); is_pos = pool.labels > 0
is_bench = np.array([pool.names[int(pool.file_of[i])].startswith('BENCH::') for i in range(pool.N)])
test_idx, _ = region_test_idx(pool, chem, refbase, is_pos, ~is_pos, is_bench, 'mixed')
org = np.array([M.org_of(pool.names[int(pool.file_of[i])]) for i in test_idx])
print('region test images', len(test_idx), 'device', dev, flush=True)
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
model = ConvFormerV2(in_ch=ck.get('in_channels', 11), dropout=hp.dropout, supcon_dim=int(os.environ.get('SUPCON_DIM', '128')),
                     sad_dim=int(os.environ.get('SAD_DIM', '0')), h=M.HEIGHT, layers=int(os.environ.get('TF_LAYERS', '2')))
model.load_state_dict(ck['model_state']); model.to(dev).eval()
ds = R.make_ds(pool, test_idx, False, hp); keys = pool.source_keys(test_idx)
attr = {}
for j, kk in enumerate(keys):
    attr.setdefault(kk, (chem[test_idx[j]], org[j], refbase[test_idx[j]]))
loader = DataLoader(ds, shuffle=False, **R.make_loader_kwargs(hp.batch, hp.num_workers, dev, R._wif))
yt, yp = R.run_inference(model, loader, dev)
t, p, ks = R.aggregate_by_position(yt, yp, keys)
a = [attr[x] for x in ks]
np.savez_compressed(OUT / 'region_sites.npz', y=np.asarray(t), p=np.asarray(p), chem=np.array([x[0] for x in a]),
                    org=np.array([x[1] for x in a]), base=np.array([str(x[2]) for x in a]))
from sklearn.metrics import roc_auc_score
print('positions', len(t), 'auroc', round(roc_auc_score(t, p), 4))
