"""Site embeddings and scores of the released all-data model at the selected contig sites (dark-modification map).
Same model input and embedding (output of model.norm) as paper/figS2_embedding/eval_embedding_bench_region.py."""
import os as _os
from pathlib import Path as _Path
REPO = str(_Path(__file__).resolve().parents[2])
RAWMOD_RESULTS = _os.environ.get('RAWMOD_RESULTS') or f'{REPO}/paper/data/runs'  # training runs; shipped metrics by default
RAWMOD_DATA = _os.environ.get('RAWMOD_DATA', 'data')  # datasets (not shipped: set this to score raw signal)
RAWMOD_WORK = _os.environ.get('RAWMOD_WORK') or f'{REPO}/paper/data/work'  # outputs of the paper scripts; shipped by default
import sys, numpy as np, pandas as pd, h5py, torch
sys.path.insert(0, f'{REPO}/scripts/evaluate')
import score_genome as sgm
D = sys.argv[1]
CKPT = _os.environ.get('CKPT') or f'{REPO}/checkpoints/rawmod.pt'
HW, L = 10, 10
def model_input(T):
    x = np.ascontiguousarray(np.transpose(T, (0, 3, 1, 2)))
    if x.shape[1] == 11: return x
    ch10 = np.zeros((x.shape[0], 1, x.shape[2], x.shape[3]), dtype=np.float32)
    ch10[:, 0, 1:] = x[:, 0, 1:] - x[:, 0, 0:1]
    cs, ce = HW * L, HW * L + L
    ref_ctr = x[:, 0, 0, cs:ce].mean(axis=1); read_ctr = x[:, 0, 1:, cs:ce].mean(axis=2)
    ch9 = np.zeros_like(ch10); ch9[:, 0, 1:, :] = (read_ctr - ref_ctr[:, None])[:, :, None]
    return np.concatenate([x, ch9, ch10], axis=1)
sites = pd.read_csv(f'{D}/sites.tsv', sep='\t').set_index('pos')
with h5py.File(f'{D}/features.h5', 'r') as F:
    print({k: F[k].shape for k in F.keys()})
    pos = F['ref_pos'][:].astype(int); keep = np.isin(pos, sites.index)
    idx = np.flatnonzero(keep); X = F['tensors'][idx].astype(np.float32); pos = pos[idx]; nr = F['n_reads'][idx]
_, first = np.unique(pos, return_index=True); X, pos, nr = X[first], pos[first], nr[first]   # one pileup per site
model, _ = sgm.load_model(CKPT, torch.device('cpu')); model.eval()
feats = []; hook = model.norm.register_forward_hook(lambda m, i, o: feats.append(o.detach().cpu().numpy()))
P = []
with torch.no_grad():
    for b in range(0, len(X), 256):
        P.append(torch.sigmoid(model(torch.from_numpy(model_input(X[b:b + 256]))).squeeze(-1)).numpy())
hook.remove(); Z, P = np.concatenate(feats), np.concatenate(P)
import umap
U = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(Z)
s = sites.loc[pos]
np.savez_compressed(f'{D}/embedding_dark_acart.npz', Z=Z, P=P, U=U, pos=pos, n_reads=nr,
                    base=s.base.to_numpy().astype(str), dorado=s.dorado.to_numpy().astype(str))
print(pd.DataFrame(dict(dorado=s.dorado.values, base=s.base.values, P=P)).groupby(['base', 'dorado']).P.describe().round(3).to_string())
