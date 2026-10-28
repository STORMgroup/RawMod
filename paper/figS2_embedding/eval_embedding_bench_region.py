#!/usr/bin/env python3
"""Site embeddings and scores of the region-holdout mixed model at up to 150 modified and 150 unmodified benchmark positions per row (nine organisms), from the same 10-read pileups as the benchmark. Writes embedding_bench_region.npz with a UMAP projection."""
import os as _os
from pathlib import Path as _Path
REPO = str(_Path(__file__).resolve().parents[2])
RAWMOD_RESULTS = _os.environ.get('RAWMOD_RESULTS') or f'{REPO}/paper/data/runs'  # training runs; shipped metrics by default
RAWMOD_DATA = _os.environ.get('RAWMOD_DATA', 'data')  # datasets (not shipped: set this to score raw signal)
RAWMOD_WORK = _os.environ.get('RAWMOD_WORK') or f'{REPO}/paper/data/work'  # outputs of the paper scripts; shipped by default

import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(f'{REPO}')
sys.path.insert(0, str(REPO / 'test' / 'benchmark'))
sys.path.insert(0, str(REPO / 'scripts' / 'evaluate'))
from bench import common as C, score as Sc  # noqa: E402
from bench.paper_table import GROUPS  # noqa: E402
import score_genome as sgm  # noqa: E402

CKPT = f'{RAWMOD_RESULTS}/region_mixed/models/mixed/mixed/best_model.pt'  # region-holdout mixed model (migration )
OUT = Path(__file__).resolve().parent
NPER = int(os.environ.get('NPER', '150'))  # sites per (row, label)
HW, L = 10, 10  # half window and samples per base, as in the images

MOD_OF_GROUP = {'5mC, CpG': '5mC', '5mC, non-CpG': '5mC', '5mC, all contexts': '5mC', '5hmC': '5hmC',
                '6mA': '6mA', '4mC': '4mC', '5hmU': '5hmU'}
ORGANISMS = [('syn_', 'ONT oligonucleotides'), ('hp26695', 'H. pylori 26695'), ('spo1', 'SPO1'),
             ('ecoli', 'E. coli'), ('anabaena', 'Anabaena'), ('arabidopsis', 'A. thaliana'),
             ('hg002', 'Human (HG002)'), ('mouse', 'Mouse brain'), ('rice', 'O. sativa')]


def organism(sample):
    return next(name for pre, name in ORGANISMS if sample.startswith(pre))


def model_input(T):
    """The model input built from stored images, as rawmod/featurize_fast.ScoreSink.model_input (bases visible)."""
    x = np.ascontiguousarray(np.transpose(T, (0, 3, 1, 2)))  # (B, C, H, W)
    ch10 = np.zeros((x.shape[0], 1, x.shape[2], x.shape[3]), dtype=np.float32)
    ch10[:, 0, 1:] = x[:, 0, 1:] - x[:, 0, 0:1]
    cs, ce = HW * L, HW * L + L
    ref_ctr = x[:, 0, 0, cs:ce].mean(axis=1)
    read_ctr = x[:, 0, 1:, cs:ce].mean(axis=2)
    ch9 = np.zeros_like(ch10)
    ch9[:, 0, 1:, :] = (read_ctr - ref_ctr[:, None])[:, :, None]
    return np.concatenate([x, ch9, ch10], axis=1)


def main():
    assert C.HOLDOUT == 'region', 'run with BENCH_HOLDOUT=region (sites inside the withheld regions)'
    rng = np.random.default_rng(0)
    S, rows = C.samples(), {r['row']: r for r in Sc.rows_cfg()}
    recs, seen = [], set()
    for group, members in GROUPS:
        for row, _ in members:
            for smp, st, y in Sc.row_sites(rows[row], S, {}):
                for lab in (1, 0):
                    idx = np.flatnonzero(y == lab)
                    for i in rng.permutation(idx)[:NPER]:
                        r = st.iloc[i]
                        k = (smp, r.strand, str(r.contig), int(r.pos))
                        if k in seen:
                            continue
                        seen.add(k)
                        recs.append(dict(row=row, sample=smp, strand=r.strand, contig=str(r.contig), pos=int(r.pos),
                                         label=lab, mod=MOD_OF_GROUP[group] if lab else 'unmodified',
                                         organism=organism(smp)))
    print('sites drawn', len(recs), flush=True)

  # gather the images, one h5 per (sample, strand)
    X = [None] * len(recs)
    by = {}
    for j, r in enumerate(recs):
        by.setdefault((r['sample'], r['strand']), []).append(j)
    for (smp, s), js in by.items():
        fn = f"features_{'plus' if s == '+' else 'minus'}.h5"
        h = C.WORK / smp / 'rawmod_features' / f'{C.TAG}_r15' / fn  # shared feature store (region benchmark)
        if not h.exists():
            h = C.WORK / smp / 'rawmod' / C.TAG / fn  # older per-column layout
        with h5py.File(h, 'r') as F:
            look = {(c, p): i for i, (c, p) in enumerate(zip(F['ref_names'][:].astype(str), F['ref_pos'][:].tolist()))}
            ii = [look[(recs[j]['contig'], recs[j]['pos'])] for j in js]
            order = np.argsort(ii)
            T = F['tensors'][np.array(ii)[order]]
            nr = F['n_reads'][np.array(ii)[order]]
        for o, t, n in zip(order, T, nr):
            X[js[o]] = t
            recs[js[o]]['n_reads'] = int(n)
    X = np.stack(X).astype(np.float32)

    dev = torch.device('cpu')
    model, _ = sgm.load_model(CKPT, dev)
    model.eval()
    feats = []
    hook = model.norm.register_forward_hook(lambda m, i, o: feats.append(o.detach().cpu().numpy()))
    P = []
    with torch.no_grad():
        for b in range(0, len(X), 256):
            P.append(torch.sigmoid(model(torch.from_numpy(model_input(X[b:b + 256]))).squeeze(-1)).numpy())
    hook.remove()
    Z, P = np.concatenate(feats), np.concatenate(P)

    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import roc_auc_score
    D = pd.DataFrame(recs)
    D['refbase'] = ['ACGT'[int(np.argmax(X[j, 0, HW * L, 2:6]))] for j in range(len(X))]  # reference row one-hot
    res = {'n_sites': len(D), 'n_mod': int(D.label.sum()), 'organisms': D.organism.value_counts().to_dict(),
           'auroc_all': float(roc_auc_score(D.label, P))}
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    neg = (D.label == 0).to_numpy()
    res['probe_organism_unmod'] = float(cross_val_score(LogisticRegression(max_iter=3000), Z[neg], D.organism[neg], cv=cv).mean())
    res['probe_organism_unmod_majority'] = float(D.organism[neg].value_counts(normalize=True).max())
    res['probe_label_all'] = float(cross_val_score(LogisticRegression(max_iter=3000), Z, D.label, cv=cv).mean())
    print(json.dumps(res, indent=1), flush=True)

    import umap
    U = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(Z)
    np.savez_compressed(OUT / 'embedding_bench_region.npz', Z=Z, P=P, U=U,
                        **{c: D[c].to_numpy().astype(str) for c in ('row', 'sample', 'organism', 'mod', 'refbase')},
                        label=D.label.to_numpy())
    json.dump(res, open(OUT / 'embedding_bench_region.json', 'w'), indent=1)
    print('wrote', OUT / 'embedding_bench_region.npz', flush=True)


if __name__ == '__main__':
    main()
