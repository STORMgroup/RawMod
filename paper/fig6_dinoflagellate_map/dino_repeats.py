#!/usr/bin/env python3
"""Scores of repeat thymines by RepeatMasker divergence from the family consensus and by repeat class."""
import os as _os
from pathlib import Path as _Path
REPO = str(_Path(__file__).resolve().parents[2])
RAWMOD_RESULTS = _os.environ.get('RAWMOD_RESULTS') or f'{REPO}/paper/data/runs'  # training runs; shipped metrics by default
RAWMOD_DATA = _os.environ.get('RAWMOD_DATA', 'data')  # datasets (not shipped: set this to score raw signal)
RAWMOD_WORK = _os.environ.get('RAWMOD_WORK') or f'{REPO}/paper/data/work'  # outputs of the paper scripts; shipped by default

import json, re
from collections import Counter
import numpy as np, pandas as pd

BASE = f'{RAWMOD_DATA}/acarterae_contig_3762'
OUT = f'{RAWMOD_WORK}/dinoflagellate/out'
N = 7733821
rng = np.random.default_rng(11)
st = {}

# genome-wide family copy number / bp
fam_n, fam_bp = Counter(), Counter()
rx = re.compile(r'Motif:([^"]+)"')
rows3762 = []
with open(f'{BASE}/external/Repeats_Filtered_Final.gff') as f:
    for line in f:
        if line.startswith('#'):
            continue
        p = line.split('\t')
        mm = rx.search(p[8])
        if not mm:
            continue
        fam = mm.group(1)
        fam_n[fam] += 1; fam_bp[fam] += int(p[4]) - int(p[3]) + 1
        if p[0] == 'contig_3762':
            rows3762.append((int(p[3]) - 1, int(p[4]), float(p[5]), fam))
rep = pd.DataFrame(rows3762, columns=['start', 'end', 'div', 'fam'])
rep['simple'] = rep.fam.str.endswith(')n') | rep.fam.str.contains('-rich')
rep['copies'] = rep.fam.map(fam_n)
print('repeat records on contig', len(rep), 'families genome-wide', len(fam_n))

# per-base tracks: interspersed flag, simple flag, min divergence, max family copy number
inter = np.zeros(N, bool); simple = np.zeros(N, bool)
div = np.full(N, np.inf, dtype=np.float32); cop = np.zeros(N, dtype=np.int64)
for s, e, d, sim, c in rep[['start', 'end', 'div', 'simple', 'copies']].itertuples(index=False):
    if sim:
        simple[s:e] = True
    else:
        inter[s:e] = True
        div[s:e] = np.minimum(div[s:e], d)
        cop[s:e] = np.maximum(cop[s:e], c)
st['interspersed_bp_frac'] = float(inter.mean()); st['simple_bp_frac'] = float(simple.mean())

# structural labels (same as dino_analysis.py)
lab = np.zeros(N, dtype=np.int8)
for line in open(f'{BASE}/Annotations_contig_3762.gff3'):
    if line.startswith('#') or not line.strip():
        continue
    p = line.split('\t')
    code = {'gene': 1, 'exon': 2, 'CDS': 3}.get(p[2])
    if code:
        s, e = int(p[3]) - 1, int(p[4])
        lab[s:e] = np.maximum(lab[s:e], code)
CATS = ['intergenic', 'intron', 'UTR', 'CDS']
st['repeat_frac_by_category'] = {k: float(inter[lab == i].mean()) for i, k in enumerate(CATS)}

df = pd.read_csv(f'{RAWMOD_WORK}/dinoflagellate/contig_3762_strand+_scores.tsv.gz', sep='\t')
pos = df.pos.values; sc = df.score.values
R = inter[pos]; S = simple[pos] & ~R; D = div[pos]; C = cop[pos]; L = lab[pos]
grp = np.where(R, 'interspersed', np.where(S, 'simple', 'unique'))
st['by_repeat_class'] = {g: dict(n=int((grp == g).sum()), mean=float(sc[grp == g].mean()),
                                 frac05=float((sc[grp == g] > .5).mean()))
                         for g in ('unique', 'simple', 'interspersed')}
print(st['by_repeat_class'])

# repeat x structure
st['repeat_x_category'] = {}
for i, k in enumerate(CATS):
    for flag, nm in ((True, 'repeat'), (False, 'unique')):
        sel = (L == i) & (R == flag)
        st['repeat_x_category'][f'{k}|{nm}'] = dict(n=int(sel.sum()), mean=float(sc[sel].mean()))
print(json.dumps(st['repeat_x_category'], indent=1))

# divergence bins (interspersed only) -- young (low div) vs old
dbins = [0, 5, 10, 15, 20, 25, 45]
st['by_divergence'] = []
for a, b in zip(dbins[:-1], dbins[1:]):
    sel = R & (D >= a) & (D < b)
    st['by_divergence'].append(dict(lo=a, hi=b, n=int(sel.sum()), mean=float(sc[sel].mean()),
                                    frac05=float((sc[sel] > .5).mean())))
print(st['by_divergence'])
from scipy.stats import spearmanr
st['rho_div_score_interspersed'] = float(spearmanr(D[R], sc[R])[0])

# family copy-number bins
cbins = [1, 10, 100, 1000, 10000, 10**7]
st['by_copies'] = []
for a, b in zip(cbins[:-1], cbins[1:]):
    sel = R & (C >= a) & (C < b)
    st['by_copies'].append(dict(lo=a, hi=b, n=int(sel.sum()), mean=float(sc[sel].mean())))
print(st['by_copies'])

# circular-shift null for repeat vs unique difference (annotation shifted)
obs = sc[R].mean() - sc[grp == 'unique'].mean()
null = []
for k in rng.integers(200_000, N - 200_000, 1000):
    q = (pos - k) % N
    r2 = inter[q]; u2 = ~inter[q] & ~simple[q]
    null.append(sc[r2].mean() - sc[u2].mean())
null = np.array(null)
st['perm_repeat_minus_unique'] = dict(obs=float(obs), null_mean=float(null.mean()), null_sd=float(null.std()),
                                      z=float((obs - null.mean()) / null.std()),
                                      p=float((1 + (np.abs(null - null.mean()) >= abs(obs - null.mean())).sum()) / 1001))
# divergence effect null: shift the divergence track, correlation among repeat sites
nd = []
for k in rng.integers(200_000, N - 200_000, 300):
    q = (pos - k) % N
    r2 = inter[q]
    nd.append(spearmanr(div[q][r2], sc[r2])[0])
nd = np.array(nd)
st['perm_rho_div'] = dict(obs=st['rho_div_score_interspersed'], null_mean=float(nd.mean()),
                          null_sd=float(nd.std()),
                          z=float((st['rho_div_score_interspersed'] - nd.mean()) / nd.std()))
print(st['perm_repeat_minus_unique'], st['perm_rho_div'])

# repeat content of intergenic gaps (from dino_analysis gap table)
gaps = pd.read_csv(f'{OUT}/dino_intergenic_gaps.tsv', sep='\t')
gaps['repeat_frac'] = [inter[s:e].mean() for s, e in zip(gaps.start, gaps.end)]
gaps['young_frac'] = [(inter[s:e] & (div[s:e] < 10)).mean() for s, e in zip(gaps.start, gaps.end)]
gaps.to_csv(f'{OUT}/dino_intergenic_gaps.tsv', sep='\t', index=False)
st['gap_rho_mean_vs_repeat_frac'] = float(spearmanr(gaps.repeat_frac, gaps['mean'])[0])
st['gap_rho_mean_vs_young_frac'] = float(spearmanr(gaps.young_frac, gaps['mean'])[0])
print(gaps[gaps.kind != 'within'])
print('gap rho repeat', st['gap_rho_mean_vs_repeat_frac'], 'young', st['gap_rho_mean_vs_young_frac'])

# 10-kb window tracks
WIN = 10_000; nw = N // WIN + 1
ar = np.arange(N) // WIN
wrep = np.bincount(ar, weights=inter.astype(float), minlength=nw) / WIN
wyoung = np.bincount(ar, weights=(inter & (div < 10)).astype(float), minlength=nw) / WIN
wi = pos // WIN
wmean = np.bincount(wi, weights=sc, minlength=nw) / np.maximum(np.bincount(wi, minlength=nw), 1)
okw = np.bincount(wi, minlength=nw) >= 200
st['window_rho_vs_repeat'] = float(spearmanr(wmean[okw], wrep[okw])[0])
st['window_rho_vs_young_repeat'] = float(spearmanr(wmean[okw], wyoung[okw])[0])
print('window rho repeat', st['window_rho_vs_repeat'], 'young', st['window_rho_vs_young_repeat'])

np.savez_compressed(f'{OUT}/dino_repeat_cache.npz', wrep=wrep, wyoung=wyoung,
                    div_lo=np.array(dbins[:-1]), div_mean=np.array([d['mean'] for d in st['by_divergence']]),
                    div_n=np.array([d['n'] for d in st['by_divergence']]))
json.dump(st, open(f'{OUT}/dino_repeat_stats.json', 'w'), indent=1)
