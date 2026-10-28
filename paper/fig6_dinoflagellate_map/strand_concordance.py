#!/usr/bin/env python3
"""Concordance of the forward- and reverse-strand maps in 10-kb windows and at individual positions. Usage: strand_concordance.py [MINUS_SCORES PLUS_SCORES]."""
import os as _os
from pathlib import Path as _Path
REPO = str(_Path(__file__).resolve().parents[2])
RAWMOD_RESULTS = _os.environ.get('RAWMOD_RESULTS') or f'{REPO}/paper/data/runs'  # training runs; shipped metrics by default
RAWMOD_DATA = _os.environ.get('RAWMOD_DATA', 'data')  # datasets (not shipped: set this to score raw signal)
RAWMOD_WORK = _os.environ.get('RAWMOD_WORK') or f'{REPO}/paper/data/work'  # outputs of the paper scripts; shipped by default

import json, re, sys
from collections import Counter
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr

BASE = f'{RAWMOD_DATA}/acarterae_contig_3762'
W = f'{RAWMOD_WORK}/dinoflagellate/strand'
MINUS = sys.argv[1] if len(sys.argv) > 1 else f'{RAWMOD_WORK}/dinoflagellate/contig_3762_strand-_scores.tsv.gz'
PLUS = sys.argv[2] if len(sys.argv) > 2 else f'{RAWMOD_WORK}/dinoflagellate/contig_3762_strand+_scores.tsv.gz'
CATS = ['intergenic', 'intron', 'UTR', 'CDS']
rng = np.random.default_rng(7)
out = {'plus_file': PLUS, 'minus_file': MINUS}

seq = ''.join(l.strip() for l in open(f'{BASE}/contig_3762.fasta') if not l.startswith('>')).upper()
N = len(seq)
sb = np.frombuffer(seq.encode(), dtype=np.uint8)
isGC = (sb == ord('G')) | (sb == ord('C'))

# structural labels exactly as dino_analysis.py (later type overrides earlier)
rows = []
for line in open(f'{BASE}/Annotations_contig_3762.gff3'):
    if line.startswith('#') or not line.strip():
        continue
    p = line.rstrip('\n').split('\t')
    rows.append((p[2], int(p[3]) - 1, int(p[4]), p[6]))
gff = pd.DataFrame(rows, columns=['type', 'start', 'end', 'strand'])
lab = np.zeros(N, dtype=np.int8)
for t, code in (('gene', 1), ('exon', 2), ('CDS', 3)):
    for s, e in gff.loc[gff.type == t, ['start', 'end']].values:
        lab[s:e] = code

# repeat classes exactly as dino_repeats.py
fam_n = Counter(); rx = re.compile(r'Motif:([^"]+)"'); rep = []
for line in open(f'{BASE}/external/Repeats_Filtered_Final.gff'):
    if line.startswith('#'):
        continue
    p = line.split('\t'); mm = rx.search(p[8])
    if not mm:
        continue
    fam_n[mm.group(1)] += 1
    if p[0] == 'contig_3762':
        rep.append((int(p[3]) - 1, int(p[4]), float(p[5]), mm.group(1)))
inter = np.zeros(N, bool); simple = np.zeros(N, bool); div = np.full(N, np.inf, dtype=np.float32)
for s, e, d, fam in rep:
    if fam.endswith(')n') or '-rich' in fam:
        simple[s:e] = True
    else:
        inter[s:e] = True; div[s:e] = np.minimum(div[s:e], d)

def load(f):
    d = pd.read_csv(f, sep='\t')
    d = d[d.contig == 'contig_3762']
    return d.pos.values.astype(np.int64), d.score.values.astype(np.float64), d.n_reads.values

P = {}
for nm, f in (('plus', PLUS), ('minus', MINUS)):
    pos, sc, nr = load(f)
    P[nm] = (pos, sc)
    refb = np.unique(sb[pos]); base = ''.join(chr(b) for b in refb)
    out[f'{nm}_sites'] = dict(n=int(len(pos)), ref_bases=base, median_reads=float(np.median(nr)),
                              mean=float(sc.mean()), frac05=float((sc > .5).mean()),
                              frac_of_ref_base=float(len(pos) / (sb == refb[0]).sum()))
print(json.dumps({k: out[k] for k in ('plus_sites', 'minus_sites')}, indent=1))

# ------------------------------------------------------------ 1. 10-kb windows
WIN = 10_000; nw = N // WIN + 1
MINSITES = 200
def wmean(pos, sc):
    wi = pos // WIN
    s = np.bincount(wi, weights=sc, minlength=nw); c = np.bincount(wi, minlength=nw)
    return np.where(c >= MINSITES, s / np.maximum(c, 1), np.nan), c
wp, cp = wmean(*P['plus']); wm, cm = wmean(*P['minus'])
wgc = np.bincount(np.arange(N) // WIN, weights=isGC.astype(float), minlength=nw) / WIN
wcds = np.bincount(np.arange(N) // WIN, weights=(lab == 3).astype(float), minlength=nw) / WIN
ok = ~np.isnan(wp) & ~np.isnan(wm)
res = {}
def corr(a, b):
    return dict(spearman=float(spearmanr(a, b)[0]), pearson=float(pearsonr(a, b)[0]), n=int(len(a)))
res['raw'] = corr(wp[ok], wm[ok])
res['plus_vs_gc'] = corr(wp[ok], wgc[ok]); res['minus_vs_gc'] = corr(wm[ok], wgc[ok])
res['plus_vs_cds'] = corr(wp[ok], wcds[ok]); res['minus_vs_cds'] = corr(wm[ok], wcds[ok])
def resid(y, x, deg):
    return y - np.polyval(np.polyfit(x, y, deg), x)
res['gc_regressed_linear'] = corr(resid(wp[ok], wgc[ok], 1), resid(wm[ok], wgc[ok], 1))
res['gc_regressed_quadratic'] = corr(resid(wp[ok], wgc[ok], 2), resid(wm[ok], wgc[ok], 2))
# non-parametric: subtract the mean of the GC bin (2% bins) on each strand
gb = np.minimum((wgc[ok] * 50).astype(int), 49)
def binresid(y):
    s = np.bincount(gb, weights=y, minlength=50); c = np.bincount(gb, minlength=50)
    return y - s[gb] / c[gb]
res['gc_binned_2pct'] = corr(binresid(wp[ok]), binresid(wm[ok]))
# both GC and CDS fraction regressed (linear)
X = np.column_stack([np.ones(ok.sum()), wgc[ok], wcds[ok]])
def mresid(y):
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
res['gc_and_cds_regressed'] = corr(mresid(wp[ok]), mresid(wm[ok]))
# autocorrelation-aware null: circularly shift the minus track by random whole windows
sh = []
pw, mw = wp[ok], wm[ok]
for k in rng.integers(20, ok.sum() - 20, 2000):
    sh.append(spearmanr(pw, np.roll(mw, k))[0])
sh = np.array(sh)
res['shift_null_spearman'] = dict(mean=float(sh.mean()), sd=float(sh.std()), max=float(sh.max()),
                                  p=float((1 + (sh >= res['raw']['spearman']).sum()) / 2001))
# 100-kb windows as well (coarser)
WIN2 = 100_000; nw2 = N // WIN2 + 1
def wmean2(pos, sc):
    wi = pos // WIN2
    s = np.bincount(wi, weights=sc, minlength=nw2); c = np.bincount(wi, minlength=nw2)
    return np.where(c >= 2000, s / np.maximum(c, 1), np.nan)
a, b = wmean2(*P['plus']), wmean2(*P['minus']); ok2 = ~np.isnan(a) & ~np.isnan(b)
res['raw_100kb'] = corr(a[ok2], b[ok2])
out['windows'] = res
print(json.dumps(res, indent=1))
pd.DataFrame(dict(win_start=np.arange(nw)[ok] * WIN, plus=wp[ok], minus=wm[ok], n_plus=cp[ok], n_minus=cm[ok],
                  gc=wgc[ok], cds_frac=wcds[ok])).to_csv(f'{W}/windows10kb_both_strands.tsv', sep='\t', index=False,
                                                        float_format='%.5f')

# ------------------------------------------------------------ 2. class means
def classes(pos):
    L = lab[pos]; R = inter[pos]; S = simple[pos] & ~R; D = div[pos]
    g = {k: L == i for i, k in enumerate(CATS)}
    g['unique'] = ~R & ~S; g['simple repeat'] = S; g['interspersed repeat'] = R
    for lo, hi in ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 45)):
        g[f'interspersed div {lo}-{hi}%'] = R & (D >= lo) & (D < hi)
    for i, k in enumerate(CATS):
        g[f'{k} | unique'] = (L == i) & ~R & ~S
        g[f'{k} | interspersed'] = (L == i) & R
    return g
tab = []
gp, gm = classes(P['plus'][0]), classes(P['minus'][0])
for k in gp:
    sp, sm = P['plus'][1][gp[k]], P['minus'][1][gm[k]]
    tab.append(dict(cls=k, n_plus=int(gp[k].sum()), mean_plus=sp.mean(), frac05_plus=(sp > .5).mean(),
                    n_minus=int(gm[k].sum()), mean_minus=sm.mean(), frac05_minus=(sm > .5).mean()))
tab = pd.DataFrame(tab)
cds_p = tab.loc[tab.cls == 'CDS', 'mean_plus'].item(); cds_m = tab.loc[tab.cls == 'CDS', 'mean_minus'].item()
tab['fold_cds_plus'] = tab.mean_plus / cds_p; tab['fold_cds_minus'] = tab.mean_minus / cds_m
tab.to_csv(f'{W}/class_means_both_strands.tsv', sep='\t', index=False, float_format='%.4f')
print(tab.to_string(float_format=lambda x: f'{x:.4f}'))
core = tab[tab.cls.isin(CATS + ['unique', 'simple repeat', 'interspersed repeat'])]
divb = tab[tab.cls.str.startswith('interspersed div')]
out['class_rank_concordance'] = dict(
    structural_and_repeat_classes=dict(n=int(len(core)), spearman=float(spearmanr(core.mean_plus, core.mean_minus)[0]),
                                       pearson=float(pearsonr(core.mean_plus, core.mean_minus)[0])),
    all_classes=dict(n=int(len(tab)), spearman=float(spearmanr(tab.mean_plus, tab.mean_minus)[0])),
    divergence_bins=dict(spearman=float(spearmanr(divb.mean_plus, divb.mean_minus)[0]),
                         order_plus=divb.mean_plus.round(4).tolist(), order_minus=divb.mean_minus.round(4).tolist()))
# divergence-score rho within interspersed repeats, both strands
for nm in ('plus', 'minus'):
    pos, sc = P[nm]; R = inter[pos]
    out[f'rho_div_score_interspersed_{nm}'] = float(spearmanr(div[pos][R], sc[R])[0])
print(json.dumps({k: v for k, v in out.items() if k.startswith(('class_rank', 'rho_div'))}, indent=1))

# ------------------------------------------------------------ 3. circular-shift null, both strands
NPERM = 1000
out['perm_ig_minus_cds'] = {}
for nm in ('plus', 'minus'):
    pos, sc = P[nm]
    def cm_(c):
        s = np.bincount(c, weights=sc, minlength=4); n = np.bincount(c, minlength=4); return s / n
    m = cm_(lab[pos]); obs = m[0] - m[3]
    null = []
    for k in rng.integers(200_000, N - 200_000, NPERM):
        mm = cm_(lab[(pos - k) % N]); null.append(mm[0] - mm[3])
    null = np.array(null)
    out['perm_ig_minus_cds'][nm] = dict(obs=float(obs), fold=float(m[0] / m[3]), null_mean=float(null.mean()),
                                        null_sd=float(null.std()), z=float((obs - null.mean()) / null.std()),
                                        p=float((1 + (np.abs(null - null.mean()) >= abs(obs - null.mean())).sum()) / (1 + NPERM)))
print(json.dumps(out['perm_ig_minus_cds'], indent=1))
json.dump(out, open(f'{W}/strand_concordance.json', 'w'), indent=1)
print('wrote', f'{W}/strand_concordance.json')
