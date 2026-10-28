#!/usr/bin/env python3
"""A. carterae contig_3762: mean 5hmU score by annotation class (CDS, intron, intergenic, repeat), with a circular-shift null and residualization on 5-mer context and local GC content. Writes dino_stats.json."""
import os as _os
from pathlib import Path as _Path
REPO = str(_Path(__file__).resolve().parents[2])
RAWMOD_RESULTS = _os.environ.get('RAWMOD_RESULTS') or f'{REPO}/paper/data/runs'  # training runs; shipped metrics by default
RAWMOD_DATA = _os.environ.get('RAWMOD_DATA', 'data')  # datasets (not shipped: set this to score raw signal)
RAWMOD_WORK = _os.environ.get('RAWMOD_WORK') or f'{REPO}/paper/data/work'  # outputs of the paper scripts; shipped by default

import json
import numpy as np, pandas as pd

BASE = f'{RAWMOD_DATA}/acarterae_contig_3762'
OUT = f'{RAWMOD_WORK}/dinoflagellate/out'
GFF = f'{BASE}/Annotations_contig_3762.gff3'
CATS = ['intergenic', 'intron', 'UTR', 'CDS']
rng = np.random.default_rng(7)
stats = {}

# ---------------------------------------------------------------- reference
seq = ''.join(l.strip() for l in open(f'{BASE}/contig_3762.fasta') if not l.startswith('>'))
N = len(seq)
sb = np.frombuffer(seq.upper().encode(), dtype=np.uint8)
print('contig bp', N)

# ---------------------------------------------------------------- annotation
rows = []
for line in open(GFF):
    if line.startswith('#') or not line.strip():
        continue
    p = line.rstrip('\n').split('\t')
    rows.append((p[2], int(p[3]) - 1, int(p[4]), p[6]))
gff = pd.DataFrame(rows, columns=['type', 'start', 'end', 'strand'])
lab = np.zeros(N, dtype=np.int8)  # 0 intergenic
for t, code in (('gene', 1), ('exon', 2), ('CDS', 3)):  # later overrides earlier
    for s, e in gff.loc[gff.type == t, ['start', 'end']].values:
        lab[s:e] = code
genes = gff[gff.type == 'gene'].sort_values('start').reset_index(drop=True)

# ---------------------------------------------------------------- scores
df = pd.read_csv(f'{RAWMOD_WORK}/dinoflagellate/contig_3762_strand+_scores.tsv.gz', sep='\t')
pos = df.pos.values
sc = df.score.values.astype(np.float64)
cat = lab[pos]
stats['n_sites'] = int(len(df))
stats['n_T_ref_plus'] = int((sb == ord('T')).sum())
stats['median_reads'] = float(np.median(df.n_reads))
stats['genome_mean'] = float(sc.mean())
stats['genome_frac05'] = float((sc > 0.5).mean())


def cat_means(values, c):
    s = np.bincount(c, weights=values, minlength=4)
    n = np.bincount(c, minlength=4)
    return s / n, n


m, n = cat_means(sc, cat)
f5, _ = cat_means((sc > 0.5).astype(float), cat)
stats['categories'] = {k: dict(n=int(n[i]), mean=float(m[i]), frac05=float(f5[i]),
                               fold_vs_cds=float(m[i] / m[3]))
                       for i, k in enumerate(CATS)}
print(pd.DataFrame(stats['categories']).T)

# ---------------------------------------------------------------- 2. context
code = np.full(256, -1, dtype=np.int64)
for i, b in enumerate(b'ACGT'):
    code[b] = i
enc = code[sb]
ok = np.ones(len(pos), bool)
kmer = np.zeros(len(pos), dtype=np.int64)
for off in (-2, -1, 1, 2):
    q = pos + off
    good = (q >= 0) & (q < N)
    v = np.where(good, enc[np.clip(q, 0, N - 1)], -1)
    ok &= v >= 0
    kmer = kmer * 4 + np.maximum(v, 0)
cs = np.concatenate([[0], np.cumsum((sb == ord('G')) | (sb == ord('C')))])
lo, hi = np.clip(pos - 50, 0, N), np.clip(pos + 51, 0, N)
gc = (cs[hi] - cs[lo]) / (hi - lo)
gcbin = np.minimum((gc * 20).astype(int), 19)
stats['gc_by_category'] = {k: float(gc[cat == i].mean()) for i, k in enumerate(CATS)}

adj = {}
for name, key in (('5mer', kmer), ('gc', gcbin), ('5mer_x_gc', kmer * 20 + gcbin)):
    kk = key[ok]
    s = np.bincount(kk, weights=sc[ok]); c = np.bincount(kk)
    expect = s[kk] / c[kk]
    resid = sc[ok] - expect + sc[ok].mean()
    mm, _ = cat_means(resid, cat[ok])
  # variance explained by the context alone
    r2 = 1 - ((sc[ok] - expect) ** 2).sum() / ((sc[ok] - sc[ok].mean()) ** 2).sum()
    adj[name] = dict(r2=float(r2), **{k: float(mm[i]) for i, k in enumerate(CATS)},
                     fold_ig_vs_cds=float(mm[0] / mm[3]))
stats['context_adjusted'] = adj
print(json.dumps(adj, indent=1))

# 5-mer composition difference between intergenic and CDS (JS divergence)
def dist(i):
    h = np.bincount(kmer[ok & (cat == i)], minlength=1024).astype(float)
    return h / h.sum()
P, Q = dist(0), dist(3)
M = (P + Q) / 2
js = 0.5 * np.nansum(np.where(P > 0, P * np.log2(P / M), 0)) + \
     0.5 * np.nansum(np.where(Q > 0, Q * np.log2(Q / M), 0))
stats['js_5mer_intergenic_vs_cds_bits'] = float(js)

# ---------------------------------------------------------------- 3. circular shift null
NPERM = 2000
obs = dict(ig_cds=m[0] - m[3], ig_intron=m[0] - m[1], intron_cds=m[1] - m[3])
null = {k: [] for k in obs}
shifts = rng.integers(200_000, N - 200_000, NPERM)
for k in shifts:
    c2 = lab[(pos - k) % N]
    mm, _ = cat_means(sc, c2)
    null['ig_cds'].append(mm[0] - mm[3]); null['ig_intron'].append(mm[0] - mm[1])
    null['intron_cds'].append(mm[1] - mm[3])
stats['perm'] = {}
for k in obs:
    a = np.array(null[k])
    stats['perm'][k] = dict(obs=float(obs[k]), null_mean=float(a.mean()), null_sd=float(a.std()),
                            p_two_sided=float((1 + (np.abs(a - a.mean()) >= abs(obs[k] - a.mean())).sum()) / (1 + NPERM)),
                            z=float((obs[k] - a.mean()) / a.std()))
print(json.dumps(stats['perm'], indent=1))
np.save(f'{OUT}/_perm_ig_cds.npy', np.array(null['ig_cds']))

# ---------------------------------------------------------------- 4. block bootstrap
BLK = 50_000
blk = pos // BLK
nb = blk.max() + 1
S = np.zeros((nb, 4)); C = np.zeros((nb, 4))
np.add.at(S, (blk, cat), sc); np.add.at(C, (blk, cat), 1)
boot = []
for _ in range(2000):
    idx = rng.integers(0, nb, nb)
    mm = S[idx].sum(0) / C[idx].sum(0)
    boot.append(mm / mm[3])
boot = np.array(boot)
stats['bootstrap_fold_vs_cds_95ci'] = {k: [float(np.percentile(boot[:, i], 2.5)),
                                           float(np.percentile(boot[:, i], 97.5))]
                                       for i, k in enumerate(CATS)}
bootm = []
for _ in range(2000):
    idx = rng.integers(0, nb, nb)
    bootm.append(S[idx].sum(0) / C[idx].sum(0))
bootm = np.array(bootm)
stats['bootstrap_mean_95ci'] = {k: [float(np.percentile(bootm[:, i], 2.5)),
                                    float(np.percentile(bootm[:, i], 97.5))]
                                for i, k in enumerate(CATS)}
print(stats['bootstrap_fold_vs_cds_95ci'])

# ---------------------------------------------------------------- 5. arrays & switches
g = genes[['start', 'end', 'strand']].values
arrays, switches = [], []
a0 = 0
for i in range(1, len(g) + 1):
    if i == len(g) or g[i][2] != g[i - 1][2]:
        arrays.append(dict(start=int(g[a0][0]), end=int(max(x[1] for x in g[a0:i])),
                           strand=g[a0][2], n_genes=int(i - a0)))
        if i < len(g):
            kind = 'convergent' if (g[i - 1][2] == '+' and g[i][2] == '-') else 'divergent'
            left = int(max(x[1] for x in g[a0:i]))
            switches.append(dict(kind=kind, gap_start=left, gap_end=int(g[i][0]),
                                 mid=int((left + g[i][0]) // 2)))
        a0 = i
stats['arrays'] = arrays; stats['switches'] = switches
print(arrays); print(switches)

# intergenic sites: in switch gap vs within-array gap vs contig-end flank
inter = cat == 0
region = np.full(len(pos), 'array-internal', dtype='<U16')
for s in switches:
    region[(pos >= s['gap_start']) & (pos < s['gap_end'])] = s['kind']
region[pos < arrays[0]['start']] = 'contig end'
region[pos >= arrays[-1]['end']] = 'contig end'
stats['intergenic_by_region'] = {}
for r in ('array-internal', 'convergent', 'divergent', 'contig end'):
    sel = inter & (region == r)
    stats['intergenic_by_region'][r] = dict(n=int(sel.sum()), mean=float(sc[sel].mean()),
                                            frac05=float((sc[sel] > 0.5).mean()))
print(stats['intergenic_by_region'])
# null for switch gaps: random intergenic gaps between same-strand genes, length-matched
gaps = [(int(g[i - 1][1]), int(g[i][0])) for i in range(1, len(g))
        if g[i][2] == g[i - 1][2] and g[i][0] - g[i - 1][1] > 2000]
gap_means = []
for s_, e_ in gaps:
    sel = (pos >= s_) & (pos < e_)
    if sel.sum() > 200:
        gap_means.append((e_ - s_, sc[sel].mean()))
gap_means = np.array(gap_means)
stats['within_array_gaps'] = dict(n=int(len(gap_means)),
                                  mean_of_means=float(gap_means[:, 1].mean()),
                                  pct95=float(np.percentile(gap_means[:, 1], 95)),
                                  max=float(gap_means[:, 1].max()))
sw_means = []
for s in switches:
    sel = (pos >= s['gap_start']) & (pos < s['gap_end'])
    s['mean'] = float(sc[sel].mean()); s['n_sites'] = int(sel.sum())
    s['rank_among_gaps'] = float((gap_means[:, 1] < s['mean']).mean())
    sw_means.append(s['mean'])
print('switch means', sw_means, 'within-array gap means pct95', stats['within_array_gaps'])

# profile around switch-gap midpoints (+-150 kb, 5 kb bins) + genome background
W, B = 150_000, 5_000
edges = np.arange(-W, W + 1, B)
prof = {}
for kind in ('convergent', 'divergent'):
    ys = []
    for s in switches:
        if s['kind'] != kind:
            continue
        d = pos - s['mid']
        sel = np.abs(d) < W
        b = ((d[sel] + W) // B).astype(int)
        ys.append(np.bincount(b, weights=sc[sel], minlength=len(edges) - 1) /
                  np.maximum(np.bincount(b, minlength=len(edges) - 1), 1))
    prof[kind] = np.mean(ys, 0)
# null band: random midpoints
nullprof = []
for _ in range(500):
    mid = rng.integers(W, N - W)
    d = pos - mid; sel = np.abs(d) < W
    b = ((d[sel] + W) // B).astype(int)
    nullprof.append(np.bincount(b, weights=sc[sel], minlength=len(edges) - 1) /
                    np.maximum(np.bincount(b, minlength=len(edges) - 1), 1))
nullprof = np.array(nullprof)

# oriented gene-body metaplot: 5 kb flanks (10 bins) + scaled body (40 bins)
FL, NBF, NBB = 5_000, 10, 40
meta_s = np.zeros(NBF * 2 + NBB); meta_c = np.zeros_like(meta_s)
meta_by = {'+': [np.zeros_like(meta_s), np.zeros_like(meta_s)],
           '-': [np.zeros_like(meta_s), np.zeros_like(meta_s)]}
order = np.argsort(pos)
ps, ss = pos[order], sc[order]
for s_, e_, st in g:
    i0, i1 = np.searchsorted(ps, s_ - FL), np.searchsorted(ps, e_ + FL)
    p_, v_ = ps[i0:i1], ss[i0:i1]
    rel = np.where(p_ < s_, (p_ - (s_ - FL)) / FL * NBF,
          np.where(p_ >= e_, NBF + NBB + (p_ - e_) / FL * NBF,
                   NBF + (p_ - s_) / max(e_ - s_, 1) * NBB))
    b = np.clip(rel.astype(int), 0, NBF * 2 + NBB - 1)
    if st == '-':
        b = NBF * 2 + NBB - 1 - b
    cs_ = np.bincount(b, weights=v_, minlength=len(meta_s))
    cc_ = np.bincount(b, minlength=len(meta_s))
    meta_s += cs_; meta_c += cc_
    meta_by[st][0] += cs_; meta_by[st][1] += cc_
meta = meta_s / meta_c

# windowed track (10 kb) for genome map
WIN = 10_000
wi = pos // WIN
nw = N // WIN + 1
wsum = np.bincount(wi, weights=sc, minlength=nw); wcnt = np.bincount(wi, minlength=nw)
wf = np.bincount(wi, weights=(sc > 0.5).astype(float), minlength=nw)
wmean = np.where(wcnt >= 200, wsum / np.maximum(wcnt, 1), np.nan)
wfrac = np.where(wcnt >= 200, wf / np.maximum(wcnt, 1), np.nan)
wcds = np.bincount(np.arange(N) // WIN, weights=(lab == 3).astype(float), minlength=nw) / WIN
wgenic = np.bincount(np.arange(N) // WIN, weights=(lab > 0).astype(float), minlength=nw) / WIN
wgc = np.bincount(np.arange(N) // WIN, weights=((sb == ord('G')) | (sb == ord('C'))).astype(float),
                  minlength=nw) / WIN
from scipy.stats import spearmanr
okw = ~np.isnan(wmean)
stats['window10kb'] = dict(
    n=int(okw.sum()),
    rho_vs_cds_frac=float(spearmanr(wmean[okw], wcds[okw])[0]),
    rho_vs_genic_frac=float(spearmanr(wmean[okw], wgenic[okw])[0]),
    rho_vs_gc=float(spearmanr(wmean[okw], wgc[okw])[0]),
)
print(stats['window10kb'])

np.savez_compressed(f'{OUT}/dino_cache.npz', m=m, f5=f5, n=n, edges=edges,
                    prof_conv=prof['convergent'], prof_div=prof['divergent'],
                    null_lo=np.percentile(nullprof, 2.5, 0), null_hi=np.percentile(nullprof, 97.5, 0),
                    null_mid=nullprof.mean(0), meta=meta,
                    meta_plus=meta_by['+'][0] / meta_by['+'][1],
                    meta_minus=meta_by['-'][0] / meta_by['-'][1],
                    wmean=wmean, wfrac=wfrac, wcds=wcds, wgenic=wgenic, wgc=wgc,
                    boot=boot, perm_ig_cds=np.array(null['ig_cds']),
                    gstart=g[:, 0].astype(int), gend=g[:, 1].astype(int),
                    gstrand=(g[:, 2] == '+').astype(int),
                    ctx_adj=np.array([[adj[k][c] for c in CATS] for k in ('5mer', 'gc', '5mer_x_gc')]))
json.dump(stats, open(f'{OUT}/dino_stats.json', 'w'), indent=1)
print('wrote stats')
