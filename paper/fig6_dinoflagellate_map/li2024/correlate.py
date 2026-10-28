#!/usr/bin/env python3
"""Correlation of the contig map with the bisulfite CpG methylation of Li et al. (2024) lifted onto the contig."""
import os as _os
from pathlib import Path as _Path
REPO = str(_Path(__file__).resolve().parents[3])
RAWMOD_RESULTS = _os.environ.get('RAWMOD_RESULTS') or f'{REPO}/paper/data/runs'  # training runs; shipped metrics by default
RAWMOD_DATA = _os.environ.get('RAWMOD_DATA', 'data')  # datasets (not shipped: set this to score raw signal)
RAWMOD_WORK = _os.environ.get('RAWMOD_WORK') or f'{REPO}/paper/data/work'  # outputs of the paper scripts; shipped by default
import os, gzip, numpy as np, pandas as pd, pysam
from scipy.stats import spearmanr, rankdata, mannwhitneyu
from pathlib import Path
W = Path(__file__).resolve().parent; P = W.parent
MODEL = os.environ.get('MODEL', 'all')
SC = lambda s: f'{RAWMOD_WORK}/dinoflagellate/contig_3762_strand{s}_scores.tsv.gz'
SEQ = pysam.FastaFile(f'{RAWMOD_DATA}/acarterae_contig_3762/contig_3762.fasta').fetch('contig_3762').upper()
L = len(SEQ); rng = np.random.default_rng(0)
print('model =', MODEL)

lf = np.load(W / 'lift.npz'); names = list(lf['names'])
lchr, lpos, lstr, lmq = lf['chrom'], lf['pos'], lf['strand'], lf['mapq']
ok = (lchr >= 0) & (lmq >= 5)
print(f'contig bases lifted (MAPQ>=5): {ok.mean():.1%}')
# inverse map: their (chrom, 0-based pos) -> our pos
inv = {}
for ci, n in enumerate(names):
    sel = ok & (lchr == ci)
    if sel.sum() == 0:
        continue
    a = np.full(lpos[sel].max() + 2, -1, np.int64); a[lpos[sel]] = np.nonzero(sel)[0]; inv[n] = a

# ---- WGBS, both replicates summed
cols = ['chr', 'base', 'pos', 'ctx', 'dinuc', 'lev', 'mc', 'cov']
wg = pd.concat([pd.read_csv(W / f'wgbs{r}.sub.CGmap.gz', sep='\t', names=cols, usecols=['chr', 'base', 'pos', 'ctx', 'mc', 'cov'])
                for r in (1, 2)])
wg = wg.groupby(['chr', 'pos', 'base', 'ctx'], as_index=False)[['mc', 'cov']].sum()
wg['p0'] = wg.pos - 1
wg['our'] = -1
for n, a in inv.items():
    m = (wg.chr == n) & (wg.p0 < len(a))
    wg.loc[m, 'our'] = a[wg.loc[m, 'p0'].to_numpy()]
wg = wg[wg.our >= 0].copy()
ours = np.array([SEQ[i] for i in wg.our])
st = lstr[wg.our.to_numpy()]
expect = np.where((wg.base == 'C') == (st == 1), 'C', 'G')
print(f'WGBS cytosines lifted: {len(wg):,}; base agrees with our contig: {(ours == expect).mean():.1%}')
wg = wg[ours == expect]
for c in ('CG', 'CHG', 'CHH'):
    s = wg[(wg.ctx == c) & (wg['cov'] >= 3)]
    print(f'  {c:3s}: {len(s):,} sites cov>=3, mean level {(s.mc / s["cov"]).mean():.3f} (pooled {s.mc.sum() / s["cov"].sum():.3f})')

# ---- 0. liftover sanity: WGBS mCG vs Dorado 5mC per CpG (Dorado both strands summed at the C of CpG)
import re
rows = []
with gzip.open(P / 'cpg_pileup.bed.gz', 'rt') as f:
    for ln in f:
        x = re.split(r'\s+', ln.strip())
        rows.append((int(x[1]) - (1 if x[5] == '-' else 0), x[3], int(x[9]), int(x[11])))
m = pd.DataFrame(rows, columns=['cpg', 'code', 'nvalid', 'nmod'])
g = m.groupby(['cpg', 'code'])[['nvalid', 'nmod']].sum().unstack('code')
dor = pd.DataFrame({'nvalid': g['nvalid']['m'], 'f5mC': g['nmod']['m'] / g['nvalid']['m'], 'f5hmC': g['nmod']['h'] / g['nvalid']['h']}).dropna()
dor = dor[dor.nvalid >= 10]
cg = wg[wg.ctx == 'CG'].copy()
cg['cpg'] = np.where(SEQ_C := np.array([SEQ[i] for i in cg.our]) == 'C', cg.our, cg.our - 1)
cgs = cg.groupby('cpg')[['mc', 'cov']].sum(); cgs = cgs[cgs['cov'] >= 5]
j = cgs.join(dor, how='inner')
j['wgbs'] = j.mc / j['cov']
print(f'\n0. CpGs with WGBS cov>=5 and Dorado >=10 calls: {len(j):,}; WGBS mean {j.wgbs.mean():.3f}, Dorado 5mC {j.f5mC.mean():.3f}, '
      f'5mC+5hmC {(j.f5mC + j.f5hmC).mean():.3f}')
print(f'   per-CpG rho WGBS vs Dorado 5mC {spearmanr(j.wgbs, j.f5mC).correlation:+.3f}, vs 5mC+5hmC '
      f'{spearmanr(j.wgbs, j.f5mC + j.f5hmC).correlation:+.3f}, vs Dorado 5hmC {spearmanr(j.wgbs, j.f5hmC).correlation:+.3f}')
jw = j.groupby(j.index // 10000).agg(w=('wgbs', 'mean'), d=('f5mC', 'mean'), n=('wgbs', 'size'))
jw = jw[jw.n >= 20]
print(f'   10-kb window rho WGBS vs Dorado 5mC {spearmanr(jw.w, jw.d).correlation:+.3f} (n={len(jw)})')

# ---- T scores
t = pd.concat([pd.read_csv(SC(s), sep='\t', usecols=['pos', 'score']).assign(strand=s) for s in '+-'], ignore_index=True)
allcpg = np.array([i for i in range(L - 1) if SEQ[i] == 'C' and SEQ[i + 1] == 'G'])
i = np.clip(np.searchsorted(allcpg, t.pos.to_numpy()), 1, len(allcpg) - 1)
t['d_any'] = np.minimum(np.abs(t.pos - allcpg[i - 1]), np.abs(t.pos - allcpg[i]))

# ---- 1. windows
gc_cum = np.r_[0, np.cumsum([c in 'GC' for c in SEQ])]
cg_cum = np.r_[0, np.cumsum([SEQ[k:k + 2] == 'CG' for k in range(L)])]


def window_table(w, tsub):
    T = tsub.groupby(tsub.pos // w).score.mean().rename('T')
    cols = [T]
    for c in ('CG', 'CHG', 'CHH'):
        s = wg[wg.ctx == c]; b = s.our // w
        cols.append((s.mc.groupby(b).sum() / s['cov'].groupby(b).sum()).rename('m' + c))
        cols.append(s['cov'].groupby(b).sum().rename('cov' + c))
    df = pd.concat(cols, axis=1).dropna()
    df = df[df.covCG >= (50 if w == 1000 else 500)]
    df['mCH'] = (df.mCHG * df.covCHG + df.mCHH * df.covCHH) / (df.covCHG + df.covCHH)
    s_ = df.index.to_numpy() * w; e = np.minimum(s_ + w, L)
    df['gc'] = (gc_cum[e] - gc_cum[s_]) / (e - s_); df['cgdens'] = (cg_cum[e] - cg_cum[s_]) / (e - s_)
    return df


def partial(df, x, y='T'):
    Z = np.c_[np.ones(len(df)), rankdata(df.gc), rankdata(df.cgdens)]
    rx = rankdata(df[x]); ry = rankdata(df[y])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]; ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return np.corrcoef(ex, ey)[0, 1]


SUM = []
print('\n1. windows (rho; partial = after GC and CpG density; p_shift from 500 circular shifts of the WGBS track)')
for w in (1000, 10000):
    for lab, tsub in [('all T', t), ('T >10 bp from CpG', t[t.d_any > 10])]:
        df = window_table(w, tsub)
        for x in ('mCG', 'mCHG', 'mCHH', 'mCH'):
            r = spearmanr(df['T'], df[x]).correlation; pr = partial(df, x)
            v = df[x].to_numpy(); nul = [partial(df.assign(**{x: np.roll(v, rng.integers(50, len(df) - 50))}), x) for _ in range(500)]
            p = (np.sum(np.abs(nul) >= abs(pr)) + 1) / 501
            print(f'  {w // 1000:>2d} kb  {lab:18s} {x:5s} n={len(df):5d}  rho {r:+.3f}  partial {pr:+.3f}  p_shift {p:.3f}')
            SUM.append(dict(model=MODEL, window=w, tsites=lab, mark=x, n=len(df), rho=r, partial=pr, p_shift=p))
        if lab == 'all T':
            print(f'        (mCG vs mCH across windows rho {spearmanr(df.mCG, df.mCH).correlation:+.3f})')
            df.to_csv(W / f'windows_{w // 1000}kb_{MODEL}.tsv', sep='\t')

# ---- 2. TEs
te = []
for g_ in ('known', 'denovo'):
    for ln in open(W / f'rm_{g_}.sub.gff'):
        x = ln.rstrip('\n').split('\t')
        if len(x) < 9:
            continue
        a = dict(kv.split('=', 1) for kv in x[8].strip(';').split(';') if '=' in kv)
        te.append((g_, x[0], int(x[3]) - 1, int(x[4]), a.get('Class', '?'), float(a.get('PercDiv', 'nan'))))
te = pd.DataFrame(te, columns=['lib', 'chr', 's', 'e', 'cls', 'div'])
te['len'] = te.e - te.s
# T scores in their coordinates
ok_t = ok[t.pos.to_numpy()]
tt = t[ok_t].copy(); tt['chr'] = np.array(names)[lchr[tt.pos.to_numpy()]]; tt['tp'] = lpos[tt.pos.to_numpy()]
out = []
for n in te.chr.unique():
    a = tt[tt.chr == n].sort_values('tp'); tp = a.tp.to_numpy(); cs = np.r_[0, np.cumsum(a.score.to_numpy())]
    b = wg[wg.chr == n]
    sub = te[te.chr == n].copy()
    lo = np.searchsorted(tp, sub.s.to_numpy()); hi = np.searchsorted(tp, sub.e.to_numpy())
    sub['nT'] = hi - lo; sub['T'] = (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1)
    for c in ('CG', 'CHG', 'CHH'):
        bc = b[b.ctx == c].sort_values('p0'); bp = bc.p0.to_numpy()
        cm = np.r_[0, np.cumsum(bc.mc.to_numpy())]; cv = np.r_[0, np.cumsum(bc['cov'].to_numpy())]
        l2 = np.searchsorted(bp, sub.s.to_numpy()); h2 = np.searchsorted(bp, sub.e.to_numpy())
        sub['cov' + c] = cv[h2] - cv[l2]; sub['m' + c] = (cm[h2] - cm[l2]) / np.maximum(cv[h2] - cv[l2], 1)
    out.append(sub)
te = pd.concat(out)
te['covCH'] = te.covCHG + te.covCHH; te['mCH'] = (te.mCHG * te.covCHG + te.mCHH * te.covCHH) / te.covCH.clip(lower=1)
te = te[(te.nT >= 30) & (te.covCG >= 20) & (te.covCH >= 100)]
# T-score background: all lifted T outside every TE
inte = np.zeros(L, bool)
print(f'\n2. TEs with >=30 scored T, >=20 CG and >=100 CH calls: {len(te):,} ({te.lib.value_counts().to_dict()})')
for x in ('mCG', 'mCH', 'mCHG', 'mCHH', 'div', 'len'):
    print(f'   per-TE rho T vs {x:4s} {spearmanr(te["T"], te[x]).correlation:+.3f}')
for x in ('mCG', 'mCH'):
    q1, q3 = te[x].quantile([.25, .75])
    H_, Lo = te[te[x] >= q3]['T'], te[te[x] <= q1]['T']
    print(f'   top vs bottom quartile {x}: T {H_.mean():.3f} vs {Lo.mean():.3f} (diff {H_.mean() - Lo.mean():+.3f}, MWU p {mannwhitneyu(H_, Lo).pvalue:.2g})')
# their 5hmU-rich classes (Fig. 4 of Li et al.)
RICH = ('Maverick', 'Novosib', 'Sola', 'CR1', 'CRE', 'R2', 'Ngaro', 'ERVK')
te['rich'] = te.cls.apply(lambda c: any(r in c for r in RICH))
print(f'   their 5hmU-rich classes: T {te[te.rich]["T"].mean():.3f} (n={te.rich.sum()}) vs other TEs {te[~te.rich]["T"].mean():.3f} '
      f'(n={(~te.rich).sum()}), MWU p {mannwhitneyu(te[te.rich]["T"], te[~te.rich]["T"]).pvalue:.2g}')
te['top'] = te.cls.str.split('/').str[:2].str.join('/')
cl = te.groupby('top').agg(n=('T', 'size'), T=('T', 'mean'), mCG=('mCG', 'mean'), mCH=('mCH', 'mean'), div=('div', 'median'))
print(cl[cl.n >= 15].sort_values('T', ascending=False).round(3).to_string())
cl = cl[cl.n >= 15]
print(f'   across classes (n={len(cl)}): rho T vs mCG {spearmanr(cl["T"], cl.mCG).correlation:+.3f}, vs mCH {spearmanr(cl["T"], cl.mCH).correlation:+.3f}')
te.to_csv(W / f'te_{MODEL}.tsv.gz', sep='\t', index=False)
pd.DataFrame(SUM).to_csv(W / f'windows_summary_{MODEL}.tsv', sep='\t', index=False)
