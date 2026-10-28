#!/usr/bin/env python3
"""Correlation of the contig map with the 5hmU-DIP and chemical pull-down profiles of Li et al. (2024), lifted onto the contig, in 1-kb and 10-kb windows and per transposable element, with GC, CpG and TE density controlled."""
import os as _os
from pathlib import Path as _Path
REPO = str(_Path(__file__).resolve().parents[3])
RAWMOD_RESULTS = _os.environ.get('RAWMOD_RESULTS') or f'{REPO}/paper/data/runs'  # training runs; shipped metrics by default
RAWMOD_DATA = _os.environ.get('RAWMOD_DATA', 'data')  # datasets (not shipped: set this to score raw signal)
RAWMOD_WORK = _os.environ.get('RAWMOD_WORK') or f'{REPO}/paper/data/work'  # outputs of the paper scripts; shipped by default
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata, mannwhitneyu
from pathlib import Path
W = Path(__file__).resolve().parent; MODEL = os.environ.get('MODEL', 'all'); L = 7733821
rng = np.random.default_rng(2)
lf = np.load(W / 'lift.npz'); names = list(lf['names']); ok = (lf['chrom'] >= 0) & (lf['mapq'] >= 5)
inv = {}
for ci, n in enumerate(names):
    sel = ok & (lf['chrom'] == ci)
    if sel.sum():
        a = np.full(lf['pos'][sel].max() + 2, -1, np.int64); a[lf['pos'][sel]] = np.nonzero(sel)[0]; inv[n] = a
SAMPLES = {'DIP': ['5hmU-DIP-1', '5hmU-DIP-2'], 'DIPin': ['DIP-input-1', 'DIP-input-2'], 'fU': ['5fU-DIP-1', '5fU-DIP-2'],
           'chem': ['5hmU-chem-1', '5hmU-chem-2'], 'chemin': ['input-chem-1', 'input-chem-2']}
reads = {}; tot = {}
for grp, ss in SAMPLES.items():
    for s in ss:
        f = W / 'dip' / f'{s}.pos.gz'
        if not (W / 'dip' / f'{s}.done').exists():
            continue
        st = open(W / 'dip' / f'{s}.pos.stats').read().split()
        tot[s] = int(st[st.index('mapq10_primary') + 1])
        reads[s] = pd.read_csv(f, sep='\t', names=['chr', 'p', 'str'])
print({s: f'{tot[s] / 1e6:.1f} M genome-wide, {len(reads[s]) / 1e3:.0f} k on chr16/38/41' for s in reads})
# per-base our-coordinate read-start counts
cnt = {}
for s, d in reads.items():
    o = np.full(len(d), -1)
    for n, a in inv.items():
        m = (d.chr == n).to_numpy() & (d.p.to_numpy() < len(a))
        o[m] = a[d.p.to_numpy()[m]]
    o = o[o >= 0]; cnt[s] = np.bincount(o, minlength=L)
    print(f'  {s}: {len(o):,} read starts lifted onto contig_3762')


def pooled(grp):
    ss = [s for s in SAMPLES[grp] if s in cnt]
    return (sum(cnt[s] for s in ss), sum(tot[s] for s in ss)) if ss else (None, None)


def enrich(ip, inp, idx_s, idx_e, ipn, inn):
    ci = np.r_[0, np.cumsum(ip)]; cn = np.r_[0, np.cumsum(inp)]
    a = ci[idx_e] - ci[idx_s]; b = cn[idx_e] - cn[idx_s]
    return np.log2(((a + 1) / ipn) / ((b + 1) / inn)), a, b


# ---- TE mask and score table
mask = np.zeros(L, bool); te = []
for g in ('known', 'denovo'):
    for ln in open(W / f'rm_{g}.sub.gff'):
        x = ln.rstrip('\n').split('\t')
        if len(x) < 9 or x[0] not in inv:
            continue
        a = inv[x[0]][int(x[3]) - 1:int(x[4])]; a = a[a >= 0]
        if len(a) < 100:
            continue
        mask[a] = True
        attr = dict(kv.split('=', 1) for kv in x[8].strip(';').split(';') if '=' in kv)
        te.append((a.min(), a.max() + 1, len(a), attr.get('Class', '?'), float(attr.get('PercDiv', 'nan'))))
te = pd.DataFrame(te, columns=['s', 'e', 'nlift', 'cls', 'div'])
te = te[(te.e - te.s) < 2 * te.nlift]  # contiguous after liftover
SC = lambda s: f'{RAWMOD_WORK}/dinoflagellate/contig_3762_strand{s}_scores.tsv.gz'
t = pd.concat([pd.read_csv(SC(s), sep='\t', usecols=['pos', 'score']) for s in '+-'], ignore_index=True)
tsum = np.bincount(t.pos, weights=t.score, minlength=L); tn = np.bincount(t.pos, minlength=L)
import pysam
SEQ = pysam.FastaFile(f'{RAWMOD_DATA}/acarterae_contig_3762/contig_3762.fasta').fetch('contig_3762').upper()
gc = np.array([c in 'GC' for c in SEQ], float); cg = np.array([SEQ[k:k + 2] == 'CG' for k in range(L)], float)
print('model =', MODEL)

ipD, nD = pooled('DIP'); inD, ninD = pooled('DIPin'); ipC, nC = pooled('chem'); inC, ninC = pooled('chemin'); ipF, nF = pooled('fU')
print('\na. agreement')
for w in (1000, 10000):
    s_ = np.arange(0, L - w, w); e_ = s_ + w
    lifted = np.array([ok[a:b].mean() for a, b in zip(s_, e_)])
    keep = lifted > 0.8
    tab = {}
    if ipD is not None and inD is not None:
        for s in SAMPLES['DIP']:
            if s in cnt:
                tab[s] = enrich(cnt[s], inD, s_, e_, tot[s], ninD)[0]
        tab['DIP'] = enrich(ipD, inD, s_, e_, nD, ninD)[0]
        if ipF is not None:
            tab['5fU'] = enrich(ipF, inD, s_, e_, nF, ninD)[0]
    if ipC is not None and inC is not None:
        tab['chem'] = enrich(ipC, inC, s_, e_, nC, ninC)[0]
    tab = pd.DataFrame(tab)[keep]
    print(f'  {w // 1000} kb windows lifted >80%: {len(tab)}')
    print(tab.corr('spearman').round(3).to_string())
    teF = np.array([mask[a:b].mean() for a, b in zip(s_, e_)])[keep]
    for c in tab:
        print(f'    {c} vs TE fraction rho {spearmanr(tab[c], teF).correlation:+.3f}')

print('\nb. windows: T score vs enrichment (partial on GC, CpG and TE density; 500 circular shifts)')
SUM = []
for w in (1000, 10000):
    s_ = np.arange(0, L - w, w); e_ = s_ + w
    df = pd.DataFrame({'T': np.add.reduceat(tsum, s_)[:len(s_)] / np.maximum(np.add.reduceat(tn, s_)[:len(s_)], 1),
                       'nT': np.add.reduceat(tn, s_)[:len(s_)], 'gc': [gc[a:b].mean() for a, b in zip(s_, e_)],
                       'cgdens': [cg[a:b].mean() for a, b in zip(s_, e_)], 'te': [mask[a:b].mean() for a, b in zip(s_, e_)],
                       'lift': [ok[a:b].mean() for a, b in zip(s_, e_)]})
    if ipD is not None and inD is not None:
        df['DIP'], df['ipn'], df['inn'] = enrich(ipD, inD, s_, e_, nD, ninD)
    if ipC is not None and inC is not None:
        df['chem'] = enrich(ipC, inC, s_, e_, nC, ninC)[0]
    if ipF is not None and inD is not None:
        df['5fU'] = enrich(ipF, inD, s_, e_, nF, ninD)[0]
    df = df[(df.lift > 0.8) & (df.nT >= (200 if w == 1000 else 2000))]
    if 'inn' in df:
        df = df[df.inn >= (20 if w == 1000 else 200)]
    df.to_csv(W / f'dip_windows_{w // 1000}kb_{MODEL}.tsv', sep='\t')
    Z = np.c_[np.ones(len(df)), rankdata(df.gc), rankdata(df.cgdens), rankdata(df.te)]

    def part(x, d):
        rx, ry = rankdata(d[x]), rankdata(d['T'])
        return np.corrcoef(rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0], ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0])[0, 1]
    for x in [c for c in ('DIP', 'chem', '5fU') if c in df]:
        r = spearmanr(df['T'], df[x]).correlation; pr = part(x, df); v = df[x].to_numpy()
        nul = [part(x, df.assign(**{x: np.roll(v, rng.integers(50, len(df) - 50))})) for _ in range(500)]
        p = (np.sum(np.abs(nul) >= abs(pr)) + 1) / 501
        q = df[x].quantile([.2, .8]).to_numpy()
        hi, lo = df[df[x] >= q[1]]['T'].mean(), df[df[x] <= q[0]]['T'].mean()
        print(f'  {w // 1000:>2d} kb n={len(df):5d} {x:5s} rho {r:+.3f} partial {pr:+.3f} p_shift {p:.3f}  T top-20% {hi:.3f} vs bottom-20% {lo:.3f}')
        SUM.append(dict(model=MODEL, window=w, assay=x, n=len(df), rho=r, partial=pr, p_shift=p, T_top20=hi, T_bot20=lo))
pd.DataFrame(SUM).to_csv(W / f'dip_summary_{MODEL}.tsv', sep='\t', index=False)

if ipD is not None and inD is not None:
    print('\nc. TEs (lifted contiguously, >=100 lifted bp)')
    te['nT'] = [tn[a:b].sum() for a, b in zip(te.s, te.e)]; te['T'] = [tsum[a:b].sum() for a, b in zip(te.s, te.e)] / te.nT.clip(lower=1)
    te['DIP'], te['ipn'], te['inn'] = enrich(ipD, inD, te.s.to_numpy(), te.e.to_numpy(), nD, ninD)
    tt = te[(te.nT >= 30) & (te.inn >= 10)].copy()
    print(f'  {len(tt):,} TEs; per-TE rho T vs DIP {spearmanr(tt["T"], tt.DIP).correlation:+.3f}; DIP vs div {spearmanr(tt.DIP, tt["div"]).correlation:+.3f}; '
          f'T vs div {spearmanr(tt["T"], tt["div"]).correlation:+.3f}')
    q1, q3 = tt.DIP.quantile([.25, .75])
    H_, Lo = tt[tt.DIP >= q3]['T'], tt[tt.DIP <= q1]['T']
    print(f'  top vs bottom DIP quartile: T {H_.mean():.3f} vs {Lo.mean():.3f}, MWU p {mannwhitneyu(H_, Lo).pvalue:.2g}')
    tt['top'] = tt.cls.str.split('/').str[:2].str.join('/')
    cl = tt.groupby('top').agg(n=('T', 'size'), T=('T', 'mean'), DIP=('DIP', 'median'), div=('div', 'median'))
    cl = cl[cl.n >= 15].sort_values('DIP', ascending=False)
    print(cl.round(3).to_string())
    print(f'  across classes (n={len(cl)}): rho T vs DIP {spearmanr(cl["T"], cl.DIP).correlation:+.3f}')
    tt.to_csv(W / f'dip_te_{MODEL}.tsv.gz', sep='\t', index=False)
