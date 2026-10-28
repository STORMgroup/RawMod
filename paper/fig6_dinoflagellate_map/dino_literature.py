#!/usr/bin/env python3
"""Comparison with the published findings: exon versus flank, gene-start profile and TE classes, using the repeat annotation of Li et al. (2024) lifted onto the contig."""
import os as _os
from pathlib import Path as _Path
REPO = str(_Path(__file__).resolve().parents[2])
RAWMOD_RESULTS = _os.environ.get('RAWMOD_RESULTS') or f'{REPO}/paper/data/runs'  # training runs; shipped metrics by default
RAWMOD_DATA = _os.environ.get('RAWMOD_DATA', 'data')  # datasets (not shipped: set this to score raw signal)
RAWMOD_WORK = _os.environ.get('RAWMOD_WORK') or f'{REPO}/paper/data/work'  # outputs of the paper scripts; shipped by default

import sys, json, re
import numpy as np, pandas as pd

BASE = f'{RAWMOD_DATA}/acarterae_contig_3762'
SCORES, OUTP = sys.argv[1], sys.argv[2]
N = 7733821
NPERM = 1000
rng = np.random.default_rng(7)
res = {'scores': SCORES}

seq = ''.join(l.strip() for l in open(f'{BASE}/contig_3762.fasta') if not l.startswith('>')).upper()
assert len(seq) == N
df = pd.read_csv(SCORES, sep='\t')
pos = df.pos.values.astype(np.int64); sc = df.score.values.astype(np.float64)
track = np.full(N, np.nan); track[pos] = sc
res['n_sites'] = int(len(pos)); res['mean_score'] = float(sc.mean())

rows = [l.rstrip('\n').split('\t') for l in open(f'{BASE}/Annotations_contig_3762.gff3')
        if not l.startswith('#') and l.strip()]
genes = sorted((int(p[3]) - 1, int(p[4]), p[6]) for p in rows if p[2] == 'gene')
exon = np.zeros(N, bool); gene = np.zeros(N, bool)
for p in rows:
    s, e = int(p[3]) - 1, int(p[4])
    if p[2] == 'exon':
        exon[s:e] = True
    elif p[2] == 'gene':
        gene[s:e] = True

# ---------------------------------------------------------------- 1. exons vs +-1 kb flanks
near = np.zeros(N, bool)
for p in rows:
    if p[2] == 'exon':
        s, e = int(p[3]) - 1, int(p[4])
        near[max(0, s - 1000):s] = True; near[e:min(N, e + 1000)] = True
flank = near & ~exon


def ratio_exon(shift=0):
    t = np.roll(track, shift)
    return np.nanmean(t[exon]) / np.nanmean(t[flank])


# ---------------------------------------------------------------- 2. gene-start metaprofile
BIN, W = 250, 5000
edges = np.arange(-W, W + BIN, BIN)
starts = [(s if st == '+' else e - 1, st) for s, e, st in genes]
# signed distance (upstream negative, strand-aware) to the nearest gene start, per base
dist = np.full(N, np.inf)
for x, st in starts:
    lo, hi = max(0, x - W), min(N, x + W + 1)
    d = np.arange(lo, hi) - x
    if st == '-':
        d = -d
    better = np.abs(d) < np.abs(dist[lo:hi])
    dist[lo:hi][better] = d[better]
core = np.abs(dist) <= 1000
ring = (np.abs(dist) > 1000) & (np.abs(dist) <= W)


def ratio_start(shift=0):
    t = np.roll(track, shift)
    return np.nanmean(t[core]) / np.nanmean(t[ring])


prof = []
for a, b in zip(edges[:-1], edges[1:]):
    m = (dist >= a) & (dist < b)
    prof.append(dict(bin_start=int(a), mean=float(np.nanmean(track[m])), n=int(np.isfinite(track[m]).sum())))
res['gene_start_profile'] = prof

shifts = rng.integers(200_000, N - 200_000, NPERM)
for name, fn in (('exon_vs_flank', ratio_exon), ('gene_start_core_vs_ring', ratio_start)):
    obs = fn(0)
    null = np.array([fn(int(k)) for k in shifts])
    res[name] = dict(ratio=float(obs), null_mean=float(null.mean()), null_sd=float(null.std()),
                     z=float((obs - null.mean()) / null.std()),
                     p_two_sided=float(((np.abs(null - null.mean()) >= abs(obs - null.mean())).sum() + 1) / (NPERM + 1)))
res['exon_vs_flank']['exon_mean'] = float(np.nanmean(track[exon]))
res['exon_vs_flank']['flank_mean'] = float(np.nanmean(track[flank]))
res['gene_start_core_vs_ring']['core_mean'] = float(np.nanmean(track[core]))
res['gene_start_core_vs_ring']['ring_mean'] = float(np.nanmean(track[ring]))
for st in ('+', '-'):
    sub = [x for x in starts if x[1] == st]
    m_core = np.zeros(N, bool); m_ring = np.zeros(N, bool)
    for x, _ in sub:
        m_core[max(0, x - 1000):x + 1001] = True
        m_ring[max(0, x - W):x + W + 1] = True
    m_ring &= ~m_core
    res['gene_start_core_vs_ring'][f'genes{st}'] = dict(n_genes=len(sub),
        ratio=float(np.nanmean(track[m_core]) / np.nanmean(track[m_ring])))

# ---------------------------------------------------------------- 3. strand-switch regions
out = []
run_end = genes[0][1]
for i in range(1, len(genes)):
    s, e = run_end, genes[i][0]
    run_end = max(run_end, genes[i][1]) if genes[i][2] == genes[i - 1][2] else genes[i][1]
    if e - s < 1000:
        continue
    vals = track[s:e]; vals = vals[np.isfinite(vals)]
    if len(vals) < 100:
        continue
    kind = 'within' if genes[i][2] == genes[i - 1][2] else ('convergent' if genes[i - 1][2] == '+' else 'divergent')
    out.append(dict(kind=kind, start=s, end=e, length=e - s, n=len(vals), mean=float(vals.mean())))
gaps = pd.DataFrame(out)
w = gaps[gaps.kind == 'within']['mean'].values
res['within_gaps'] = dict(n=int(len(w)), mean=float(w.mean()), p95=float(np.percentile(w, 95)))
res['junctions'] = [dict(r._asdict() if hasattr(r, '_asdict') else r, pct=float((w < r['mean']).mean() * 100))
                    for r in gaps[gaps.kind != 'within'].to_dict('records')]

# ---------------------------------------------------------------- 4. short-read uniqueness
code = np.frombuffer(seq.encode(), dtype=np.uint8)
lut = np.full(256, 4, np.uint8)
for i, c in enumerate(b'ACGT'):
    lut[c] = i
c = lut[code].astype(np.uint64)
isn = c == 4
c[isn] = 0
rc = (3 - c)[::-1]


def kmer_hash(arr, k):
    """Polynomial rolling hash mod 2^64 of every k-mer start."""
    n = len(arr)
  # prefix H[i] = sum_{j<i} (arr[j]+1)*B^(i-1-j) mod 2^64 (one sequential pass)
    with np.errstate(over='ignore'):
        Hl = [0] * (n + 1)
        a = 0
        M = (1 << 64) - 1
        b = 1000003
        al = arr.tolist()
        for i in range(n):
            a = (a * b + al[i] + 1) & M
            Hl[i + 1] = a
        H = np.array(Hl, dtype=np.uint64)
        bk = np.uint64(pow(b, k, 1 << 64))
        return H[k:] - H[:-k] * bk


res['uniqueness'] = {}
for k in (100, 150):
    hf = kmer_hash(c, k)  # k-mer starting at i on +
    hr = kmer_hash(rc, k)[::-1]  # reverse-complement k-mer covering the same span
    canon = np.minimum(hf, hr)
    _, inv, cnt = np.unique(canon, return_inverse=True, return_counts=True)
    uniq_start = cnt[inv] == 1
    hasn = np.convolve(isn.astype(np.int64), np.ones(k, np.int64), 'valid') > 0
    uniq_start &= ~hasn
  # a base is "mappable" if the k-mer centred on it is unique
    mapp = np.zeros(N, bool)
    off = k // 2
    mapp[off:off + len(uniq_start)] = uniq_start
    cls = {}
    for r in gaps.to_dict('records'):
        cls.setdefault(r['kind'], []).append(float(mapp[r['start']:r['end']].mean()))
    cls_out = {kk: dict(n=len(v), mean=float(np.mean(v)), values=[round(x, 4) for x in v] if kk != 'within' else None)
               for kk, v in cls.items()}
    cls_out['genes'] = float(mapp[gene].mean()); cls_out['exons'] = float(mapp[exon].mean())
    cls_out['contig'] = float(mapp.mean())
    res['uniqueness'][f'k{k}'] = cls_out
    print(k, json.dumps(cls_out)[:400], flush=True)

# young interspersed repeats per region (RepeatMasker, Judd et al.)
rx = re.compile(r'Motif:([^"]+)"')
young = np.zeros(N, bool); inter = np.zeros(N, bool)
for line in open(f'{BASE}/external/Repeats_Filtered_Final.gff'):
    if line.startswith('#'):
        continue
    p = line.split('\t')
    if p[0] != 'contig_3762':
        continue
    mm = rx.search(p[8]); fam = mm.group(1) if mm else ''
    if fam.endswith(')n') or '-rich' in fam:
        continue
    s, e = int(p[3]) - 1, int(p[4])
    inter[s:e] = True
    if float(p[5]) <= 5:
        young[s:e] = True
res['young_repeats'] = {kk: float(np.mean([young[r['start']:r['end']].mean() for r in gaps.to_dict('records') if r['kind'] == kk]))
                        for kk in ('within', 'divergent', 'convergent')}
res['young_repeats']['contig'] = float(young.mean())
res['interspersed'] = {kk: float(np.mean([inter[r['start']:r['end']].mean() for r in gaps.to_dict('records') if r['kind'] == kk]))
                       for kk in ('within', 'divergent', 'convergent')}
res['interspersed']['contig'] = float(inter.mean())

# repeat-content control: does repeat (and young-repeat) content of a gap predict its score?
# Fit mean ~ interspersed frac + young(<10%) frac on the within-array gaps, predict the junctions.
div10 = np.zeros(N, bool)
for line in open(f'{BASE}/external/Repeats_Filtered_Final.gff'):
    if line.startswith('#'):
        continue
    p = line.split('\t')
    if p[0] != 'contig_3762':
        continue
    mm = rx.search(p[8]); fam = mm.group(1) if mm else ''
    if fam.endswith(')n') or '-rich' in fam:
        continue
    if float(p[5]) < 10:
        div10[int(p[3]) - 1:int(p[4])] = True
gaps['inter'] = [inter[s:e].mean() for s, e in zip(gaps.start, gaps.end)]
gaps['young10'] = [div10[s:e].mean() for s, e in zip(gaps.start, gaps.end)]
wg = gaps[gaps.kind == 'within']
X = np.c_[np.ones(len(wg)), wg.inter, wg.young10]
beta, *_ = np.linalg.lstsq(X, wg['mean'].values, rcond=None)
resid_sd = float(np.std(wg['mean'].values - X @ beta, ddof=3))
res['repeat_control'] = dict(beta=[float(b) for b in beta], resid_sd=resid_sd, junctions=[])
for r in gaps[gaps.kind != 'within'].to_dict('records'):
    pred = float(beta[0] + beta[1] * r['inter'] + beta[2] * r['young10'])
    res['repeat_control']['junctions'].append(dict(kind=r['kind'], start=int(r['start']), inter=float(r['inter']),
        young10=float(r['young10']), observed=float(r['mean']), predicted=pred,
        z_resid=float((r['mean'] - pred) / resid_sd)))
print(json.dumps(res['repeat_control'], indent=1), flush=True)

json.dump(res, open(f'{OUTP}.json', 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k not in ('gene_start_profile', 'uniqueness')}, indent=1, default=float))
