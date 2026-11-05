"""Paired cluster bootstrap of the per-type mean AUROC difference, RawMod minus UniMeth-Agn (python -m bench.paired_diff, after 2_score.sh).
Same resampling as ci_and_called.py: per row, (contig, pos) clusters drawn with replacement, shared by both tools,
so the interval is on the paired difference. Per-type mean = mean over the rows of a type, then over the 5 types."""
import os, sys, numpy as np
from scipy.stats import rankdata
from . import common as C, score as Sc
from .paper_table import GROUPS, RAWMOD_COLS

def auroc(y, s):
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def cluster_picks(order, starts, lens, nk, reps, rng):
    out = []
    for _ in range(reps):
        pick = rng.integers(0, nk, nk)
        L = lens[pick]; tot = int(L.sum())
        off = np.repeat(np.r_[0, np.cumsum(L)[:-1]], L)
        out.append(order[np.repeat(starts[pick], L) + (np.arange(tot) - off)])
    return out
REPS = int(os.environ.get('CI_REPS', '2000'))
A, B = 'rawmod_' + RAWMOD_COLS[0], os.environ.get('CI_OTHER', 'unimeth_agn_mod_reg')
rng = np.random.default_rng(0)
S = C.samples(); rows = {r['row']: r for r in Sc.rows_cfg()}; cache = {}
types, va, vb, pa, pb = [], [], [], [], []
for g, members in GROUPS:
    for row, _ in members:
        blocks = Sc.row_sites(rows[row], S, cache)
        y = np.concatenate([b[2] for b in blocks])
        key = np.concatenate([(b[1]['contig'].astype(str) + ':' + b[1]['pos'].astype(str)).to_numpy() for b in blocks])
        uk, kidx = np.unique(key, return_inverse=True)
        order = np.argsort(kidx, kind='mergesort')
        starts = np.searchsorted(kidx[order], np.arange(len(uk))); ends = np.r_[starts[1:], len(order)]
        sa = np.concatenate([Sc.site_scores(smp, A, sites)[0] for smp, sites, _ in blocks])
        sb = np.concatenate([Sc.site_scores(smp, B, sites)[0] for smp, sites, _ in blocks])
        picks = cluster_picks(order, starts, ends - starts, len(uk), REPS, rng)
        types.append(g.split(',')[0]); pa.append(auroc(y, sa)); pb.append(auroc(y, sb))
        va.append([auroc(y[i], sa[i]) for i in picks]); vb.append([auroc(y[i], sb[i]) for i in picks])
        print(f'{row:26s} {g.split(",")[0]:5s} rawmod={pa[-1]:.3f} agn={pb[-1]:.3f}', flush=True)
types = np.array(types); va = np.array(va); vb = np.array(vb); pa = np.array(pa); pb = np.array(pb)
T = sorted(set(types), key=list(types).index)
def permod(v): return np.mean([np.nanmean(v[types == t], axis=0) for t in T], axis=0)
def worst(v): return np.min([np.nanmean(v[types == t], axis=0) for t in T], axis=0)
for name, f in (('per-type mean', permod), ('worst type', worst)):
    pt_a, pt_b = f(pa[:, None])[0], f(pb[:, None])[0]
    d = f(va) - f(vb)
    lo, hi = np.nanpercentile(d, [2.5, 97.5]); lo_a, hi_a = np.nanpercentile(f(va), [2.5, 97.5]); lo_b, hi_b = np.nanpercentile(f(vb), [2.5, 97.5])
    print(f'{name}: RawMod {pt_a:.3f} [{lo_a:.3f},{hi_a:.3f}]  Agn {pt_b:.3f} [{lo_b:.3f},{hi_b:.3f}]  '
          f'diff {pt_a-pt_b:+.4f} [{lo:+.4f},{hi:+.4f}]  P(diff<=0)={np.mean(d<=0):.3f}  reps={REPS}')
