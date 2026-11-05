"""Per-row 95% CIs and called-only AUROC for the main benchmark table.

  python -m bench.ci            (after 2_score.sh; BENCH_HOLDOUT=region for the paper)   -> <paper dir>/ci_rows.tsv
  python -m bench.ci_table      renders the supplementary tables from that file

CI: cluster bootstrap over reference positions. A two-library row scores the SAME reference position in both
libraries, so resampling sites independently would understate the interval; we resample (contig, pos) keys with
replacement and take every block of the row that key appears in.
Called-only AUROC: the benchmark scores an uncalled read as probability 0 out of K=10, so a tool that calls few
reads is pushed towards 0. The called-only column recomputes the AUROC over the sites where the tool emitted at
least one call, which is the comparison that does not mix applicability with accuracy.
"""
import os, sys, numpy as np, pandas as pd
from scipy.stats import rankdata
from . import common as C, score as Sc
from .paper_table import MODELS, MAIN_DROP, GROUPS

REPS = int(os.environ.get('CI_REPS', '500'))
rng = np.random.default_rng(0)

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

def main():
    S = C.samples(); rows = {r['row']: r for r in Sc.rows_cfg()}; cache = {}
    cols = [m for m, _ in MODELS if m not in MAIN_DROP]   # main-table columns
    out = []
    for g, members in GROUPS:
        for row, _ in members:
            blocks = Sc.row_sites(rows[row], S, cache)
            y = np.concatenate([b[2] for b in blocks])
            key = np.concatenate([(b[1]['contig'].astype(str) + ':' + b[1]['pos'].astype(str)).to_numpy() for b in blocks])
            uk, kidx = np.unique(key, return_inverse=True)
            # index lists per unique key, for the cluster bootstrap
            order = np.argsort(kidx, kind='mergesort')
            starts = np.searchsorted(kidx[order], np.arange(len(uk)))
            ends = np.r_[starts[1:], len(order)]
            rec = {'row': row, 'mod': g.split(',')[0], 'n_sites': len(y), 'n_pos': int(y.sum()), 'n_keys': len(uk)}
            boot_idx = None
            for m in cols:
                sc, cl, missing = [], [], False
                for smp, sites, _ in blocks:
                    try:
                        mp, _f, called = Sc.site_scores(smp, m, sites)
                    except FileNotFoundError:          # tool never run on this sample -> '--' in the table
                        missing = True; break
                    sc.append(mp); cl.append(called)
                if missing:
                    for k in ('', '::frac_called', '::called', '::lo', '::hi'):
                        rec[m + k] = np.nan
                    continue
                sc = np.concatenate(sc); cl = np.concatenate(cl)
                a = auroc(y, sc)
                rec[m] = a
                rec[m + '::frac_called'] = float(cl.mean())
                rec[m + '::called'] = auroc(y[cl], sc[cl]) if cl.sum() > 20 else np.nan
                if np.isnan(a):
                    rec[m + '::lo'] = rec[m + '::hi'] = np.nan; continue
                if boot_idx is None:
                    boot_idx = cluster_picks(order, starts, ends - starts, len(uk), REPS, rng)
                vals = np.array([auroc(y[b], sc[b]) for b in boot_idx])
                vals = vals[~np.isnan(vals)]
                rec[m + '::lo'], rec[m + '::hi'] = np.percentile(vals, [2.5, 97.5])
            out.append(rec)
            print(f"{row:26s} n={len(y):7d} keys={len(uk):7d} rawmod={rec[cols[0]]:.3f} "
                  f"[{rec[cols[0]+'::lo']:.3f},{rec[cols[0]+'::hi']:.3f}]", flush=True)
    D = pd.DataFrame(out)
    out = os.environ.get('CI_OUT') or str(C.WORK / f'paper_{C.PAPER_TAG}' / 'ci_rows.tsv')
    D.to_csv(out, sep='\t', index=False, float_format='%.4f')
    print('wrote', out)


if __name__ == '__main__':
    main()
