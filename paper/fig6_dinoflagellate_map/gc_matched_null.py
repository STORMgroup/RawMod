#!/usr/bin/env python3
"""GC-matched null for the strand-switch junctions: scores of random intergenic windows with the same GC content and length. Usage: gc_matched_null.py [PLUS_SCORES MINUS_SCORES]."""
import os as _os
from pathlib import Path as _Path
REPO = str(_Path(__file__).resolve().parents[2])
RAWMOD_RESULTS = _os.environ.get('RAWMOD_RESULTS') or f'{REPO}/paper/data/runs'  # training runs; shipped metrics by default
RAWMOD_DATA = _os.environ.get('RAWMOD_DATA', 'data')  # datasets (not shipped: set this to score raw signal)
RAWMOD_WORK = _os.environ.get('RAWMOD_WORK') or f'{REPO}/paper/data/work'  # outputs of the paper scripts; shipped by default

import json, os, sys
import numpy as np, pandas as pd

BASE = f'{RAWMOD_DATA}/acarterae_contig_3762'
W = f'{RAWMOD_WORK}/dinoflagellate/strand'
PLUS = sys.argv[1] if len(sys.argv) > 1 else f'{RAWMOD_WORK}/dinoflagellate/contig_3762_strand+_scores.tsv.gz'
MINUS = sys.argv[2] if len(sys.argv) > 2 else f'{RAWMOD_WORK}/dinoflagellate/contig_3762_strand-_scores.tsv.gz'
NDRAW, TOL, STEP = 1000, 0.02, 100
rng = np.random.default_rng(2026)

seq = ''.join(l.strip() for l in open(f'{BASE}/contig_3762.fasta') if not l.startswith('>')).upper()
N = len(seq)
sb = np.frombuffer(seq.encode(), dtype=np.uint8)
cgc = np.concatenate([[0], np.cumsum((sb == ord('G')) | (sb == ord('C')))])
rows = [l.rstrip('\n').split('\t') for l in open(f'{BASE}/Annotations_contig_3762.gff3') if not l.startswith('#') and l.strip()]
g = sorted([(int(p[3]) - 1, int(p[4]), p[6]) for p in rows if p[2] == 'gene'])
genic = np.zeros(N, bool)
for s, e, _ in g:
    genic[s:e] = True
cgen = np.concatenate([[0], np.cumsum(genic)])

def gaps(pos, sc):
    """dino_gaps.py verbatim (gap list, kind, mean, gc)."""
    out = []; run_end = g[0][1]
    for i in range(1, len(g)):
        s, e = run_end, g[i][0]
        run_end = max(run_end, g[i][1]) if g[i][2] == g[i - 1][2] else g[i][1]
        if e - s < 1000:
            continue
        sel = (pos >= s) & (pos < e)
        if sel.sum() < 100:
            continue
        kind = 'within' if g[i][2] == g[i - 1][2] else ('convergent' if g[i - 1][2] == '+' else 'divergent')
        gc = (cgc[e] - cgc[s]) / (e - s)
        out.append(dict(kind=kind, start=s, end=e, length=e - s, n=int(sel.sum()), mean=float(sc[sel].mean()),
                        frac05=float((sc[sel] > .5).mean()), gc=gc))
    return pd.DataFrame(out)

def analyse(name, f):
    d = pd.read_csv(f, sep='\t'); pos = d.pos.values.astype(np.int64); sc = d.score.values.astype(np.float64)
    G = gaps(pos, sc)
    J = G[G.kind != 'within'].reset_index(drop=True); Wg = G[G.kind == 'within'].reset_index(drop=True)
  # cumulative site score / count tracks (all sites, intergenic sites)
    cs_all = np.zeros(N + 1); cn_all = np.zeros(N + 1); cs_ig = np.zeros(N + 1); cn_ig = np.zeros(N + 1)
    np.add.at(cs_all, pos + 1, sc); np.add.at(cn_all, pos + 1, 1)
    ig = ~genic[pos]
    np.add.at(cs_ig, pos[ig] + 1, sc[ig]); np.add.at(cn_ig, pos[ig] + 1, 1)
    for a in (cs_all, cn_all, cs_ig, cn_ig):
        np.cumsum(a, out=a)
  # within-gap regression mean ~ gc + log10(length)
    X = np.column_stack([np.ones(len(Wg)), Wg.gc, np.log10(Wg.length)])
    beta, *_ = np.linalg.lstsq(X, Wg['mean'].values, rcond=None)
    rsd = float(np.std(Wg['mean'].values - X @ beta, ddof=3))
    res = dict(strand=name, file=f, n_within=int(len(Wg)), within_mean=float(Wg['mean'].mean()),
               within_gc_mean=float(Wg.gc.mean()), within_gc_sd=float(Wg.gc.std()),
               contig_gc=float(cgc[N] / N), regression=dict(beta_const=float(beta[0]), beta_gc=float(beta[1]),
                                                              beta_log10len=float(beta[2]), resid_sd=rsd),
               junctions=[])
    jint = [(int(r.start), int(r.end)) for r in J.itertuples()]
    for r in J.itertuples():
        L_ = int(r.length)
        starts = np.arange(0, N - L_, STEP)
        gcs = (cgc[starts + L_] - cgc[starts]) / L_
        ok = np.abs(gcs - r.gc) <= TOL
        for s, e in jint:  # no overlap with any junction region
            ok &= (starts + L_ <= s) | (starts >= e)
        rec = dict(kind=r.kind, start=int(r.start), end=int(r.end), length=L_, gc=float(r.gc), n_sites=int(r.n),
                   mean=float(r.mean), frac05=float(r.frac05),
                   pct_among_within_gaps=float((Wg['mean'] < r.mean).mean() * 100))
        m = np.abs(Wg.gc - r.gc) <= TOL
        rec['gc_matched_within_gaps'] = dict(n=int(m.sum()), pct=float((Wg['mean'][m] < r.mean).mean() * 100) if m.sum() else None,
                                             mean=float(Wg['mean'][m].mean()) if m.sum() else None)
        pred = beta[0] + beta[1] * r.gc + beta[2] * np.log10(L_)
        rec['regression_resid_z'] = float((r.mean - pred) / rsd); rec['regression_pred'] = float(pred)
        for tol in (TOL, 0.05):
            okt = ok if tol == TOL else (np.abs(gcs - r.gc) <= tol) & np.array([
                all((st + L_ <= s) | (st >= e) for s, e in jint) for st in starts])
            el = starts[okt]
            if len(el) == 0:
                rec[f'null_tol{tol}'] = dict(n_eligible=0); continue
            draw = rng.choice(el, NDRAW, replace=True)
  # distinct non-overlapping eligible windows (coverage of the eligible set)
            n_nonover = 0; last = -10**9
            for st in el:
                if st >= last + L_:
                    n_nonover += 1; last = st
            out_ = dict(n_eligible_starts=int(len(el)), n_nonoverlapping=int(n_nonover),
                        eligible_span_kb=[int(el.min() // 1000), int(el.max() // 1000)])
            for lbl, cs_, cn_ in (('intergenic_sites', cs_ig, cn_ig), ('all_sites', cs_all, cn_all)):
                nn = cn_[draw + L_] - cn_[draw]; keep = nn >= 100
                mu = (cs_[draw + L_] - cs_[draw])[keep] / nn[keep]
                out_[lbl] = dict(n_draws=int(keep.sum()), null_mean=float(mu.mean()), null_sd=float(mu.std()),
                                 null_p95=float(np.percentile(mu, 95)), null_max=float(mu.max()),
                                 pct=float((mu < r.mean).mean() * 100), z=float((r.mean - mu.mean()) / mu.std()),
                                 median_sites_per_draw=float(np.median(nn[keep])))
            rec[f'null_tol{tol}'] = out_
  # block-mosaic null: junction cut into 5-kb blocks; each block replaced by a random intergenic
  # (>=90% non-genic bp) segment of the same length with GC within +-2 pct points of THAT block, drawn
  # anywhere outside the junction regions -> 1,000 mosaics with the junction's length and GC profile.
        B = 5000; tot_s = np.zeros(NDRAW); tot_n = np.zeros(NDRAW); n_el = []
        for b0 in range(int(r.start), int(r.end), B):
            b1 = min(int(r.end), b0 + B); lb = b1 - b0
            if lb < 1000:
                continue
            gb = (cgc[b1] - cgc[b0]) / lb
            st = np.arange(0, N - lb, STEP)
            okb = np.abs((cgc[st + lb] - cgc[st]) / lb - gb) <= TOL
            okb &= (cgen[st + lb] - cgen[st]) / lb <= 0.10
            for s_, e_ in jint:
                okb &= (st + lb <= s_) | (st >= e_)
            el = st[okb]; n_el.append(len(el))
            if len(el) == 0:
                continue
            dr = rng.choice(el, NDRAW, replace=True)
            tot_s += cs_ig[dr + lb] - cs_ig[dr]; tot_n += cn_ig[dr + lb] - cn_ig[dr]
        mu = tot_s[tot_n > 0] / tot_n[tot_n > 0]
        rec['null_mosaic5kb'] = dict(n_blocks=len(n_el), min_eligible_per_block=int(min(n_el)),
                                     median_eligible_per_block=float(np.median(n_el)), null_mean=float(mu.mean()),
                                     null_sd=float(mu.std()), null_p95=float(np.percentile(mu, 95)),
                                     null_max=float(mu.max()), pct=float((mu < r.mean).mean() * 100),
                                     z=float((r.mean - mu.mean()) / mu.std()))
        res['junctions'].append(rec)
    return res

out = {}
for nm, f in (('plus', PLUS), ('minus', MINUS)):
    if not os.path.exists(f):
        print('missing', f); continue
    out[nm] = analyse(nm, f)
    r = out[nm]
    print(f"\n=== {nm} strand: {r['n_within']} within gaps, contig GC {r['contig_gc']:.3f}, within-gap GC "
          f"{r['within_gc_mean']:.3f}+-{r['within_gc_sd']:.3f}; regression beta gc={r['regression']['beta_gc']:.3f} "
          f"log10len={r['regression']['beta_log10len']:.3f} resid sd={r['regression']['resid_sd']:.3f}")
    for j in r['junctions']:
        t = j['null_tol0.02']
        print(f"{j['kind']:10s} {j['start']:>8d} len {j['length']:>6d} GC {j['gc']:.3f} mean {j['mean']:.3f} | "
              f"pct among within {j['pct_among_within_gaps']:5.1f} | GC-matched within n={j['gc_matched_within_gaps']['n']} "
              f"pct={j['gc_matched_within_gaps']['pct']} | regr z={j['regression_resid_z']:+.2f} | "
              f"null(+-2% GC) elig={t.get('n_eligible_starts', 0)} nonover={t.get('n_nonoverlapping', 0)} "
              f"ig-sites pct={t.get('intergenic_sites', {}).get('pct')} z={t.get('intergenic_sites', {}).get('z')} "
              f"all-sites pct={t.get('all_sites', {}).get('pct')} z={t.get('all_sites', {}).get('z')} | "
              f"+-5%: ig pct={j['null_tol0.05'].get('intergenic_sites', {}).get('pct')} | mosaic5kb pct="
              f"{j['null_mosaic5kb']['pct']} z={j['null_mosaic5kb']['z']:+.2f} null={j['null_mosaic5kb']['null_mean']:.3f}"
              f"+-{j['null_mosaic5kb']['null_sd']:.3f} max={j['null_mosaic5kb']['null_max']:.3f} minElig={j['null_mosaic5kb']['min_eligible_per_block']}")
json.dump(out, open(f'{W}/gc_matched_null.json', 'w'), indent=1)
print('wrote', f'{W}/gc_matched_null.json')
