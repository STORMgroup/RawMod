"""The basecall-error baseline in the SAME unit as the RawMod strain-typing number:
median over windows of the oriented AUROC, with read length matched inside each window.

RawMod's figure is a median over 1200 bp windows of the AUROC separating native from PCR
reads of that window. Here the per-read score is a Dorado basecall statistic instead of a
RawMod embedding, so the two can be compared directly."""
import os as _os
from pathlib import Path as _Path
REPO = str(_Path(__file__).resolve().parents[2])
RAWMOD_RESULTS = _os.environ.get('RAWMOD_RESULTS') or f'{REPO}/paper/data/runs'  # training runs; shipped metrics by default
RAWMOD_DATA = _os.environ.get('RAWMOD_DATA', 'data')  # datasets (not shipped: set this to score raw signal)
RAWMOD_WORK = _os.environ.get('RAWMOD_WORK') or f'{REPO}/paper/data/work'  # outputs of the paper scripts; shipped by default
import sys, numpy as np, pandas as pd, pysam
from scipy.stats import rankdata
ROOT = _os.environ.get('SPO1_DIR') or f'{RAWMOD_DATA}/SPO1'   # one directory per barcode: reads.tsv (read_id, group), windows.tsv, basecalled/reads_refined.sorted.bam
W = f'{RAWMOD_WORK}/basecall_errors'
FEATS = ['identity', 'qs', 'subT', 'subFar', 'subT_minus_far']


def auroc(pos, neg):
    x = np.concatenate([pos, neg]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    ok = np.isfinite(x); x, y = x[ok], y[ok]
    if y.sum() < 5 or (1 - y).sum() < 5:
        return np.nan
    r = rankdata(x)
    return (r[y == 1].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (1 - y).sum())


rng = np.random.default_rng(0)
POOL = {f: [] for f in FEATS}
POOLC = {f: [] for f in FEATS}
print(f"{'dataset':10s} {'windows':>8s}" + ''.join(f'{f:>18s}' for f in FEATS))
for ds, (a, b) in [('bc01', ('mod_native', 'unmod_pcr')), ('bc02', ('mod_native', 'unmod_pcr')),
                   ('bc03', ('mod_native', 'unmod_pcr')), ('bc04', ('mod_native', 'unmod_pcr')),
                   ('bc05', ('mod_native', 'unmod_pcr')), ('runctrl', ('mod_run1', 'mod_run2')),
                   ('pcrctrl', ('pcr_bc01', 'pcr_bc05'))]:
    try:
        df = pd.read_csv(f'{W}/bc_strain_typing_{ds}_reads.tsv.gz', sep='\t')
        win = pd.read_csv(f'{ROOT}/{ds}/windows.tsv', sep='\t')
    except Exception:
        continue
    bam = pysam.AlignmentFile(f'{ROOT}/{ds}/basecalled/reads_refined.sorted.bam')
    span = {}
    for r in bam.fetch(until_eof=True):
        if not r.is_unmapped and not r.is_secondary and not r.is_supplementary:
            span[r.query_name] = (r.reference_start, r.reference_end)
    df = df[df.read.isin(span)].copy()
    df['rs'] = [span[x][0] for x in df.read]; df['re'] = [span[x][1] for x in df.read]
    med = {f: [] for f in FEATS}; nwin = 0
    for _, w in win.iterrows():
        lo, hi = int(w.start), int(w.end)
        ov = np.minimum(df.re, hi) - np.maximum(df.rs, lo)
        s = df[ov >= 0.6 * (hi - lo)]                                  # read covers >=60% of the window
        if (s.group == a).sum() < 10 or (s.group == b).sum() < 10:
            continue
        # length-match inside the window, deciles of pooled length
        e = np.quantile(s.length, np.linspace(0, 1, 6)); e[0] -= 1; e[-1] += 1
        keep = []
        for lo, hi in zip(e[:-1], e[1:]):
            t = s[(s.length > lo) & (s.length <= hi)]
            ia, ib = t.index[t.group == a], t.index[t.group == b]
            n = min(len(ia), len(ib))
            if n:
                keep += list(rng.choice(ia, n, replace=False)) + list(rng.choice(ib, n, replace=False))
        m = s.loc[keep]
        if (m.group == a).sum() < 10:
            continue
        nwin += 1
        for f in FEATS:
            v = auroc(m[f][m.group == a].values, m[f][m.group == b].values)
            # orient: report the separation, not the direction
            med[f].append(max(v, 1 - v) if np.isfinite(v) else np.nan)
    tgt = POOL if ds.startswith('bc') else POOLC
    for f in FEATS: tgt[f] += [v for v in med[f] if np.isfinite(v)]
    print(f'{ds:10s} {nwin:>8d}' + ''.join(f'{np.nanmedian(med[f]):>18.3f}' if nwin else f'{"--":>18s}' for f in FEATS))

print()
print(f"{'POOLED':10s} {len(POOL[FEATS[0]]):>8d}" + ''.join(f'{np.median(POOL[f]):>18.3f}' for f in FEATS) + '   native vs PCR')
print(f"{'CONTROLS':10s} {len(POOLC[FEATS[0]]):>8d}" + ''.join(f'{np.median(POOLC[f]):>18.3f}' for f in FEATS) + '   null')
