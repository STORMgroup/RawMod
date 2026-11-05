#!/usr/bin/env python3
"""Supplementary calibration figure: reliability diagram of the region-holdout mixed model on the withheld-region test set.

Input: <work>/calibration/region_sites.npz (per-position probability and label, chemistry, organism, base; written by
score_region_sites.py and shipped). Output: fig_supp_calibration.pdf/.png (a: reliability curve with 10 equal-width
bins and the count per bin; b: per-chemistry ECE) and <work>/calibration/calibration_region.json (ECE, maximum
calibration error, Brier score, overall and per chemistry).
"""
import os as _os
from pathlib import Path as _Path
REPO = str(_Path(__file__).resolve().parents[2])
RAWMOD_RESULTS = _os.environ.get('RAWMOD_RESULTS') or f'{REPO}/paper/data/runs'  # training runs; shipped metrics by default
RAWMOD_DATA = _os.environ.get('RAWMOD_DATA', 'data')  # datasets (not shipped: set this to score raw signal)
RAWMOD_WORK = _os.environ.get('RAWMOD_WORK') or f'{REPO}/paper/data/work'  # outputs of the paper scripts; shipped by default

import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, f'{REPO}/paper/common'); sys.path.insert(0, f'{REPO}/scripts/train')
import make_paper_figures as P
from run_matched_loco import CHEMS, CHEM_BASES, CHEM_ORGS   # the chemistry -> base / organism tables of the trainer

WORK = _Path(f'{RAWMOD_WORK}/calibration')
d = np.load(WORK / 'region_sites.npz', allow_pickle=True)
y, p, chem = d['y'].astype(int), d['p'].astype(float), d['chem'].astype(str)
EDGES = np.linspace(0, 1, 11)


def reliability(y, p):
    b = np.clip(np.digitize(p, EDGES[1:-1]), 0, 9)
    rows = []
    for k in range(10):
        m = b == k
        if m.sum():
            rows.append((k, int(m.sum()), float(p[m].mean()), float(y[m].mean())))
    n = np.array([r[1] for r in rows]); conf = np.array([r[2] for r in rows]); acc = np.array([r[3] for r in rows])
    ece = float((n / n.sum() * np.abs(acc - conf)).sum())
    mce = float(np.abs(acc - conf).max())
    brier = float(np.mean((p - y) ** 2))
    return rows, ece, mce, brier


res = {}
rows, ece, mce, brier = reliability(y, p)
res['all'] = dict(n=int(len(y)), n_pos=int(y.sum()), ece=ece, mce=mce, brier=brier, bins=rows)
# per chemistry: its positives against the negatives of the same organisms and base, as the read-depth curve does
org = d['org'].astype(str)
base = np.array([b[2:-1] if b.startswith("b'") else b for b in d['base'].astype(str)])   # S1 -> str
for c in CHEMS:
    pos = (y == 1) & (chem == c)
    cb = [x.decode() if isinstance(x, bytes) else str(x) for x in CHEM_BASES[c]]
    co = [x.decode() if isinstance(x, bytes) else str(x) for x in CHEM_ORGS[c]]
    neg = (y == 0) & np.isin(org, co) & np.isin(base, cb)
    sel = pos | neg
    if pos.sum() > 20 and neg.sum() > 20:
        r, e, m, b = reliability(y[sel], p[sel])
        res[c] = dict(n=int(sel.sum()), n_pos=int(pos.sum()), ece=e, mce=m, brier=b, bins=r)
json.dump(res, open(WORK / 'calibration_region.json', 'w'), indent=1)
for k, v in res.items():
    print(f'{k:6s} n={v["n"]:6d} pos={v["n_pos"]:6d} ECE {v["ece"]:.3f} MCE {v["mce"]:.3f} Brier {v["brier"]:.3f}')

matplotlib.rcdefaults(); plt.rcParams.update(P.STYLE)
fig = plt.figure(figsize=(P.TW * 0.78, 2.8))
gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.0], width_ratios=[1.0, 0.8], hspace=0.08, wspace=0.35)
ax = fig.add_subplot(gs[0, 0]); P.plabel(ax, 'a', x=-0.16)
axn = fig.add_subplot(gs[1, 0], sharex=ax)
ax.plot([0, 1], [0, 1], color='#a0a0a0', lw=0.8, ls='--', zorder=0)
conf = [r[2] for r in rows]; acc = [r[3] for r in rows]; n = np.array([r[1] for r in rows])
ax.plot(conf, acc, '-o', color=P.BLUE, ms=3.5, lw=1.3, mec='black', mew=0.3, zorder=3)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_ylabel('Fraction modified'); ax.tick_params(axis='x', labelbottom=False)
ax.text(0.03, 0.95, f'ECE {ece:.3f}\nBrier {brier:.3f}', ha='left', va='top', fontsize=P.FS - 1.5, transform=ax.transAxes)
# positions per bin, in its own strip so that no count sits on the curve
ctr = np.array([(r[0] + 0.5) / 10 for r in rows])
axn.bar(ctr, n, 0.08, color=P.BLUE, **P.EDGE)
axn.set_yscale('log'); axn.set_ylim(150, 3e5); axn.set_yticks([1e3, 1e4, 1e5])
axn.yaxis.set_minor_locator(plt.NullLocator())
axn.set_yticklabels(['$10^3$', '$10^4$', '$10^5$'], fontsize=P.FS - 1.5)
axn.set_ylabel('Positions', fontsize=P.FS - 1.5)
axn.set_xlabel('Site score (bin mean)')
ax = fig.add_subplot(gs[:, 1]); P.plabel(ax, 'b', x=-0.2)
cs = [c for c in CHEMS if c in res]
vals = [res[c]['ece'] for c in cs]
ax.bar(np.arange(len(cs)), vals, 0.6, color=[P.CHEM_COL[c] for c in cs], **P.EDGE)
for i, v in enumerate(vals):
    ax.text(i, v + 0.004, f'{v:.3f}', ha='center', va='bottom', fontsize=P.FS - 1.5)
ax.set_xticks(np.arange(len(cs))); ax.set_xticklabels(cs, rotation=40, ha='right', rotation_mode='anchor'); ax.tick_params(axis='x', top=False, bottom=False)
ax.set_ylabel('ECE'); ax.set_ylim(0, max(vals) * 1.35)
P.save(fig, 'fig_supp_calibration.pdf')
