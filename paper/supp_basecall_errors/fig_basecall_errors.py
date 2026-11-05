"""Dorado basecalling on fully 5hmU-substituted phage DNA, against PCR copies of the same genome.

(a) per-read identity, native vs PCR, one violin per barcode plus the two null controls
(b) substitution+deletion rate against distance to the nearest thymine of the read strand,
    native and PCR on the same axes, showing the degradation is not localised to the modified base
(c) per-window AUROC of Dorado read identity as a strain typer, against RawMod on the same windows

Inputs: the per-read error tables and the error profiles written by bc_strain_typing.py and profile_ties.py
(shipped in paper/data/work/basecall_errors/). Panel (c) lists the per-window medians of window_auroc.py (Dorado
statistics) and of the SPO1 arm of the read-origin figure (RawMod). Writes fig_basecall_errors.{pdf,png}.
"""
import os as _os
from pathlib import Path as _Path
REPO = str(_Path(__file__).resolve().parents[2])
RAWMOD_RESULTS = _os.environ.get('RAWMOD_RESULTS') or f'{REPO}/paper/data/runs'  # training runs; shipped metrics by default
RAWMOD_DATA = _os.environ.get('RAWMOD_DATA', 'data')  # datasets (not shipped: set this to score raw signal)
RAWMOD_WORK = _os.environ.get('RAWMOD_WORK') or f'{REPO}/paper/data/work'  # outputs of the paper scripts; shipped by default

import json, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
sys.path.insert(0, f'{REPO}/paper/common')
import make_paper_figures as P                      # noqa: E402  style + palette
import matplotlib.pyplot as plt                     # noqa: E402

W = f'{RAWMOD_WORK}/basecall_errors'   # bc_strain_typing.py and profile_ties.py outputs; shipped
matplotlib.rcdefaults(); plt.rcParams.update(P.STYLE)
J = json.load(open(f'{W}/bc_strain_typing.json'))
PROF = 10
BCS = ['bc01', 'bc02', 'bc03', 'bc04', 'bc05']
CTRL = ['runctrl', 'pcrctrl']

fig, axes = plt.subplots(1, 3, figsize=(P.TW, 3.0))

# ── (a) read identity ────────────────────────────────────────────────────────
ax = axes[0]
pos, lab, dat, col = [], [], [], []
for i, ds in enumerate(BCS + CTRL):
    df = pd.read_csv(f'{W}/bc_strain_typing_{ds}_reads.tsv.gz', sep='\t')
    gs = sorted(df.group.unique(), key=lambda g: 0 if ('native' in g or g.endswith('run1') or g.endswith('bc01')) else 1)
    for j, g in enumerate(gs):
        pos.append(i + (j - 0.5) * 0.34)
        dat.append(df.identity[df.group == g].values)
        col.append(P.PURPLE if ('native' in g or 'run' in g) else P.BLUE)
    lab.append(ds.replace('bc0', 'bc').replace('runctrl', 'native/native').replace('pcrctrl', 'PCR/PCR'))
vp = ax.violinplot(dat, positions=pos, widths=0.3, showmedians=True, showextrema=False)
for b, c in zip(vp['bodies'], col):
    b.set_facecolor(c); b.set_edgecolor('black'); b.set_linewidth(0.5); b.set_alpha(1.0)
vp['cmedians'].set_color('black'); vp['cmedians'].set_linewidth(0.8)
ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, fontsize=P.FS - 1.5, rotation=45, ha='right', rotation_mode='anchor')
ax.set_ylabel('Read identity'); ax.set_ylim(0.88, 1.0)
ax.set_title('(a) Basecalling\naccuracy', loc='left')
h = [plt.Line2D([], [], marker='s', ls='', mfc=P.PURPLE, mec='black', mew=0.5, ms=4, label='native (5hmU)'),
     plt.Line2D([], [], marker='s', ls='', mfc=P.BLUE, mec='black', mew=0.5, ms=4, label='PCR (T)')]
ax.legend(handles=h, loc='lower left', fontsize=P.FS - 2)

# ── (b) error against distance to the nearest T ──────────────────────────────
# profile_ties.py: same reads as (a); only loci inside the amplicon windows covered by both groups, so native and
# PCR are compared at the same positions; a base with Ts at the same distance on both sides is its own class
# ('both'), so -d and +d hold the same kind of base; |d| >= 4 is left out (few loci per amplicon).
ax = axes[1]
PT = json.load(open(f'{W}/profile_ties_span.json'))
CL = ['-3', '-2', '-1', '0', '1', '2', '3']; XB = 4.4
for ds in BCS:
    for g, R in PT[ds].items():
        rate = np.array([R[c][1] / R[c][0] for c in CL]); rb = R['both1'][1] / R['both1'][0]
        colr = P.PURPLE if 'native' in g else P.BLUE
        ax.plot(np.arange(-3, 4), rate, color=colr, lw=1.0, alpha=0.65, marker='o', ms=2.5, mec='black', mew=0.3)
        ax.plot([XB], [rb], ls='', color=colr, alpha=0.65, marker='D', ms=2.8, mec='black', mew=0.3)
ax.axvline(3.7, color='#a0a0a0', lw=0.6)
ax.set_yscale('log')
ax.set_xlabel('Offset from the nearest T (bp)')
ax.set_ylabel('Error rate (log)')
ax.set_xticks([-3, -2, -1, 0, 1, 2, 3, XB]); ax.set_xticklabels(['$-3$', '$-2$', '$-1$', '0', '1', '2', '3', 'T·X·T'])
ax.set_xlim(-3.5, XB + 0.5)
ax.set_yticks([0.006, 0.01, 0.02, 0.04, 0.08])
ax.set_yticklabels(['0.006', '0.01', '0.02', '0.04', '0.08'])
ax.minorticks_off()
ax.set_title('(b) Parallel curves:\na constant ratio', loc='left')

# ── (c) strain typing per window ─────────────────────────────────────────────
ax = axes[2]
names = ['Dorado identity', 'Dorado qscore', 'Dorado sub. at T', 'RawMod T + CpG', 'RawMod T only']
vals = [0.827, 0.827, 0.768, 0.824, 0.792]     # medians over the 21 windows: window_auroc.py (Dorado), read-origin SPO1 arm (RawMod)
nulls = [0.589, 0.549, 0.550, 0.52, 0.52]       # each statistic's own null (native vs. native, PCR vs. PCR)
cols = [P.BLUE, P.BLUE, P.BLUE, P.PURPLE, P.PURPLE]
x = np.arange(len(names))
ax.bar(x, vals, width=0.62, color=cols, **P.EDGE)
for xi, n in zip(x, nulls):
    ax.plot([xi - 0.31, xi + 0.31], [n, n], color='black', lw=1.0, ls=(0, (2, 1.2)))
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=P.FS - 1.5, rotation=45, ha='right', rotation_mode='anchor')
ax.set_ylim(0.4, 1.0); ax.set_ylabel('Median AUROC per window')
ax.set_title('(c) Telling the\nlibraries apart', loc='left')
ax.plot([], [], color='black', lw=1.0, ls=(0, (2, 1.2)), label='own null')
ax.legend(loc='upper right', fontsize=P.FS - 2)

fig.tight_layout(pad=0.4, w_pad=1.0)
P.save(fig, 'fig_basecall_errors.pdf')
print('wrote fig_basecall_errors')
