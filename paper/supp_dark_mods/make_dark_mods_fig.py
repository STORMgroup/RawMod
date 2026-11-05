#!/usr/bin/env python3
"""Supplementary figure: modifications that only RawMod can call.

Row 1 (benchmark sites, ground truth): the UMAP of Fig. S5 (region-holdout mixed model). One panel per released
single-sample caller; a modified site is coloured by its modification where the tool has a released model for the
row's modification and context (the N/A pattern of tab:supp-bench-matched), and drawn black where the tool has none.
Row 2 (A. carterae contig_3762, no ground truth): UMAP of the released all-data model's embeddings of + strand sites,
labelled by Dorado's all-context calls (>= 10 reads; called >= 50%, unmodified <= 10%); T and G have no Dorado model.
Inputs: the Figure S2 embedding of the benchmark sites and the A. carterae embedding of embed_sites.py (both
shipped). Writes fig_supp_dark_mods.{pdf,png}.
"""
import os as _os
from pathlib import Path as _Path
REPO = str(_Path(__file__).resolve().parents[2])
RAWMOD_RESULTS = _os.environ.get('RAWMOD_RESULTS') or f'{REPO}/paper/data/runs'  # training runs; shipped metrics by default
RAWMOD_DATA = _os.environ.get('RAWMOD_DATA', 'data')  # datasets (not shipped: set this to score raw signal)
RAWMOD_WORK = _os.environ.get('RAWMOD_WORK') or f'{REPO}/paper/data/work'  # outputs of the paper scripts; shipped by default

import sys, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, f'{REPO}/paper/common')
import make_paper_figures as P
DARK = f'{RAWMOD_WORK}/dark_mods/embedding_dark_acart.npz'   # embed_sites.py output; shipped

z = np.load(f'{RAWMOD_WORK}/figures/embedding_bench_region.npz')   # the Figure S2 embedding (figS2_embedding/)
o = np.random.default_rng(0).permutation(len(z['U']))
U, MOD, ROW = z['U'][o], z['mod'][o], z['row'][o]
CPG = {'arabidopsis_cpg_chr4', 'hg002_cpg', 'mouse_cpg', 'rice_cpg', 'ecoli_mssi_5mC_reg', 'syn_5mC'}
NONCPG = {'arabidopsis_noncpg_chr4', 'ecoli_dcm_5mC_reg'}
SIXMA = {'anabaena_6mA_reg', 'ecoli_dam_6mA_reg', 'hp26695_6mA_reg', 'syn_6mA'}
COVER = {   # rows each released single-sample caller has a model for (tab:supp-bench-matched)
    'Dorado': CPG | NONCPG | SIXMA | {'syn_5hmC', 'hp26695_4mC_reg'},
    'UniMeth': CPG | NONCPG | SIXMA,
    'DeepMod2': CPG, 'Rockfish': CPG, 'MethyNano': CPG | NONCPG,
    'RawMod': set(ROW)}
UNMOD = '#d9d9d9'; S, EW = 3.0, 0.1

def clean(ax, UU):
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(UU[:, 0].min() - 0.5, UU[:, 0].max() + 0.5); ax.set_ylim(UU[:, 1].min() - 0.5, UU[:, 1].max() + 0.5)

fig = plt.figure(figsize=(P.TW, 6.6))
gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.15], hspace=0.35, wspace=0.06)
tools = list(COVER)
for i, t in enumerate(tools):
    ax = fig.add_subplot(gs[i // 3, i % 3])
    mod = MOD != 'unmodified'; can = np.isin(ROW, list(COVER[t])) & mod; cannot = mod & ~can
    ax.scatter(U[~mod, 0], U[~mod, 1], s=S, c=UNMOD, lw=0, rasterized=True, zorder=1)
    ax.scatter(U[cannot, 0], U[cannot, 1], s=S, c='black', lw=0, rasterized=True, zorder=2)
    for c in P.CHEMS:
        m = can & (MOD == c)
        ax.scatter(U[m, 0], U[m, 1], s=S, c=P.CHEM_COL[c], lw=EW, edgecolors='black', rasterized=True, zorder=3)
    clean(ax, U)
    n_mod = mod.sum(); ax.set_title(f'{t}\n{can.sum() / n_mod:.0%} of modified sites', fontsize=P.FS - 1, pad=2, linespacing=1.1)
    if i == 0: P.plabel(ax, 'a', x=-0.08, y=1.18)
ax = fig.add_subplot(gs[0:2, 3]); ax.axis('off')
h = [Line2D([], [], ls='', marker='o', ms=4, mfc=P.CHEM_COL[c], mec='black', mew=0.3, label=c) for c in P.CHEMS]
h += [Line2D([], [], ls='', marker='o', ms=4, mfc='black', mec='black', label='no model'),
      Line2D([], [], ls='', marker='o', ms=4, mfc=UNMOD, mec=UNMOD, label='unmodified')]
ax.legend(handles=h, loc='center left', frameon=False, fontsize=P.FS - 1, handletextpad=0.2, labelspacing=0.4,
          title='Modified site,\ntool has a model', title_fontsize=P.FS - 1)

try:
    d = np.load(DARK)
except FileNotFoundError:
    d = None
if d is not None:
    UD, PD, DL, BS = d['U'], d['P'], d['dorado'], d['base']
    LAB = [('5mC CpG', P.BLUE), ('5mC non-CpG', '#8fb0e6'), ('5hmC', P.PURPLE), ('6mA', P.AQUA), ('4mC', P.GOLD),
           ('unmodified', '#a6a6a6'), ('no Dorado model', 'black')]
    ax = fig.add_subplot(gs[2, 0]); P.plabel(ax, 'b', x=-0.08, y=1.05)
    ZO = {'no Dorado model': 1, 'unmodified': 2}
    for lab, col in LAB:
        m = DL == lab
        ax.scatter(UD[m, 0], UD[m, 1], s=S, c=col, lw=0 if lab in ZO else EW, edgecolors='black', rasterized=True,
                   zorder=ZO.get(lab, 3))
    clean(ax, UD); ax.set_title('Dorado call', fontsize=P.FS - 1, pad=2)
    ax = fig.add_subplot(gs[2, 1])
    sc = ax.scatter(UD[:, 0], UD[:, 1], s=S, c=PD, cmap='Blues', vmin=0, vmax=1, lw=EW, edgecolors='black', rasterized=True)
    clean(ax, UD); ax.set_title('RawMod score', fontsize=P.FS - 1, pad=2)
    cax = ax.inset_axes([0.15, -0.09, 0.7, 0.04]); cb = fig.colorbar(sc, cax=cax, orientation='horizontal', ticks=[0, 0.5, 1])
    cb.ax.tick_params(labelsize=P.FS - 1.5, length=2, pad=1); cb.outline.set_linewidth(0.4)
    ax = fig.add_subplot(gs[2, 2])
    dark = (DL == 'no Dorado model') & (PD >= 0.5)
    ax.scatter(UD[~dark, 0], UD[~dark, 1], s=S, c=UNMOD, lw=0, rasterized=True, zorder=1)
    for b, col, nm in (('T', P.RED, 'T'), ('G', 'white', 'G (no known mark)')):
        m = dark & (BS == b)
        ax.scatter(UD[m, 0], UD[m, 1], s=S + 1, c=col, lw=0.3, edgecolors='black', rasterized=True, zorder=2 if b == 'G' else 3,
                   label=f'{nm} ({m.sum()} of {(BS == b).sum()})')
    clean(ax, UD); ax.set_title('No Dorado model,\nRawMod score $\\geq$ 0.5', fontsize=P.FS - 1, pad=2, linespacing=1.1)
    dark_h, dark_l = ax.get_legend_handles_labels()
    ax = fig.add_subplot(gs[2, 3]); ax.axis('off')
    h = [Line2D([], [], ls='', marker='o', ms=4, mfc=c, mec='black' if l not in ('unmodified', 'no Dorado model') else c, mew=0.3,
                label={'no Dorado model': 'no Dorado model (T, G)', 'unmodified': 'Dorado unmodified'}.get(l, l)) for l, c in LAB]
    lg = ax.legend(handles=h, loc='upper left', bbox_to_anchor=(0.0, 1.02), frameon=False, fontsize=P.FS - 1.5, handletextpad=0.2,
                   labelspacing=0.3, title='Dorado call', title_fontsize=P.FS - 1, alignment='left')
    ax.add_artist(lg)
    ax.legend(dark_h, dark_l, loc='lower left', bbox_to_anchor=(0.0, -0.08), frameon=False, fontsize=P.FS - 1.5, handletextpad=0.2,
              markerscale=1.5, labelspacing=0.3, title='Right panel', title_fontsize=P.FS - 1, alignment='left')
for d in (P.HERE, P.TEX):   # as make_paper_figures.save, at 300 dpi for the rasterized points
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / 'fig_supp_dark_mods.pdf', dpi=300, bbox_inches='tight', pad_inches=0.02)
fig.savefig(P.HERE / 'fig_supp_dark_mods.png', dpi=300, bbox_inches='tight', pad_inches=0.02)
print('wrote fig_supp_dark_mods.pdf')
