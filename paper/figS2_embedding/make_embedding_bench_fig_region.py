#!/usr/bin/env python3
"""Figure S2: UMAP of the benchmark site embeddings coloured by modification, reference base and RawMod score, with one panel per organism."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import sys
from pathlib import Path as _Path
REPO = str(_Path(__file__).resolve().parents[2])
sys.path.insert(0, f'{REPO}/paper/common')
import make_paper_figures as P

z = np.load(P.FIGDATA / 'embedding_bench_region.npz')
order = np.random.default_rng(0).permutation(len(z['U']))  # random draw order: no class always on top
U, MODS, RB, ORGS, LAB, SC = (z[k][order] for k in ('U', 'mod', 'refbase', 'organism', 'label', 'P'))

UNMOD = '#b4b4b4'
MOD = [('unmodified', 'Unmodified', UNMOD)] + [(c, c, P.CHEM_COL[c]) for c in P.CHEMS]
BASE = [('A', 'A', P.BLUE), ('C', 'C', P.PURPLE), ('G', 'G', P.AQUA), ('T', 'T', P.GOLD)]
ORG = ['ONT oligonucleotides', 'H. pylori 26695', 'SPO1', 'E. coli', 'Anabaena', 'A. thaliana',
       'Human (HG002)', 'Mouse brain', 'O. sativa']
ORG_TEX = {'ONT oligonucleotides': 'ONT oligos', 'H. pylori 26695': r'$\mathit{H.\ pylori}$ 26695',
           'SPO1': 'Phage SPO1', 'E. coli': r'$\mathit{E.\ coli}$', 'Anabaena': r'$\mathit{Anabaena}$',
           'A. thaliana': r'$\mathit{A.\ thaliana}$', 'Human (HG002)': 'Human (HG002)', 'Mouse brain': 'Mouse brain',
           'O. sativa': r'$\mathit{O.\ sativa}$'}
MOD_C, UNMOD_C = P.PURPLE, P.BLUE  # panel d: modified / unmodified
S, EW = 4.0, 0.12  # marker size and edge width


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(U[:, 0].min() - 0.5, U[:, 0].max() + 0.5); ax.set_ylim(U[:, 1].min() - 0.5, U[:, 1].max() + 0.5)


def legend(ax, spec, ncol):
    h = [Line2D([], [], ls='', marker='o', ms=3.8, mfc=c, mec='black', mew=0.3, label=n) for _, n, c in spec]
    ax.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=ncol, frameon=False,
              handletextpad=0.1, columnspacing=0.7, labelspacing=0.2, fontsize=P.FS - 1)


def cat_panel(ax, lab, spec, title, letter, ncol):
    for key, _, col in spec:
        m = lab == key
        ax.scatter(U[m, 0], U[m, 1], s=S, c=col, lw=EW, edgecolors='black', rasterized=True,
                   zorder=1 if key == 'unmodified' else 2)
    clean(ax); ax.set_title(title, fontsize=P.FS); P.plabel(ax, letter, x=-0.04, y=1.02)
    legend(ax, [s for s in spec if (lab == s[0]).any()], ncol)


fig = plt.figure(figsize=(P.TW, 6.3))
gs = GridSpec(2, 1, height_ratios=[1.12, 1.0], hspace=0.28, figure=fig)
top = gs[0].subgridspec(1, 3, wspace=0.08)
bot = gs[1].subgridspec(2, 5, wspace=0.05, hspace=0.32)

cat_panel(fig.add_subplot(top[0]), MODS, MOD, 'Modification', 'a', 3)
cat_panel(fig.add_subplot(top[1]), RB, BASE, 'Reference base', 'b', 4)
ax = fig.add_subplot(top[2])
sc = ax.scatter(U[:, 0], U[:, 1], s=S, c=SC, cmap='Blues', vmin=0, vmax=1, lw=EW, edgecolors='black', rasterized=True)
clean(ax); ax.set_title('RawMod score', fontsize=P.FS); P.plabel(ax, 'c', x=-0.04, y=1.02)
cax = ax.inset_axes([0.15, -0.075, 0.7, 0.035])
cb = fig.colorbar(sc, cax=cax, orientation='horizontal', ticks=[0, 0.5, 1])
cb.ax.tick_params(labelsize=P.FS - 1, length=2, pad=1); cb.outline.set_linewidth(0.4)

for i, org in enumerate(ORG):
    ax = fig.add_subplot(bot[i // 5, i % 5])
    o = ORGS == org
    ax.scatter(U[~o, 0], U[~o, 1], s=1.2, c='#e3e3e3', lw=0, rasterized=True, zorder=1)
    for lab, col in ((0, UNMOD_C), (1, MOD_C)):
        m = o & (LAB == lab)
        ax.scatter(U[m, 0], U[m, 1], s=S, c=col, lw=EW, edgecolors='black', rasterized=True, zorder=2 + lab)
    clean(ax)
    mods = [k for k, _, _ in MOD if k != 'unmodified' and (MODS[o] == k).any()]
    ax.set_title(f"{ORG_TEX[org]}\n{', '.join(mods)}", fontsize=P.FS - 1, pad=2, linespacing=1.1)
    if i == 0:
        P.plabel(ax, 'd', x=-0.12, y=1.2)
ax = fig.add_subplot(bot[1, 4]); ax.axis('off')
h = [Line2D([], [], ls='', marker='o', ms=4.2, mfc=c, mec='black', mew=0.3, label=n)
     for n, c in (('Modified', MOD_C), ('Unmodified', UNMOD_C), ('Other organisms', '#e3e3e3'))]
ax.legend(handles=h, loc='center', frameon=False, handletextpad=0.2, labelspacing=0.5, fontsize=P.FS - 1)

for ext in ('pdf', 'png'):
    fig.savefig(P.HERE / f'fig_supp_embedding_bench_region.{ext}', dpi=300, bbox_inches='tight', pad_inches=0.02)
print('wrote fig_supp_embedding_bench_region.pdf')
