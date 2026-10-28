#!/usr/bin/env python3
"""Motivation figure (column width). (a) A modification-specific caller sees the raw signal of one read and runs one model per modification. (b) RawMod sees the signal pileup of every read at the site and runs one model for any modification. Both panels show the same 5mC site of the ONT oligonucleotide library (Figure 1 data): one read against the expected current, and the deviation of all 15 reads from it."""
import os as _os
from pathlib import Path as _Path
REPO = str(_Path(__file__).resolve().parents[2])
RAWMOD_WORK = _os.environ.get('RAWMOD_WORK') or f'{REPO}/paper/data/work'  # outputs of the paper scripts; shipped by default

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, f'{REPO}/paper/common')
import make_paper_figures as P
from make_paper_figures import STYLE, FS, COL, CHEM_COL, BLUE, GREY, plabel, save

L, HALF = 10, 5                       # samples per base; bases shown on each side of the site
SEQ = 'CAGTAGCGAACATGCGAGCAT'         # reference 21-mer of the example site (5mers_rand_ref_adapter_13, + strand, position 74)
sc = np.load(P.FIGDATA / 'signal_contrast.npz', allow_pickle=True)
X = sc['5mC_ex_mod']                  # (16, 210): expected current (row 0) + 15 reads, 21 bases x 10 samples
cols = slice((10 - HALF) * L, (11 + HALF) * L)
ref, reads = X[0, cols], X[1:, cols]
t = (np.arange(reads.shape[1]) + 0.5) / L - HALF - 0.5          # base coordinate of each sample, site at 0
bases = list(SEQ[10 - HALF:11 + HALF])
ticks = np.arange(-HALF, HALF + 1)
INK, SOFT = 'black', '#6a6a6a'


def arrow(fig, x0, y0, x1, y1):
    fig.add_artist(FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure, arrowstyle='-|>', mutation_scale=7,
                                   lw=0.8, color=INK, shrinkA=0, shrinkB=0))


def main():
    matplotlib.rcdefaults(); plt.rcParams.update(STYLE)
    fig = plt.figure(figsize=(COL, 3.0))
    gs = fig.add_gridspec(2, 1, left=0.13, right=0.55, top=0.90, bottom=0.11, hspace=0.62, height_ratios=[1, 1.15])

  # ---- (a) one read against the expected current -----------------------------------------------
    ax = fig.add_subplot(gs[0]); ax.grid(False)
    ax.axvspan(-0.5, 0.5, color='k', alpha=0.08, lw=0)
    ax.step(np.r_[t, t[-1] + 1 / L], np.r_[ref, ref[-1]], where='post', color=BLUE, lw=1.0, label='expected (unmodified)')
    ax.plot(t, reads[0], color=INK, lw=0.7, label='one read')
    ax.set_xlim(-HALF - 0.5, HALF + 0.5); ax.set_ylim(-2.1, 2.1)
    ax.set_xticks(ticks); ax.set_xticklabels(bases, fontsize=FS - 1.5, family='monospace')
    ax.set_yticks([-2, 0, 2]); ax.set_ylabel('Current (norm.)', labelpad=1)
    ax.tick_params(length=2, pad=1.5)
    ax.legend(loc='upper left', fontsize=FS - 2.5, frameon=False, handlelength=1.3, labelspacing=0.15,
              handletextpad=0.4, borderaxespad=0.2)
    ax.set_title('One read at the site', fontsize=FS - 0.5, pad=2, loc='left')
    plabel(ax, 'a', x=-0.30, y=1.04)

  # ---- (b) the pileup: deviation of every read from the expected current -----------------------
    ax2 = fig.add_subplot(gs[1]); ax2.grid(False)
    dlt = reads - ref
    im = ax2.imshow(dlt, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2, interpolation='nearest',
                    extent=(-HALF - 0.5, HALF + 0.5, reads.shape[0] + 0.5, 0.5))
    ax2.axvline(-0.5, color='k', lw=0.7); ax2.axvline(0.5, color='k', lw=0.7)
    ax2.set_xticks(ticks); ax2.set_xticklabels(bases, fontsize=FS - 1.5, family='monospace')
    ax2.set_yticks([1, 5, 10, 15]); ax2.set_ylabel('Read', labelpad=1)
    ax2.tick_params(length=2, pad=1.5, right=False, top=False)
    ax2.set_xlabel('Reference base (queried C at the centre)', labelpad=1)
    ax2.set_title('All 15 reads at the site', fontsize=FS - 0.5, pad=2, loc='left')
    plabel(ax2, 'b', x=-0.30, y=1.04)
    cax = fig.add_axes([0.558, gs[1].get_position(fig).y0, 0.012, gs[1].get_position(fig).height])
    cb = fig.colorbar(im, cax=cax); cb.outline.set_linewidth(0.6); cb.set_ticks([-2, 0, 2])
    cb.ax.tick_params(length=2, width=0.6, pad=1, labelsize=FS - 2); cb.set_label('Δ current', labelpad=1, fontsize=FS - 2)

  # ---- right column: what each input feeds --------------------------------------------------------
    pa, pb = gs[0].get_position(fig), gs[1].get_position(fig)
    xr = 0.71                                                    # text column
    ya = pa.y0 + pa.height / 2
    arrow(fig, pa.x1 + 0.01, ya, xr - 0.015, ya)
    fig.text(xr, pa.y1 + 0.005, 'Modification-specific\ncallers', fontsize=FS - 1, fontweight='bold', va='bottom', ha='left')
    models = [('5mC', 'model'), ('5hmC', 'model'), ('6mA', 'model'), ('4mC', 'model')]
    ys = np.linspace(pa.y1 - 0.02, pa.y0 + 0.02, len(models) + 1)
    for (c, w), y in zip(models, ys[:-1]):
        fig.add_artist(plt.Rectangle((xr, y - 0.011), 0.018, 0.022, fc=CHEM_COL[c], ec='none', transform=fig.transFigure))
        fig.text(xr + 0.03, y, f'{c} {w}', fontsize=FS - 1.5, va='center', ha='left')
    fig.text(xr + 0.03, ys[-1], '5hmU: no model', fontsize=FS - 1.5, va='center', ha='left', color=SOFT)
    fig.text(xr, pa.y0 - 0.055, 'one call per read and model;\nreads combined afterwards', fontsize=FS - 2.5, va='top', ha='left', color=SOFT)

    yb = pb.y0 + pb.height / 2
    arrow(fig, cax.get_position().x1 + 0.07, yb, xr - 0.015, yb)
    fig.text(xr, pb.y1 + 0.005, 'RawMod', fontsize=FS - 1, fontweight='bold', va='bottom', ha='left')
    fig.text(xr, yb + 0.065, 'one model', fontsize=FS - 1.5, va='center', ha='left')
    fig.text(xr, yb + 0.005, 'one score per site:\nmodified or not', fontsize=FS - 1.5, va='center', ha='left')
    fig.text(xr, yb - 0.075, 'for 5mC, 5hmC, 6mA, 4mC, 5hmU,\nand for modifications it\nhas never seen', fontsize=FS - 2.5, va='center', ha='left', color=SOFT)
    save(fig, 'fig_teaser.pdf')


if __name__ == '__main__':
    main()
