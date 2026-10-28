#!/usr/bin/env python3
"""Figure 7: the two read-origin figures combined (panels a-d from fig_tumor_normal_joint.py, e-h from fig_method_comparison.py)."""
import os as _os
from pathlib import Path as _Path
REPO = str(_Path(__file__).resolve().parents[2])
RAWMOD_RESULTS = _os.environ.get('RAWMOD_RESULTS') or f'{REPO}/paper/data/runs'  # training runs; shipped metrics by default
RAWMOD_DATA = _os.environ.get('RAWMOD_DATA', 'data')  # datasets (not shipped: set this to score raw signal)
RAWMOD_WORK = _os.environ.get('RAWMOD_WORK') or f'{REPO}/paper/data/work'  # outputs of the paper scripts; shipped by default

import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, f'{REPO}/paper/common')
import fig_tumor_normal_joint as J
import fig_method_comparison as M
from make_paper_figures import STYLE, FS, TW, EDGE, plabel, save


def inside_legend(ax, key):
    """Replace the legend above the axes by one inside its upper-left corner."""
    ax.get_legend().remove()
    leg = ax.legend(loc='upper right' if key == 'colo' else 'lower left', fontsize=FS - 1.5, handletextpad=0.2,
                    borderpad=0.25, labelspacing=0.25, handlelength=1.0, borderaxespad=0.3, title=J.D_LAB[key])
    leg.get_title().set_fontsize(FS - 1); leg.get_title().set_fontweight('bold')


def main():
    colo, hs = J.load_tables()
    matplotlib.rcdefaults(); plt.rcParams.update(STYLE)
    # The second cell-line pair is optional; without it the figure shows the COLO829 arm alone.
    sets = [('colo', colo)] + ([('hs', hs)] if hs is not None else [])
    # The flow-cell control is measured on the second cell-line pair, so it is absent
    # whenever that pair is.
    parts = [pd.read_csv(M.AUROC, sep='\t')] + ([pd.read_csv(M.FCNULL, sep='\t')] if M.FCNULL.exists() else [])
    tab = pd.concat(parts, ignore_index=True)
    panels = [p for p in M.PANELS if (tab.dataset == p[0]).any()]

    fig = plt.figure(figsize=(TW, 4.45))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.3], hspace=0.47, left=0.075, right=0.995, top=0.91, bottom=0.14)
    # One example panel per cell-line pair; the row shrinks when only one is present.
    ratios = ([1, 1] if hs is not None else [1]) + [1.2, 1.3]
    top = gs[0].subgridspec(1, len(ratios), width_ratios=ratios, wspace=0.42)
    bot = gs[1].subgridspec(1, max(len(panels), 1), wspace=0.16)
    axs = [fig.add_subplot(top[i]) for i in range(len(ratios))]
    bxs = [fig.add_subplot(bot[i]) for i in range(len(panels))]

  # ---- top row: examples and dose-response (fig_tumor_normal_joint) ----
    Zc, yc = J.read_scores(J.COLO_EMB, 'x27'); J.scatter_panel(axs[0], Zc, yc, 'colo')
    if hs is not None:
        hs_best = hs.sort_values('ari_vs_sample', ascending=False).iloc[0]['group']
        Zh, yh = J.read_scores(J.HS_EMB, str(hs_best)); J.scatter_panel(axs[1], Zh, yh, 'hs')
    ax = axs[-2]; ax.grid(True, axis='both')
    for key, df in sets:
        r, _ = spearmanr(df.n_diff, df.ari_vs_sample)
        ax.scatter(df.n_diff, df.ari_vs_sample, s=9, color=J.D_COL[key], marker=J.D_MARK[key], lw=0.3, ec='black',
                   label=f'{J.D_LAB[key]}, ρ={r:.2f}')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), borderaxespad=0.0, fontsize=FS - 1.5, handletextpad=0.2,
              borderpad=0.25, labelspacing=0.25, handlelength=1.0)
    ax.axhline(0, color='black', lw=0.8, ls=':'); ax.set_xlabel('Differential CpGs'); ax.set_ylabel('ARI')
    ax.set_ylim(-0.1, 1.06); ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax = axs[-1]; w = 0.38; xs = np.arange(len(J.TIERS))
    for i, (key, df) in enumerate(sets):
        med = [df[(df.n_diff >= lo) & (df.n_diff < hi)].ari_vs_sample.median() for lo, hi, _ in J.TIERS]
        ax.bar(xs + (i - 0.5) * w, med, w, color=J.D_COL[key], label=J.D_LAB[key], **EDGE)
    ax.set_xticks(xs); ax.set_xticklabels([t[2] for t in J.TIERS], fontsize=FS - 1.5, rotation=35, ha='right', rotation_mode='anchor')
    ax.set_xlabel('Differential CpGs'); ax.set_ylabel('Median ARI'); ax.set_ylim(0, 1.05); ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, 1.0), borderaxespad=0.0, fontsize=FS - 1.5,
              borderpad=0.25, handlelength=1.0, handletextpad=0.3, columnspacing=0.6)

  # ---- bottom row: every method on every dataset (fig_method_comparison) ----
    for ax, (ds, title, model) in zip(bxs, panels):
        M.method_panel(ax, tab[tab.dataset == ds], title, model)
    bxs[0].set_ylabel('AUROC')
    for ax in bxs[1:]:
        ax.tick_params(labelleft=False)
    h = [plt.Line2D([], [], marker='o', ls='', ms=3.6, color=c, label=l) for c, l in ((M.FREE, 'Unsupervised'), (M.SUP, 'Supervised'))]
    bxs[0].legend(handles=h, loc='lower right', fontsize=FS - 1.5, handletextpad=0.3, borderpad=0.3,
                  labelspacing=0.25, borderaxespad=0.3)

    J.raise_for_ticklabels(fig, (axs[-2], axs[-1]))
    J.panel_letters(fig, [(ax, c, -0.22) for ax, c in zip(axs, 'abcd')])
    for ax, s in zip(bxs, 'abcdefgh'[len(axs):]):
        plabel(ax, s, x=-0.12, y=1.2)
    save(fig, 'fig_readorigin.pdf'); print('wrote fig_readorigin.pdf')


if __name__ == '__main__':
    main()
