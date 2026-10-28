#!/usr/bin/env python3
"""Figure 7e-h: per-window oriented AUROC of every method on each dataset, with the batch control."""

from __future__ import annotations
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

sys.path.insert(0, f'{REPO}/paper/common')
from make_paper_figures import STYLE, FS, TW, plabel, save  # noqa: E402

AUROC = Path(f'{RAWMOD_WORK}/read_origin/auroc.tsv')
# flow-cell null (run_hs_fcnull.sh): one HS578 line, flow cell 1 vs 2, tumor and
# normal lines as separate points; no biological difference exists
FCNULL = Path(f'{RAWMOD_WORK}/read_origin/auroc_fcnull.tsv')

# Three hues, Okabe-Ito, by WHAT EACH METHOD WAS GIVEN -- not by which method it
# is, which is already positional. validate_palette.js --pairs all: ALL PASS
# (worst CVD dE 8.6 tritan / 11.0 deutan; normal-vision floor 18.7).
# two hues: unsupervised vs supervised. MethylBERT (fine-tuned on
# held-out folds of these reads) and ROCIT (trained on HG008) are both supervised.
FREE, SUP = '#4b71bb', '#aa4499'  # paper palette blue/purple (was Okabe-Ito '#0072B2', '#009E73')
XFER = SUP
METHODS = [('RawMod', 'RawMod', FREE),
           ('dorado', 'Dorado', FREE),
           ('dorado+RawMod', 'Dorado+RawMod', FREE),
           ('MethylBERT', 'MethylBERT', SUP),
  # ROCIT appears ONLY as its own pipeline trains it: on HG008, from
  # somatic variants, then applied here with no labels from us. The
  # variant where we handed it our true labels is not shown -- ROCIT
  # derives its labels itself, and on HS578 the arms are different
  # donors so that stack cannot run at all, which makes the oracle
  # version a protocol the method could never execute here.
           ('ROCIT', 'ROCIT', XFER)]

# Third title line is the dorado modification model (dorado omitted for width), because it is not the same
# one everywhere: the human lines use the CpG-only caller, while SPO1 is given
# dorado's all-context cytosine and adenine models, i.e. the best it ships.
PANELS = [('SPO1', 'SPO1   5hmU\nnative vs. PCR',
           r'5mC$\_$5hmC@v3'),
          ('COLO829', 'COLO829   5mC/CpG\ncell typing',
           r'5mCG$\_$5hmCG@v3'),
          ('HS578', 'HS578   5mC/CpG\ntumor/normal typing',
           r'5mCG$\_$5hmCG@v3'),
          ('HS578_fc', 'HS578\nflow cell typing',
           r'5mCG$\_$5hmCG@v3')]

# SPO1 arms (unify_auroc.SPO1_RUN): RawMod is one arm over T sites only, dorado
# is every call it makes with the all-context 5mC_5hmC@v3 + 6mA@v3 models (all C
# + all A), and dorado+RawMod is late fusion of dorado at every C and RawMod at T (option E).
# The site-matched and per-site-class numbers live in tab:methodcmp, not here.


def strip(ax, x, v, colr, seed):
    """Every window as a point, median as a bar. A box would hide how bimodal
    these are: most windows are solved or untouched, few in between."""
    r = np.random.default_rng(seed)
    ax.scatter(x + r.uniform(-0.26, 0.26, len(v)), v, s=4.5, color=colr,
               alpha=0.55, lw=0.2, ec='white', zorder=3)
    ax.hlines(np.median(v), x - 0.36, x + 0.36, color='black', lw=1.6, zorder=6)


def method_panel(ax, tab, title, model):
    for i, (key, _name, colr) in enumerate(METHODS):
        g = tab[tab.method == key]
        if g.empty:  # cannot be run here -- say so
            ax.text(i, 0.70, 'n/a', ha='center', va='center',
                    fontsize=FS - 1.5, color='#999999', style='italic')
            continue
        strip(ax, i, g.auroc.to_numpy(), colr, seed=i)
  # the SAMPLING null on every panel, E[max(AUC, 1-AUC)] for the window's
  # read count. tab.null on SPO1 is floored at the 0.78 library-identity
  # control, which is RawMod's own floor, not a chance level; that control
  # is reported in the text, not drawn here.
    n = tab.n_reads.to_numpy(float)
    nul = np.median(0.5 + np.sqrt((n + 1.0) / (3.0 * n ** 2)) * np.sqrt(2.0 / np.pi)) if len(n) else np.nan
  # the line stays, the number does not: its value is in tab:methodcmp, and
  # three different nulls spelled out on three panels was noise
    ax.axhline(nul, color='black', lw=0.9, ls='--', zorder=2)
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([m[1] for m in METHODS], fontsize=FS - 1.5, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_xlim(-0.6, len(METHODS) - 0.4)
    ax.set_ylim(0.42, 1.03)
    ax.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax.grid(True, axis='y')
    ax.set_title(title, fontsize=FS - 1, pad=13, linespacing=1.0)
    ax.text(0.5, 1.012, model, transform=ax.transAxes, ha='center', va='bottom',
            fontsize=FS - 1.5, color='#555555')


def main():
    matplotlib.rcdefaults(); plt.rcParams.update(STYLE)
    tab = pd.concat([pd.read_csv(f, sep='\t') for f in (AUROC, FCNULL) if Path(f).exists()],
                    ignore_index=True)

    fig, axs = plt.subplots(1, 4, figsize=(TW, 3.4),
                            gridspec_kw=dict(wspace=0.16))
    for ax, (ds, title, model) in zip(axs, PANELS):
        method_panel(ax, tab[tab.dataset == ds], title, model)
    axs[0].set_ylabel('AUROC')
    for ax in axs[1:]:
        ax.tick_params(labelleft=False)

    h = [plt.Line2D([], [], marker='o', ls='', ms=3.6, color=c, label=l)
         for c, l in ((FREE, 'Unsupervised'), (SUP, 'Supervised'))]
  # a figure-level legend in its own band above the titles, so it can never
  # collide with the panel letters or the panel titles
    fig.legend(handles=h, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0),
               frameon=False, fontsize=FS - 1.5, handletextpad=0.15,
               columnspacing=1.6, borderaxespad=0.0)

    for ax, s in zip(axs, 'abcd'):
        plabel(ax, s, x=-0.10, y=1.20)

  # save() crops with bbox_inches='tight', so what LaTeX receives is the axes
  # extent, not figsize; without this it is magnified at width=\linewidth.
    fig.subplots_adjust(left=0.055, right=0.997, top=0.74, bottom=0.135)
    save(fig, 'fig_method_comparison.pdf')


if __name__ == '__main__':
    main()
