#!/usr/bin/env python3
"""Figure 5a-d: example windows in embedding PC space and the agreement of the label-free partition with read origin against the number of differential CpGs."""

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
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, f'{REPO}/paper/common')
from make_paper_figures import STYLE, FS, TW, EDGE, plabel, save  # noqa: E402

RES = Path(f'{RAWMOD_RESULTS}')
HS = Path(f'{RAWMOD_WORK}/read_origin/hs578')
# window definitions and read origins; the truth tables are shipped with the repository
WORK = Path(_os.environ['RAWMOD_DATA'] + '/COLO829') if _os.environ.get('RAWMOD_DATA') else Path(f'{REPO}/paper/data/COLO829')
COLO_EMB = Path(f'{RAWMOD_WORK}/read_origin/data/colo3_rawmod.npz')
HS_EMB = Path(f'{RAWMOD_WORK}/read_origin/data/hs_rawmod.npz')
TIERS = [(0, 5, '<5'), (5, 10, '5–9'), (10, 20, '10–19'),
         (20, 50, '20–49'), (50, 10 ** 9, '≥50')]
# Two DIFFERENT things are coloured in this figure, so they get two different
# validated pairs -- reusing one pair for both would imply a mapping between
# them that does not exist.
#
#   a, b   which cell line a READ came from (the arms within one dataset)
#   c, d   which DATASET a point came from
#
# Both pairs are colourblind-safe, and so is their union. Checked with the
# dataviz validator (validate_palette.js, light mode, --pairs all) over all
# four together -- lightness band, chroma floor, CVD separation, normal-vision
# floor and surface contrast all PASS; worst all-pairs CVD is violet/blue at
# OKLab dE 9.5 (deutan), and those two never share a panel.
#
# The paper's house BLUE/ORANGE (#6799ce/#db8043) are deliberately NOT used
# here: they are too light to clear the contrast check at scatter-marker size.
# the paper palette (make_paper_figures.py: RawHash2 hues, aqua for green); these four pass
# validate_palette.js --pairs all (worst CVD dE 7.0 deutan blue/purple, legal with the marker shapes + legends;
# normal-vision floor 17.5). Old Okabe-Ito hexes were '#D55E00', '#0072B2' / '#785EF0', '#009E73'.
ARM_CANCER, ARM_OTHER = '#aa4499', '#4b71bb'  # purple, blue
D_COL = {'colo': '#1baf7a', 'hs': '#d4a017'}  # aqua, gold
D_MARK = {'colo': 'o', 'hs': '^'}  # shape backs up colour in c

GRID_FRACS = (1 / 3, 2 / 3)  # decorative grid for the PC panels

D_LAB = {'colo': 'COLO829', 'hs': 'HS578'}
# COLO829's arms are two cell types; HS578's are tumour and normal. Same
# underlying 'tumor'/'normal' group keys in the npz, different biology, so
# the DISPLAYED names differ per panel.
ARMS = {
    'colo': [('tumor', ARM_CANCER, 'Melanoma'),
             ('normal', ARM_OTHER, 'LCL')],
    'hs': [('tumor', ARM_CANCER, 'Tumor'),
           ('normal', ARM_OTHER, 'Normal')],
}


def read_scores(npz, window):
    """Reads x (sites*dims) matrix for one window, mean-imputed, then PCA to 2-d."""
    d = np.load(npz, allow_pickle=True)
    k = np.array([str(x) == window for x in d['window']])
    emb, rid, si = d['emb'][k], d['read_id'][k], d['site_i'][k]
    sites = sorted(set(int(x) for x in si))
    order = {s: i for i, s in enumerate(sites)}
    acc = {}
    for e, r, s in zip(emb, rid, si):
        acc.setdefault(r, {})[order[int(s)]] = e
    need = int(round(0.6 * len(sites)))
    reads = [r for r, v in acc.items() if len(v) >= need]
    X = np.full((len(reads), len(sites) * emb.shape[1]), np.nan, np.float32)
    for i, r in enumerate(reads):
        for j, e in acc[r].items():
            X[i, j * emb.shape[1]:(j + 1) * emb.shape[1]] = e
    colm = np.nanmean(X, 0)
    X = np.where(np.isnan(X), np.where(np.isnan(colm), 0.0, colm), X)
    g = dict(zip(d['read_id'][k], map(str, d['group'][k])))
    y = np.array([g[r] for r in reads])
    Z = PCA(n_components=2, random_state=0).fit_transform(
        StandardScaler().fit_transform(X))
    return Z, y


def load_tables():
    """Per-window ARI and differential-CpG count for both datasets."""
    out = []
    for ph, tag in (('phase2', 'colo2'), ('phase3', 'colo3')):
        d = pd.read_csv(Path(f'{RAWMOD_WORK}/read_origin/runs') / f'{tag}_rawmod' / 'discovery_summary.tsv', sep='\t')
        t = pd.read_csv(WORK / f'windows_{ph}_truth.tsv', sep='\t')
        out.append(d.merge(t, left_on='group', right_on='name', how='left'))
    colo = pd.concat(out, ignore_index=True)[['n_diff', 'ari_vs_sample']].copy()

    # The second cell-line pair is optional: without its inputs the figure is drawn from COLO829 alone.
    tp, dp = HS / 'hs578_chr1_truth.tsv', Path(f'{RAWMOD_WORK}/read_origin/runs') / 'hs_rawmod' / 'discovery_summary.tsv'
    if not (tp.exists() and dp.exists()):
        return colo, None
    hs = pd.read_csv(dp, sep='\t').merge(pd.read_csv(tp, sep='\t'), left_on='group', right_on='name')[
        ['n_diff', 'ari_vs_sample', 'group']].copy()
    return colo, hs


def scatter_panel(ax, Z, y, key):
    """Reads in PC space, coloured by which cell line they came from.

    The dataset name rides on the legend title rather than a set_title, so the
    two arm names sit directly under the name of the pair they belong to.
    """
    for lab, cc, nm in ARMS[key]:
        m = y == lab
        ax.scatter(Z[m, 0], Z[m, 1], s=11, color=cc, lw=0.3, ec='black', label=nm)
    ax.set_xlabel('PC 1'); ax.set_ylabel('PC 2')
  # The grid follows the tick locator, and PC units are arbitrary, so an
  # automatic grid lands somewhere different in each panel. The tick labels
  # are hidden anyway, i.e. the grid here is purely a visual reference, so
  # pin it to the same fractions of each panel's own range: a and b then
  # carry an identical backdrop.
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xticks([x0 + f * (x1 - x0) for f in GRID_FRACS])
    ax.set_yticks([y0 + f * (y1 - y0) for f in GRID_FRACS])
    ax.grid(True, axis='both'); ax.tick_params(labelbottom=False, labelleft=False)
    leg = ax.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, 1.0),
                    borderaxespad=0.0, fontsize=FS - 0.5, handletextpad=0.1,
                    columnspacing=0.4, borderpad=0.2, handlelength=1.0,
                    title=D_LAB[key])
    leg.get_title().set_fontsize(FS + 0.5)
    leg.get_title().set_fontweight('bold')


def raise_for_ticklabels(fig, axes):
    """Lift these axes by one tick-label row.

    a and b hide their tick labels, so 'PC 1' sits directly under the spine,
    while c and d carry a row of numbers first and their xlabel ends up lower.
    All four share a gridspec row, so the only way to level the xlabels is to
    move c and d's bottom up by exactly the height that row costs.
    """
    fig.canvas.draw()
    pad = plt.rcParams['xtick.major.pad'] * fig.dpi / 72.0
    for ax in axes:
        h = max((t.get_window_extent().height for t in ax.get_xticklabels()
                 if t.get_text()), default=0.0)
        dy = (h + pad) / fig.bbox.height
        b = ax.get_position()
        ax.set_position([b.x0, b.y0 + dy, b.width, b.height - dy])


def panel_letters(fig, items):
    """Put every panel letter on ONE figure-level y, above the tallest legend.

    plabel() works in axes coordinates, so once c and d have been raised their
    axes are shorter and a shared axes-fraction y would no longer line up.
    """
    fig.canvas.draw()
    tops = [ax.get_legend().get_window_extent().y1 / fig.bbox.height
            for ax, _, _ in items if ax.get_legend() is not None]
    y_fig = max(tops) + 0.015
    for ax, letter, x in items:
        b = ax.get_position()
        plabel(ax, letter, x=x, y=(y_fig - b.y0) / b.height)


def main():
    colo, hs = load_tables()
  # AFTER load_tables(): it imports make_nci_slides, which sets its own
  # (slide-sized) rcParams at import time and would otherwise win.
    matplotlib.rcdefaults(); plt.rcParams.update(STYLE)

  # example windows: the clearest separation each dataset offers, so the
  # scatter shows what a solved window looks like (the rate is panels c/d)
    hs_best = hs.sort_values('ari_vs_sample', ascending=False).iloc[0]['group'] if hs is not None else None
    sets = [('colo', colo)] + ([('hs', hs)] if hs is not None else [])

    fig, axs = plt.subplots(1, 4, figsize=(TW, 1.72),
                            gridspec_kw=dict(width_ratios=[1, 1, 1.28, 1.52],
                                             wspace=0.30))
  # save() crops with bbox_inches='tight', so the PDF that reaches LaTeX is
  # the AXES extent, not figsize -- with matplotlib's default margins that is
  # only 0.775*TW, and \includegraphics[width=\linewidth] then magnifies the
  # whole figure (fonts included) by 1/0.775. Push the axes out to the figure
  # edges so the crop lands on TW and the figure is placed 1:1.
    fig.subplots_adjust(left=0.043, right=0.997, top=0.80, bottom=0.175)

    Zc, yc = read_scores(COLO_EMB, 'x27')
    scatter_panel(axs[0], Zc, yc, 'colo')

    if hs is not None and HS_EMB.exists():
        Zh, yh = read_scores(HS_EMB, str(hs_best))
        scatter_panel(axs[1], Zh, yh, 'hs')
    else:
        axs[1].axis('off')

  # c: dose-response, both datasets on one axis
    ax = axs[2]
    ax.grid(True, axis='both')
    for key, df in sets:
        ax.scatter(df.n_diff, df.ari_vs_sample, s=9, color=D_COL[key],
                   marker=D_MARK[key], lw=0.3, ec='black', label=D_LAB[key])
    ax.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, 1.0),
              borderaxespad=0.0, fontsize=FS - 0.5, handletextpad=0.1,
              columnspacing=0.4, borderpad=0.2, handlelength=1.0)
    ax.axhline(0, color='black', lw=0.8, ls=':')
    ax.set_xlabel('Differential CpGs'); ax.set_ylabel('ARI')
    ax.set_ylim(-0.1, 1.06); ax.set_yticks(np.arange(0, 1.01, 0.25))
    txt = []
    for key, df in sets:
        r, _ = spearmanr(df.n_diff, df.ari_vs_sample)
        txt.append((D_LAB[key], r, D_COL[key]))
    for i, (nm, r, cc) in enumerate(txt):
        ax.text(0.95, 0.30 - 0.165 * i, f'{nm}  ρ = {r:.2f}',
                transform=ax.transAxes, ha='right', fontsize=FS - 1.5,
                color='black', bbox=dict(fc='white', ec=cc, lw=0.8, pad=1.4))

  # d: median ARI by tier -- the rate, not the anecdote
    ax = axs[3]
    w = 0.38
    xs = np.arange(len(TIERS))
    for i, (key, df) in enumerate(sets):
        med = [df[(df.n_diff >= lo) & (df.n_diff < hi)].ari_vs_sample.median()
               for lo, hi, _ in TIERS]
        ax.bar(xs + (i - 0.5) * w, med, w, color=D_COL[key],
               label=D_LAB[key], **EDGE)
    ax.set_xticks(xs)
    ax.set_xticklabels([t[2] for t in TIERS], fontsize=FS - 1.5)
    ax.set_xlabel('Differential CpGs'); ax.set_ylabel('Median ARI')
    ax.set_ylim(0, 1.05); ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, 1.0),
              borderaxespad=0.0, fontsize=FS - 0.5, borderpad=0.2,
              handlelength=1.0, handletextpad=0.3, columnspacing=0.4)

    raise_for_ticklabels(fig, (axs[2], axs[3]))
    panel_letters(fig, [(axs[0], 'a', -0.24), (axs[1], 'b', -0.24),
                        (axs[2], 'c', -0.30), (axs[3], 'd', -0.26)])

    save(fig, 'fig_tumor_normal_joint.pdf')
    print('wrote fig_tumor_normal_joint.pdf')
    for key, df in sets:
        r, p = spearmanr(df.n_diff, df.ari_vs_sample)
        print(f'  {D_LAB[key]}: n={len(df)} rho={r:.3f} p={p:.3g} '
              f'median ARI={df.ari_vs_sample.median():.3f}')


if __name__ == '__main__':
    main()
