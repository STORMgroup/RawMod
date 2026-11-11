#!/usr/bin/env python3
"""Shared figure style of the paper: palette, font sizes, figure widths, panel labels and the save() helper. Every figure script imports this module."""
import os as _os
from pathlib import Path as _Path
REPO = str(_Path(__file__).resolve().parents[2])
RAWMOD_RESULTS = _os.environ.get('RAWMOD_RESULTS') or f'{REPO}/paper/data/runs'  # training and evaluation runs
RAWMOD_DATA = _os.environ.get('RAWMOD_DATA', 'data')  # datasets
RAWMOD_WORK = _os.environ.get('RAWMOD_WORK') or f'{REPO}/paper/data/work'  # outputs of the paper scripts

import sys, json, glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

MODDIR = Path(__file__).resolve().parent          # this module (the bundled fonts live here)
# Figures and tables are written here; the defaults need no configuration.
HERE = Path(_os.environ.get('RAWMOD_FIGS') or f'{REPO}/paper/figures')
TEX = HERE
TABLES = Path(_os.environ.get('RAWMOD_TABLES') or f'{REPO}/paper/tables')
# Inputs: the npz/json/tsv written by the eval_* and analysis scripts. RAWMOD_WORK defaults to
# the results shipped with the repository, so the figure scripts run without recomputing anything.
HERE.mkdir(parents=True, exist_ok=True)
FIGDATA = Path(f'{RAWMOD_WORK}/figures')
DINODATA = Path(f'{RAWMOD_WORK}/dinoflagellate/out')
RES = Path(f'{RAWMOD_RESULTS}')

# Drop Times New Roman .TTF files into common/fonts/ (or point RAWMOD_FONTS at a folder) to match the paper.
for _d in (MODDIR / 'fonts', _os.environ.get('RAWMOD_FONTS', '')):
    for f in glob.glob(str(Path(_d) / '*.[tT][tT][fF]')) if _d else []:
        fm.fontManager.addfont(f)
# The paper is set in Times New Roman. Where it is not installed, fall back to a
# metric-compatible serif rather than filling the log with missing-font warnings.
FONT = next((n for n in ('Times New Roman', 'Liberation Serif', 'Nimbus Roman', 'DejaVu Serif')
             if any(f.name == n for f in fm.fontManager.ttflist)), 'serif')

# Base font size at print size. Figures are drawn at the width of the one-column RECOMB
# line (TW) and printed at that width, so FS is the printed size; the smallest text used
# anywhere (FS - 1.5) is 8 pt.
FS = 9.5
# Plot style follows the RawHash2 paper figures (STORMgroup/RawHash2 test/figures/*/plot.py):
# boxed axes, inward ticks on all four sides, light horizontal grid, black-edged bars, and the
# RawHash2 colour palette.  Text is Times New Roman bold.
STYLE = {
    'figure.facecolor': 'white',
    'font.family': FONT, 'font.weight': 'bold',
    'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
    'font.size': FS, 'axes.titlesize': FS, 'axes.labelsize': FS,
    'xtick.labelsize': FS - 0.5, 'ytick.labelsize': FS - 0.5, 'legend.fontsize': FS - 1,
    'mathtext.fontset': 'custom', 'mathtext.rm': f'{FONT}:bold',
    'mathtext.it': f'{FONT}:bold:italic', 'mathtext.bf': f'{FONT}:bold',
    'mathtext.default': 'bf',
    'axes.linewidth': 0.8, 'axes.edgecolor': 'black',
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.minor.visible': False, 'ytick.minor.visible': False,
    'xtick.major.pad': 3, 'ytick.major.pad': 3, 'axes.labelpad': 2,
    'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': '#b0b0b0', 'grid.linestyle': '-',
    'grid.linewidth': 0.4, 'axes.axisbelow': True,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'legend.frameon': True, 'legend.framealpha': 1.0, 'legend.edgecolor': 'black', 'legend.fancybox': False,
    'legend.handlelength': 1.4, 'legend.handletextpad': 0.4, 'legend.columnspacing': 1.0,
    'legend.borderpad': 0.3,
    'axes.spines.top': True, 'axes.spines.right': True,
}
plt.rcParams.update(STYLE)
COL, TW = 3.45, 6.27  # half-width and full-width figures (inches); TW is the RECOMB text width
# RawHash2 palette: ["#6799ce", "#aa4499", "#db8043", "#f0c041", "#4b71bb", "#a3a3a3", "#5FA137"]
# Since RawHash2 hues with its green swapped for aqua so the set passes the dataviz colourblind
# validator (worst pair blue/purple deutan dE 7.0, legal with markers + legends; normal-vision floor 17.5).
# Two-group comparisons are BLUE vs PURPLE. Aqua and gold sit below 3:1 on white: keep black marker edges + legends.
BLUE, PURPLE, AQUA, GOLD, RED = '#4b71bb', '#aa4499', '#1baf7a', '#d4a017', '#c0392b'
GREEN, TEAL, YELLOW = AQUA, AQUA, GOLD
DBLUE, GREY, LIGHT = '#34508a', '#a3a3a3', '#a9c8e8'
EDGE = dict(edgecolor='black', linewidth=0.6)
CHEM_COL = {'5mC': BLUE, '5hmC': PURPLE, '6mA': AQUA, '4mC': GOLD, '5hmU': RED}
CHEMS = ['5mC', '5hmC', '6mA', '4mC', '5hmU']


def save(fig, name):
    for d in (HERE, TEX):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(HERE / name.replace('.pdf', '.png'), dpi=250, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print('wrote', name)


def plabel(ax, s, x=-0.18, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=FS + 3, fontweight='bold',
            va='bottom', ha='left')


def barlabels(ax, bars, vals, y, fmt='{:.2f}×', fs=FS - 1.5):
    """RawHash2-style value labels: white, bold, vertical, inside each bar, all centred on the same
    height y (data units) so they read as one row."""
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, y, fmt.format(v), rotation=90,
                ha='center', va='center', color='white', fontsize=fs, fontweight='bold')




def fig_overview():
    sc = np.load(FIGDATA / 'signal_contrast.npz', allow_pickle=True)  # from signal_contrast.py
    FH, NSHOW = 1.95, 8   # height (in); reads shown per pileup (of 15) so each row stays legible
    fig = plt.figure(figsize=(TW, FH))  # architecture is Fig. 2 (Architecture2.svg)
    TOP, BOT = 1 - 0.15 / FH, 0.42 / FH   # margins in inches: panel letters above, x labels below
    L, HALF = 10, 4
    cols = slice((10 - HALF) * L, (11 + HALF) * L)  # +-4 bases around candidate
    t = (np.arange((2 * HALF + 1) * L) + 0.5) / L - HALF - 0.5

  # ---- (a) one 5mC site: pileups of the modified and the unmodified library + read medians
    xm, xu = sc['5mC_ex_mod'], sc['5mC_ex_unmod']
    gs = fig.add_gridspec(3, 1, left=0.06, right=0.44, top=TOP, bottom=BOT, hspace=0.12,
                          height_ratios=[1, 1, 1.35])
    heat = []
    for i, (x, nm) in enumerate(((xm, 'Modified (5mC)'), (xu, 'Unmodified'))):
        ax = fig.add_subplot(gs[i]); ax.grid(False)
        valid = np.abs(x[1:]).sum(1) > 1e-6
        dlt = (x[1:][valid] - x[0])[:NSHOW, cols]
        im = ax.imshow(dlt, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2, interpolation='nearest',
                       extent=(-HALF - 0.5, HALF + 0.5, len(dlt) + 0.5, 0.5))
        ax.axvline(-0.5, color='k', lw=0.8); ax.axvline(0.5, color='k', lw=0.8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.tick_params(right=False, top=False, bottom=False)
        ax.set_ylabel('Reads', labelpad=0)
        ax.text(0.02, 0.93, nm, transform=ax.transAxes, ha='left', va='top', fontsize=FS - 1.5,
                bbox=dict(fc='white', ec='black', lw=0.5, pad=1.0))
        heat.append(ax)
    plabel(heat[0], 'a', x=-0.14, y=0.98)
    p0, p1 = heat[0].get_position(), heat[1].get_position()   # one colour bar centred on both pileups
    cax = fig.add_axes([0.448, p1.y0, 0.007, p0.y1 - p1.y0])
    cb = fig.colorbar(im, cax=cax); cb.outline.set_linewidth(0.6)
    cb.set_ticks([-2, 0, 2]); cb.ax.tick_params(length=2, width=0.6, pad=1, direction='in')
    cb.set_label('Δ current', labelpad=0, fontsize=FS - 1)
    ax = fig.add_subplot(gs[2])
    for x, lab, colr in ((xm, 'Modified', BLUE), (xu, 'Unmodified', GREY)):
        valid = np.abs(x[1:]).sum(1) > 1e-6
        d = (x[1:][valid] - x[0])[:, cols]
        q1, mu, q3 = np.percentile(d, [25, 50, 75], axis=0)
        ax.fill_between(t, q1, q3, color=colr, alpha=0.3, lw=0)
        ax.plot(t, mu, color='black' if colr == BLUE else '#606060', lw=0.4)
        ax.plot(t, mu, color=colr, lw=1.3, label=lab)
    ax.axvspan(-0.5, 0.5, color='k', alpha=0.08, lw=0)
    ax.set_xlim(-HALF - 0.5, HALF + 0.5); ax.set_xticks(range(-4, 5, 2))
    ax.set_xlabel('Base offset from 5mC')
    ax.set_ylabel('Δ current', labelpad=1)
    ax.set_ylim(-1.45, 1.95)
    ax.legend(loc='upper center', ncol=2, borderaxespad=0.2, handlelength=0.9, columnspacing=0.6,
              fontsize=FS - 1.5, borderpad=0.2)

  # ---- (b) all matched sites: excess signal difference per base offset, per modification
    ax = fig.add_axes([0.575, BOT, 0.415, TOP - BOT])
    plabel(ax, 'b', x=-0.175, y=0.98)   # left of the y label, which spans the short axis
    off = np.arange(-10, 11)
    for c in CHEMS:
        r = (sc[f'{c}_diff'] / sc[f'{c}_null']).reshape(21, 10).mean(1)
        ax.plot(off, r, '-o', ms=2.6, lw=1.2, color=CHEM_COL[c], mec='black', mew=0.3, label=c)
    ax.axhline(1, color='black', lw=0.8, ls='--')
    ax.axvline(0, color='black', lw=0.6, ls=':')
    ax.set_xlim(-10.5, 10.5); ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_ylim(0.8, 3.45); ax.set_yticks([1, 1.5, 2, 2.5, 3])
    ax.set_xlabel('Base offset from modified base')
    ax.set_ylabel('Difference (× noise)', labelpad=2)
    ax.legend(loc='upper right', ncol=2, borderaxespad=0.3, columnspacing=0.7, handlelength=1.0,
              labelspacing=0.2, fontsize=FS - 1.5)   # upper right is empty: the peaks sit at offsets -1 to +2

    save(fig, 'fig_overview.pdf')




# ============================================================== Supplementary Fig. S5 dinoflagellate
def fig_dino():
    c = np.load(DINODATA / 'dino_cache.npz'); rc = np.load(DINODATA / 'dino_repeat_cache.npz')
    st = json.load(open(DINODATA / 'dino_stats.json')); rs = json.load(open(DINODATA / 'dino_repeat_stats.json'))
    WIN = 10_000
    wm = c['wmean']; xs = (np.arange(len(wm)) + 0.5) * WIN / 1e6

    fig = plt.figure(figsize=(TW, 5.6))
    top = fig.add_gridspec(3, 1, left=0.085, right=0.99, top=0.94, bottom=0.56, hspace=0.10,
                           height_ratios=[2.3, 1.0, 0.9])
    axs = [fig.add_subplot(top[0])]
    axs += [fig.add_subplot(top[i], sharex=axs[0]) for i in (1, 2)]
    for a_ in axs:
        a_.grid(False)
    ax = axs[0]
    ax.fill_between(xs, np.nanmean(wm), wm, where=~np.isnan(wm), color=BLUE, lw=0, alpha=0.35)
    ax.plot(xs, wm, color=DBLUE, lw=0.8)
    ax.axhline(np.nanmean(wm), color='black', lw=0.8, ls='--')
    hot = wm >= np.nanpercentile(wm, 95)
    ax.scatter(xs[hot], np.full(hot.sum(), 0.84), marker='v', s=10, color=RED, lw=0)
    ax.set_ylim(0.05, 0.9); ax.set_ylabel('Modification score')
    plabel(ax, 'a', x=-0.09, y=1.0)
    handles = [Line2D([], [], marker='v', color=RED, lw=0, ms=4),
               Rectangle((0, 0), 1, 1, color='#A0A0A0'), Rectangle((0, 0), 1, 1, color=TEAL)]
    ax.legend(handles, ['Top 5% windows',
                        'Interspersed repeats', 'Young repeats (<10% div.)'],
              loc='lower right', ncol=3, bbox_to_anchor=(1.0, 1.02), borderaxespad=0.0, columnspacing=1.0)
    ax = axs[1]
    ax.fill_between(xs, 0, rc['wrep'][:len(xs)], color='#A0A0A0', lw=0)
    ax.fill_between(xs, 0, rc['wyoung'][:len(xs)], color=TEAL, lw=0)
    ax.set_ylabel('Repeats'); ax.set_ylim(0, 1); ax.set_yticks([0, 1])
    ax = axs[2]
    ax.fill_between(xs, 0, c['wcds'][:len(xs)], color='#404040', lw=0)
    ax.set_ylabel('CDS'); ax.set_ylim(0, 0.6); ax.set_yticks([0, 0.5])
    ax.set_xlabel('contig_3762 position (Mb)'); ax.set_xlim(0, 7.74)
    for a_ in axs[:2]:
        plt.setp(a_.get_xticklabels(), visible=False)
    for a_ in axs[1:]:
        a_.yaxis.set_label_coords(-0.05, 0.5)
    axs[0].yaxis.set_label_coords(-0.05, 0.5)

    bot = fig.add_gridspec(1, 4, left=0.07, right=0.99, top=0.40, bottom=0.13, wspace=0.48)
    CATS = ['intergenic', 'intron', 'UTR', 'CDS']
  # (b) annotation class relative to CDS
    ax = fig.add_subplot(bot[0]); plabel(ax, 'b', x=-0.36, y=1.03)
    raw = np.array([st['categories'][k]['fold_vs_cds'] for k in CATS])
    ci = np.array([st['bootstrap_fold_vs_cds_95ci'][k] for k in CATS])
    adj = np.array([st['context_adjusted']['5mer_x_gc'][k] for k in CATS]); adj = adj / adj[3]
    xx = np.arange(4)
    b1 = ax.bar(xx - 0.2, raw, 0.38, color=BLUE, label='Observed', **EDGE)
    ax.errorbar(xx[:3] - 0.2, raw[:3], yerr=[raw[:3] - ci[:3, 0], ci[:3, 1] - raw[:3]], fmt='none',
                ecolor='k', lw=0.8, capsize=2)
    b2 = ax.bar(xx + 0.2, adj, 0.38, color=PURPLE, label='Adjusted', **EDGE)
    ylab = 0.5 * min(raw.min(), adj.min()); barlabels(ax, b1, raw, ylab); barlabels(ax, b2, adj, ylab)
    ax.axhline(1, color='black', lw=0.8, ls=':')
    ax.set_xticks(xx); ax.set_xticklabels(['Inter.', 'Intron', 'UTR', 'CDS'], rotation=40, ha='right', rotation_mode='anchor'); ax.tick_params(axis='x', top=False)
    ax.set_ylim(0, 1.95); ax.set_yticks([0, 0.5, 1.0, 1.5]); ax.set_ylabel('Score relative to CDS')
    ax.text(0.0, max(ci[0, 1], adj[0]) + 0.03, f"z = {st['perm']['ig_cds']['z']:.1f}", ha='center', va='bottom',
            fontsize=FS - 1.5)
    ax.legend(loc='upper right', borderaxespad=0.3)
  # (c) repeat divergence
    ax = fig.add_subplot(bot[1]); plabel(ax, 'c', x=-0.36, y=1.03)
    bd = rs['by_divergence']
    lab = [f"{d['lo']}–{d['hi']}" if d['hi'] < 45 else f"≥{d['lo']}" for d in bd]
    ax.bar(range(len(bd)), [d['mean'] for d in bd], 0.7, color=GREEN, **EDGE)
    ax.axhline(rs['by_repeat_class']['unique']['mean'], color='k', lw=0.8, ls='--', label='Non-repeat mean')
    ax.set_xticks(range(len(bd))); ax.set_xticklabels(lab, rotation=40, ha='right', rotation_mode='anchor')
    ax.set_xlabel('Divergence (%)'); ax.set_ylabel('Modification score')
    ax.set_ylim(0.15, 0.395); ax.set_yticks([0.15, 0.20, 0.25, 0.30, 0.35]); ax.tick_params(axis='x', top=False)
    pr = rs['perm_rho_div']
    lg = ax.legend(loc='upper center', borderaxespad=0.3, handlelength=1.2, fontsize=FS - 1.5,
                   title=f"ρ = {pr['obs']:.2f}, z = {pr['z']:.1f}", title_fontsize=FS - 1.5)
    lg.get_title().set_fontweight('bold')
  # (d) intergenic gaps by array context
    ax = fig.add_subplot(bot[2]); plabel(ax, 'd', x=-0.36, y=1.03)
    g = pd.read_csv(DINODATA / 'dino_intergenic_gaps.tsv', sep='\t')
    w_ = g[g.kind == 'within']
    ax.grid(True, axis='both')
    ax.scatter(np.log10(w_.length), w_['mean'], s=9, color=GREY, lw=0.3, ec='black', label='Within array')
    for kind, colr, mk in (('divergent', PURPLE, 'D'), ('convergent', BLUE, 's')):
        s_ = g[g.kind == kind]
        ax.scatter(np.log10(s_.length), s_['mean'], s=26, color=colr, marker=mk, lw=0.6, ec='k',
                   label=kind.capitalize(), zorder=3)
    ax.set_xlabel('Gap length (log$_{10}$ bp)'); ax.set_ylabel('Modification score')
    ax.set_ylim(0.08, 0.66)
    ax.legend(loc='lower right', borderaxespad=0.3, handletextpad=0.1, labelspacing=0.25, fontsize=FS - 1.5)
  # (e) repeat x structure
    ax = fig.add_subplot(bot[3]); plabel(ax, 'e', x=-0.36, y=1.03)
    rx = rs['repeat_x_category']
    cats4 = ['intergenic', 'intron', 'UTR', 'CDS']
    xx = np.arange(4)
    vu = np.array([rx[f'{k}|unique']['mean'] for k in cats4]); vr = np.array([rx[f'{k}|repeat']['mean'] for k in cats4])
    ref = rx['CDS|unique']['mean']  # relative to non-repeat CDS
    b1 = ax.bar(xx - 0.2, vu, 0.38, color='#7f7f7f', label='Non-repeat', **EDGE)
    b2 = ax.bar(xx + 0.2, vr, 0.38, color=GREEN, label='Repeat', **EDGE)
    ylab = 0.5 * min(vu.min(), vr.min()); barlabels(ax, b1, vu / ref, ylab); barlabels(ax, b2, vr / ref, ylab)
    ax.set_xticks(xx); ax.set_xticklabels(['Inter.', 'Intron', 'UTR', 'CDS'], rotation=40, ha='right', rotation_mode='anchor'); ax.tick_params(axis='x', top=False)
    ax.set_ylim(0, 0.43); ax.set_yticks([0, 0.1, 0.2, 0.3]); ax.set_ylabel('Modification score')
    ax.legend(loc='upper right', borderaxespad=0.3)
    save(fig, 'fig_dino.pdf')


# ============================================================== Supplementary dino figure
def fig_supp_dino():
    c = np.load(DINODATA / 'dino_cache.npz'); rs = json.load(open(DINODATA / 'dino_repeat_stats.json'))
    al = np.load(DINODATA / 'dino_align_cache.npz')
    fig, axs = plt.subplots(1, 4, figsize=(TW, 2.7), gridspec_kw=dict(wspace=0.62))
    ax = axs[0]; plabel(ax, 'a', x=-0.34)
    m = c['meta']; x = np.arange(len(m))
    ax.axvspan(10, 50, color='#EBEBEB', lw=0)
    ax.plot(x, m, color=DBLUE, lw=1.3)
    ax.set_xticks([10, 50]); ax.set_xticklabels(['Start', 'End'])
    ax.set_ylabel('Modification score'); ax.set_xlabel('Gene body ± 5 kb')
    ax = axs[1]; plabel(ax, 'b', x=-0.34)
    bc = [d for d in rs['by_copies'] if d['n'] > 0]
    ax.bar(range(len(bc)), [d['mean'] for d in bc], 0.7, color=GREEN, **EDGE)
    ax.set_xticks(range(len(bc)))
    ax.set_xticklabels(['10$^1$', '10$^2$', '10$^3$', '10$^4$'])
    ax.set_xlabel('Copy number (≥)'); ax.set_ylabel('Modification score')
    ok = ~np.isnan(c['wmean']) & (al['depth'][:len(c['wmean'])] > 0)
    ax = axs[2]; plabel(ax, 'c', x=-0.34)
    ax.grid(True, axis='both')
    ax.scatter(al['err'][:len(ok)][ok] * 100, c['wmean'][ok], s=3, color=BLUE, lw=0)
    ax.set_xlabel('Align. error (%)'); ax.set_ylabel('10 kb score')
    ax.set_xlim(3, 9)
    ax = axs[3]; plabel(ax, 'd', x=-0.34)
    e = c['edges']; mid = (e[:-1] + e[1:]) / 2e3
    ax.fill_between(mid, c['null_lo'], c['null_hi'], color='#DDDDDD', lw=0, label='Random 95%')
    ax.plot(mid, c['prof_div'], color=PURPLE, lw=1.3, label='Divergent')
    ax.plot(mid, c['prof_conv'], color=BLUE, lw=1.3, label='Convergent')
    ax.axvline(0, color='k', lw=0.6, ls=':'); ax.grid(False)
    ax.set_xlabel('Distance (kb)'); ax.set_ylabel('Modification score')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), borderaxespad=0.1, handlelength=1.0,
              labelspacing=0.3, fontsize=FS - 1.5)
    save(fig, 'fig_supp_dino.pdf')


if __name__ == '__main__':
    todo = sys.argv[1:] or ['overview', 'dino', 'supp']
    for t in todo:
        {'overview': fig_overview, 'dino': fig_dino, 'supp': fig_supp_dino}[t]()
