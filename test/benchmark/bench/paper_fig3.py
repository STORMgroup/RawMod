#!/usr/bin/env python3
"""Main benchmark figure (Fig. 3): one small RawMod model against every tool, at the level of the tool.

  (a) one threshold across all rows. Each tool that covers more than one modification gives one score per site with no
      knowledge of the modification (the maximum over all of its models; RawMod its single score), and the sites of
      every row are pooled (at most 1,000 positives and 1,000 negatives per row, seed 0: exactly the pooled AUROC of
      tables/benchmark_agnostic.tex). The ROC curve asks whether one cut-off works for every modification at once.
  (b) size against accuracy on the modified datasets: each tool at the parameters of every model it needs to cover its
      modifications (bench.paper_table.PARAMS: learned weights of the modification model only, no basecaller, no k-mer
      table; MethyNano's species models count once, a run uses one), against the mean over the five modifications of
      its modification-agnostic score (maximum over all of its models; the rows of each modification averaged first),
      exactly the 'Mean over the five modifications' row of tables/benchmark_agnostic.tex.
  (c) library memorization: bench.library_control's library fingerprint, the AUROC between the native and the control
      library at unmodified background positions of each two-library row (same base, held-out region, far from every
      known modified site), with the models of the tool that claim the row's modification. 0.5 = the tool cannot tell
      the libraries apart where nothing is modified. Small dots: rows; large: mean over the rows the tool covers.
      Tools trained on these libraries (RawMod, UniMeth-FT, UniMeth-Agn) are grouped above the released ones.
  Comparative methods (Uncalled4, NovaMod) score a sample against its control library by design: (a) only.

Reads bench.paper_table's outputs in --dir (paper_tables.tsv, pooled_sites.tsv.gz) and
library_control's library_control.tsv; no rescoring.

  python -m bench.paper_fig3 --dir DIR [--name fig_benchmark_main]
"""
import argparse
from pathlib import Path

import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.transforms import blended_transform_factory
from sklearn.metrics import roc_auc_score, roc_curve

from . import common as C
from .paper_curves import STY, FIGDIR, style, sty, name as _name


def name(f):
    return _name(f).replace('†', '').replace('*', '')
from .paper_table import families, params, GROUPS, TOOLS, COMPARATIVE

SHORT = {'rawmod': 'RawMod', 'dorado_6mA': 'Dorado\n6mA', 'dorado_4mC_5mC': 'Dorado\n4mC+5mC',
         'dorado_5mC_5hmC': 'Dorado\n5mC+5hmC', 'dorado_5mCG_5hmCG': 'Dorado\n5mCG', 'unimeth_5mC': 'UniMeth\n5mC',
         'unimeth_6mA': 'UniMeth\n6mA', 'unimeth_ft_5hmC': 'FT†\n5hmC', 'unimeth_ft_4mC': 'FT†\n4mC',
         'unimeth_ft_5hmU': 'FT†\n5hmU', 'deepmod2': 'DeepMod2\n5mCG', 'rockfish': 'Rockfish\n5mCG',
         'methynano': 'MethyNano\n5mC', 'uncalled4': 'Uncalled4*', 'novamod_pooled': 'NovaMod-pool*',
         'rawmod_region': 'RawMod', 'unimeth_reg_5hmC': 'FT†\n5hmC', 'unimeth_reg_4mC': 'FT†\n4mC', 'unimeth_reg_5hmU': 'FT†\n5hmU'}
HALO = [pe.withStroke(linewidth=2.2, foreground='white')]  # labels stay legible over grid lines
ROC_TOOLS = ('dorado', 'unimeth', 'unimeth_ft', 'unimeth_reg', 'unimeth_agn_mod_reg', 'uncalled4')
AGN = 'unimeth_agn_mod_reg'  # fine-tuned by us as one agnostic model: hollow marker, dashed curve
OURS_DATA = ('unimeth_ft', 'unimeth_reg', AGN)  # besides RawMod, trained on the benchmark's training libraries
ONE_MODEL = {'methynano': 1}  # species models: a run uses one
# (b) label placement per tool: (dx, dy in points, ha); default right of the point
# label offsets in points (dx, dy, ha) for the narrow panel (b): each label sits in free space and inside the axes
CLAB = {'deepmod2': (0, 10, 'center'), 'unimeth_reg': (6, -10, 'right'), 'unimeth_ft': (6, -10, 'right'), 'unimeth': (-6, 0, 'right')}
CLAB = {**CLAB, AGN: (0, 11, 'center'), 'dorado': (-5, 0, 'right')}


def nparam(col):
    p = params(col)
    return float(p[:-1]) * 1e6 if p.endswith('M') else 0.0


INK, OFF, HILITE = '#1f1f1f', '#9a9a9a', '#e3ebf7'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default=str(C.WORK / f'paper_{C.PAPER_TAG}'))
    ap.add_argument('--name', default='fig_benchmark_main' + (f'_holdout-{C.HOLDOUT}' if C.HOLDOUT else ''))
    a = ap.parse_args()
    d = Path(a.dir)
    pooled = pd.read_csv(d / 'pooled_sites.tsv.gz', sep='\t')
    fam = families(C.subtools())

    style()
    W, H = 6.27, 2.75                         # the RECOMB text width; printed 1:1
    fig = plt.figure(figsize=(W, H))
  # fixed axes boxes in inches: (left, width). The gaps hold the y labels of (b) and the tool names of (c),
  # so no panel's labels reach into its neighbour; (b) is the narrowest because it holds only 8 points
    y0, y1 = 0.52, 2.40
    box = {'a': (0.44, 1.50), 'b': (2.46, 1.35), 'c': (4.72, 1.47)}
    bx, cx, lx = (fig.add_axes([l / W, y0 / H, w / W, (y1 - y0) / H]) for l, w in box.values())

  # (a) one threshold across all datasets: tools that cover more than one modification
    y = pooled['label'].to_numpy()
    for f in [t for t, _ in TOOLS if t.startswith('rawmod') or t in ROC_TOOLS]:
        if f not in pooled:
            continue
        s = pooled[f].to_numpy()
        fpr, tpr, _ = roc_curve(y, s)
        col, _, lw = sty(f)
        hot = f.startswith('rawmod')
        k = 1 if hot else len(fam[f])
        bx.plot(fpr, tpr, color=STY['rawmod'][0] if hot else col, ls='--' if f == AGN else '-', lw=1.9 if hot else 1.2,
                zorder=4 if hot else 3,
                label=f"{'RawMod' if hot else name(f)} ({k}): {roc_auc_score(y, s):.3f}")
    bx.plot([0, 1], [0, 1], color='#9a9a9a', lw=0.6, ls=':')
    bx.set_xlim(0, 1); bx.set_ylim(0, 1.0)
    bx.set_xlabel('False-positive rate'); bx.set_ylabel('True-positive rate', labelpad=1)
    bx.legend(loc='lower right', frameon=True, framealpha=0.95, edgecolor='#cccccc', fontsize=8, handlelength=1.4, borderpad=0.3, labelspacing=0.25)
    
  # (b) one point per tool (comparative methods excluded); x = parameters of all models the tool runs
    mark_of = {row: g.split(',')[0] for g, members in GROUPS for row, _ in members}
    t1 = pd.read_csv(d / 'paper_tables.tsv', sep='\t')
    t1['auroc'] = pd.to_numeric(t1['agnostic'], errors='coerce')  # the tool as a whole, as in benchmark_agnostic.tex
    acc = t1.assign(mark=t1['row'].map(mark_of)).groupby(['tool', 'mark'])['auroc'].mean().groupby('tool').mean()
    tools = [f for f, _ in TOOLS if f in acc.index and f not in COMPARATIVE]
    size = {f: sum(nparam(t) for t in fam[f][:ONE_MODEL.get(f, len(fam[f]))]) for f in tools}
    for f in tools:  # a RawMod column scored outside config/rawmod_models.tsv has no family entry: it is the one model
        if f.startswith('rawmod') and not size[f]:
            size[f] = 230697
    for f in tools:
        hot = f.startswith('rawmod')
        col = STY['rawmod'][0] if hot else sty(f)[0]
        if f == AGN:
            cx.scatter(size[f], acc[f], s=26, facecolor='white', edgecolor=col, lw=1.1, zorder=3)
        else:
            cx.scatter(size[f], acc[f], s=46 if hot else 24, color=col, edgecolor='white', lw=0.6, zorder=4 if hot else 3)
        k = len(fam[f][:ONE_MODEL.get(f, len(fam[f]))])
        lab = ('RawMod' if hot else name(f).replace('*', '')) + (f' ({k})' if k > 1 else '')
  # RawMod sits at the far left of the x axis and at nearly the same accuracy as UniMeth-Agn, whose
  # label runs leftwards: put RawMod's label below its point so the two never overlap
        dx, dy, ha = CLAB.get(f, (6, 0, 'left') if hot else (5, 0, 'left'))
        cx.annotate(lab, (size[f], acc[f]), xytext=(dx, dy), textcoords='offset points', ha=ha, va='center',
                    fontsize=8, color=col if hot else INK, path_effects=HALO)
    cx.set_xscale('log'); cx.set_xlim(4e4, 4e9)
    cx.set_xticks([1e5, 1e7, 1e9]); cx.xaxis.set_minor_locator(LogLocator(base=10, subs=(1.0,), numticks=12))
    cx.xaxis.set_minor_formatter(NullFormatter())
    cx.set_ylim(0.4, 1.06); cx.set_yticks(np.arange(0.4, 1.01, 0.1))  # headroom for the label above UniMeth-Agn
    cx.axhline(0.5, color='#9a9a9a', lw=0.6, ls=':', zorder=1)
    cx.set_xlabel('Parameters (all models)'); cx.set_ylabel('Mean AUROC (5 types)', labelpad=1)

  # (c) library memorization: native vs. control AUROC at unmodified positions, per row (small) and mean (large)
    lc = pd.read_csv(C.WORK / f'library_control_{C.PAPER_TAG}' / 'library_control.tsv', sep='\t')
    lc = lc[(lc['status'] == 'ok') & lc['fingerprint'].notna()]
    have = set(lc['tool'])
    order = ([f for f in tools if f.startswith('rawmod')] + [f for f in tools if f in OURS_DATA] +
             [f for f in tools if not f.startswith('rawmod') and f not in OURS_DATA])
    order = [f for f in order if f in have]
    ntrained = sum(f.startswith('rawmod') or f in OURS_DATA for f in order)
    rng = np.random.default_rng(0)
    print('library fingerprint (mean over rows, rows):')
    for i, f in enumerate(order):
        yy = len(order) - 1 - i
        hot = f.startswith('rawmod')
        col = STY['rawmod'][0] if hot else sty(f)[0]
  # library invariance: 1 - 2|AUROC - 0.5|, so 1 = the tool cannot tell the native library
  # from its control at unmodified positions and 0 = it separates them perfectly (either way round)
        v = 1.0 - 2.0 * np.abs(lc.loc[lc['tool'] == f, 'fingerprint'].to_numpy() - 0.5)
        lx.scatter(v, yy + rng.uniform(-0.18, 0.18, len(v)), s=7, color=col, alpha=0.55, lw=0, zorder=2)
        if f == AGN:
            lx.scatter(v.mean(), yy, s=40, facecolor='white', edgecolor=col, lw=1.3, zorder=4)
        else:
            lx.scatter(v.mean(), yy, s=48 if hot else 34, color=col, edgecolor='white', lw=0.6, zorder=4)
        print(f'  {f:22s} {v.mean():.3f}  {len(v)}')
    lx.set_yticks(range(len(order)))
    lx.set_yticklabels([('RawMod' if f.startswith('rawmod') else name(f).replace('*', '')) for f in order[::-1]],
                       fontsize=8)
    for t, f in zip(lx.get_yticklabels(), order[::-1]):
        if f.startswith('rawmod'):
            t.set_color(STY['rawmod'][0])
    lx.axhline(len(order) - ntrained - 0.5, color='#9a9a9a', lw=0.6)
    lx.text(0.02, len(order) + 0.75, 'Trained on\nbenchmark libraries', ha='left', va='top', linespacing=1.0, fontsize=8, color='#5a5a5a',
            transform=blended_transform_factory(lx.transAxes, lx.transData))
    lx.text(0.02, len(order) - ntrained - 0.6, 'Released models', ha='left', va='top', fontsize=8, color='#5a5a5a',
            transform=blended_transform_factory(lx.transAxes, lx.transData))
    lx.axvline(1.0, color='#9a9a9a', lw=0.6, ls=':', zorder=1)
    lx.set_xlim(0.0, 1.04); lx.set_ylim(-0.6, len(order) + 0.9)
    lx.grid(axis='y', visible=False); lx.tick_params(axis='y', length=0)
    lx.set_xlabel('Library invariance')
    print('tool-level (b) / parameters:')
    for f in tools:
        print(f"  {f:22s} {acc[f]:.3f}  {size[f] / 1e6:.2f}M")

  # panel titles start at the left edge of each panel's labels, not of its axes, so a long title never
  # runs over the next panel's y label
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
  # pull (c) left until its tool names sit a fixed small gap right of (b), and give it the freed width
    lab0 = min(t.get_window_extent(r).x0 for t in lx.get_yticklabels()) / fig.dpi   # inches
    shift = lab0 - (box['b'][0] + box['b'][1] + 0.12)
    pos = lx.get_position()
    lx.set_position([pos.x0 - shift / W, pos.y0, pos.width + shift / W, pos.height])
    fig.canvas.draw()
  # one baseline for the three x labels: (b)'s tick labels carry superscripts and push its label lowest,
  # so put every label where (b)'s sits (all axes share their bottom edge, so one axes-fraction y fits all)
    ylab = min(ax.xaxis.label.get_window_extent(r).y1 for ax in (bx, cx, lx))
    for ax in (bx, cx, lx):
        ax.xaxis.set_label_coords(0.5, ax.transAxes.inverted().transform((0, ylab))[1])
        ax.xaxis.label.set_va('top')
    fig.canvas.draw()
    for ax, t in ((bx, '(a) One threshold, all datasets'), (cx, '(b) Size against accuracy'), (lx, '(c) Library invariance')):
        x0 = ax.get_tightbbox(r).transformed(fig.transFigure.inverted()).x0
        fig.text(x0, (y1 + 0.10) / H, t, ha='left', va='bottom', fontweight='bold')

    out = d
    dirs = [out] + ([FIGDIR] if str(FIGDIR) not in ('', '.') and FIGDIR.is_dir() else [])
    for dd in dirs:
        fig.savefig(dd / f'{a.name}.pdf', bbox_inches='tight', pad_inches=0.02)
        fig.savefig(dd / f'{a.name}.png', dpi=250, bbox_inches='tight', pad_inches=0.02)
    print('wrote ' + ' and '.join(str(dd / f'{a.name}.pdf') for dd in dirs))


if __name__ == '__main__':
    main()
