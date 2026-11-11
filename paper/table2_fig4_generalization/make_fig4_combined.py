#!/usr/bin/env python3
"""Figure 4 of the preprint: generalization (a, b) and few-shot detection (c, d) in one figure.

(a) zero-shot AUROC of each withheld modification over three seeds, (b) accuracy against reads per
pileup, (c, d) 5hmU and 6mA AUROC against the number of labeled sites, for retraining and fine-tuning.
Reads the same shipped metrics as make_region_figures.py and uses its drawing helpers, so the numbers
match Supplementary Table S1 and Supplementary Figure S1.

    python paper/table2_fig4_generalization/make_fig4_combined.py [OUT.pdf]

OUT defaults to $RAWMOD_FIGS/fig_generalization_fewshot.pdf. Set RAWMOD_FONTS to a folder of Times New
Roman .ttf files to match the paper's typeface.
"""
import os, sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'common'))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import make_paper_figures as P
import make_region_figures as R

PREPRINT_TW = 8.5 - 2 * 0.675          # two-column text width of the preprint (in)
CHEMS = P.CHEMS


def panel_zero_shot(ax):
    x = np.arange(len(CHEMS) + 1); w = 0.6
    st = [R.seed_stats(f'loco_{c}') for c in CHEMS]
    v = np.array([s[0] for s in st] + [np.mean([s[0] for s in st])])
    lo = np.array([s[1] for s in st] + [np.nan]); hi = np.array([s[2] for s in st] + [np.nan])
    ax.bar(x, v, w, color=P.BLUE, **P.EDGE)
    ax.errorbar(x[:-1], v[:-1], yerr=[v[:-1] - lo[:-1], hi[:-1] - v[:-1]], fmt='none', ecolor='black', elinewidth=0.8, capsize=2)
    for i, val in enumerate(v):
        top = hi[i] if np.isfinite(hi[i]) else val
        ax.text(x[i], top + 0.015, f'{val:.3f}', ha='center', va='bottom', fontsize=P.FS - 1.5)
    for i in range(1, len(x)):
        ax.axvline(i - 0.5, color='#a0a0a0', lw=0.8, ls='--', zorder=0)
    mixed = R.seed_stats('mixed')[0]
    ax.axhline(mixed, color='k', lw=0.8, ls='--'); ax.axhline(0.5, color='k', lw=0.8, ls=':')
    ax.set_xticks(x); ax.set_xticklabels(CHEMS + ['Mean']); ax.tick_params(axis='x', top=False, bottom=False)
    ax.set_xlim(-0.5, len(x) - 0.5); ax.set_ylim(0.4, 1.12); ax.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax.set_ylabel('Zero-shot AUROC'); ax.set_xlabel('Held-out modification')
    ax.text(len(x) - 0.55, mixed + 0.012, f'In-distribution ({mixed:.3f})', ha='right', va='bottom', fontsize=P.FS - 1.5)
    print('zero-shot', dict(zip(CHEMS + ['Mean'], v.round(3))), 'mixed', round(mixed, 3))


def panel_depth(ax):
    d = pd.DataFrame(json.load(open(P.FIGDATA / 'read_depth_results_region.json')))
    g = d.groupby('k').mean(numeric_only=True)
    ax.grid(True, axis='both')
    for c in CHEMS:
        if f'auroc_{c}' in g:
            ax.plot(g.index, g[f'auroc_{c}'], '-o', ms=2.8, lw=1.2, color=P.CHEM_COL[c], mec='black', mew=0.3, label=c)
    ax.plot(g.index, g['auroc'], '-s', ms=3, lw=1.5, color='black', label='All')
    ymin = np.floor(min(g[[k for k in g if k.startswith('auroc')]].min()) * 20) / 20 - 0.01
    ax.set_xticks([1, 3, 6, 10, 15]); ax.set_ylim(ymin, 1.01); ax.set_xlim(0.3, 15.7)
    ax.set_xlabel('Reads per pileup'); ax.set_ylabel('AUROC')
    ax.legend(ncol=2, loc='lower right', borderaxespad=0.3, columnspacing=0.6, handlelength=1.2, labelspacing=0.15)


X0 = 3.0
xs = lambda s: np.where(s == 0, X0, s)


def panel_fewshot(ax, chem):
    rows, r = R.panel_a(chem)
    d = pd.DataFrame(rows)
    for ser, lab, col, mk in (('A_pool', 'Added to training pool', P.BLUE, 's'), ('A_finetune', 'Fine-tuned from LOCO model', P.PURPLE, 'o')):
        s = d[d.series == ser].sort_values('pos_sites')
        x = xs(s.pos_sites.values)
        ax.fill_between(x, s.auroc_min, s.auroc_max, color=col, alpha=0.25, lw=0)
        ax.plot(x, s.auroc_mean, '-' + mk, ms=3, lw=1.3, color=col, mec='black', mew=0.3, label=lab)
    if np.isfinite(r):
        ax.axhline(r, color='k', lw=0.8, ls='--')
        ax.text(X0 / 1.3, r - 0.008, f'Supervised ({r:.3f})', ha='left', va='top', fontsize=P.FS - 1.5)
    ax.set_xscale('log'); ax.minorticks_off()
    ax.set_xticks([X0, 10, 100, 1000, 10000]); ax.set_xticklabels(['0', '10', '100', '1k', '10k'])
    ax.set_xlim(X0 / 1.5, 25000); ax.set_ylim(0.3 if (d.auroc_min < 0.4).any() else 0.4, 1.02)
    ax.axhline(0.5, color='k', lw=0.8, ls=':', label='Chance'); ax.grid(True, axis='both')
    ax.set_xlabel(f'Labeled {chem} sites'); ax.set_ylabel(f'{chem} AUROC')
    print(chem, 'supervised ref', round(r, 3), d.groupby('series').auroc_mean.max().round(3).to_dict())


def main(out):
    fig = plt.figure(figsize=(PREPRINT_TW, 4.9))
    top = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], left=0.07, right=0.995, top=0.95, bottom=0.585, wspace=0.22)
    bot = fig.add_gridspec(1, 2, left=0.07, right=0.995, top=0.445, bottom=0.085, wspace=0.22)
    a = fig.add_subplot(top[0]); b = fig.add_subplot(top[1]); c = fig.add_subplot(bot[0]); dd = fig.add_subplot(bot[1])
    panel_zero_shot(a); panel_depth(b)
    panel_fewshot(c, '5hmU')
    panel_fewshot(dd, '6mA')
    # one legend for both few-shot panels, in the empty lower-right corner of (d)
    dd.legend(loc='lower right', borderaxespad=0.3, handlelength=1.6, labelspacing=0.15)
    # panel letters at one fixed figure-x per column, so a/c and b/d line up
    for ax, s in ((a, 'a'), (b, 'b'), (c, 'c'), (dd, 'd')):
        bb = ax.get_position()
        fig.text(bb.x0 - 0.062, bb.y1 + 0.012, s, fontsize=P.FS + 3, fontweight='bold', va='bottom', ha='left')
    fig.savefig(out, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(str(out).replace('.pdf', '.png'), dpi=200, bbox_inches='tight', pad_inches=0.02)
    print('wrote', out)


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.environ.get('RAWMOD_FIGS', str(HERE.parent / 'figures')), 'fig_generalization_fewshot.pdf')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    main(out)
