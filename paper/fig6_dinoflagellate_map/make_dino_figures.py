#!/usr/bin/env python3
"""Supplementary Figures S5 and S6 (the dinoflagellate map) from the outputs of the analysis scripts."""
import os as _os
from pathlib import Path as _Path
REPO = str(_Path(__file__).resolve().parents[2])
RAWMOD_RESULTS = _os.environ.get('RAWMOD_RESULTS') or f'{REPO}/paper/data/runs'  # training runs; shipped metrics by default
RAWMOD_DATA = _os.environ.get('RAWMOD_DATA', 'data')  # datasets (not shipped: set this to score raw signal)
RAWMOD_WORK = _os.environ.get('RAWMOD_WORK') or f'{REPO}/paper/data/work'  # outputs of the paper scripts; shipped by default

import numpy as np, matplotlib.pyplot as plt, matplotlib.ticker as mt
import sys; sys.path.insert(0, f'{REPO}/paper/common'); import make_paper_figures as P
YLIM = {'fig_dino.pdf': {0: (0.34, 1.00), 4: (0.5, 0.92), 5: (0.35, 1.0), 6: (0, 1.0)}}

def panel_b_observed_only(ax):
    """Redraw panel b without the context-adjusted bars: under a base-masked model the
    adjustment leaves IG/CDS at 1.10-1.14x (observed 1.11x), so the second bar only
    repeats the first. The control stays in the text and Supplementary Table S7."""
    import json
    st = json.load(open(P.DINODATA / 'dino_stats.json'))
    cats = ['intergenic', 'intron', 'UTR', 'CDS']
    raw = np.array([st['categories'][k]['fold_vs_cds'] for k in cats])
    ci = np.array([st['bootstrap_fold_vs_cds_95ci'][k] for k in cats])
    ax.cla(); ax.grid(True, axis='y'); ax.set_axisbelow(True)
    P.plabel(ax, 'b', x=-0.36, y=1.03)
    xx = np.arange(4)
    b1 = ax.bar(xx, raw, 0.6, color=P.BLUE, **P.EDGE)
    ax.errorbar(xx[:3], raw[:3], yerr=[raw[:3] - ci[:3, 0], ci[:3, 1] - raw[:3]], fmt='none',
                ecolor='k', lw=0.8, capsize=2)
    P.barlabels(ax, b1, raw, 0.5 * raw.min())
    ax.axhline(1, color='black', lw=0.8, ls=':')
    ax.set_xticks(xx); ax.set_xticklabels(['Inter.', 'Intron', 'UTR', 'CDS'], rotation=40, ha='right', rotation_mode='anchor'); ax.tick_params(axis='x', top=False)
    ax.set_ylim(0, 1.45); ax.set_yticks([0, 0.5, 1.0]); ax.set_ylabel('Score relative to CDS')
    ax.text(0.0, ci[0, 1] + 0.03, f"z = {st['perm']['ig_cds']['z']:.1f}", ha='center', va='bottom',
            fontsize=P.FS - 1.5)
    ax.text(1.0, ci[1, 1] + 0.03, f"z = {st['perm']['intron_cds']['z']:.1f}", ha='center', va='bottom',
            fontsize=P.FS - 1.5)


def save(fig, name):
    if name == 'fig_dino.pdf':
        panel_b_observed_only(fig.axes[3])
    for i, lim in YLIM.get(name, {}).items():
        ax = fig.axes[i]; ax.set_ylim(*lim); ax.yaxis.set_major_locator(mt.MaxNLocator(5))
    out = P.HERE / name
    fig.savefig(out, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(str(out).replace('.pdf', '.png'), dpi=250, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig); print('wrote', out)

P.save = save
P.fig_dino(); P.fig_supp_dino()
