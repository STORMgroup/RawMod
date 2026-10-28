#!/usr/bin/env python3
"""Supplementary leave-out table: RawMod in distribution (the released mixed model, as in tables/benchmark_models.tex)
against RawMod with the row's organism or library left out of training (checkpoints/rawmod_leaveout/, registered in
config/rawmod_models.tsv as logo_<group>). Same sites as the main table (BENCH_HOLDOUT=positions: positions the released
model never trained on). The best model trained for the row's modification is shown for reference (paper_models.tsv,
own == True).

  rawmod_bench.py leaveout --holdout positions      (or: BENCH_HOLDOUT=positions python -m bench.paper_leaveout --dir DIR)
writes <dir>/tables/benchmark_leaveout.tex and <dir>/leaveout.tsv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import common as C
from . import score as Sc
from .paper_table import GROUPS, COMPARATIVE

# row prefix -> (leave-out checkpoint, what it withholds); rows with no entry were never in training (mixed is already leave-out)
import os as _os
_LP = _os.environ.get('BENCH_LEAVEOUT_PREFIX', 'logo')
LEAVE = {'ecoli': (_LP + '_logo_ecoli', r'\textit{E.~coli} (all samples)'),
         'anabaena': (_LP + '_logo_anabaena', r'\textit{Anabaena}'),
         'hp26695': (_LP + '_logo_hp26695', r'\textit{H.~pylori} 26695 (native, WGA)'),
         'arabidopsis': (_LP + '_logo_plant', r'\textit{A.~thaliana}'),
         'hg002': (_LP + '_logo_mammal', 'HG001, HG002'),
         'syn_5mC': (_LP + '_logo_ont5mC', '5mC oligonucleotides, control'),
         'syn_5hmC': (_LP + '_logo_ont5hmC', '5hmC oligonucleotides, control'),
         'syn_6mA': (_LP + '_logo_ont6mA', '6mA oligonucleotides, control'),
         'spo1': (_LP + '_logo_spo1', r'\phage (all barcodes)')}
from .paper_table import RAWMOD_COLS
from .models import subtool_name
BASE = subtool_name(RAWMOD_COLS[0])  # the in-distribution column: the model the main table scores (BENCH_RAWMOD; default rawmod)
ONLY_SOURCE = {'hp26695_4mC', 'syn_5hmC', 'spo1_5hmU', 'hp26695_4mC_reg', 'spo1_5hmU_reg'}  # withheld data = the only training source of the modification


def site_scores(smp, t, sites):
    s = pd.read_csv(C.sites_path(smp, t), sep='\t', dtype={'contig': str})
    return sites[['contig', 'pos', 'strand']].merge(s, on=['contig', 'pos', 'strand'], how='left')['sum_p'].fillna(0).to_numpy()


def heat(v):
    t = min(max((v - 0.5) / 0.5, 0.0), 1.0)
    return rf'\cellcolor{{yellow!{round(200 * t)}!red!35}}' if t < 0.5 else rf'\cellcolor{{green!{round(200 * t - 100)}!yellow!35}}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default=str(C.WORK / f'paper_{C.PAPER_TAG}'))
    d = Path(ap.parse_args().dir)
    pm = pd.read_csv(d / 'paper_models.tsv', sep='\t')
    pm['auroc'] = pd.to_numeric(pm['auroc'], errors='coerce'); pm['own'] = pm['own'].astype(str) == 'True'
    nulls = pd.read_csv(C.WORK / f'nulls_{C.PAPER_TAG}.tsv', sep='\t').set_index('row')['ref11'].to_dict()
    S, rows = C.samples(), {r['row']: r for r in Sc.rows_cfg()}
    recs, body = [], []
    for g, members in GROUPS:
        body.append(rf'\multicolumn{{5}}{{@{{}}l}}{{\textbf{{{g}}}}} \\')
        for row, lab in members:
            blocks = Sc.row_sites(rows[row], S, {})
            y = np.concatenate([b[2] for b in blocks])
            key = next((k for k in LEAVE if row.startswith(k)), None)
            ck, held = LEAVE[key] if key else ('', 'none (never in training)')
            ind = roc_auc_score(y, np.concatenate([site_scores(s, BASE, st) for s, st, _ in blocks]))
            lo = roc_auc_score(y, np.concatenate([site_scores(s, 'rawmod_' + ck if ck else BASE, st) for s, st, _ in blocks]))
  # best model trained for the modification: no RawMod column and no comparative method (Uncalled4 / NovaMod score a
  # sample against its control library, they have no model of the modification; columns added )
            own = pm[(pm['row'] == row) & pm['own'] & ~pm['model'].str.startswith('rawmod') & ~pm['model'].isin(COMPARATIVE)]
            best = own.loc[own['auroc'].idxmax()] if len(own) else None
            recs.append({'row': row, 'held_out': held, 'checkpoint': ck or 'mixed', 'in_distribution': ind, 'leave_out': lo,
                         'best_specialist': best['model'] if best is not None else '', 'best_specialist_auroc': best['auroc'] if best is not None else np.nan})
            lab2 = lab + (r'$^{\ddagger}$' if nulls.get(row, 0) >= 0.9 else '') + (r'$^{\P}$' if row in ONLY_SOURCE else '')
            bs = f"{heat(best['auroc'])}{best['auroc']:.3f}" if best is not None else r'\cellcolor{red!25}--'
            body.append(rf'\hspace{{3pt}}{lab2} & {held} & {heat(ind)}{ind:.3f} & {heat(lo)}{lo:.3f} & {bs} \\')
            print(f'{row:<20} in-dist {ind:.3f}  leave-out {lo:.3f}  [{ck or "mixed"}]', flush=True)
    df = pd.DataFrame(recs)
    df.to_csv(d / 'leaveout.tsv', sep='\t', index=False, float_format='%.4f')
    mi, ml = df['in_distribution'].mean(), df['leave_out'].mean()
    n = len(df)
    foot = [r'\midrule',
            rf'\textbf{{Average AUROC}} & & {heat(mi)}{mi:.3f} & {heat(ml)}{ml:.3f} & \\',
            rf'Rows with AUROC $\ge$0.9 & & {int((df.in_distribution >= 0.9).sum())}/{n} & {int((df.leave_out >= 0.9).sum())}/{n} & \\']
    head = [r'\setlength{\tabcolsep}{4pt}%', r'\begin{tabular}{@{}l l ccc@{}}', r'\toprule',
            r'\textbf{Dataset} & \textbf{Withheld from training} & '
            + (r'\textbf{\shortstack{\sysname\\region held out}} & ' if BASE != 'rawmod' else r'\textbf{\shortstack{\sysname\\in distribution}} & ') +
            r'\textbf{\shortstack{\sysname\\leave-out}} & \textbf{\shortstack{Best model trained\\for the modification}} \\', r'\midrule']
    (d / 'tables').mkdir(exist_ok=True)
    (d / 'tables' / 'benchmark_leaveout.tex').write_text('\n'.join(head + body + foot + [r'\bottomrule', r'\end{tabular}']) + '\n')
    print(f'mean in-distribution {mi:.3f}  leave-out {ml:.3f}\nwrote {d / "tables" / "benchmark_leaveout.tex"}')


if __name__ == '__main__':
    main()
