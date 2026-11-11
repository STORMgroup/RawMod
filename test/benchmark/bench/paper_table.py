#!/usr/bin/env python3
"""Paper tables: one column per TOOL (not subtool), drop-in LaTeX for the RawMod paper.

  tables/benchmark_models.tex    Table 1: one column per MODEL (each Dorado / UniMeth variant separately) except MAIN_DROP;
  tables/benchmark_models_full.tex  the same with every column (supplement). Every model scored on
                                 every row on all of the row's sites (a model with no call at the row's base scores 0.5);
                                 bold = the model is trained for the row's modification. Footer: average over all rows,
                                 over the model's own rows, and rows >= 0.9.
  tables/benchmark.tex           matched evaluation. A tool is scored on a row only with the models it ships for
                                 that row's modification (config/claims.tsv); with several such models the site
                                 score is their maximum (e.g. Dorado 5mC = max of 5mCG_5hmCG, 5mC_5hmC, 4mC_5mC),
                                 a rule fixed in advance, not the best model picked after scoring. A CpG-only model
                                 on a row of mixed context is scored on that row's CpG sites only ($^{c}$). N/A = the
                                 tool has no model for the modification (or context). Bottom: modifications covered
                                 and models needed.
  tables/benchmark_agnostic.tex  modification-agnostic evaluation. Every tool gives ONE score per site, "is anything
                                 modified here", with no knowledge of which mark the row holds: the maximum over all
                                 of its models (RawMod: its single score). Every row is scored for every tool on all
                                 of the row's sites. Summary rows: mean per-row AUROC, the same over rows whose
                                 sequence-only null is < 0.75, and a pooled AUROC with ONE threshold across all rows
                                 (each row subsampled to <= 1,000 positives and <= 1,000 negatives, seed 0; each
                                 modification type weighted equally).

Rows whose reference-11-mer null (bench.nulls) is >= 0.9 are marked $^{\\ddagger}$: sequence alone separates them.
'--' = output not in yet. Cells are coloured by AUROC (red = N/A or <= 0.5, green = 1.0); the best number per row is\nbold. Matched table ends with each tool's average AUROC over the rows it scores, and how many rows that is.

  python -m bench.paper_table [--out DIR]       (default $BENCH_WORK/paper; also writes paper_tables.tsv)
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
from sklearn.metrics import roc_auc_score

from . import common as C
from . import score as Sc
from . import upright

# RawMod columns: BENCH_RAWMOD=name[,name,...] picks entries of config/rawmod_models.tsv (bench.models); default the
# paper's model 'rawmod'. Each is its own column (subtool 'rawmod' or 'rawmod_<name>'), never a max over checkpoints.
# BENCH_HOLDOUT=region (the region-holdout benchmark): the RawMod column ('rawmod') is the model trained without the held-out
# regions, the UniMeth-FT columns are the fine-tunes trained without them (unimeth_reg_*), and the E. coli and
# A. thaliana, H. pylori, Anabaena and SPO1 rows use the reads of the held-out region (rows *_reg, *_chr4); every row is scored inside its region only.
REGION = C.HOLDOUT == 'region'
RAWMOD_COLS = [n.strip() for n in os.environ.get('BENCH_RAWMOD', 'rawmod').split(',') if n.strip()]
FT = 'unimeth_reg' if REGION else 'unimeth_ft'


def _rawmod_cols():
    from .models import subtool_name
    esc = lambda n: n.replace('_', r'\_')
    single = len(RAWMOD_COLS) == 1
    return [(subtool_name(n), r'\sysname' if n == 'rawmod' or single else r'\sysname{} (' + esc(n) + ')') for n in RAWMOD_COLS]


# UniMeth-Agn: one modification-agnostic UniMeth fine-tuned by us on RawMod's training images
# without the held-out regions (bench.unimeth_agn_labels --exclude-region); region benchmark only (no clean positions run)
AGN = [('unimeth_agn_mod_reg', r'UniMeth-Agn$^{\dagger}$')] if REGION else []
TOOLS = _rawmod_cols() + [('dorado', 'Dorado'), ('unimeth', 'UniMeth'), (FT, r'UniMeth-FT$^{\dagger}$'), ('deepmod2', 'DeepMod2'),
         ('rockfish', 'Rockfish'), ('methynano', 'MethyNano')] + AGN + [('uncalled4', r'Uncalled4$^{*}$'),
         ('novamod', r'NovaMod$^{*}$')]
# per-model table: one column per model; a family name means "max over that family's models" (MethyNano species)
MODELS = _rawmod_cols() + [('dorado_6mA', r'\shortstack{Dorado\\6mA}'), ('dorado_4mC_5mC', r'\shortstack{Dorado\\4mC+5mC}'),
          ('dorado_5mC_5hmC', r'\shortstack{Dorado\\5mC+5hmC}'), ('dorado_5mCG_5hmCG', r'\shortstack{Dorado\\5mCG+5hmCG}'),
          ('unimeth_5mC', r'\shortstack{UniMeth\\5mC}'), ('unimeth_6mA', r'\shortstack{UniMeth\\6mA}'),
          (f'{FT}_5hmC', r'\shortstack{UniMeth-FT$^{\dagger}$\\5hmC}'), (f'{FT}_4mC', r'\shortstack{UniMeth-FT$^{\dagger}$\\4mC}'),
          (f'{FT}_5hmU', r'\shortstack{UniMeth-FT$^{\dagger}$\\5hmU}'),
          ('deepmod2', r'\shortstack{DeepMod2\\5mCG}'), ('rockfish', r'\shortstack{Rockfish\\5mCG}'),
          ('methynano', r'\shortstack{MethyNano$^{\S}$\\5mC}')] + [(f, rf'\shortstack{{{n}\\all}}') for f, n in AGN] + [('uncalled4', r'\shortstack{Uncalled4$^{*}$\\vs.\ control}'),
          ('novamod', r'\shortstack{NovaMod$^{*}$\\own control}')]
# comparative methods (sample vs. an unmodified control, no model of the modification): the last columns of every
# table, under their own column-group heading
COMPARATIVE = ('uncalled4', 'novamod')
# columns that need the row's unmodified control library: blank ('') on one-sample rows, and their summaries are over
# the rows they score (all five modifications have such a row)
NEEDS_CONTROL = ('novamod',)
# columns left out of Table 1 but kept in benchmark_models_full.tex
MAIN_DROP = ('deepmod2', 'methynano')
# parameters of each model column: learned network weights of the modification model, counted from
# the checkpoints we ran (floating-point tensors of the state dict). Every tool starts from the same basecall, so no
# basecaller is counted (Dorado's mod model is a standalone conv_lstm_v2 on signal + called sequence; the sup basecaller
# it is run with has 78.7M), and k-mer tables of expected current are not counted (Dorado's refine_kmer_levels, RawMod's
# level table). RawMod: 230,697 trainable parameters incl. its SupCon projection and Deep SAD head (every released
# checkpoint; the saved state also holds normalization statistics, which are not parameters). Dorado mod
# models 1,802,682-1,802,939. UniMeth(-FT): 100.3M (all variants within 2k). DeepMod2 bilstm R10.4.1_5kHz_v5.0
# 1,236,225; Rockfish rf_5kHz 34,731,269; MethyNano 5,974,401 (each species model); NovaMod model_v1_cnn_t 4,299,051
# (own-control and pooled alike). Uncalled4 has no learned model (a 9-mer pore model): '--'.
PARAMS = {'rawmod': '0.23M', 'dorado': '1.80M', 'unimeth': '100.3M', 'unimeth_reg': '100.3M', 'unimeth_ft': '100.3M',
          'deepmod2': '1.24M', 'rockfish': '34.7M', 'methynano': '6.0M', 'uncalled4': '--', 'novamod': '4.3M',
          'unimeth_agn_mod_reg': '100.3M'}


def params(col):
    """Parameter count of a table column (subtool or family), by its longest matching PARAMS prefix."""
    keys = [k for k in PARAMS if col == k or col.startswith(k + '_')]
    return PARAMS[max(keys, key=len)] if keys else '--'


def param_cell(txt, allvals):
    """Parameters row cell, coloured on a log scale from green (fewest parameters in the table) to red (most), with the
    smallest count in bold; '--' stays white."""
    import math
    vals = [float(v[:-1]) for v in allvals if v != '--']
    if txt == '--' or not vals:
        return txt
    lo, hi = math.log10(min(vals)), math.log10(max(vals))
    t = 1.0 - (math.log10(float(txt[:-1])) - lo) / (hi - lo) if hi > lo else 1.0
    col = (rf'\cellcolor{{yellow!{round(200 * t)}!red!35}}' if t < 0.5
           else rf'\cellcolor{{green!{round(200 * t - 100)}!yellow!35}}')
    return col + (rf'\textbf{{{txt}}}' if float(txt[:-1]) == min(vals) else txt)


def colgroups(cols):
    """Header row splitting the columns into modification callers and comparative methods (empty if none is comparative)."""
    n = len(cols); k = sum(f in COMPARATIVE for f, _ in cols)
    assert all(f in COMPARATIVE for f, _ in cols[n - k:]), 'comparative columns must come last'
    if not k:
        return []
    return [rf'& \multicolumn{{{n - k}}}{{c}}{{\textit{{Modification callers}}}} & '
            rf'\multicolumn{{{k}}}{{c}}{{\textit{{Comparative}}}} \\',
            rf'\cmidrule(lr){{2-{n - k + 1}}}\cmidrule(lr){{{n - k + 2}-{n + 1}}}']


# (group heading, [(row, label)]); a two-sample row's label names its unmodified control library ('vs.')
GROUPS = [
    ('5mC, CpG', [('arabidopsis_cpg', r'\textit{A.~thaliana}'), ('hg002_cpg', 'HG002'), ('rice_cpg', r'\textit{O.~sativa}'),
                  ('mouse_cpg', 'Mouse brain'), ('ecoli_mssi_5mC_ko', r'\textit{E.~coli} M.SssI vs.\ $\Delta$\textit{dam}$\Delta$\textit{dcm}')]),
    ('5mC, non-CpG', [('ecoli_dcm_5mC_ko', r'\textit{E.~coli} Dcm vs.\ $\Delta$\textit{dam}$\Delta$\textit{dcm}'),
                      ('arabidopsis_noncpg', r'\textit{A.~thaliana}')]),
    ('5mC, all contexts', [('syn_5mC', r'ONT Benchmark -- 5mC vs.\ control')]),
    ('5hmC', [('syn_5hmC', r'ONT Benchmark -- 5hmC vs.\ control')]),
    ('6mA', [('ecoli_dam_6mA_ko', r'\textit{E.~coli} Dam vs.\ $\Delta$\textit{dam}$\Delta$\textit{dcm}'),
             ('anabaena_6mA', r'\textit{Anabaena}'), ('hp26695_6mA', r'\textit{H.~pylori} 26695 vs.\ WGA (GCATG)'),
             ('syn_6mA', r'ONT Benchmark -- 6mA vs.\ control')]),
    ('4mC', [('hp26695_4mC', r'\textit{H.~pylori} 26695 vs.\ WGA (TCTTC)')]),
    ('5hmU', [('spo1_5hmU', r'\phage native vs.\ PCR')]),
]
if REGION:  # the rows of the held-out regions replace the E. coli 0-1 Mb and A. thaliana chromosome 1 rows
    _SWAP = {'ecoli_mssi_5mC_ko': 'ecoli_mssi_5mC_reg', 'ecoli_dcm_5mC_ko': 'ecoli_dcm_5mC_reg',
             'ecoli_dam_6mA_ko': 'ecoli_dam_6mA_reg', 'arabidopsis_cpg': 'arabidopsis_cpg_chr4',
             'arabidopsis_noncpg': 'arabidopsis_noncpg_chr4', 'anabaena_6mA': 'anabaena_6mA_reg',
             'hp26695_6mA': 'hp26695_6mA_reg', 'hp26695_4mC': 'hp26695_4mC_reg', 'spo1_5hmU': 'spo1_5hmU_reg'}
    GROUPS = [(g, [(_SWAP.get(r, r), lab) for r, lab in m]) for g, m in GROUPS]
# rows whose dataset was never in any RawMod training set (marked in the tables)
NEVER_TRAINED = {'rice_cpg', 'mouse_cpg'}


def claims():
    cl = {c[0]: {'marks': set(c[1].split(',')), 'context': c[2]} for c in C._rows(C.CONFIG / 'claims.tsv')}
    from . import models
    for st, m in models.load().items():  # registered RawMod checkpoints: marks from the registry
        cl.setdefault(st, {'marks': set(m['marks'].split(',')), 'context': 'any'})
    return cl


def families(T):
    """Column -> its subtools. A RawMod column is exactly one checkpoint; any other tool is every subtool of its family."""
    return {f: [f] if T.get(f, {}).get('family') == 'rawmod' else [t for t in T if T[t]['family'] == f] for f, _ in TOOLS}


def row_marks(r):
    return {'6mA', '4mC', '5mC'} if r['chem'] == 'mixed' else {r['chem']}


def is_cpg(blocks, S):
    """Per block: bool array, the site is the C of a CpG on its own strand."""
    out = []
    for smp, sites, _ in blocks:
        fa, m = pysam.FastaFile(S[smp]['ref']), np.zeros(len(sites), bool)
        for i, (c, p, s) in enumerate(zip(sites['contig'], sites['pos'], sites['strand'])):
            p = int(p)
            m[i] = (fa.fetch(c, p, p + 2).upper() == 'CG') if s == '+' else (fa.fetch(c, max(p - 1, 0), p + 1).upper() == 'CG')
        out.append(m)
    return out


def family_score(blocks, subtools):
    """Site score = max mean_P over `subtools`; None if any of them is not in yet for any block."""
    if any(not C.sites_path(smp, t).exists() for smp, _, _ in blocks for t in subtools):
        return None
    return np.concatenate([np.max([Sc.site_scores(smp, t, sites)[0] for t in subtools], axis=0) for smp, sites, _ in blocks])


def auroc(y, s):
    return roc_auc_score(y, s) if len(y) and 0 < y.sum() < len(y) else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(C.WORK / f'paper_{C.PAPER_TAG}'))
    a = ap.parse_args()
    out = Path(a.out); (out / 'tables').mkdir(parents=True, exist_ok=True)
    S, T, CL = C.samples(), C.subtools(), claims()
    rows = {r['row']: r for r in Sc.rows_cfg()}
    npath = C.WORK / f'nulls_{C.PAPER_TAG}.tsv'
    nulls = pd.read_csv(npath, sep='\t').set_index('row')['ref11'].to_dict() if npath.exists() else {}
    fam = families(T)
    cache, recs, pool, mrecs = {}, [], [], []

    def own(r, m):
        """Is model (or family) m trained for row r's modification and context?"""
        return any(row_marks(r) & CL[t]['marks'] and not (CL[t]['context'] == 'cpg' and r['context'] == 'noncpg')
                   for t in (fam.get(m) or [m]))
    rng = np.random.default_rng(0)
    for _, members in GROUPS:
        for row, _ in members:
            r = rows[row]
            need = [r['pos_sample']] + ([r['neg_sample']] if r['neg_sample'] else [])
            if not all(C.readsel_path(x).exists() for x in need):
                for f, _ in TOOLS:
                    use = [t for t in fam[f] if row_marks(r) & CL[t]['marks']
                           and not (CL[t]['context'] == 'cpg' and r['context'] == 'noncpg')]
                    recs.append({'row': row, 'tool': f, 'matched': '--' if use else 'N/A', 'agnostic': '--',
                                 'null_ref11': nulls.get(row, np.nan)})
                print(f'[{row}] read selection not built yet: all cells pending')
                for m, _ in MODELS:
                    mrecs.append({'row': row, 'model': m, 'auroc': '--', 'own': own(r, m)})
                continue
            blocks = Sc.row_sites(r, S, cache)
            y = np.concatenate([b[2] for b in blocks])
            cpg = None
            take = np.concatenate([rng.permutation(np.flatnonzero(y == k))[:1000] for k in (0, 1)])
            pool.append((row, y[take], take))
            for m, _ in MODELS:
                if m in NEEDS_CONTROL and r['type'] == 'one':
                    mrecs.append({'row': row, 'model': m, 'auroc': '', 'own': own(r, m)}); continue
                sc = family_score(blocks, fam.get(m) or [m])
                mrecs.append({'row': row, 'model': m, 'auroc': '--' if sc is None else auroc(y, sc), 'own': own(r, m)})
            for f, _ in TOOLS:
  # matched
                use = [t for t in fam[f] if row_marks(r) & CL[t]['marks']
                       and not (CL[t]['context'] == 'cpg' and r['context'] == 'noncpg')]
                rec = {'row': row, 'tool': f, 'matched_models': ','.join(use), 'null_ref11': nulls.get(row, np.nan)}
                if f in NEEDS_CONTROL and r['type'] == 'one':
                    recs.append({**rec, 'matched': '', 'agnostic': ''}); continue
                if not use:
                    rec['matched'] = 'N/A'
                else:
                    sc = family_score(blocks, use)
                    if sc is None:
                        rec['matched'] = '--'
                    else:
                        keep = np.ones(len(y), bool)
                        if all(CL[t]['context'] == 'cpg' for t in use) and r['context'] == 'any':
                            cpg = np.concatenate(is_cpg(blocks, S)) if cpg is None else cpg
                            keep, rec['cpg_only'] = cpg, True
                        v = auroc(y[keep], sc[keep])
                        rec['matched'] = 'N/A' if np.isnan(v) else v
                        rec['n_matched'] = int(keep.sum())
  # agnostic: all of the tool's models, all sites
                sc = family_score(blocks, fam[f])
                rec['agnostic'] = '--' if sc is None else auroc(y, sc)
                if sc is not None:
                    rec['_pool'] = sc[take]
                recs.append(rec)
            print(f'[{row}] ' + '  '.join(f"{x['tool']}={x['matched'] if isinstance(x['matched'], str) else round(x['matched'], 3)}"
                                         f"/{x['agnostic'] if isinstance(x['agnostic'], str) else round(x['agnostic'], 3)}"
                                         for x in recs if x['row'] == row))
    df = pd.DataFrame(recs)

    def heat(v):
        """Cell colour: red = N/A or chance (<= 0.5), through yellow, to green = 1.0; '--' (pending) stays white."""
        if isinstance(v, str):
            return r'\cellcolor{red!25}' if v == 'N/A' else ''
        t = min(max((v - 0.5) / 0.5, 0.0), 1.0)  # 0.5 -> red, 0.75 -> yellow, 1.0 -> green
        if t < 0.5:
            return rf'\cellcolor{{yellow!{round(200 * t)}!red!35}}'
        return rf'\cellcolor{{green!{round(200 * t - 100)}!yellow!35}}'

    def cell(v, best, suffix=''):
        txt = fmt(v) + suffix
        if best is not None and not isinstance(v, str) and v >= best - 5e-4:
            txt = rf'\textbf{{{txt}}}'
        return heat(v) + txt

    def fmt(v):
        return v if isinstance(v, str) else f'{v:.3f}'

    def label(row, lab):
        return (lab + (r'$^{\ddagger}$' if nulls.get(row, 0) >= 0.9 else '')
                + (r'$^{\circ}$' if row in NEVER_TRAINED else ''))

    def body(col, mark_cpg):
        L = []
        for g, members in GROUPS:
            L.append(rf'\multicolumn{{{len(TOOLS) + 1}}}{{@{{}}l}}{{\textbf{{{g}}}}} \\')
            for row, lab in members:
                x = df[df['row'] == row].set_index('tool')
                vals = [x.loc[f, col] for f, _ in TOOLS]
                num = [v for v in vals if not isinstance(v, str)]
                best = max(num) if num else None
                cells = [cell(x.loc[f, col], best, r'$^{c}$' if mark_cpg and x.loc[f].get('cpg_only') is True else '')
                         for f, _ in TOOLS]
                L.append(rf'\hspace{{3pt}}{label(row, lab)} & ' + ' & '.join(cells) + r' \\')
        return L

    head = [r'\setlength{\tabcolsep}{4pt}%', r'\begin{tabular}{@{}l ' + 'c' * len(TOOLS) + '@{}}', r'\toprule', *colgroups(TOOLS),
            r'\textbf{Dataset} & ' + ' & '.join(rf'\textbf{{{n}}}' for _, n in TOOLS) + r' \\', r'\midrule']
  # matched table + coverage footer
    allmarks = ['5mC', '5hmC', '6mA', '4mC', '5hmU']
    cov = {f: sorted({m for t in fam[f] for m in CL[t]['marks']}, key=allmarks.index) for f, _ in TOOLS}
    nrows = sum(len(m) for _, m in GROUPS)
    avg = []
    for f, _ in TOOLS:
        v = [x for x in df[df['tool'] == f]['matched'] if not isinstance(x, str)]
        avg.append(float(np.mean(v)) if v else '--')
    num = [v for v in avg if not isinstance(v, str)]
    foot = [r'\midrule',
            r'\textbf{Average AUROC} & ' + ' & '.join(cell(v, max(num) if num else None) for v in avg) + r' \\',
            r'Rows scored & ' + ' & '.join(f"{sum(1 for x in df[df['tool'] == f]['matched'] if not isinstance(x, str))}/{nrows}"
                                          for f, _ in TOOLS) + r' \\',
            r'Modifications & ' + ' & '.join(f'{len(cov[f])}/5' for f, _ in TOOLS) + r' \\',
            r'Models & ' + ' & '.join(str(len(fam[f])) for f, _ in TOOLS) + r' \\']
    (out / 'tables' / 'benchmark.tex').write_text('\n'.join(head + body('matched', True) + foot + [r'\bottomrule', r'\end{tabular}']) + '\n')

  # agnostic table + summary rows
    clean = {row for row in df['row'] if nulls.get(row, 1) < 0.75}
    summ = []
    for name, sel in (('Mean AUROC, all rows', None), ('Mean AUROC, null $<$ 0.75', clean)):
        cells = []
        for f, _ in TOOLS:
            v = df[(df['tool'] == f) & (df['row'].isin(sel) if sel else True)]['agnostic']
            cells.append('' if f in NEEDS_CONTROL else '--' if any(isinstance(x, str) for x in v) else np.mean(v.astype(float)))
        summ.append((name, cells))
  # per-modification summaries (5mC has 8 of the 15 rows): mean, harmonic mean and minimum of the five
  # per-modification means of each tool's agnostic score
    mark_of = {row: g.split(',')[0] for g, members in GROUPS for row, _ in members}
    pm = {}
    for f, _ in TOOLS:
        x = df[df['tool'] == f]
        x = x[x['agnostic'] != '']
        pm[f] = (None if any(isinstance(a, str) for a in x['agnostic'])
                 else x.assign(mark=x['row'].map(mark_of), a=x['agnostic'].astype(float))
                       .groupby('mark', sort=False)['a'].mean().to_numpy())
    for name, fn in (('Mean over the five modifications', np.mean),
                     ('Harmonic mean over the five modifications', lambda p: len(p) / np.sum(1.0 / p)),
                     ('Worst modification', np.min)):
        summ.append((name, ['--' if pm[f] is None else float(fn(pm[f])) for f, _ in TOOLS]))
    cells = []
    pooled = {'row': np.concatenate([[p[0]] * len(p[1]) for p in pool]) if pool else [],
              'label': np.concatenate([p[1] for p in pool]) if pool else []}
  # each modification type carries equal weight (split equally between its modified and unmodified sites), so the
  # 5mC rows (8 of 15) do not dominate; bench.paper_fig3 weights its ROC curves the same way
    if pool:
        y_pool = np.concatenate([p[1] for p in pool])
        m_pool = np.concatenate([[mark_of[p[0]]] * len(p[1]) for p in pool])
        w_pool = np.zeros(len(y_pool))
        for m in np.unique(m_pool):
            for k in (0, 1):
                sel = (m_pool == m) & (y_pool == k)
                w_pool[sel] = 1.0 / max(int(sel.sum()), 1)
    for f, _ in TOOLS:
        x = df[df['tool'] == f]
        if f in NEEDS_CONTROL:
            cells.append(''); continue
        if x['_pool'].isna().any() if '_pool' in x else True:
            cells.append('--'); continue
        yy = np.concatenate([p[1] for p in pool]); ss = np.concatenate(x['_pool'].tolist())
        cells.append(roc_auc_score(yy, ss, sample_weight=w_pool))
        pooled[f] = ss
    summ.append(('Pooled AUROC, one threshold, types weighted equally', cells))
  # the pooled sites behind that row (bench.paper_fig3 draws their ROC curves)
    pd.DataFrame(pooled).to_csv(out / 'pooled_sites.tsv.gz', sep='\t', index=False, float_format='%.8g')
    sfoot = [r'\midrule']
    for name, cells in summ:
        num = [v for v in cells if not isinstance(v, str)]
        best = max(num) if num else None
        sfoot.append(name + ' & ' + ' & '.join(cell(v, best) for v in cells) + r' \\')
    (out / 'tables' / 'benchmark_agnostic.tex').write_text('\n'.join(head + body('agnostic', False) + sfoot + [r'\bottomrule', r'\end{tabular}']) + '\n')

    mdf = pd.DataFrame(mrecs)
    def write_models(MODELS, fname, row_mean=True):
        """One per-model table over the columns MODELS (Table 1 drops MAIN_DROP; the supplement has every column)."""
        mhead = [r'\setlength{\tabcolsep}{3pt}%', r'\begin{tabular}{@{}l ' + 'c' * len(MODELS) + '@{}}', r'\toprule', *colgroups(MODELS),
                 r'\textbf{Dataset} & ' + ' & '.join(rf'\textbf{{{n}}}' for _, n in MODELS) + r' \\', r'\midrule']
        L = []
        for g, members in GROUPS:
            L.append(rf'\multicolumn{{{len(MODELS) + 1}}}{{@{{}}l}}{{\textbf{{{g}}}}} \\')
            for row, lab in members:
                x = mdf[mdf['row'] == row].set_index('model')
                cells = []
                for m, _ in MODELS:
                    v, o = x.loc[m, 'auroc'], bool(x.loc[m, 'own'])
                    cells.append(heat(v) + (rf'\textbf{{{fmt(v)}}}' if o else fmt(v)))
                L.append(rf'\hspace{{3pt}}{label(row, lab)} & ' + ' & '.join(cells) + r' \\')
        def mmean(sel):
            out = []
            for m, _ in MODELS:
                v = mdf[(mdf['model'] == m) & sel(mdf)]['auroc']
                if m in NEEDS_CONTROL:
                    out.append(''); continue
                out.append('--' if len(v) == 0 or any(isinstance(a, str) for a in v) else float(np.mean(v.astype(float))))
            return out
        allr = mmean(lambda d: d['row'] == d['row'])
        n90 = []
        for m, _ in MODELS:
            v = mdf[mdf['model'] == m]['auroc']
            v = v[v != '']
            n90.append('--' if any(isinstance(a, str) for a in v) else f"{sum(float(a) >= 0.9 for a in v)}/{len(v)}")
  # per-modification summaries: 5mC has 8 of the 15 rows, so the row average is weighted towards 5mC.
  # Average the rows of each modification first, then summarise the five per-modification means.
        mark_of = {row: g.split(',')[0] for g, members in GROUPS for row, _ in members}
        per_mark = []
        for m, _ in MODELS:
            v = mdf[mdf['model'] == m]
            v = v[v['auroc'] != '']
            if any(isinstance(a, str) for a in v['auroc']):
                per_mark.append(None); continue
            v = v.assign(mark=v['row'].map(mark_of), a=v['auroc'].astype(float))
            per_mark.append(v.groupby('mark', sort=False)['a'].mean().to_numpy())
        macro = [('--' if p is None else float(np.mean(p))) for p in per_mark]
        worst = [('--' if p is None else float(np.min(p))) for p in per_mark]

        def brow(name, vals):
            num = [v for v in vals if not isinstance(v, str)]
            best = max(num) if num else None
            return name + ' & ' + ' & '.join(heat(v) + (rf'\textbf{{{fmt(v)}}}' if best is not None and not isinstance(v, str)
                                                        and v >= best - 5e-4 else fmt(v)) for v in vals) + r' \\'
  # three summary rows: per-modification mean first, then the row mean, then the worst modification;
  # own-row means, rows >= 0.9 (Fig. 3a) and the harmonic mean are reported in the text / benchmark_summary.tsv only
        mfoot = [r'\midrule',
                 brow(r'\textbf{Mean over the five modifications}', macro)] + (
  # the 15-row mean is weighted towards 5mC (8 of the 15 rows), so the main table reports the
  # per-modification mean and the worst modification only; the row mean stays in the full table
  # and in the agnostic table.
                 [brow(r'Mean over the 15 rows', allr)] if row_mean else []) + [
                 brow(r'Worst modification', worst),
                 r'\midrule', r'Parameters & ' + ' & '.join(param_cell(params(m), [params(x) for x, _ in MODELS]) for m, _ in MODELS) + r' \\']
        (out / 'tables' / fname).write_text('\n'.join(mhead + L + mfoot + [r'\bottomrule', r'\end{tabular}']) + '\n')
    write_models(MODELS, 'benchmark_models_full.tex')
    write_models([c for c in MODELS if c[0] not in MAIN_DROP], 'benchmark_models.tex', row_mean=False)
    # upright layout for the supplement: abbreviated headers; the full model table in two panels
    for name in ('benchmark', 'benchmark_agnostic', 'benchmark_models_full'):
        upright.fit(out / 'tables' / f'{name}.tex')
    # Table 1 of the preprint: every model except MethyNano on all 15 rows, at the page width
    from . import paper_table_all
    paper_table_all.main(str(out / 'tables' / 'benchmark_models_full.tex'), str(out / 'tables' / 'benchmark_models_all.tex'))
    mdf.to_csv(out / 'paper_models.tsv', sep='\t', index=False, float_format='%.4f')
    df.drop(columns=['_pool'], errors='ignore').to_csv(out / 'paper_tables.tsv', sep='\t', index=False, float_format='%.4f')
    for name, cells in summ:
        print(f'{name:32s} ' + '  '.join(f'{f}={fmt(v)}' for (f, _), v in zip(TOOLS, cells)))
    print(f"wrote {out / 'tables' / 'benchmark_models.tex'}\nwrote {out / 'tables' / 'benchmark.tex'}\nwrote {out / 'tables' / 'benchmark_agnostic.tex'}\nwrote {out / 'paper_tables.tsv'}")


if __name__ == '__main__':
    main()
