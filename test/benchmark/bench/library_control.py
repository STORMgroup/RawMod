#!/usr/bin/env python3
"""Library control for the two-library rows (native vs. unmodified control at the same positions) .

A two-library row compares the same reference positions in two sequencing libraries, so a tool earns AUROC either by
detecting the modification or by telling the libraries apart. This measures and removes the second part.

Background positions of a row: positions of the row's base, on either strand, scorable (K selected reads) in BOTH
libraries, inside the row's held-out region (score._holdout, as the row's own sites), not a row site and more than
WIN bp from every known modified site of the organism (MOTIFS), or WIN_DESIGNED bp from every designed site of the
ONT oligos (they are dense: 6 bp would leave ~50 positions in the held-out oligos). Up to NBG
per row (seeded). SPO1 has none (every thymine of the native phage carries 5hmU): n/a.

  library fingerprint      AUROC(native background > control background), per tool. 0.5 = the tool cannot tell the
                           libraries apart where nothing is modified.
  baseline-corrected AUROC every site score is replaced by its mid-rank percentile among the SAME library's background
                           scores (per tool), then the row AUROC is recomputed. A library-wide shift cancels; the
                           modification signal does not. One-library rows are unchanged by construction.

Matched models only (the Table 1 cell rule of bench.paper_table); comparative methods (Uncalled4, NovaMod) are left
out: they score a sample against its control library by design. RawMod's background scores come from images of the
benchmark's selected reads at the background positions (`cands` writes the candidate BEDs; the featurization job writes
<sample>/<rawmod subtool>_lc/sites_<TAG>.tsv.gz = its row sites + background sites).

  BENCH_HOLDOUT=region python -m bench.library_control cands
  BENCH_HOLDOUT=region python -m bench.library_control rawmod   (GPU; the RawMod background scores)
  BENCH_HOLDOUT=region python -m bench.library_control table [--extra unimeth_agn_mod_reg]
  BENCH_HOLDOUT=region python -m bench.library_control within [--extra unimeth_agn_mod_reg]

  within-library AUROC     each library on its own: the row's sites against that library's background. native: the
                           modified sites (detection with no library cue); control: the SAME sites, unmodified there,
                           0.5 if the tool reads the modification, away from 0.5 if it scores the sequence context
                           (motif) itself. -> OUT/within_library.tsv
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
from sklearn.metrics import roc_auc_score

from . import common as C
from . import score as Sc
from .paper_table import GROUPS, TOOLS, COMPARATIVE, claims, families, row_marks, is_cpg
from . import paper_table as _PT

# bp kept from known motif sites (10 = half of RawMod's 21-base window, so no known modified base is inside any
# image) / from designed oligo sites (dense; 10 would leave none in the held-out oligos); positions per row at most
WIN, WIN_DESIGNED, NBG = 10, 3, 5000
OUT = C.WORK / f'library_control_{C.PAPER_TAG}'
# known modified sites per native library (motif, modified offset); both strands. H. pylori 26695: REBASE motifs with
# a characterised MTase (the row motifs among them); E. coli: Dam, Dcm and, for the M.SssI library, CpG.
MOTIFS = {'hp26695': [('GCATG', 2), ('TCTTC', 1), ('GATC', 1), ('GANTC', 1), ('CATG', 1), ('GAAGA', 4), ('GTAC', 2),
                      ('CCTC', 0), ('GCGC', 1), ('GAGG', 2), ('TCGA', 3), ('ACGT', 0), ('CCGG', 0), ('GGCC', 0),
                      ('TGCA', 3), ('CTRYAG', 4)],
          'ecoli': [('GATC', 1), ('CCWGG', 1)], 'ecoli_mssi': [('CG', 0)]}
IU = {'W': '[AT]', 'N': '[ACGT]', 'R': '[AG]', 'Y': '[CT]', 'S': '[CG]'}
COMP = str.maketrans('ACGTRYWSN', 'TGCAYRWSN')


def two_library_rows():
    rows = {r['row']: r for r in Sc.rows_cfg()}
    return [rows[row] for _, mem in GROUPS for row, _ in mem if rows[row]['type'] == 'two']


def _motif_mask(seq, motifs):
    """bool array over the contig: within WIN bp of a known modified site (either strand)."""
    m = np.zeros(len(seq), bool)
    rx = lambda s: ''.join(IU.get(ch, ch) for ch in s)
    for mot, off in motifs:
        hits = [x.start() + off for x in re.finditer(f'(?={rx(mot)})', seq)]
        rc = mot[::-1].translate(COMP)
        hits += [x.start() + len(mot) - 1 - off for x in re.finditer(f'(?={rx(rc)})', seq)]
        for p in hits:
            m[max(0, p - WIN):p + WIN + 1] = True
    return m


def _known(r):
    if r['row'].startswith('hp26695'):
        return MOTIFS['hp26695']
    if r['row'].startswith('ecoli_mssi'):
        return MOTIFS['ecoli_mssi']
    if r['row'].startswith('ecoli'):
        return MOTIFS['ecoli']
    return []


def background(r, S, cache):
    """-> blocks [(native sample, bg df, ones), (control sample, bg df, zeros)] at the SAME positions, or None."""
    if r['row'].startswith('spo1'):
        return None
    R = C.WORK / 'rows' / r['row']
    gt = Sc.read_bed(R / 'gt.bed')
    sc = lambda s: cache.setdefault(s, Sc.Scorable(s))
    a, b = sc(r['pos_sample']), sc(r['neg_sample'])
    fa = pysam.FastaFile(S[r['pos_sample']]['ref'])
    designed = set(zip(gt['contig'], gt['pos']))
    rng = np.random.default_rng(0)
    parts = []
    for (ctg, st), (s0, e0) in a.iv.items():
        if (ctg, st) not in b.iv:
            continue
        seq = fa.fetch(ctg).upper()
        cov = np.zeros(len(seq) + 1, np.int8)
        for iv in (a.iv[(ctg, st)], b.iv[(ctg, st)]):
            d = np.zeros(len(seq) + 1, np.int32)
            np.add.at(d, np.clip(iv[0], 0, len(seq)), 1); np.add.at(d, np.clip(iv[1], 0, len(seq)), -1)
            cov += (np.cumsum(d) > 0).astype(np.int8)
        pos = np.flatnonzero(cov[:len(seq)] == 2)
        want = r['base'] if st == '+' else r['base'].translate(COMP)
        arr = np.frombuffer(seq.encode(), dtype='S1')
        pos = pos[arr[pos] == want.encode()]
        mot = _known(r)
        if mot:
            pos = pos[~_motif_mask(seq, mot)[pos]]
        if designed:  # ONT oligos and row positives: keep WIN bp away
            dz = np.zeros(len(seq), bool)
            for c, p in designed:
                if c == ctg:
                    dz[max(0, p - WIN_DESIGNED):p + WIN_DESIGNED + 1] = True
            pos = pos[~dz[pos]]
        parts.append(pd.DataFrame({'contig': ctg, 'pos': pos.astype(np.int64), 'strand': st}))
    bg = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=['contig', 'pos', 'strand'])
    blocks = Sc._holdout(r, [(r['pos_sample'], bg, np.ones(len(bg), int)), (r['neg_sample'], bg, np.zeros(len(bg), int))])
    keep = blocks[0][1].merge(blocks[1][1], on=['contig', 'pos', 'strand'])
    if len(keep) > NBG:
        keep = keep.iloc[np.sort(rng.choice(len(keep), NBG, replace=False))].reset_index(drop=True)
    if not len(keep):
        return None
    return [(r['pos_sample'], keep, np.ones(len(keep), int)), (r['neg_sample'], keep.copy(), np.zeros(len(keep), int))]


def cands():
    """Background positions of every two-library row -> per-sample candidate BEDs for RawMod's featurization."""
    S, cache = C.samples(), {}
    OUT.mkdir(parents=True, exist_ok=True)
    per = {}
    for r in two_library_rows():
        bl = background(r, S, cache)
        if bl is None:
            print(f"[{r['row']}] no background (n/a)"); continue
        bl[0][1].to_csv(OUT / f"bg_{r['row']}.tsv", sep='\t', index=False)
        print(f"[{r['row']}] {len(bl[0][1]):,} background positions")
        for smp, df, _ in bl:
            per.setdefault(smp, []).append(df)
    for smp, dfs in per.items():
        d = pd.concat(dfs).drop_duplicates()
        od = OUT / 'rawmod_cands' / smp
        od.mkdir(parents=True, exist_ok=True)
        for st, nm in (('+', 'plus'), ('-', 'minus')):
            x = d[d.strand == st].sort_values(['contig', 'pos'])
            x.assign(end=x.pos + 1)[['contig', 'pos', 'end']].to_csv(od / f'cand_{nm}.bed', sep='\t', header=False, index=False)
        print(f'{smp}: {len(d):,} RawMod background candidates -> {od}')


def rawmod_lc(names):
    """RawMod at the background positions: for every registered
    checkpoint in `names` (config/rawmod_models.tsv) and every sample with background candidates, pileup images of the
    selected reads at the background positions (<sample>/<subtool>_lc/features_<strand>.h5; reused from another checkpoint
    with the same reads per image, as in bench.steps.rawmod), scored -> <sample>/<subtool>_lc/sites_<TAG>.tsv.gz = the row
    sites of the checkpoint's own site file + the background sites."""
    import os, subprocess
    from . import models, settings as ST
    from .steps import py, REPO
    S, reg = C.samples(), models.load()
    for name in names:
        st = models.subtool_name(name)
        m = reg[st]
        nread = models.reads_per_image(m['checkpoint'])
        for d in sorted((OUT / 'rawmod_cands').iterdir()):
            smp = d.name
            out = C.WORK / smp / f'{st}_lc'; out.mkdir(parents=True, exist_ok=True)
            if C.sites_path(smp, f'{st}_lc').exists():
                print(f'[skip] {smp} {st}_lc: {C.sites_path(smp, f"{st}_lc")} exists'); continue
            ref = C.WORK / smp / 'refine' / C.INPUTS
            parts = []
            for s, sgn, flag in (('plus', '+', ['--strand', '+']), ('minus', '-', ['--strand', '-', '--orient', 'read'])):
                bed = d / f'cand_{s}.bed'
                if not bed.stat().st_size:
                    continue
                h5 = out / f'features_{s}.h5'
                if not h5.exists() and nread == 15:  # images of another checkpoint at the same positions and reads
                    for other in (C.WORK / smp).glob('rawmod*_lc'):
                        if other != out and (other / f'features_{s}.h5').exists():
                            h5.symlink_to(other / f'features_{s}.h5'); print(f'reusing {other / f"features_{s}.h5"}'); break
                if not h5.exists():
                    subprocess.run([py(), str(REPO / 'rawmod/featurization.py'), '--pod5', str(S[smp]['sub_pod5']), '--bam',
                                    str(ref / 'reads_refined.bam'), '--peaks', str(ref / 'peaks_refined.tsv'), '--output', f'{h5}.tmp',
                                    '--level-table', str(ST.level_table()), '--gt', '--candidate-bed', str(bed), '--read-select',
                                    str(C.readsel_path(smp)), '--min-reads', '1', '--max-reads', str(nread), '--max-images-per-base', '1',
                                    '--min-mapq', '0', '--normalize', '--half-window', '10', '--L', '10', *flag], check=True)
                    Path(f'{h5}.tmp').rename(h5)
                sc = out / f'{smp}_{s}_scores.tsv.gz'
                if not sc.exists():
                    subprocess.run([py(), str(REPO / 'scripts/evaluate/score_genome.py'), '--h5', str(h5), '--dataset', f'{smp}_{s}',
                                    '--checkpoint', str(m['checkpoint']), '--out-dir', str(out)] + (['--legacy-ch9'] if m['legacy_ch9'] else []),
                                   check=True, env={**os.environ, 'PILEUP_MASK_BASES': '1' if m['mask_bases'] else '0'})
                x = pd.read_csv(sc, sep='\t', dtype={'contig': str})
                x = x.groupby(['contig', 'pos']).agg(score=('score', 'max'), n_reads=('n_reads', 'max')).reset_index()
                parts.append(pd.DataFrame({'contig': x.contig, 'pos': x.pos.astype('int64'), 'strand': sgn, 'n_called': x.n_reads,
                                           'sum_p': x.score * C.K, 'n_mod': (x.score >= 0.5) * C.K}))
            bg = pd.concat(parts, ignore_index=True)
            rows = pd.read_csv(C.sites_path(smp, st), sep='\t', dtype={'contig': str})
            mm = pd.concat([rows, bg], ignore_index=True).drop_duplicates(['contig', 'pos', 'strand'], keep='first')
            mm.to_csv(C.sites_path(smp, f'{st}_lc'), sep='\t', index=False, float_format='%.6g')
            print(f'{smp}: {len(bg):,} background + {len(rows):,} row sites -> {C.sites_path(smp, f"{st}_lc")}', flush=True)


def _pct(bg, x):
    """mid-rank percentile of x among bg (both 1-d)."""
    bg = np.sort(bg)
    return (np.searchsorted(bg, x, 'left') + np.searchsorted(bg, x, 'right')) / (2.0 * len(bg))


def _score(blocks, use, lc=False):
    """per-block site scores (max over `use`); lc: RawMod from its library-control site file (row + background)."""
    if lc:
        use = [u + '_lc' if C.subtools().get(u, {}).get('family') == 'rawmod' else u for u in use]
    if any(not C.sites_path(smp, t).exists() for smp, _, _ in blocks for t in use):
        return None
    return [np.max([Sc.site_scores(smp, t, sites)[0] for t in use], axis=0) for smp, sites, _ in blocks]


def table(extra):
    S, T, CL, cache = C.samples(), C.subtools(), claims(), {}
    fam = families(T)
    extra = [e for e in extra if e not in dict(TOOLS)]  # an --extra that is already a Table 1 column (UniMeth-Agn since ) is not duplicated
    tools = [(f, n) for f, n in TOOLS if f not in COMPARATIVE] + [(e, e) for e in extra]
    for e in extra:
        fam[e] = [e]
    recs = []
    for r in two_library_rows():
        blocks = Sc.row_sites(r, S, cache)
        y = np.concatenate([b[2] for b in blocks])
        p = OUT / f"bg_{r['row']}.tsv"
        bg = None
        if p.exists():
            d = pd.read_csv(p, sep='\t', dtype={'contig': str})
            bg = [(r['pos_sample'], d, None), (r['neg_sample'], d, None)]
        for f, _ in tools:
            use = [t for t in fam[f] if row_marks(r) & CL[t]['marks']
                   and not (CL[t]['context'] == 'cpg' and r['context'] == 'noncpg')]
            rec = {'row': r['row'], 'tool': f}
            if not use:
                recs.append({**rec, 'plain': np.nan, 'fingerprint': np.nan, 'corrected': np.nan, 'status': 'N/A'}); continue
            cpg_only = all(CL[t]['context'] == 'cpg' for t in use) and r['context'] == 'any'
            sc = _score(blocks, use)
            if sc is None:
                recs.append({**rec, 'status': 'missing'}); continue
            keep = np.concatenate(is_cpg(blocks, S)) if cpg_only else np.ones(len(y), bool)
            s_all = np.concatenate(sc)
            rec['plain'] = roc_auc_score(y[keep], s_all[keep]) if 0 < y[keep].sum() < keep.sum() else np.nan
            if bg is None:
                recs.append({**rec, 'fingerprint': np.nan, 'corrected': np.nan, 'status': 'no background'}); continue
            bsc = _score(bg, use, lc=True)
            if bsc is None:
                recs.append({**rec, 'fingerprint': np.nan, 'corrected': np.nan, 'status': 'background missing'}); continue
            bkeep = np.concatenate(is_cpg(bg, S))[:len(bg[0][1])] if cpg_only else np.ones(len(bg[0][1]), bool)
            if bkeep.sum() < 50:  # the held-out oligos give 81-98
                recs.append({**rec, 'fingerprint': np.nan, 'corrected': np.nan, 'status': 'background too small'}); continue
            nb, cb = bsc[0][bkeep], bsc[1][bkeep]
            rec['fingerprint'] = roc_auc_score(np.r_[np.ones(len(nb)), np.zeros(len(cb))], np.r_[nb, cb])
            corr = np.concatenate([_pct(nb, sc[0]), _pct(cb, sc[1])])
            rec['corrected'] = roc_auc_score(y[keep], corr[keep]) if 0 < y[keep].sum() < keep.sum() else np.nan
            rec['n_background'] = int(bkeep.sum())
            rec['status'] = 'ok'
            recs.append(rec)
        print(f"[{r['row']}] " + '  '.join(f"{x['tool']}={x.get('plain', np.nan):.3f}/{x.get('fingerprint', np.nan):.3f}"
                                          f"/{x.get('corrected', np.nan):.3f}" for x in recs if x['row'] == r['row']))
    D = pd.DataFrame(recs)
    OUT.mkdir(parents=True, exist_ok=True)
    D.to_csv(OUT / 'library_control.tsv', sep='\t', index=False, float_format='%.4f')
    print(f"-> {OUT / 'library_control.tsv'}")
    return D


def within(extra):
    S, T, CL, cache = C.samples(), C.subtools(), claims(), {}
    fam = families(T)
    extra = [e for e in extra if e not in dict(TOOLS)]  # as in table()
    tools = [f for f, _ in TOOLS if f not in COMPARATIVE] + list(extra)
    for e in extra:
        fam[e] = [e]
    recs = []
    for r in two_library_rows():
        p = OUT / f"bg_{r['row']}.tsv"
        if not p.exists():
            continue
        d = pd.read_csv(p, sep='\t', dtype={'contig': str})
        bg = [(r['pos_sample'], d, None), (r['neg_sample'], d, None)]
        blocks = Sc.row_sites(r, S, cache)
        for f in tools:
            use = [t for t in fam[f] if row_marks(r) & CL[t]['marks']
                   and not (CL[t]['context'] == 'cpg' and r['context'] == 'noncpg')]
            if not use:
                continue
            cpg_only = all(CL[t]['context'] == 'cpg' for t in use) and r['context'] == 'any'
            sc, bsc = _score(blocks, use), _score(bg, use, lc=True)
            if sc is None or bsc is None:
                continue
            rec = {'row': r['row'], 'tool': f, 'n_bg': len(d)}
            for i, lib in ((0, 'native'), (1, 'control')):
                keep = is_cpg([blocks[i]], S)[0] if cpg_only else np.ones(len(sc[i]), bool)
                if lib == 'native':  # the modified sites only
                    keep &= blocks[0][2].astype(bool)
                bkeep = is_cpg([bg[i]], S)[0] if cpg_only else np.ones(len(bsc[i]), bool)
                s, b = sc[i][keep], bsc[i][bkeep]
                rec[lib] = roc_auc_score(np.r_[np.ones(len(s)), np.zeros(len(b))], np.r_[s, b])
                rec[f'n_{lib}'] = len(s)
            recs.append(rec)
        print(f"[{r['row']}] " + '  '.join(f"{x['tool']}={x['native']:.3f}/{x['control']:.3f}" for x in recs
                                          if x['row'] == r['row']), flush=True)
    D = pd.DataFrame(recs)
    D.to_csv(OUT / 'within_library.tsv', sep='\t', index=False, float_format='%.4f')
    print(f"-> {OUT / 'within_library.tsv'}")
    within_latex(D)
    return D


def within_latex(D):
    """tables/supp_bench_within.tex: native / control within-library AUROC per row and tool, and mean |control - 0.5|."""
    from .paper_table import GROUPS
    tools = [t for t in NAMES if t in set(D['tool'])]
    labels = {row: lab for _, mem in GROUPS for row, lab in mem}
    X = D.set_index(['row', 'tool'])
    cell = lambda r, t: (f"{X.loc[(r, t), 'native']:.3f} / {X.loc[(r, t), 'control']:.3f}" if (r, t) in X.index else '--')
    A = [r'\begin{tabular}{@{}l ' + 'c' * len(tools) + '@{}}', r'\toprule',
         r'\textbf{Row} & ' + ' & '.join(rf'\textbf{{{NAMES[t]}}}' for t in tools) + r' \\', r'\midrule']
    for r in dict.fromkeys(D['row']):
        A.append(rf'{labels.get(r, r)} & ' + ' & '.join(cell(r, t) for t in tools) + r' \\')
    dev = (D['control'] - 0.5).abs().groupby(D['tool'])
    A += [r'\midrule', r'Mean $|$control $-$ 0.5$|$ & ' + ' & '.join(f'{dev.mean()[t]:.3f}' for t in tools) + r' \\',
          r'Rows & ' + ' & '.join(str(dev.size()[t]) for t in tools) + r' \\', r'\bottomrule', r'\end{tabular}']
    out = OUT / 'supp_bench_within.tex'
    out.write_text('\n'.join(A) + '\n')
    print(f'-> {out}')


NAMES = {'rawmod': r'\sysname', 'dorado': 'Dorado', 'unimeth': 'UniMeth', 'unimeth_reg': r'UniMeth-FT$^{\dagger}$',
         'deepmod2': 'DeepMod2', 'rockfish': 'Rockfish', 'methynano': 'MethyNano', 'unimeth_agn_mod_reg': 'UniMeth-Agn'}
MARK = {'5mC, CpG': '5mC', '5mC, non-CpG': '5mC', '5mC, all contexts': '5mC', '5hmC': '5hmC', '6mA': '6mA', '4mC': '4mC',
        '5hmU': '5hmU'}


def latex(D, onelib):
    """tables/supp_bench_libcontrol.tex: (a) library fingerprint, (b) baseline-corrected AUROC (plain in brackets) and
    the per-modification summaries with the corrected two-library rows. onelib: {(row, tool): AUROC} of the
    one-library rows (unchanged by the correction) and of the rows without a background (SPO1), from the Table 1 run."""
    from .paper_table import GROUPS
    tools = [t for t in NAMES if t in set(D['tool'])]
    labels = {row: lab for _, mem in GROUPS for row, lab in mem}
    rows = [r for r in dict.fromkeys(D['row']) if not r.startswith('spo1')]
    f = lambda v: '--' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:.3f}'
    X = D.set_index(['row', 'tool'])
    get = lambda r, t, c: X.loc[(r, t), c] if (r, t) in X.index and c in X.columns else np.nan
    head = [r'\begin{tabular}{@{}l ' + 'c' * len(tools) + '@{}}', r'\toprule',
            r'\textbf{Row} & ' + ' & '.join(rf'\textbf{{{NAMES[t]}}}' for t in tools) + r' \\', r'\midrule']
    A = head + [rf'\multicolumn{{{len(tools) + 1}}}{{@{{}}l}}{{\textit{{(a) Native vs.\ control at unmodified background positions}}}} \\']
    for r in rows:
        n = int(np.nanmax([get(r, t, 'n_background') for t in tools]))
        A.append(rf'\hspace{{3pt}}{labels.get(r, r)} ($n{{=}}{n:,}$) & ' + ' & '.join(f(get(r, t, 'fingerprint')) for t in tools) + r' \\')
    A += [r'\midrule', rf'\multicolumn{{{len(tools) + 1}}}{{@{{}}l}}{{\textit{{(b) Baseline-corrected AUROC (matched AUROC in brackets)}}}} \\']
    for r in rows:
        A.append(rf'\hspace{{3pt}}{labels.get(r, r)} & ' + ' & '.join(
            '--' if np.isnan(get(r, t, 'plain')) else f"{f(get(r, t, 'corrected'))} ({f(get(r, t, 'plain'))})" for t in tools) + r' \\')
  # summaries over all 15 rows: corrected two-library rows + unchanged one-library rows
    mark_of = {row: MARK[g] for g, mem in GROUPS for row, _ in mem}
    summ = {}
    for t in tools:
        res = {}
        for col in ('plain', 'corrected'):
            vals, ok = {}, True
            for g, mem in GROUPS:
                for row, _ in mem:
                    v = get(row, t, col) if row in rows else onelib.get((row, t), np.nan)
                    if row in rows and np.isnan(v) and not np.isnan(get(row, t, 'plain')):
                        ok = False
                    vals[row] = 0.5 if np.isnan(v) else v  # a model without a call at the row's base: 0.5, as in Table 1
            pm = pd.Series(vals).groupby(pd.Series(mark_of)).mean()
            res[col] = (pm.mean(), pm.min()) if ok else (np.nan, np.nan)
        summ[t] = res
    A += [r'\midrule',
          r'Mean over the five modifications & ' + ' & '.join(f(summ[t]['plain'][0]) for t in tools) + r' \\',
          r'\quad baseline-corrected & ' + ' & '.join(f(summ[t]['corrected'][0]) for t in tools) + r' \\',
          r'Worst modification & ' + ' & '.join(f(summ[t]['plain'][1]) for t in tools) + r' \\',
          r'\quad baseline-corrected & ' + ' & '.join(f(summ[t]['corrected'][1]) for t in tools) + r' \\',
          r'\bottomrule', r'\end{tabular}']
    out = OUT / 'supp_bench_libcontrol.tex'
    out.write_text('\n'.join(A) + '\n')
    print(f'-> {out}')
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['cands', 'rawmod', 'table', 'within'])
    ap.add_argument('--extra', default='', help='comma-separated extra subtools as their own columns')
    ap.add_argument('--rawmod', default=','.join(_PT.RAWMOD_COLS), help='rawmod mode: config/rawmod_models.tsv entries (default BENCH_RAWMOD)')
    a = ap.parse_args()
    if a.mode == 'cands':
        cands()
    elif a.mode == 'rawmod':
        rawmod_lc([x for x in a.rawmod.split(',') if x])
    elif a.mode == 'within':
        within([e for e in a.extra.split(',') if e])
    else:
        ext = [e for e in a.extra.split(',') if e]
        D = table(ext)
  # unchanged rows: Table 1's matched AUROC of every tool (paper_tables.tsv of the region run) and, for the extra
  # one-model columns, their AUROC on every row (they claim all five modifications, so matched = all sites)
        P = pd.read_csv(C.WORK / f'paper_{C.PAPER_TAG}' / 'paper_tables.tsv', sep='\t')
        onelib = {(r, t): float(v) for r, t, v in zip(P['row'], P['tool'], P['matched'])
                  if str(v) not in ('N/A', '--', '', 'nan')}
        ext = [e for e in ext if e not in dict(TOOLS)]
        if ext:
            S, cache = C.samples(), {}
            for r in Sc.rows_cfg():
                if r['row'] not in {row for _, mem in GROUPS for row, _ in mem}:
                    continue
                blocks = Sc.row_sites(r, S, cache)
                y = np.concatenate([b[2] for b in blocks])
                for e in ext:
                    onelib[(r['row'], e)] = roc_auc_score(y, np.concatenate(_score(blocks, [e])))
        print(latex(D, onelib))


if __name__ == '__main__':
    main()
