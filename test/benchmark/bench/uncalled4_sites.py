#!/usr/bin/env python3
"""Uncalled4 (Kovaka et al. 2025), a comparative signal-alignment method, scored per site like every other column.

Uncalled4 aligns each read's signal to the reference k-mers (`uncalled4 align`, DTW guided by the move table) and
detects a modification as a difference between a sample and an unmodified control, used as released: every read
(sample and control) is aligned with the bundled R10.4.1 pore model (trained by the Uncalled4 authors on
whole-genome-amplified, i.e. unmodified, D. melanogaster DNA); no model is trained here. Uncalled4 is NOT held out:
its control is the whole control library, including its reads over the held-out region (a comparative method needs
a control of the very sites it tests; RawMod's region holdout does not apply to it).

Site score: Uncalled4's comparison -- a two-sample Kolmogorov-Smirnov statistic D between the currents of the site's
K selected reads and the currents of the control library's reads at the same reference position and strand (the k-mer
whose shift base is the site, i.e. Uncalled4's seq.pos), all aligned with the same model. The control reads are the
control library's reads over the region that are in no benchmark read set (pool.*, bench.uncalled4_controls) plus the
control read set's own benchmark reads at the site other than the K scored there (the whole control library, as a
comparative tool sees it; no read is compared with itself). Where fewer than MIN_POOL control reads cover a site, and
on one-sample rows (no control library), D is taken against Normal(mean, sd) of the k-mer in the released model
instead (column ref = pool | model). D is in [0, 1]; mean |z| against the model is kept for reference.
Every alignment uses --min-aln-length 50: the oligonucleotide reads are ~110 bases and the default (100) drops ~45%.
Output: $BENCH_WORK/<sample>/<subtool>/<TAG>/sites.tsv (bench.adapters.uncalled4).

  python -m bench.uncalled4_sites --sample spo1_bc07_reg
"""
import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import common as C
from . import rawmod_cands
from . import score as Sc
from . import settings as ST
from .aggregate import Selection

BUNDLED = 'dna_r10.4.1_400bps_9mer'
# control libraries with a comparison read set (bench.uncalled4_controls); a benchmark sample maps to one by row
CONTROL_LIBS = ('syn_control', 'ecoli_dm', 'hp26695_wga', 'spo1_bc01')


def control_lib(sample):
    """The control library of the two-sample rows this sample is in, or None (one-sample rows only)."""
    libs = set()
    for r in Sc.rows_cfg():
        if r['neg_sample'] and sample in (r['pos_sample'], r['neg_sample']):
            n = r['neg_sample']
            for suf in ('_wreg', '_w1m', '_regC', '_reg'):  # _regC: whitelisted clean samples (config/read_whitelist.tsv)
                n = n[:-len(suf)] if n.endswith(suf) else n
            libs.add(n)
    libs &= set(CONTROL_LIBS)  # replicate-2 oligo rows (not in Table 1) have no comparison read set
    if len(libs) > 1:
        raise SystemExit(f'{sample}: rows with different control libraries {sorted(libs)}')
    return libs.pop() if libs else None


def model_for(sample):
    """(released model .npz, control library or None). $UNCALLED4_MODELS/bundled/model.npz is a copy of the released model."""
    p = Path(ST.need('UNCALLED4_MODELS', 'Uncalled4 models (bench.uncalled4_controls)')) / 'bundled' / 'model.npz'
    if not p.exists():
        raise SystemExit(f'{sample}: {p} missing; run python -m bench.uncalled4_controls --control bundled')
    return str(p), control_lib(sample)


MIN_COUNT = 5  # a control k-mer seen fewer times keeps the released model's level (its sd can be 0)
MIN_POOL = 5  # control reads needed at a site for the two-sample test
MIN_ALN = 50  # uncalled4 align --min-aln-length (default 100 drops ~45% of the ~110-base oligonucleotide reads)
ALN = f'align_bundled_m{MIN_ALN}.tsv'  # alignments with the released model
COLS = ['seq.name', 'seq.pos', 'seq.strand', 'seq.kmer', 'aln.read_id', 'dtw.current']


def load_model(m):
    z = np.load(m, allow_pickle=True)
    mu, sd = z['current.mean'].astype(float), z['current.stdv'].astype(float)
    base = np.load(Path(ST.need('UNCALLED4_MODELS', 'Uncalled4 models')) / 'bundled' / 'model.npz', allow_pickle=True)
    assert (base['kmer'] == z['kmer']).all()
    few = z['count'] < MIN_COUNT
    mu = np.where(few, base['current.mean'], mu); sd = np.where(few, base['current.stdv'], sd)
    print(f'{m}: {few.sum():,} of {len(few):,} k-mers below {MIN_COUNT} control instances use the released level')
    return dict(zip(z['kmer'].tolist(), zip(mu.tolist(), sd.tolist())))


def align(ref, pod5, bam, model, tsv):
    """uncalled4 align -> TSV (read id, reference position, strand, k-mer, current); skipped if tsv[.gz] exists."""
    if tsv.exists() or Path(f'{tsv}.gz').exists():
        return tsv if tsv.exists() else Path(f'{tsv}.gz')
    Path(tsv).parent.mkdir(parents=True, exist_ok=True)
    cmd = (f"{ST.need('UNCALLED4', 'Uncalled4 signal alignment')} align --ref {ref} --reads {pod5} --bam-in {bam} "
           f"-m {model} --min-aln-length {MIN_ALN} -p {len(os.sched_getaffinity(0))} --tsv-cols aln.read_id,dtw.current "
           f"--tsv-out {tsv}.tmp && mv {tsv}.tmp {tsv}")
    print(cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)
    return tsv


def read_aligned(tsv, candf, sel=None):
    """Rows of an alignment TSV at candidate sites (contig, pos, strand); with sel, only each site's selected reads."""
    parts = []
    for ch in pd.read_csv(tsv, sep='\t', chunksize=2_000_000, na_values=['*'],  # '*' = no current at that k-mer
                          dtype={'seq.name': str, 'seq.pos': np.int64, 'seq.strand': str, 'seq.kmer': str,
                                 'aln.read_id': str, 'dtw.current': float}, usecols=COLS):
        ch = ch.dropna(subset=['dtw.current'])
        df = pd.DataFrame({'read_id': ch['aln.read_id'], 'contig': ch['seq.name'], 'pos': ch['seq.pos'],
                           'kmer': ch['seq.kmer'], 'cur': ch['dtw.current'], 'strand': ch['seq.strand']})
        if sel is not None:
            ust = df['strand'].to_numpy()
            df = sel.keep(df.drop(columns='strand').assign(ustrand=ust))  # only the K selected reads of each site
            df = df[df['strand'] == df['ustrand']].drop(columns='ustrand')
        parts.append(df.merge(candf, on=['contig', 'pos', 'strand'], how='inner')[['contig', 'pos', 'strand', 'read_id', 'kmer', 'cur']])
    return pd.concat(parts, ignore_index=True).drop_duplicates(['contig', 'pos', 'strand', 'read_id'])  # one value per read


def control_sample(sample):
    """The control (negative) read set of the two-sample rows this sample is in."""
    ns = {r['neg_sample'] for r in Sc.rows_cfg() if r['neg_sample'] and sample in (r['pos_sample'], r['neg_sample'])}
    ns = {n for n in ns if control_lib(n) is not None}
    if len(ns) != 1:
        raise SystemExit(f'{sample}: expected one control read set, got {sorted(ns)}')
    return ns.pop()


def control_pool(lib, model, candf, out, sample):
    """The comparison reads at each candidate site: the control library's unused reads over the region (pool.*) plus the control read set's own benchmark reads at the site that are NOT among the K selected there. The
    same set serves the positive and the negative library at a position, and no scored read is compared with itself.
    Cached per sample, keyed by the unused read set."""
    from .uncalled4_controls import SOURCES
    from .readsel import whitelist
    import hashlib
    md = Path(ST.need('UNCALLED4_MODELS', 'Uncalled4 models')) / lib
    cs = control_sample(sample)
  # Opt-in read whitelist of the control read set (config/read_whitelist.tsv ): the comparison reads are the
  # whitelisted control reads at the site other than the K selected there, and the library's unused reads (pool.*) are
  # not used -- for the clean SPO1 row the pool.bam of the bc01 library holds no primer-anchored PCR read.
    clean = whitelist(cs) is not None
    key = 'whitelist' if clean else hashlib.md5((md / 'pool_ids.txt').read_bytes()).hexdigest()[:10]
    cache = out / f'control_pool_{key}_bundled.tsv.gz'
    if cache.exists():
        return pd.read_csv(cache, sep='\t', dtype={'contig': str})
    if clean:
        unused = pd.DataFrame(columns=['contig', 'pos', 'strand', 'read_id', 'kmer', 'cur'])
    else:
        tsv = align(C.samples()[SOURCES[lib][2]]['ref'], md / 'pool.pod5', md / 'pool.bam', model, md / f'pool_{ALN}')
        unused = read_aligned(tsv, candf)
        if tsv.suffix != '.gz':
            subprocess.run(['gzip', '-1', '-f', str(tsv)], check=True)
    CS = C.samples()[cs]
    ctsv = align(CS['ref'], CS['sub_pod5'], CS['sub_bam'], model, C.WORK / cs / 'uncalled4' / C.TAG / ALN)
    csel = Selection(cs)
    allc = read_aligned(ctsv, candf)
    allc['read_id'] = allc['read_id'].map(csel.read_idx).fillna(-1).astype(np.int64)  # Selection.keep returns read indices
    if clean:
        allc = allc[allc['read_id'] >= 0]  # the selection's read_ids are the whitelist: drop every other read
    sel = read_aligned(ctsv, candf, csel)[['contig', 'pos', 'strand', 'read_id']].assign(_sel=True)
    other = allc.merge(sel, on=['contig', 'pos', 'strand', 'read_id'], how='left')
    other = other[other['_sel'].isna()]
    print(f'{sample}: comparison reads = {len(unused):,} unused + {len(other):,} unselected {"whitelisted " if clean else ""}{cs} values at the candidate sites')
    pool = pd.concat([unused, other], ignore_index=True)[['contig', 'pos', 'strand', 'cur']]
    pool.to_csv(f'{cache}.tmp', sep='\t', index=False, float_format='%.6g', compression='gzip')
    os.replace(f'{cache}.tmp', cache)
    return pool


def ks2(a, b):
    """Two-sample KS statistic D of sorted arrays a, b."""
    x = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, x, side='right') / len(a) - np.searchsorted(b, x, side='right') / len(b))))


def run(sample, subtool):
    S = C.samples()[sample]
    out = C.WORK / sample / subtool / C.TAG
    if (out / 'sites.tsv').exists():
        print(f'{sample} {subtool}: done')
        return
    out.mkdir(parents=True, exist_ok=True)
    model, lib = model_for(sample)
    rawmod_cands.build(sample, out=out)
    cand = []
    for name, st in (('plus', '+'), ('minus', '-')):
        b = out / f'cand_{name}.bed'
        if b.stat().st_size:
            d = pd.read_csv(b, sep='\t', header=None, usecols=[0, 1], names=['contig', 'pos'], dtype={'contig': str})
            cand.append(d.assign(strand=st))
    candf = pd.concat(cand, ignore_index=True).drop_duplicates()
    tsv = align(S['ref'], S['sub_pod5'], S['sub_bam'], model, out / ALN)
    d = read_aligned(tsv, candf, Selection(sample))
    lev = load_model(model)
    mu_sd = d['kmer'].map(lev)
    d = d[mu_sd.notna()]; mu_sd = mu_sd[mu_sd.notna()]
    mu = np.array([v[0] for v in mu_sd]); sd = np.array([v[1] for v in mu_sd])
    d = d.assign(F=norm.cdf((d['cur'].to_numpy() - mu) / sd), az=np.abs((d['cur'].to_numpy() - mu) / sd))
    pool = {}
    if lib is not None:
        p = control_pool(lib, model, candf, out, sample)
        pool = {k: np.sort(g['cur'].to_numpy()) for k, g in p.groupby(['contig', 'pos', 'strand'], sort=False)}
    rows = []
    for (c, st, pos), g in d.groupby(['contig', 'strand', 'pos'], sort=True):
        x = np.sort(g['cur'].to_numpy()); ctrl = pool.get((c, pos, st))
        if ctrl is not None and len(ctrl) >= MIN_POOL:
            D, ref, nc = ks2(x, ctrl), 'pool', len(ctrl)
        else:
            F = np.sort(g['F'].to_numpy()); n = len(F); i = np.arange(1, n + 1)
            D, ref, nc = float(np.max(np.maximum(i / n - F, F - (i - 1) / n))), 'model', 0 if ctrl is None else len(ctrl)
        rows.append((c, pos, st, D, len(x), nc, ref, g['az'].mean()))
    res = pd.DataFrame(rows, columns=['contig', 'pos', 'strand', 'score', 'n_reads', 'n_ctrl', 'ref', 'mean_abs_z'])
    res.to_csv(out / 'sites.tsv.tmp', sep='\t', index=False, float_format='%.6g')
    os.replace(out / 'sites.tsv.tmp', out / 'sites.tsv')
    if tsv.suffix != '.gz':
        subprocess.run(['gzip', '-1', '-f', str(tsv)], check=True)
    (out / 'PROVENANCE.txt').write_text(f'uncalled4 align -m {model} (released) --min-aln-length {MIN_ALN} (control library: {lib}, '
                                        f'not held out); two-sample KS vs control reads at the site (>= {MIN_POOL}), else KS '
                                        f'vs Normal(k-mer mean, sd)\n')
    print(f'{sample} {subtool}: {len(res):,} sites (released model, control {lib}), median reads {res["n_reads"].median():.0f}, '
          f'pool test at {(res["ref"] == "pool").mean():.1%} (median {res["n_ctrl"].median():.0f} control reads)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sample', nargs='+', required=True)
    ap.add_argument('--subtool', default='uncalled4')
    a = ap.parse_args()
    for s in a.sample:
        run(s, a.subtool)


if __name__ == '__main__':
    main()
