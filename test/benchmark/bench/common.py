"""Shared configuration and conventions for the site-level tool benchmark.

Conventions:
  * A site is (contig, 0-based pos, strand). Both strands are scored; the strand of a call is the
    orientation of the read that made it.
  * Every site is scored from EXACTLY `K` reads (default 10) of the sample's read subset, the same
    reads for every tool: the K reads covering it on that strand whose `read_rank` is lowest. Sites
    with fewer than K covering reads are not scored.
  * Only primary alignments with mean qscore >= MIN_QS (BAM qs tag) are eligible. The rice and mouse
    subsets were already filtered at 10 by their basecaller; this applies the same filter everywhere.
  * A selected read that a tool did not call at the site counts as unmodified (probability 0).
  * Site score = mean probability over the K reads (primary, for AUROC); call frequency = fraction
    of the K reads with probability > 0.5 (for F1 at 0.5).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from . import settings as _ST

HERE = _ST.HERE  # test/benchmark
CONFIG = HERE / 'config'
WORK = Path(os.environ.get('BENCH_WORK') or _ST.get('BENCH_WORK'))
K = int(os.environ.get('BENCH_K', '10'))
SEED = int(os.environ.get('BENCH_SEED', '0'))
MIN_QS = float(os.environ.get('BENCH_MIN_QS', '10'))  # reads with mean basecall qscore (qs tag) below this are never selected
NEG_CAP = 50000  # one-sample rows: at most this many negatives (seeded, all tools)
INPUTS = os.environ.get('BENCH_INPUTS', 'own1')  # own1 = the benchmark's own basecall of every sample (step basecall)
# The basecalling model of step basecall. Another model (e.g. hac or fast) is run as its own inputs tag, so every
# derived file lands beside the sup results: BENCH_INPUTS=hac1 BENCH_BASECALL_MODEL=dna_r10.4.1_e8.2_400bps_hac@v5.0.0.
BASECALL_MODEL = os.environ.get('BENCH_BASECALL_MODEL', 'dna_r10.4.1_e8.2_400bps_sup@v5.0.0')
# BENCH_READSEL_FROM=own1: score the reads selected under that inputs tag instead of reselecting them from the new
# basecalls, so two basecalling models are compared on the same reads.
READSEL_FROM = os.environ.get('BENCH_READSEL_FROM', '')
# BENCH_REFINE=none scores the mapping Remora derives from the raw move table without its level-table refinement
# (a control for the value of refinement); the derived files carry a _noref suffix and a separate refine/ directory.
REFINE = os.environ.get('BENCH_REFINE', 'remora')
if REFINE not in ('remora', 'none'):
    raise SystemExit(f'BENCH_REFINE={REFINE!r}: remora or none')
TAG = f'{INPUTS}_k{K}_seed{SEED}_q{MIN_QS:g}' + ('_noref' if REFINE == 'none' else '')  # every derived file carries it
# BENCH_HOLDOUT=positions: score only positions the benchmarked RawMod checkpoint never trained on (bench.holdout). Site files
# are unchanged; only the scored site set shrinks, so paper outputs go to paper_<TAG>_holdout-<name>.
HOLDOUT = os.environ.get('BENCH_HOLDOUT', '')
PAPER_TAG = TAG + (f'_holdout-{HOLDOUT}' if HOLDOUT else '')
SAMTOOLS = os.environ.get('SAMTOOLS') or _ST.get('SAMTOOLS')
COMP = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}


def read_rank(read_id: str, seed: int = SEED) -> int:
    """Seeded 64-bit rank of a read. The K lowest-ranked reads covering a site are the ones scored,
    for every tool; featurization.py --rank-reads uses this same function."""
    return int.from_bytes(hashlib.blake2b(f'{seed}:{read_id}'.encode(), digest_size=8).digest(), 'big')


def _rows(path: Path):
    for line in open(path):
        if line.strip() and not line.startswith('#'):
            yield line.rstrip('\n').split('\t')


def samples() -> dict:
    """sample -> basecall BAM, read-subset pod5 and reference (the sheet bench.datasets.samples_tsv builds)."""
    if 'SAMPLES_TSV' not in os.environ:
        from . import datasets
        os.environ['SAMPLES_TSV'] = str(datasets.samples_tsv())
    S = {c[0]: {'sample': c[0], 'sub_bam': c[1], 'sub_pod5': c[2], 'ref': c[3]}
         for c in _rows(Path(os.environ['SAMPLES_TSV']))}
    if INPUTS != 'own1':
        # A second basecall of the same reads lives in the sample directory that holds the sheet's basecall,
        # so samples that share one basecall (a whitelisted read set of another sample) share the new one too.
        for r in S.values():
            r['sub_bam'] = str(Path(r['sub_bam']).resolve().parent.parent / f'{INPUTS}.moves.bam')
    return S


def subtools() -> dict:
    out = {c[0]: {'subtool': c[0], 'family': c[1], 'arg': c[2],
                  'codes': None if c[3] == '*' else set(c[3].split(',')),
                  'bases': set(c[4]), 'context': c[5]}
           for c in _rows(CONFIG / 'subtools.tsv')}
  # extra RawMod checkpoints registered in config/rawmod_models.tsv (bench.models): one subtool each, rawmod_<name>
    from . import models
    for st, m in models.load().items():
        out.setdefault(st, {'subtool': st, 'family': 'rawmod', 'arg': str(m['checkpoint']), 'codes': None,
                            'bases': set('ACGT'), 'context': 'any'})
    return out


def refine_dir(sample: str) -> Path:
    """Refinement outputs of this sample's basecall (one directory per inputs tag and refinement mode)."""
    return WORK / sample / 'refine' / (INPUTS + ('_noref' if REFINE == 'none' else ''))


def readsel_path(sample: str, inputs: str = INPUTS) -> Path:
    return WORK / sample / f'readsel_{inputs}_k{K}_seed{SEED}_q{MIN_QS:g}.npz'


def sites_path(sample: str, subtool: str) -> Path:
    return WORK / sample / subtool / f'sites_{TAG}.tsv.gz'
