"""Positions the benchmarked RawMod checkpoint trained on, per benchmark sample (BENCH_HOLDOUT=positions).

Source: test/data/holdout/rawmod_seen_mixed.tsv.gz, written by bench.rawmod_seen (every image of the training
pool with its split). A position counts as SEEN if any training image (split == 'train') of the same organism sits on it, on either
strand and under either label. config/holdout_positions.tsv maps each benchmark sample to a regular expression over the
training-pool file names of its organism; samples with no entry (mouse, rice) have no training data and keep every site.
Organism matching matters because contig names repeat across organisms (human and mouse both have 'chr1').
"""
from functools import lru_cache

import pandas as pd

from . import common as C
from .settings import DATA_DIR

SEEN = {'positions': 'rawmod_seen_mixed.tsv.gz'}


@lru_cache(maxsize=1)
def _table():
    d = pd.read_csv(DATA_DIR / 'holdout' / SEEN[C.HOLDOUT], sep='\t', usecols=['file', 'contig', 'pos', 'split'],
                    dtype={'file': str, 'contig': str, 'pos': 'int64', 'split': str})
    return d[d['split'] == 'train']


@lru_cache(maxsize=1)
def _map():
    return {r[0]: r[1] for r in C._rows(C.CONFIG / f'holdout_{C.HOLDOUT}.tsv')}


@lru_cache(maxsize=None)
def _seen_for(sample):
    pat = _map().get(sample)
    if not pat:
        return frozenset()
    d = _table()
    d = d[d['file'].str.contains(pat, regex=True)]
    return frozenset(zip(d['contig'], d['pos'].astype(int)))


@lru_cache(maxsize=None)
def _region_for(sample):
    """BENCH_HOLDOUT=region: {contig: [(start, end), ...]} of the sample's held-out region, or None if the sample was
    never in any training set (config/holdout_region.tsv)."""
    bed = {r[0]: r[1] for r in C._rows(C.CONFIG / 'holdout_region.tsv')}.get(sample)
    if not bed:
        return None
    out = {}
    for line in open(DATA_DIR / 'holdout' / 'regions' / bed):
        f = line.split()
        if f and not line.startswith('#'):
            out.setdefault(f[0], []).append((int(f[1]), int(f[2])))
    return out


def in_region(sample, contigs, positions):
    """Boolean mask: which (contig, pos) lie inside the sample's held-out region (all True if it has none)."""
    reg = _region_for(sample)
    if reg is None:
        return [True] * len(contigs)
    return [any(a <= int(p) < b for a, b in reg.get(c, ())) for c, p in zip(contigs, positions)]


def seen_positions(samples):
    """Union of trained (contig, pos) over the organisms of `samples`."""
    out = set()
    for s in samples:
        out |= _seen_for(s)
    return out
