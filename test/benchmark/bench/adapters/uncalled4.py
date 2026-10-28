"""Uncalled4: per-site scores from step tool (bench.uncalled4_sites) -- the one-sample KS statistic of the site's K
selected reads against the control pore model (trained on the row's unmodified control library, or the released
WGA-trained model on one-sample rows). Yields sites directly: contig, pos, strand, score, n_reads."""
import pandas as pd

from .. import common as C

PER_SITE = True


def iter_sites(sample, subtool):
    df = pd.read_csv(C.WORK / sample['sample'] / subtool['subtool'] / C.TAG / 'sites.tsv', sep='\t', dtype={'contig': str})
    yield df[['contig', 'pos', 'strand', 'score', 'n_reads']]
