"""NovaMod: per-read reconstruction scores (score_recon; high = unlike unmodified DNA) at the candidate sites, from step
tool (bench.novamod). bench.aggregate keeps the K selected reads of each site."""
import pandas as pd

from .. import common as C


def iter_calls(sample, subtool):
    p = C.WORK / sample['sample'] / subtool['subtool'] / C.TAG / 'calls.tsv.gz'
    for ch in pd.read_csv(p, sep='\t', dtype={'read_id': str, 'contig': str}, usecols=['read_id', 'contig', 'pos', 'prob'],
                          chunksize=2_000_000):
        yield ch
