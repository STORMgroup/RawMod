"""Pick + strand sites of the A. carterae contig by what Dorado can say about them, for the dark-modification map.
Dorado labels come from modkit pileups of two all-context basecalls (6mA + 5mC_5hmC; 4mC_5mC). A site counts as
called when >= 10 valid reads and modified fraction >= 0.5, and as unmodified at >= 10 reads and fraction <= 0.1.
T and G have no Dorado model. Writes candidates.tsv (contig, 0-based pos) and sites.tsv (pos, base, dorado label)."""
import os as _os
from pathlib import Path as _Path
REPO = str(_Path(__file__).resolve().parents[2])
RAWMOD_RESULTS = _os.environ.get('RAWMOD_RESULTS') or f'{REPO}/paper/data/runs'  # training runs; shipped metrics by default
RAWMOD_DATA = _os.environ.get('RAWMOD_DATA', 'data')  # datasets (not shipped: set this to score raw signal)
RAWMOD_WORK = _os.environ.get('RAWMOD_WORK') or f'{REPO}/paper/data/work'  # outputs of the paper scripts; shipped by default
import sys, numpy as np, pandas as pd, pysam
D = sys.argv[1]; REF = sys.argv[2]   # work directory (holds bc/passA.bed and bc/passB.bed), contig FASTA
S = pysam.FastaFile(REF).fetch('contig_3762').upper()
COLS = ['chrom', 'start', 'end', 'code', 'score', 'strand', 's2', 'e2', 'rgb', 'nvalid', 'frac', 'nmod', 'ncanon',
        'nother', 'ndel', 'nfail', 'ndiff', 'nnocall']
def bed(fn):
    b = pd.read_csv(fn, sep='\t', header=None, names=COLS, usecols=range(18))
    return b[b.strand == '+']
A = bed(f'{D}/bc/passA.bed'); B = bed(f'{D}/bc/passB.bed')
rng = np.random.default_rng(0)
def pick(df, n): return df.iloc[rng.permutation(len(df))[:n]].start.to_numpy()
rows = []
def add(pos, base, lab):
    rows.extend(dict(pos=int(p), base=base, dorado=lab) for p in pos)
a = A[A.code == 'a']; m = A[A.code == 'm']; h = A[A.code == 'h']; c4 = B[B.code.astype(str) == '21839']
cpg = lambda df: df[[S[p + 1:p + 2] == 'G' for p in df.start]]
non = lambda df: df[[S[p + 1:p + 2] != 'G' for p in df.start]]
called = lambda df: df[(df.nvalid >= 10) & (df.frac >= 50)]          # modkit fraction is a percentage
unmod = lambda df: df[(df.nvalid >= 10) & (df.frac <= 10)]
add(pick(called(a), 500), 'A', '6mA'); add(pick(unmod(a), 500), 'A', 'unmodified')
add(pick(cpg(called(m)), 300), 'C', '5mC CpG'); add(pick(non(called(m)), 300), 'C', '5mC non-CpG')
add(pick(called(h), 300), 'C', '5hmC'); add(pick(called(c4), 300), 'C', '4mC')
cu = unmod(m); cu = cu[cu.start.isin(unmod(c4).start) & cu.start.isin(unmod(h).start)]
add(pick(cu, 500), 'C', 'unmodified')
cov = set(A[A.nvalid >= 10].start)                                   # covered positions (any A/C call row)
T = np.array([i for i in range(20, len(S) - 20) if S[i] == 'T'])
G = np.array([i for i in range(20, len(S) - 20) if S[i] == 'G'])
add(rng.choice(T, 1500, replace=False), 'T', 'no Dorado model'); add(rng.choice(G, 400, replace=False), 'G', 'no Dorado model')
d = pd.DataFrame(rows).drop_duplicates('pos', keep='first')
d.to_csv(f'{D}/sites.tsv', sep='\t', index=False)
d[['pos']].assign(contig='contig_3762')[['contig', 'pos']].to_csv(f'{D}/candidates.tsv', sep='\t', index=False, header=False)
print(d.groupby(['base', 'dorado']).size().to_string())
print('called counts available:', {k: len(v) for k, v in dict(a=called(a), m_cpg=cpg(called(m)), m_non=non(called(m)), h=called(h), c4=called(c4)).items()})
