#!/usr/bin/env python3
"""Gene arrays of the contig as blocks of same-strand genes: block lengths, scores and junction types."""
import os as _os
from pathlib import Path as _Path
REPO = str(_Path(__file__).resolve().parents[2])
RAWMOD_RESULTS = _os.environ.get('RAWMOD_RESULTS') or f'{REPO}/paper/data/runs'  # training runs; shipped metrics by default
RAWMOD_DATA = _os.environ.get('RAWMOD_DATA', 'data')  # datasets (not shipped: set this to score raw signal)
RAWMOD_WORK = _os.environ.get('RAWMOD_WORK') or f'{REPO}/paper/data/work'  # outputs of the paper scripts; shipped by default

import numpy as np, pandas as pd
BASE = f'{RAWMOD_DATA}/acarterae_contig_3762'
W = f'{RAWMOD_WORK}/dinoflagellate/strand'
F = {'plus': f'{RAWMOD_WORK}/dinoflagellate/contig_3762_strand+_scores.tsv.gz', 'minus': f'{RAWMOD_WORK}/dinoflagellate/contig_3762_strand-_scores.tsv.gz'}
rows = []
for line in open(f'{BASE}/Annotations_contig_3762.gff3'):
    if line.startswith('#') or not line.strip():
        continue
    p = line.rstrip('\n').split('\t'); rows.append((p[2], int(p[3]) - 1, int(p[4]), p[6]))
gff = pd.DataFrame(rows, columns=['type', 'start', 'end', 'strand'])
N = 7733821
lab = np.zeros(N, dtype=np.int8)
for t, code in (('gene', 1), ('exon', 2), ('CDS', 3)):
    for s, e in gff.loc[gff.type == t, ['start', 'end']].values:
        lab[s:e] = code
g = gff[gff.type == 'gene'].sort_values('start')[['start', 'end', 'strand']].values
arrays = []; a0 = 0
for i in range(1, len(g) + 1):
    if i == len(g) or g[i][2] != g[i - 1][2]:
        arrays.append((int(g[a0][0]), int(max(x[1] for x in g[a0:i])), g[a0][2], i - a0)); a0 = i
D = {nm: pd.read_csv(f, sep='\t') for nm, f in F.items()}
tab = []
for s, e, st, ng in arrays:
    row = dict(array=f'{s / 1e6:.2f}-{e / 1e6:.2f} Mb', strand=st, n_genes=ng, gc=None)
    for nm, d in D.items():
        pos = d.pos.values; sc = d.score.values; sel = (pos >= s) & (pos < e); c = lab[pos[sel]]; v = sc[sel]
        for ci, cn in ((0, 'ig'), (1, 'intron'), (3, 'CDS')):
            row[f'{cn}_{nm}'] = v[c == ci].mean() if (c == ci).sum() > 100 else np.nan
        row[f'all_{nm}'] = v.mean()
    tab.append(row)
tab = pd.DataFrame(tab).drop(columns='gc')
print(tab.to_string(float_format=lambda x: f'{x:.3f}'))
tab.to_csv(f'{W}/array_blocks.tsv', sep='\t', index=False, float_format='%.4f')
