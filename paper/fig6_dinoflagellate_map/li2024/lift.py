#!/usr/bin/env python3
"""Per-base lift-over of contig_3762 onto the Li et al. (2024) assembly from a minimap2 PAF alignment with cs tags. Writes lift.npz."""
import os, re, numpy as np
from pathlib import Path
W = Path(os.environ.get('RAWMOD_DATA', 'data')) / 'acarterae_li2024'  # contig_vs_v1.paf (minimap2 -c --cs contig_3762 vs the Li assembly)
L = 7733821
chrom = np.full(L, -1, np.int16); pos = np.full(L, -1, np.int64); strand = np.zeros(L, np.int8); mq = np.full(L, -1, np.int16)
names = []
for ln in open(W / 'contig_vs_v1.paf'):
    x = ln.rstrip('\n').split('\t')
    if 'tp:A:P' not in x[12:]:
        continue
    qs, qe, st, tname, ts, te, q = int(x[2]), int(x[3]), x[4], x[5], int(x[7]), int(x[8]), int(x[11])
    if tname not in names:
        names.append(tname)
    ci = names.index(tname)
    cs = next(f[5:] for f in x[12:] if f.startswith('cs:Z:'))
  # walk target forward; query forward if '+', backward from qe if '-'
    t = ts; qq = qs if st == '+' else qe - 1; dq = 1 if st == '+' else -1
    for op, val in re.findall(r'([:*+\-~])([0-9a-z]+)', cs):
        if op == ':':
            n = int(val); qi = qq + dq * np.arange(n); ti = t + np.arange(n)
            keep = mq[qi] < q
            chrom[qi[keep]] = ci; pos[qi[keep]] = ti[keep]; strand[qi[keep]] = 1 if st == '+' else -1; mq[qi[keep]] = q
            t += n; qq += dq * n
        elif op == '*':
            if mq[qq] < q:
                chrom[qq] = ci; pos[qq] = t; strand[qq] = 1 if st == '+' else -1; mq[qq] = q
            t += 1; qq += dq
        elif op == '+':  # insertion in query
            qq += dq * len(val)
        elif op == '-':  # deletion from query
            t += len(val)
np.savez_compressed(W / 'lift.npz', chrom=chrom, pos=pos, strand=strand, mapq=mq, names=np.array(names))
print('aligned bases', (chrom >= 0).sum(), f'({(chrom >= 0).mean():.1%})', 'mapq>=20', (mq >= 20).mean(), 'chroms', names)
