#!/usr/bin/env python3
"""Split a genome into N chunks of equal read coverage: chunk<i>.cand lists every position of chunk i (contig, pos),
chunk<i>.reg its regions (samtools syntax). Used by timing_lib.sh to run RawMod's fused scoring in N processes.

  python split_chunks.py refined.bam N
"""
import sys

import numpy as np
import pysam

f = pysam.AlignmentFile(sys.argv[1])
n = int(sys.argv[2])
segs = []
for c, L in zip(f.references, f.lengths):
    d = np.zeros(L + 1, np.int64)
    for r in f.fetch(c):
        if not (r.is_secondary or r.is_supplementary or r.is_unmapped):
            d[r.reference_start] += 1
            d[r.reference_end] -= 1
    segs.append((c, np.cumsum(d)[:L]))
mass = np.concatenate([s for _, s in segs]).astype(float)
cut = np.searchsorted(np.cumsum(mass), np.linspace(0, mass.sum(), n + 1)[1:-1])
glob = np.concatenate([[0], cut, [len(mass)]])
off, spans = 0, [[] for _ in range(n)]
for c, s in segs:
    for i in range(n):
        a, b = max(glob[i] - off, 0), min(glob[i + 1] - off, len(s))
        if a < b:
            spans[i].append((c, a, b))
    off += len(s)
for i, sp in enumerate(spans):
    with open(f'chunk{i}.cand', 'w') as cf, open(f'chunk{i}.reg', 'w') as rf:
        for c, a, b in sp:
            cf.write(''.join(f'{c}\t{p}\n' for p in range(a, b)))
            rf.write(f'{c}:{a + 1}-{b}\n')
