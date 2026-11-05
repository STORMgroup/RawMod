"""Error rate against the distance to the nearest T on the read's own strand, with ties as their own class.

Same reads and filters as bc_strain_typing.py (primary alignments, <= NMAX reads per group, >= 200 T and >= 200
far positions). A base whose nearest Ts sit at the same distance on both sides ('both', e.g. T-X-T) is kept out of
the signed bins, so -d and +d hold the same kind of base. Offsets are in read orientation: -1 = the T is the base
immediately before (5' of) this base on the read. Also counts the distinct reference positions behind every bin. Profiles count only positions inside the
library's amplicon windows (windows.tsv) that both groups cover, so native and PCR reads are compared on the same loci.
Writes profile_ties.json: {library: {group: {class: [n, errors, distinct_positions]}}}.
"""
import os as _os
from pathlib import Path as _Path
REPO = str(_Path(__file__).resolve().parents[2])
RAWMOD_RESULTS = _os.environ.get('RAWMOD_RESULTS') or f'{REPO}/paper/data/runs'  # training runs; shipped metrics by default
RAWMOD_DATA = _os.environ.get('RAWMOD_DATA', 'data')  # datasets (not shipped: set this to score raw signal)
RAWMOD_WORK = _os.environ.get('RAWMOD_WORK') or f'{REPO}/paper/data/work'  # outputs of the paper scripts; shipped by default
import sys, json, collections
import numpy as np, pysam, pandas as pd
ROOT = _os.environ.get('SPO1_DIR') or f'{RAWMOD_DATA}/SPO1'   # one directory per barcode: reads.tsv (read_id, group), windows.tsv, basecalled/reads_refined.sorted.bam
REF = _os.environ.get('SPO1_REF') or f'{ROOT}/SPO1_FJ230960.1.fasta'
COMP = str.maketrans('ACGT', 'TGCA'); PROF = 10; NMAX = 2500
fa = pysam.FastaFile(REF); G = fa.fetch(fa.references[0]).upper()

def cls_of(rp, rev):
    win = G[rp - PROF:rp + PROF + 1]
    wins = win.translate(COMP)[::-1] if rev else win
    offs = [i - PROF for i, b in enumerate(wins) if b == 'T']
    if not offs: return None
    d = min(abs(o) for o in offs)
    if d == 0: return '0'
    if -d in offs and d in offs: return f'both{d}'
    return str(-d if -d in offs else d)

out = {}
for ds in sys.argv[1:]:
    meta = pd.read_csv(f'{ROOT}/{ds}/reads.tsv', sep='\t'); grp = dict(zip(meta.read_id, meta.group))
    bam = pysam.AlignmentFile(f'{ROOT}/{ds}/basecalled/reads_refined.sorted.bam')
    win = pd.read_csv(f'{ROOT}/{ds}/windows.tsv', sep='	'); inwin = np.zeros(len(G), bool)
    for a, b in zip(win.start, win.end): inwin[int(a):int(b)] = True
    per = collections.Counter(); res = {}
    for r in bam.fetch(until_eof=True):
        if r.is_unmapped or r.is_secondary or r.is_supplementary: continue
        g = grp.get(r.query_name)
        if g is None or per[g] >= NMAX: continue
        per[g] += 1
        seq = r.query_sequence.upper(); rev = r.is_reverse
        nT = nFar = 0; lk = {}
        for qp, rp in r.get_aligned_pairs():
            if rp is None or rp < PROF + 2 or rp > len(G) - PROF - 3: continue
            rb = G[rp]; rbs = rb.translate(COMP) if rev else rb
            ctx = G[rp - 2:rp + 3]; ctxs = ctx.translate(COMP)[::-1] if rev else ctx
            if rbs == 'T': nT += 1
            elif 'T' not in ctxs: nFar += 1
            if not inwin[rp]: continue                  # amplicon windows only
            c = cls_of(rp, rev)
            if c is None: continue
            err = 1 if (qp is None or seq[qp] != rb) else 0
            v = lk.setdefault((rp, rev), [c, 0, 0]); v[1] += 1; v[2] += err
        if nT < 200 or nFar < 200: continue          # same read filter as bc_strain_typing.py
        R = res.setdefault(g, {})                       # per locus (position, strand): class, n, errors
        for k, (c, n, e) in lk.items():
            x = R.setdefault(k, [c, 0, 0]); x[1] += n; x[2] += e
    shared = set.intersection(*(set(R) for R in res.values()))  # loci covered by every group
    out[ds] = {}
    for g, R in res.items():
        agg = {}
        for k in shared:
            c, n, e = R[k]; v = agg.setdefault(c, [0, 0, 0]); v[0] += n; v[1] += e; v[2] += 1
        out[ds][g] = agg
    print(ds, 'shared loci', len(shared), {g: sum(v[0] for v in R.values()) for g, R in out[ds].items()}, flush=True)
_os.makedirs(f'{RAWMOD_WORK}/basecall_errors', exist_ok=True)
json.dump(out, open(f'{RAWMOD_WORK}/basecall_errors/profile_ties_span.json', 'w'), indent=1)
