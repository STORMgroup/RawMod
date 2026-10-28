#!/usr/bin/env python3
"""Scores by gene strand relative to the scored strand, and by 5-mer context, on both strands of the contig."""
import os as _os
from pathlib import Path as _Path
REPO = str(_Path(__file__).resolve().parents[2])
RAWMOD_RESULTS = _os.environ.get('RAWMOD_RESULTS') or f'{REPO}/paper/data/runs'  # training runs; shipped metrics by default
RAWMOD_DATA = _os.environ.get('RAWMOD_DATA', 'data')  # datasets (not shipped: set this to score raw signal)
RAWMOD_WORK = _os.environ.get('RAWMOD_WORK') or f'{REPO}/paper/data/work'  # outputs of the paper scripts; shipped by default

import json
import numpy as np, pandas as pd
BASE = f'{RAWMOD_DATA}/acarterae_contig_3762'
W = f'{RAWMOD_WORK}/dinoflagellate/strand'
F = {'plus': f'{RAWMOD_WORK}/dinoflagellate/contig_3762_strand+_scores.tsv.gz', 'minus': f'{RAWMOD_WORK}/dinoflagellate/contig_3762_strand-_scores.tsv.gz'}
CATS = ['intergenic', 'intron', 'UTR', 'CDS']
seq = ''.join(l.strip() for l in open(f'{BASE}/contig_3762.fasta') if not l.startswith('>')).upper()
N = len(seq); sb = np.frombuffer(seq.encode(), dtype=np.uint8)
rows = []
for line in open(f'{BASE}/Annotations_contig_3762.gff3'):
    if line.startswith('#') or not line.strip():
        continue
    p = line.rstrip('\n').split('\t'); rows.append((p[2], int(p[3]) - 1, int(p[4]), p[6]))
gff = pd.DataFrame(rows, columns=['type', 'start', 'end', 'strand'])
lab = np.zeros(N, dtype=np.int8); gs = np.zeros(N, dtype=np.int8)  # gene strand: +1 / -1 / 0 intergenic, 2 both
for t, code in (('gene', 1), ('exon', 2), ('CDS', 3)):
    for s, e in gff.loc[gff.type == t, ['start', 'end']].values:
        lab[s:e] = code
for s, e, st in gff.loc[gff.type == 'gene', ['start', 'end', 'strand']].values:
    v = 1 if st == '+' else -1
    seg = gs[s:e]; seg[(seg != 0) & (seg != v)] = 2; seg[seg == 0] = v
code = np.full(256, -1, dtype=np.int64)
for i, b in enumerate(b'ACGT'):
    code[b] = i
enc = code[sb]
cs = np.concatenate([[0], np.cumsum((sb == ord('G')) | (sb == ord('C')))])
out = {}
for nm, f in F.items():
    d = pd.read_csv(f, sep='\t'); pos = d.pos.values.astype(np.int64); sc = d.score.values.astype(np.float64)
    cat = lab[pos]; g = gs[pos]
    r = {}
    for ci, cn in enumerate(CATS):
        for gv, gn in ((1, 'gene on +'), (-1, 'gene on -'), (2, 'overlapping +/-')):
            sel = (cat == ci) & (g == gv)
            if sel.sum() > 100:
                r[f'{cn} | {gn}'] = dict(n=int(sel.sum()), mean=float(sc[sel].mean()))
    r['intergenic'] = dict(n=int((cat == 0).sum()), mean=float(sc[cat == 0].mean()))
  # which physical strand carries the scanned T relative to transcription:
  # + scan: gene on + -> T on coding (sense) strand; gene on - -> template.  - scan: the reverse.
    sense = (g == 1) if nm == 'plus' else (g == -1); templ = (g == -1) if nm == 'plus' else (g == 1)
    for ci, cn in ((1, 'intron'), (3, 'CDS')):
        r[f'{cn} | T on sense strand'] = dict(n=int(((cat == ci) & sense).sum()), mean=float(sc[(cat == ci) & sense].mean()))
        r[f'{cn} | T on template strand'] = dict(n=int(((cat == ci) & templ).sum()), mean=float(sc[(cat == ci) & templ].mean()))
  # context adjustment (dino_analysis.py step 2)
    ok = np.ones(len(pos), bool); kmer = np.zeros(len(pos), dtype=np.int64)
    for off in (-2, -1, 1, 2):
        q = pos + off; good = (q >= 0) & (q < N); v = np.where(good, enc[np.clip(q, 0, N - 1)], -1)
        ok &= v >= 0; kmer = kmer * 4 + np.maximum(v, 0)
    lo, hi = np.clip(pos - 50, 0, N), np.clip(pos + 51, 0, N)
    gc = (cs[hi] - cs[lo]) / (hi - lo); gcbin = np.minimum((gc * 20).astype(int), 19)
    adj = {}
    for name, key in (('none', np.zeros(len(pos), dtype=np.int64)), ('5mer', kmer), ('gc50', gcbin), ('5mer_x_gc50', kmer * 20 + gcbin)):
        kk = key[ok]; s = np.bincount(kk, weights=sc[ok]); c = np.bincount(kk)
        expect = s[kk] / c[kk]; resid = sc[ok] - expect + sc[ok].mean()
        mm = np.bincount(cat[ok], weights=resid, minlength=4) / np.bincount(cat[ok], minlength=4)
        r2 = 1 - ((sc[ok] - expect) ** 2).sum() / ((sc[ok] - sc[ok].mean()) ** 2).sum()
        adj[name] = dict(r2=float(r2), **{k: float(mm[i]) for i, k in enumerate(CATS)}, fold_ig_vs_cds=float(mm[0] / mm[3]))
    r['context_adjusted'] = adj
    r['gc50_by_category'] = {k: float(gc[cat == i].mean()) for i, k in enumerate(CATS)}
    out[nm] = r
    print(f'=== {nm}')
    for k, v in r.items():
        if k not in ('context_adjusted', 'gc50_by_category'):
            print(f'  {k:32s} n={v["n"]:>9,} mean={v["mean"]:.4f}')
    print('  gc50 by category', {k: round(v, 3) for k, v in r['gc50_by_category'].items()})
    for k, v in adj.items():
        print(f'  adj {k:12s} r2={v["r2"]:.3f} ' + ' '.join(f'{c}={v[c]:.4f}' for c in CATS) + f' fold_ig/cds={v["fold_ig_vs_cds"]:.3f}')
json.dump(out, open(f'{W}/genestrand_and_context.json', 'w'), indent=1)
