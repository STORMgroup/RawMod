"""Intergenic regions of the contig classified by the strands of their flanking genes (within-array, divergent, convergent), with their mean score, length, GC and repeat content."""
import os as _os
from pathlib import Path as _Path
REPO = str(_Path(__file__).resolve().parents[2])
RAWMOD_RESULTS = _os.environ.get('RAWMOD_RESULTS') or f'{REPO}/paper/data/runs'  # training runs; shipped metrics by default
RAWMOD_DATA = _os.environ.get('RAWMOD_DATA', 'data')  # datasets (not shipped: set this to score raw signal)
RAWMOD_WORK = _os.environ.get('RAWMOD_WORK') or f'{REPO}/paper/data/work'  # outputs of the paper scripts; shipped by default

import numpy as np, pandas as pd
from scipy.stats import spearmanr
BASE=f'{RAWMOD_DATA}/acarterae_contig_3762'
df=pd.read_csv(f'{RAWMOD_WORK}/dinoflagellate/contig_3762_strand+_scores.tsv.gz',sep='\t')
pos=df.pos.values; sc=df.score.values
rows=[l.rstrip('\n').split('\t') for l in open(f'{BASE}/Annotations_contig_3762.gff3') if not l.startswith('#') and l.strip()]
g=sorted([(int(p[3])-1,int(p[4]),p[6]) for p in rows if p[2]=='gene'])
seq=''.join(l.strip() for l in open(f'{BASE}/contig_3762.fasta') if not l.startswith('>')).upper()
out=[]
run_end=g[0][1]
for i in range(1,len(g)):
    s,e=run_end,g[i][0]
    run_end=max(run_end,g[i][1]) if g[i][2]==g[i-1][2] else g[i][1]
    if e-s<1000: continue
    sel=(pos>=s)&(pos<e)
    if sel.sum()<100: continue
    kind='within' if g[i][2]==g[i-1][2] else ('convergent' if g[i-1][2]=='+' else 'divergent')
    sub=seq[s:e]; gc=(sub.count('G')+sub.count('C'))/len(sub)
    out.append(dict(kind=kind,start=s,end=e,length=e-s,n=int(sel.sum()),mean=float(sc[sel].mean()),frac05=float((sc[sel]>.5).mean()),gc=gc))
d=pd.DataFrame(out)
print(d[d.kind!='within'])
w=d[d.kind=='within']
print('within gaps',len(w),'length-mean rho',spearmanr(w.length,w['mean']))
print('within gaps >=20kb:',(w.length>=20000).sum(), w[w.length>=20000]['mean'].describe())
print(w.sort_values('mean',ascending=False).head(8))
print('gc-mean rho within', spearmanr(w.gc,w['mean']))
d.to_csv('dino_intergenic_gaps.tsv',sep='\t',index=False)
# within-divergent profile: 5kb bins
for _,r in d[d.kind=='divergent'].iterrows():
    b=np.arange(r.start,r.end,5000)
    print('div',r.start, [round(float(sc[(pos>=x)&(pos<x+5000)].mean()),2) for x in b])
for _,r in d[d.kind=='convergent'].iterrows():
    b=np.arange(r.start,r.end,5000)
    print('conv',r.start, [round(float(sc[(pos>=x)&(pos<x+5000)].mean()),2) for x in b])
