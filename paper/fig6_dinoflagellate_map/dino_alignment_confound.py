"""Alignment controls for the contig map: score against read mapping quality, coverage and distance to alignment ends."""
import os as _os
from pathlib import Path as _Path
REPO = str(_Path(__file__).resolve().parents[2])
RAWMOD_RESULTS = _os.environ.get('RAWMOD_RESULTS') or f'{REPO}/paper/data/runs'  # training runs; shipped metrics by default
RAWMOD_DATA = _os.environ.get('RAWMOD_DATA', 'data')  # datasets (not shipped: set this to score raw signal)
RAWMOD_WORK = _os.environ.get('RAWMOD_WORK') or f'{REPO}/paper/data/work'  # outputs of the paper scripts; shipped by default

import numpy as np, pysam, json
from scipy.stats import spearmanr, rankdata
BASE=f'{RAWMOD_DATA}/acarterae_contig_3762'; OUT='.'
N=7733821; WIN=10_000; nw=N//WIN+1
mq=np.zeros(nw); err=np.zeros(nw); cov=np.zeros(nw)
nreads=0
with pysam.AlignmentFile(f'{BASE}/preprocess/reads.bam','rb',check_sq=False) as bam:
    for r in bam.fetch(until_eof=True):
        if r.is_unmapped or r.is_secondary or r.is_supplementary or r.is_reverse: continue
        if r.reference_name!='contig_3762': continue
        s,e=r.reference_start,r.reference_end
        alen=r.query_alignment_length or 1
        e_rate=r.get_tag('NM')/alen if r.has_tag('NM') else np.nan
        w0,w1=s//WIN,(e-1)//WIN
        for w in range(w0,w1+1):
            ov=min(e,(w+1)*WIN)-max(s,w*WIN)
            cov[w]+=ov; mq[w]+=ov*r.mapping_quality; err[w]+=ov*e_rate
        nreads+=1
mq/=np.maximum(cov,1); err/=np.maximum(cov,1); depth=cov/WIN
c=np.load('dino_cache.npz'); rc=np.load('dino_repeat_cache.npz')
wm=c['wmean']; ok=~np.isnan(wm)&(depth>0)
def pcor(x,y,zs):
    X=np.column_stack([rankdata(z[ok]) for z in zs]+[np.ones(ok.sum())])
    rx=rankdata(x[ok])-X@np.linalg.lstsq(X,rankdata(x[ok]),rcond=None)[0]
    ry=rankdata(y[ok])-X@np.linalg.lstsq(X,rankdata(y[ok]),rcond=None)[0]
    return float(np.corrcoef(rx,ry)[0,1])
st=dict(n_fwd_reads=nreads,
  rho_score_mapq=float(spearmanr(wm[ok],mq[ok])[0]),
  rho_score_err=float(spearmanr(wm[ok],err[ok])[0]),
  rho_score_depth=float(spearmanr(wm[ok],depth[ok])[0]),
  rho_score_repeat=float(spearmanr(wm[ok],rc['wrep'][ok])[0]),
  rho_score_young=float(spearmanr(wm[ok],rc['wyoung'][ok])[0]),
  rho_score_cds=float(spearmanr(wm[ok],c['wcds'][ok])[0]),
  prho_repeat_given_align=pcor(wm,rc['wrep'],[mq,err,depth,c['wgc']]),
  prho_young_given_align=pcor(wm,rc['wyoung'],[mq,err,depth,c['wgc']]),
  prho_cds_given_align_repeat=pcor(wm,c['wcds'],[mq,err,depth,c['wgc'],rc['wrep']]),
  median_mapq=float(np.median(mq[ok])), median_err=float(np.median(err[ok])))
print(json.dumps(st,indent=1))
json.dump(st,open('dino_alignment_stats.json','w'),indent=1)
np.savez_compressed('dino_align_cache.npz',mq=mq,err=err,depth=depth)
