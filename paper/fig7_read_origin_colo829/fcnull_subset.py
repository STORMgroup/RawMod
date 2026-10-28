#!/usr/bin/env python3
"""Batch control: keep one sample and relabel its reads by flow cell, so that any separation found is batch rather than biology."""
import argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--npz', required=True)
ap.add_argument('--reads', required=True, help='read_id group flowcell ...')
ap.add_argument('--arm', required=True, help='tumor or normal')
ap.add_argument('--out', required=True)
a = ap.parse_args()
z = np.load(a.npz, allow_pickle=True)
fc = pd.read_csv(a.reads, sep='\t').set_index('read_id').flowcell.to_dict()
rid = z['read_id'].astype(str)
keep = (z['group'].astype(str) == a.arm) & np.array([r in fc for r in rid])
d = {k: z[k] for k in z.files}
for k in ('emb', 'read_id', 'site_i', 'group', 'window'):
    d[k] = z[k][keep]
d['group'] = np.array([fc[r] for r in rid[keep]])
np.savez_compressed(a.out, **d)
print(f'{a.arm}: {keep.sum():,} rows, flow cells {dict(zip(*np.unique(d["group"], return_counts=True)))}')
