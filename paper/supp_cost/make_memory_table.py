#!/usr/bin/env python3
"""Summarize memory_scaling.sh: per number of processes, wall time and peak memory of the fused featurization +
scoring of both strands (Supplementary Section S2.2), next to Dorado's basecalling pass.

  python make_memory_table.py [TIMING_TSV]    (default: the shipped paper/data/work/cost/memory_scaling.tsv)
"""
import re
import sys
from pathlib import Path

import pandas as pd

DEFAULT = Path(__file__).resolve().parents[1] / 'data' / 'work' / 'cost' / 'memory_scaling.tsv'


def main(path):
    t = pd.read_csv(path, sep='\t').set_index('stage')
    rows = []
    for st in t.index:
        m = re.fullmatch(r'fused_(plus|minus)_x(\d+)_gpu', st)
        if m:
            rows.append(dict(procs=int(m.group(2)), strand=m.group(1), **t.loc[st]))
    f = pd.DataFrame(rows)
    g = f.groupby('procs').agg(wall_min=('wall_s', lambda x: x.sum() / 60), host_gb=('peak_pss_sum_kb', lambda x: x.max() / 1e6),
                               gpu_gb=('gpu_peak_mib', lambda x: x.max() / 1024), gpu_busy_min=('gpu_busy_s', lambda x: x.sum() / 60))
    print('RawMod featurization + scoring, both strands')
    print(g.round(2).to_string())
    if 'dorado_basecall' in t.index:
        d = t.loc['dorado_basecall']
        print(f"Dorado basecall: {d.wall_s / 60:.1f} min, host {d.peak_pss_sum_kb / 1e6:.1f} GB, GPU {d.gpu_peak_mib / 1024:.1f} GB")
    if 'torch_cuda_baseline' in t.index:
        print(f"idle torch process with CUDA: host {t.loc['torch_cuda_baseline', 'peak_pss_sum_kb'] / 1e6:.2f} GB")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
