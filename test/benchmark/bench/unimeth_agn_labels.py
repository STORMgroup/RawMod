#!/usr/bin/env python3
"""Per-read labels for a modification-agnostic UniMeth fine-tune (UniMeth-Agn) at EXACTLY the training images of the
benchmarked RawMod checkpoint.

Each training image of RawMod (bench.holdout source table, split == 'train') is a (library, contig, pos, strand) with a
binary label, whatever the modification. Here every read of the library's benchmark read set inherits that label at
that base: ML 255 if the image is a positive, ML 0 if it is a negative. A control library (unmodified everywhere) is
labelled 0 at every training position of its organism. Labels are written under one code per base
(C+m, A+a, T+g; read orientation), which the patched UniMeth reads together with --agnostic 1. Bases without a training
image, and G (UniMeth has no question token for G), are left out of the MM tag ('?' mode), so they carry no loss and the
benchmark's held-out positions stay unseen. With --exclude-region (the region-holdout benchmark), every read whose
alignment overlaps the library's held-out region (config/holdout_region.tsv) is dropped, as for the region UniMeth-FT.

  python -m bench.unimeth_agn_labels --sample syn_5hmC --out syn_5hmC.agn.bam [--exclude-region]
"""
import argparse
import os
from array import array

import numpy as np
import pandas as pd
import pysam

from . import common as C
from .holdout import SEEN, _region_for
from .settings import DATA_DIR

# benchmark sample -> regex of its training-pool files (one library each; both strand files)
FILES = {'syn_5mC': r'^ONT::5mC\|', 'syn_5hmC': r'^ONT::5hmC\|', 'syn_6mA': r'^ONT::6mA\|', 'syn_control': r'^ONT::control\|',
         'hp26695': r'^HP::WT\|', 'hp26695_wga': r'^HP::WGA\|', 'spo1_bc07': r'^SPO1::bc07\|', 'spo1_bc01': r'^SPO1::bc01\|',
         'ecoli_wt_w1m': r'^BENCH::Ecoli_WT_5kHz\|', 'ecoli_dm_w1m': r'^BENCH::Ecoli_DM_5kHz\|',
         'ecoli_mssi_w1m': r'^BENCH::Ecoli_DM_MSssI_5kHz\|', 'anabaena': r'^BENCH::Anabaena_WT_5kHz\|',
         'hpj99': r'^BENCH::HPJ99_WT_5kHz\|', 'tdenticola': r'^BENCH::Tdenticola_WT_5kHz\|',
         'arabidopsis_chr1': r'^BENCH::arabidopsis\|', 'hg002': r'^BENCH::hg002\|'}
# control libraries carry no modification anywhere, so they are labelled unmodified at every training position of the
# organism (any library), not only at their own images -- as in UniMeth-FT. Without this, SPO1 has almost no
# unmodified T: RawMod's other PCR barcodes (bc02-05) are not among the benchmark read sets.
CONTROL = {'syn_control': r'^ONT::', 'hp26695_wga': r'^HP::', 'spo1_bc01': r'^SPO1::', 'ecoli_dm_w1m': r'^BENCH::Ecoli_'}
CODE = {'C': 'm', 'A': 'a', 'T': 'g'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sample', required=True, choices=sorted(FILES))
    ap.add_argument('--out', required=True)
    ap.add_argument('--exclude-region', action='store_true', help="drop every read overlapping the library's held-out "
                    'region (config/holdout_region.tsv); for the region-holdout benchmark')
    a = ap.parse_args()
    excl = (_region_for(a.sample) or {}) if a.exclude_region else {}

    d = pd.read_csv(DATA_DIR / 'holdout' / SEEN['positions'], sep='\t', dtype={'contig': str})
    d = d[(d['split'] == 'train') & d['file'].str.contains(CONTROL.get(a.sample, FILES[a.sample]), regex=True)]
    if a.sample in CONTROL:
        d = d.assign(label=0)
    lab = {(c, int(p), s): (255 if l > 0 else 0) for c, p, s, l in zip(d['contig'], d['pos'], d['strand'], d['label'])}
    print(f'{a.sample}: {len(lab):,} training images ({sum(v == 255 for v in lab.values()):,} positive)', flush=True)

    src = C.samples()[a.sample]['sub_bam']
    tmp = a.out + '.unsorted.bam'
    n_reads = n_mv = n_ex = 0
    n_lab = {b: [0, 0] for b in CODE}; n_g = 0; seen = set()
    with pysam.AlignmentFile(src, 'rb') as fin, pysam.AlignmentFile(tmp, 'wb', template=fin) as fout:
        for r in fin:
            if r.is_unmapped or r.is_secondary or r.is_supplementary:
                continue
            if any(s0 < r.reference_end and r.reference_start < e0 for s0, e0 in excl.get(r.reference_name, ())):
                n_ex += 1; continue  # overlaps the held-out test region
            if r.has_tag('mv') and sum(r.get_tag('mv')[1:]) != r.query_length:
                n_mv += 1; continue  # UniMeth's extractor rejects these (move table != sequence)
            fseq = (r.get_forward_sequence() or '').upper()
            if not fseq:
                continue
            L, st, ctg = len(fseq), '-' if r.is_reverse else '+', r.reference_name
            q2r = dict(r.get_aligned_pairs(matches_only=True))
            mm, ml = [], array('B')
            for b in CODE:  # one MM section per base, in this order
                keep, vals = [], []
                for j, i in enumerate(i for i, x in enumerate(fseq) if x == b):
                    p = q2r.get(L - 1 - i if r.is_reverse else i)
                    v = None if p is None else lab.get((ctg, p, st))
                    if v is not None:
                        keep.append(j); vals.append(v); n_lab[b][v == 255] += 1; seen.add((ctg, p, st))
                if keep:
                    skips = np.diff(np.r_[-1, keep]) - 1
                    mm.append(f"{b}+{CODE[b]}?," + ','.join(map(str, skips)) + ';'); ml.extend(vals)
            for i, x in enumerate(fseq):  # images whose read base is G cannot be labelled
                if x == 'G':
                    p = q2r.get(L - 1 - i if r.is_reverse else i)
                    if p is not None and (ctg, p, st) in lab:
                        n_g += 1
            r.set_tag('MM', None); r.set_tag('ML', None)
            if mm:
                r.set_tag('MM', ''.join(mm), 'Z'); r.set_tag('ML', ml)
            fout.write(r); n_reads += 1
    pysam.sort('-@', '4', '-o', a.out, tmp); os.remove(tmp); pysam.index(a.out)
    per = ', '.join(f'{b} {n[1]:,}+/{n[0]:,}-' for b, n in n_lab.items())
    print(f'{a.out}: {n_reads:,} reads; labelled read bases {per}; {len(seen):,} of {len(lab):,} images covered by a '
          f'read; {n_g:,} read bases at an image are G (unlabelled); {n_mv:,} reads dropped for a move table not '
          f'matching the sequence; {n_ex:,} reads dropped for overlapping the held-out region')


if __name__ == '__main__':
    main()
