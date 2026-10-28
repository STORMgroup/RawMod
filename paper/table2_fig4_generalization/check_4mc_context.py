#!/usr/bin/env python3
"""Sequence context of the zero-shot 4mC test set, and its AUROC by context.

The 4mC positives are H. pylori 26695 cytosines typed per site. A motif-annotated label set would instead
mark GCATG cytosines, whose signal comes from the 6mA on the neighbouring adenine. This script counts the
positives per motif and repeats the loco_4mC AUROC (a) per motif and (b) on positives and negatives more
than 3 bp from any labelled 6mA, so that transfer from a neighbouring 6mA would show as a drop.

Inputs are the per-site score dumps that region_sites_analyze.py reads (sites.tsv.gz and
loco_4mC_s<seed>.tsv.gz from dump_region_scores.py) and the H. pylori 26695 reference FASTA.

  python check_4mc_context.py SCORES_DIR hpylori_26695.fa
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

SEEDS = (42, 43, 44)
COMP = str.maketrans('ACGT', 'TGCA')


def read_fasta(path):
    with open(path) as f:
        return ''.join(l.strip() for l in f if not l.startswith('>')).upper()


def motif(c):
    """c is an 11-mer centred on the scored C, in read orientation."""
    if c[4:9] == 'TCTTC':
        return 'TCTTC'
    if c[4:9] == 'GCATG':
        return 'GCATG'
    if c[3:8] == 'CCTC' or c[5:9] == 'CCTC':
        return 'xCCTC'
    if 'GCGC' in (c[2:6], c[3:7], c[4:8]):
        return 'GCGC'
    return 'other'


def main(scores_dir, fasta):
    d = Path(scores_dir)
    ref = read_fasta(fasta)
    s = pd.read_csv(d / 'sites.tsv.gz', sep='\t', keep_default_na=False, dtype={'chem': str})
    s = s[s.member.str.startswith('HP::') & (s.in_region == 1)].copy()
    s['strand'] = s.member.str[-1]
    s['ctx'] = [ref[p - 5:p + 6] if st == '+' else ref[p - 5:p + 6].translate(COMP)[::-1]
                for p, st in zip(s.pos, s.strand)]
    assert (s.ctx.str[5] == s.refbase).all(), 'reference does not match the scored bases'
    a6 = set(s.pos[s.chem == '6mA'])
    s['near6mA'] = [any(p + k in a6 for k in (-3, -2, -1, 1, 2, 3)) for p in s.pos]
    s['motif'] = s.ctx.map(motif)
    t = s[((s.label == 1) & (s.chem == '4mC')) | ((s.label == 0) & (s.refbase == 'C'))].copy()
    pos = t[t.label == 1]
    print(f'positives {len(pos)}, negatives {(t.label == 0).sum()}')
    print(pos.motif.value_counts().to_string())
    print(f'positives within 3 bp of a labelled 6mA: {pos.near6mA.sum()} ({pos.near6mA.mean():.1%})')
    rows = []
    for seed in SEEDS:
        sc = pd.read_csv(d / f'loco_4mC_s{seed}.tsv.gz', sep='\t').set_index('site').score
        t['score'] = sc.reindex(t.site).to_numpy()
        subsets = {'all': t.label >= 0, 'far from 6mA (both classes)': ~t.near6mA}
        subsets.update({f'positives {m}': (t.label == 0) | (t.motif == m)
                        for m in ('TCTTC', 'xCCTC', 'GCGC', 'other')})
        for name, m in subsets.items():
            g = t[m]
            rows.append(dict(seed=seed, subset=name, n_pos=int(g.label.sum()), auroc=roc_auc_score(g.label, g.score)))
    r = pd.DataFrame(rows)
    print(r.pivot_table(index='subset', columns='seed', values='auroc').assign(
        mean=lambda x: x.mean(axis=1)).round(3).to_string())


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
