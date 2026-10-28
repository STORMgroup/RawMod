#!/usr/bin/env python3
"""Score EVERY image of the region-only tree with one or more checkpoints and dump per-site scores.

score_holdout_region.py builds one fold's test set and reports one AUROC. This script scores the whole region tree
once per checkpoint (images averaged to one score per source file + position, exactly as evaluate() does) and writes
a per-site table, so test-set definitions can be compared offline without another GPU pass:

  * the original region test set (core members only, every image in the region-only files);
  * in_region: only sites inside the organism's withheld region (rawmod.paths.HOLDOUT_REGIONS). The ONT oligo
    region-only files also carry the reads' images on the 27 training constructs, where the modified position is
    labelled 0 (the ground truth was cut to the region), so this column matters for the oligo rows;
  * BENCH members (E. coli, A. thaliana, human, ...), which the original scorer leaves out.

Usage (region tree in RAWMOD_STRANDRES_ROOT, same env as score_holdout_region.py):
  python dump_region_scores.py --list ckpts.tsv --out-dir DIR
ckpts.tsv: one line per checkpoint, "<tag>\\t<path to best_model.pt>". Output DIR/<tag>.tsv.gz and DIR/sites.tsv.gz.
"""
import argparse
import csv
import gzip
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts' / 'train'))
sys.path.insert(0, str(REPO / 'rawmod'))

import run_matched_loco as M  # noqa: E402
from run_convformer_v2 import ConvFormerV2  # noqa: E402
from rawmod.paths import HOLDOUT_REGIONS  # noqa: E402


def load_regions():
    reg = {}
    with open(HOLDOUT_REGIONS) as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            reg.setdefault(r['contig'], []).append((int(r['start']), int(r['end'])))
    return reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', required=True)
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    root = os.environ.get('RAWMOD_STRANDRES_ROOT', '')
    if 'region_only' not in root:
        raise SystemExit(f'RAWMOD_STRANDRES_ROOT must point at the region-only tree, got {root!r}')
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    hp = M.R.HP()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    members = M.build_members()
    pool = M.R.Group(list(members), members)
    print(f'region pool: {pool.N:,} images across {len(members)} files', flush=True)
    mod_map = M.build_umces_mod_map(M.R.UMCES_PILEUPS, M.R.UMCES_REF)
    refbase = M.ref_base_center(pool)
    chem = M.chem_array(pool, mod_map, refbase)
    idx = np.arange(pool.N, dtype=np.int64)
    keys = pool.source_keys(idx)

  # per-site metadata (one row per source key, in first-seen order, as aggregate_by_position does)
    reg = load_regions()
    site_of, first, lab, chm = {}, [], [], []
    for i, k in enumerate(keys):
        s = site_of.get(k)
        if s is None:
            s = site_of[k] = len(first); first.append(i); lab.append(0); chm.append('')
        if pool.labels[i] > 0:
            lab[s] = 1
            chm[s] = chem[i] or chm[s]
    site = np.array([site_of[k] for k in keys], dtype=np.int64)
    first = np.array(first)
    nimg = np.bincount(site)
    sp = out / 'sites.tsv.gz'
    if not sp.exists():
        with gzip.open(sp, 'wt') as fh:
            fh.write('site\tmember\torg\tcontig\tpos\trefbase\tlabel\tchem\tn_img\tin_region\n')
            for s, i in enumerate(first):
                nm = pool.names[int(pool.file_of[i])]
                c, p = str(pool.contig[i]), int(pool.ref_pos[i])
                inr = any(lo <= p < hi for lo, hi in reg.get(c, []))
                rb = refbase[i].decode() if isinstance(refbase[i], bytes) else str(refbase[i])
                fh.write(f'{s}\t{nm}\t{M.org_of(nm)}\t{c}\t{p}\t{rb}\t{lab[s]}\t{chm[s]}\t{nimg[s]}\t{int(inr)}\n')
        print(f'wrote {sp} ({len(first):,} sites)', flush=True)

    ds = M.R.make_ds(pool, idx, False, hp)
    in_ch = 10 if os.environ.get('RAWMOD_DROP_CH9', '0') == '1' else 11
    for line in open(a.list):
        if not line.strip():
            continue
        tag, ck = line.split()[:2]
        dst = out / f'{tag}.tsv.gz'
        if dst.exists():
            print(f'skip {tag} (exists)', flush=True)
            continue
        model = ConvFormerV2(in_ch=in_ch, dropout=hp.dropout, supcon_dim=int(os.environ.get('SUPCON_DIM', '128')),
                             sad_dim=int(os.environ.get('SAD_DIM', '0')), h=M.HEIGHT,
                             layers=int(os.environ.get('TF_LAYERS', '2')),
                             row_emb=os.environ.get('ROW_EMB', '1') == '1').to(device)
        state = torch.load(ck, map_location=device)
        model.load_state_dict(state.get('model_state', state.get('model', state)))
        model.eval()
        loader = DataLoader(ds, shuffle=False, **M.R.make_loader_kwargs(hp.batch, hp.num_workers, device, M.R._wif))
        _, yp = M.R.run_inference(model, loader, device)
        score = np.bincount(site, weights=yp.astype(np.float64)) / nimg
        tmp = dst.with_suffix('.tmp.gz')
        with gzip.open(tmp, 'wt') as fh:
            fh.write('site\tscore\n')
            fh.writelines(f'{s}\t{v:.6f}\n' for s, v in enumerate(score))
        tmp.rename(dst)
        print(f'DONE {tag} -> {dst}', flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
