#!/usr/bin/env python3
"""Figure 1b: per-position difference of the read-median current and dwell time between a modified library and its unmodified control, for every chemistry, relative to the noise floor. Reads the pileup features of matched positions."""
import os as _os
from pathlib import Path as _Path
REPO = str(_Path(__file__).resolve().parents[2])
RAWMOD_RESULTS = _os.environ.get('RAWMOD_RESULTS') or f'{REPO}/paper/data/runs'  # training runs; shipped metrics by default
RAWMOD_DATA = _os.environ.get('RAWMOD_DATA', 'data')  # datasets (not shipped: set this to score raw signal)
RAWMOD_WORK = _os.environ.get('RAWMOD_WORK') or f'{REPO}/paper/data/work'  # outputs of the paper scripts; shipped by default

import re
import numpy as np, h5py, pysam
from pathlib import Path
F = Path(f'{RAWMOD_DATA}/features')
HP_REF = f'{RAWMOD_DATA}/hp26695/ref.fa'  # the benchmark's 26695 reference (same contig names)
HERE = Path(__file__).resolve().parent
MAXN = 3000
rng = np.random.default_rng(0)
COMP = str.maketrans('ACGT', 'TGCA')
# chemistry -> (modified library, unmodified library, centre base, motif regex on read-oriented 9-mer or None, label)
PAIRS = {
    '5mC':  ('ONT_5mC', 'ONT_control', 'C', None),
    '5hmC': ('ONT_5hmC', 'ONT_control', 'C', None),
    '6mA':  ('HP26695_WT_5kHz', 'HP26695_WGA_5kHz', 'A', '..GCATG..'),  # centre (index 4) = A of GCATG
    '4mC':  ('HP26695_WT_5kHz', 'HP26695_WGA_5kHz', 'C', '...TCTTC.'),  # centre (index 4) = first C of TCTTC
    '5hmU': ('barcode07', 'barcode01_test', 'T', None),
}
BASES = 'ACGT'


def index(f, want_label=None):
    names = f['ref_names'][:].astype(str); pos = f['ref_pos'][:]
    keep = f['image_idx'][:] == 0
    if want_label is not None:
        keep &= f['labels'][:] == want_label
    return {(n, int(p)): i for i, (n, p) in enumerate(zip(names, pos)) if keep[i]}


def read_rows(f, idx):
    order = np.argsort(idx); s_idx = np.asarray(idx)[order]
    out = np.empty((len(idx), 16, 210, 9), np.float32)
    for s in range(0, len(s_idx), 256):
        out[order[s:s + 256]] = f['tensors'][s_idx[s:s + 256]]
    return out


def site_profile(x):
    ref = x[:, 0, :, 0]; sig = x[:, 1:, :, 0]
    valid = np.abs(sig).sum(-1) > 1e-6
    d = np.where(valid[..., None], sig - ref[:, None, :], np.nan)
    rank = np.cumsum(valid, 1) - 1
    odd = np.where(((rank % 2) == 1)[..., None], d, np.nan)
    even = np.where(((rank % 2) == 0)[..., None], d, np.nan)
    base = np.array([BASES[k] for k in x[:, 0, 105, 2:6].argmax(-1)])
    return np.nanmedian(d, 1), np.nanmedian(odd, 1), np.nanmedian(even, 1), base


fa = pysam.FastaFile(HP_REF); hp = {n: fa.fetch(n).upper() for n in fa.references}


def motif_ok(rx, n, p, st):
    s = hp[n][p - 4:p + 5]
    s = s if st == 'plus' else s.translate(COMP)[::-1]
    return re.fullmatch(rx, s) is not None


out, examples = {}, {}
for chem, (lm, lu, b, rx) in PAIRS.items():
    parts = []
    for st in ('plus', 'minus'):
        hm, hu = h5py.File(F / f'{lm}_{st}.h5', 'r'), h5py.File(F / f'{lu}_{st}.h5', 'r')
        im = index(hm, None if rx else 1); iu = index(hu)
        keys = sorted(set(im) & set(iu))
        if rx:
            keys = [k for k in keys if motif_ok(rx, *k, st)]
        rng.shuffle(keys); keys = keys[:MAXN * 2]
        if not keys:
            continue
        xm = read_rows(hm, [im[k] for k in keys]); xu = read_rows(hu, [iu[k] for k in keys])
        parts.append((xm, xu, [(st, *k) for k in keys]))
    xm = np.concatenate([p[0] for p in parts]); xu = np.concatenate([p[1] for p in parts])
    keys = sum([p[2] for p in parts], [])
    pm, pm1, pm2, bm = site_profile(xm); pu, pu1, pu2, _ = site_profile(xu)
    sel = np.nonzero(bm == b)[0]
    sel = rng.permutation(sel)[:MAXN]
    out[f'{chem}_mod'] = pm[sel]; out[f'{chem}_unmod'] = pu[sel]
    out[f'{chem}_diff'] = np.nanmean(np.abs(pm1[sel] - pu2[sel]), 0)
    out[f'{chem}_null'] = np.nanmean(np.abs(pu1[sel] - pu2[sel]), 0)
    out[f'{chem}_n'] = len(sel)
    r = (out[f'{chem}_diff'] / out[f'{chem}_null']).reshape(21, 10).mean(1)
    print(f'{chem}: n={len(sel)}  peak offset {int(np.argmax(r)) - 10}  ratio -4..+4', np.round(r[6:15], 2))
    full = (np.abs(xm[sel, 1:, :, 0]).sum(-1) > 1e-6).sum(1) >= 12
    fullu = (np.abs(xu[sel, 1:, :, 0]).sum(-1) > 1e-6).sum(1) >= 12
    score = np.where(full & fullu, np.abs(np.nanmean(pm[sel, 95:115] - pu[sel, 95:115], 1)), -1)
    j = int(np.argmax(score))
    examples[f'{chem}_ex_mod'] = xm[sel[j], :, :, 0]
    examples[f'{chem}_ex_unmod'] = xu[sel[j], :, :, 0]
    examples[f'{chem}_ex_key'] = np.array(keys[sel[j]], dtype=object)
    print('   example', keys[sel[j]], f'contrast {score[j]:.2f}')
np.savez_compressed(HERE / 'signal_contrast.npz', **out, **examples)
