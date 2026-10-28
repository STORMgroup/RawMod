#!/usr/bin/env python3
"""Score every site of a genome-wide features.h5 with a RawMod checkpoint.

Writes <out>/<dataset>_scores.tsv.gz with one row per site: contig, pos, score, label,
n_reads (and the site embedding with --embed). The released model is the default
checkpoint; pass --checkpoint to score with another one, for example a leave-out model
when the scored organism must be absent from training (checkpoints/README.md).
"""
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import gzip
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / 'scripts' / 'train', REPO / 'rawmod'):
    sys.path.insert(0, str(_p))
DEFAULT_CHECKPOINT = REPO / 'checkpoints' / 'rawmod.pt'


def load_model(ckpt, device):
    """Rebuild the ConvFormer from a checkpoint written by run_pipeline.py
    ({'model_state': ..., 'in_channels': 11, 'val_auprc': ..., 'epoch': ..., 'tag': ...}).

    Width, depth, height and the training-only heads (SupCon projection, Deep SAD,
    adversary) are inferred from the state dict, so every released checkpoint loads
    without configuration. Returns (model, 'convformer_v2').
    """
    raw = torch.load(str(ckpt), map_location=device, weights_only=False)
    sd = raw['model_state'] if isinstance(raw, dict) and 'model_state' in raw else raw
  # A DANN checkpoint carries 'adv.*'; a SupCon checkpoint carries 'proj.*'.
  # Both auxiliaries are training-only (forward() returns the logit alone at
  # eval), but the submodules must exist for a strict load, so rebuild them.
    has_adv = any(k.startswith('adv.') for k in sd)
    has_proj = any(k.startswith('proj.') for k in sd)
    has_sad = any(k.startswith('sad_head.') for k in sd)
    if has_adv or has_proj or has_sad:
        from run_convformer_v2 import ConvFormerV2
  # infer aux dims from the state dict
        proj_dim = int(sd['proj.net.2.weight'].shape[0]) if has_proj else 0
        sad_dim = int(sd['sad_head.weight'].shape[0]) if has_sad else 0
        height = int(sd['pos'].shape[1])
  # in_channels is saved by run_pipeline.py; fall back to the stem conv's
  # weight shape for older checkpoints that predate that field (e.g. the
  # RAWMOD_DROP_CH9 ablation's 10-channel models need this to not silently
  # default to IN_CH=11 and crash on load_state_dict).
        in_ch = int(raw['in_channels']) if isinstance(raw, dict) and raw.get('in_channels') \
            else int(sd['read_encoder.stem.0.0.weight'].shape[1])
  # size and depth from the weights;
  # the published model gives d_model=96, dim_ff=192, layers=2, no read head
        d_model = int(sd['norm.weight'].shape[0])
        layers = len({k.split('.')[2] for k in sd if k.startswith('encoder.layers.')})
        dim_ff = int(sd['encoder.layers.0.linear1.weight'].shape[0]) if layers else 192
        read_aux = any(k.startswith('read_head.') for k in sd)
        conv_mult = int(sd['read_encoder.stem.0.0.weight'].shape[0]) // 32
        row_emb = True
        model = ConvFormerV2(dropout=0.0, in_ch=in_ch,
                             dann_lambda=1.0 if has_adv else 0.0,  # value irrelevant at eval
                             supcon_dim=proj_dim, sad_dim=sad_dim, h=height,
                             d_model=d_model, dim_ff=dim_ff, layers=layers, read_aux=read_aux, conv_mult=conv_mult,
                             row_emb=row_emb)
        model.load_state_dict(sd)
        model.to(device).eval()
        aux = []
        if has_adv:
            aux.append('DANN adversary')
        if has_proj:
            aux.append(f'SupCon proj dim={proj_dim}')
        if has_sad:
            aux.append(f'DeepSAD dim={sad_dim}')
        print(f"  checkpoint: {' + '.join(aux)} present  h={height} in_ch={in_ch} "
              f"epoch={raw.get('epoch')} val_auprc={raw.get('val_auprc')}", flush=True)
        return model, 'convformer_v2'

    raise SystemExit(f'{ckpt}: not a RawMod ConvFormer checkpoint (no proj/sad_head/adv weights)')


def score(h5_path, ckpt, device, batch=512, workers=8, want_embed=False,
          legacy_ch9=False):
    """Stream the whole file through the model. Returns (scores, embeds|None).

    legacy_ch9=True computes the ch9 delta at the k-mer centre (attrs['center_idx'])
    rather than the window centre, to MATCH models trained before the ch9 fix. Use
    it for checkpoints of the earliest pipeline; leave False for models trained after the
    fix (e.g. the pipeline3 ablation arms).
    """
    from model import PileupDataset
    with h5py.File(h5_path, 'r') as h:
        n = h['tensors'].shape[0]
    ds = PileupDataset([str(h5_path)], np.arange(n, dtype=np.int64), [n],
                       augment=False, seed=0, signal_noise_std=0.0,
                       delta_channels=True, preload=False,
                       legacy_ch9_center=legacy_ch9,
  # PILEUP_MASK_BASES=1 for checkpoints trained with base
  # identity blanked (base-masked models); default off.
                       mask_all_bases=os.environ.get('PILEUP_MASK_BASES', '0') == '1')
    from torch.utils.data import DataLoader
    from model import make_loader_kwargs, _worker_init_fn
    try:
        lk = make_loader_kwargs(batch, workers, device, _worker_init_fn)
        loader = DataLoader(ds, shuffle=False, **lk)
    except Exception:
        loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers)

    model, arch = load_model(ckpt, device)

  # Where the 512/96-dim penultimate embedding lives differs by architecture:
  #   - ConvFormerV2/ConvFormer: self.head has no internal pooling, so the
  #     INPUT to self.head (= self.norm(pooled)) is already the flat embedding.
  #   - PileupInceptionV3 ('inception'): self.head is
  #     Sequential(AdaptiveAvgPool2d, Flatten, Dropout, Linear) -- the pool
  #     is INSIDE head, so hooking head itself would capture the pre-pool
  #     (B, 512, 1, W) conv feature map, not an embedding. Hook head[3] (the
  #     final Linear) instead; its INPUT is the pooled+flattened+dropout
  #     (B, 512) embedding (see model.py's dual-output forward for SupCon,
  #     which taps the identical point via self.head[2](...)).
    cap = {}
    hook = None
    if want_embed:
        def _grab(_mod, inp, _out):
            cap['e'] = inp[0].detach()
        target = model.head[3] if arch == 'inception' else model.head
        hook = target.register_forward_hook(_grab)

    out = np.empty(n, dtype=np.float32)
    embeds = [] if want_embed else None
    i = 0
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            logit = model(xb)
            if want_embed:
                embeds.append(cap['e'].float().cpu().numpy())
            p = torch.sigmoid(logit.squeeze(-1)).float().cpu().numpy()
            out[i:i + len(p)] = p
            i += len(p)
    if hook is not None:
        hook.remove()
    if want_embed and embeds:
        embeds = np.concatenate(embeds, 0)
    return out, embeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--checkpoint', default=None,
                    help='model to score with (default: checkpoints/rawmod.pt)')
    ap.add_argument('--tag', default=None, help='suffix for the output file')
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--save-embeddings', action='store_true')
    ap.add_argument('--legacy-ch9', action='store_true',
                    help='compute ch9 at the k-mer centre, to match models trained '
                         'before the deviation-channel fix (earliest pipeline)')
    a = ap.parse_args()

    ckpt = Path(a.checkpoint) if a.checkpoint else DEFAULT_CHECKPOINT
    if ckpt is None or not Path(ckpt).exists():
        raise SystemExit(f"no checkpoint for {a.dataset!r} (looked at {ckpt})")
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"scoring {a.dataset}\n  h5   = {a.h5}\n  ckpt = {ckpt}\n  dev  = {device}",
          flush=True)

    if a.legacy_ch9:
        print("  ch9: LEGACY (k-mer centre) to match a pre-fix checkpoint", flush=True)
    s, emb = score(a.h5, ckpt, device, a.batch, a.workers, a.save_embeddings,
                   legacy_ch9=a.legacy_ch9)

    with h5py.File(a.h5, 'r') as h:
        contigs = [x.decode() if isinstance(x, bytes) else str(x)
                   for x in h['ref_names'][:]]
        pos = h['ref_pos'][:]
        lab = h['labels'][:]
        nrd = h['n_reads'][:] if 'n_reads' in h else np.zeros(len(pos), dtype=int)

    tag = f"_{a.tag}" if a.tag else ""
    fout = out / f"{a.dataset}{tag}_scores.tsv.gz"
    with gzip.open(fout, 'wt') as fh:
        fh.write("contig\tpos\tscore\tlabel\tn_reads\n")
        for c, p, sc, l, nr in zip(contigs, pos, s, lab, nrd):
            fh.write(f"{c}\t{int(p)}\t{sc:.6f}\t{int(l)}\t{int(nr)}\n")
    print(f"wrote {fout}  ({len(s):,} sites)  mean_score={s.mean():.4f}", flush=True)

    if emb is not None:
        np.save(out / f"{a.dataset}{tag}_embed.npy", emb.astype(np.float16))
        print(f"wrote embeddings {emb.shape}", flush=True)


if __name__ == '__main__':
    main()
