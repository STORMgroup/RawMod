#!/usr/bin/env python3
"""Data pipeline shared by the RawMod trainers and scorers.

PileupDataset reads the featurized HDF5 files (one pileup image per site and strand),
make_loader_kwargs / _worker_init_fn configure the DataLoader, split_position_groups
splits sites so that no reference position is shared between training, validation
and test, and ProjectionHead / SupConLoss are the supervised-contrastive head and
loss. run_inference and aggregate_by_position score a loader and average the images
of each position; the remaining helpers build the position keys and labels they use.
"""

import os
import sys
import time
import random
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al. 2020, arXiv:2004.11362).

    Pulls together L2-normalised embeddings that share the same binary label
    (modified / unmodified) regardless of which H5 source file they came from,
    and pushes apart embeddings with different labels.

    This directly counteracts genomic-context leakage: to satisfy the loss the
    encoder must find signal-level features shared between a modified UMCES site
    and a modified ONT site (cross-dataset positive pairs) rather than
    memorising dataset-specific context such as genomic region identity.

    Parameters
    ----------
    temperature : τ — scales the cosine-similarity logits before softmax.
                  Lower values create sharper distributions (harder negatives
                  dominate). 0.07 is the value used in the SupCon paper.

    Inputs
    ------
    features : (N, D) float32 tensor of L2-normalised embeddings
               (output of ProjectionHead).
    labels   : (N,)  int64 tensor, binary {0=unmod, 1=mod}.

    Returns
    -------
    Scalar loss. Returns 0.0 (detached) when every sample in the batch shares
    the same label (no valid positive pairs exist).
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        N = features.shape[0]

  # Pairwise cosine similarity scaled by temperature: (N, N)
        sim = torch.matmul(features, features.T) / self.temperature

  # Positive mask: same label, excluding diagonal (self)
        lbl = labels.view(-1, 1)
        pos_mask = (lbl == lbl.T).float()
        pos_mask.fill_diagonal_(0.0)

  # Numerical stability: subtract per-row max before exp
        sim = sim - sim.detach().max(dim=1, keepdim=True).values

        self_mask = torch.eye(N, dtype=torch.bool, device=device)
        exp_sim   = torch.exp(sim).masked_fill(self_mask, 0.0)
        denom     = exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-8)

        log_prob  = sim - torch.log(denom)  # (N, N)

        n_pos = pos_mask.sum(dim=1)  # (N,)
        loss  = -(pos_mask * log_prob).sum(dim=1) / n_pos.clamp(min=1)

        valid = n_pos > 0
        if not valid.any():
            return torch.tensor(0.0, device=device, requires_grad=True)
        return loss[valid].mean()


_H5_HANDLES: dict = {}


def _worker_init_fn(worker_id: int):
    """Called once per DataLoader worker at startup — resets the handle cache."""
    global _H5_HANDLES
    _H5_HANDLES = {}


def _get_h5(path: str) -> h5py.File:
    """Return a cached open h5py.File for this process, opening if necessary."""
    if path not in _H5_HANDLES:
        _H5_HANDLES[path] = h5py.File(path, 'r')
    return _H5_HANDLES[path]


def _drop_h5(path: str) -> None:
    """Forget (and close) this process's cached handle for path."""
    hf = _H5_HANDLES.pop(path, None)
    if hf is not None:
        try:
            hf.close()
        except Exception:
            pass


def _h5_read(path: str, local_idx: int, attempts: int = 5):
    """Read one image + label, retrying a transient filesystem error.

    The feature files live on a shared NFS/Isilon mount. Under heavy concurrent
    load (a whole LOCO battery streaming random reads at once) a read can come
    back as OSError errno 5 even though the file is intact -- 14 of 18 training
    jobs died that way, each on a different file, within minutes
    of all starting together. Reopening the handle and retrying with a short
    backoff turns that into a pause instead of a lost job.
    """
    for attempt in range(attempts):
        try:
            hf = _get_h5(path)
            return (hf['tensors'][local_idx].astype(np.float32),
                    float(hf['labels'][local_idx]))
        except (OSError, KeyError) as e:
            _drop_h5(path)
            if attempt == attempts - 1:
                raise
            print(f"[pileup] read retry {attempt + 1}/{attempts - 1} on "
                  f"{os.path.basename(path)}[{local_idx}]: {e}", file=sys.stderr,
                  flush=True)
            time.sleep(0.5 * 2 ** attempt)


def add_delta_channels(x: np.ndarray, center_idx: int, L: int, drop_ch9: bool = False) -> np.ndarray:
    """Append the deviation channels the model is trained with to one pileup image.

    x is (C, H, W*L) channels-first with the raw signal in channel 0, the reference
    row (expected k-mer level) at row 0 and one read per further row.
      Ch 9  per-read deviation at the centre base, broadcast along the row
            (the analogue of DeepVariant's read_supports_variant)
      Ch 10 per-sample deviation of every read from the expected level
    drop_ch9 appends only Ch 10. Scoring code that builds pileups in memory
    (paper/fig7_read_origin_colo829/emb_all_reads.py) calls this too, so the
    channels match the training features exactly.
    """
    ch10 = np.zeros((1, x.shape[1], x.shape[2]), dtype=np.float32)
    ch10[0, 1:] = x[0, 1:] - x[0, 0]
    if drop_ch9:
        return np.concatenate([x, ch10], axis=0)
    cs, ce = center_idx * L, center_idx * L + L
    ref_ctr = float(x[0, 0, cs:ce].mean())
    read_ctr = x[0, 1:, cs:ce].mean(axis=1)
    ch9 = np.zeros((1, x.shape[1], x.shape[2]), dtype=np.float32)
    ch9[0, 1:, :] = (read_ctr - ref_ctr)[:, np.newaxis]
    return np.concatenate([x, ch9, ch10], axis=0)


class PileupDataset(Dataset):
    """
    Loads pileup tensors from one or more HDF5 files produced by
    featurize_pileup.py.

    Tensor on disk : (H, W, C)  where H = max_reads+1, C = 9
    Returned       : (C, H, W)  channels-first for PyTorch Conv2d

    When delta_channels=True (default), two extra channels are appended at
    load time without any re-featurization:
      Ch 9  read_supports_modification : per-read center-position signal
                deviation from expected kmer level, broadcast across the
                full row (analogous to DeepVariant's read_supports_variant).
      Ch 10 window_delta               : per-sample full-window deviation
                (observed signal minus expected kmer level) for all positions.

    HDF5 file handles are cached per-process (see _get_h5) so that
    multiprocessing DataLoader workers do not share handles and deadlock.
    """
    def __init__(self, h5_paths: list, indices: np.ndarray,
                 file_sizes: np.ndarray,
                 augment: bool = False, seed: int = 42,
                 rc_augment: bool = False,
                 signal_noise_std: float = 0.05,
                 delta_channels: bool = True,
                 preload: bool = False,
                 mask_flank_bases: bool = False,
                 mask_all_bases: bool = False,
                 legacy_ch9_center: bool = False):
        self.h5_paths   = h5_paths
        self.indices    = indices
        self.file_sizes = file_sizes
        self.augment    = augment
  # RAWMOD_DENSE_AUG=p: see __getitem__.
        self.dense_aug  = float(os.environ.get('RAWMOD_DENSE_AUG', '0'))
  # RAWMOD_TRAIN_MAX_READS=k: keep at most k read rows per training image (benchmark scores 10).
        self.train_max_reads = int(os.environ.get('RAWMOD_TRAIN_MAX_READS', '0'))
  # RAWMOD_READ_JITTER="img_gain,read_gain,read_offset". Per-image and
  # per-read multiplicative gain and per-read additive offset on the signal channel, simulating the residual
  # scale/level differences that survive per-read MAD normalisation and differ between libraries, flow cells
  # and pore states. 0,0,0 (default) leaves training unchanged.
        _j = os.environ.get('RAWMOD_READ_JITTER', '')
        self.read_jitter = tuple(float(v) for v in _j.split(',')) if _j else (0.0, 0.0, 0.0)
        self.rc_augment = rc_augment
        self.signal_noise_std = signal_noise_std
        self.delta_channels   = delta_channels
        self.rng        = np.random.default_rng(seed)
        self.offsets    = np.concatenate([[0], np.cumsum(file_sizes)])
        self._mem       = None  # (N, C, H, W) float32 when preload=True
        self._mem_y     = None  # (N,) float32 labels aligned to self.indices

        with h5py.File(h5_paths[0], 'r') as hf:
            self.n_channels       = int(hf.attrs.get('n_channels', 9))
            self.window_positions = int(hf.attrs.get('W', 0))
            self.samples_per_base = int(hf.attrs.get('L', 0))
  # Which window position is the candidate base (0-based).
  #
  # Do NOT read attrs['center_idx'] here. That attribute is the K-MER
  # centre used for the level-table lookup (6 for a 9-mer), which is a
  # different quantity that merely also indexes window positions. The
  # candidate base sits at the WINDOW centre = half_window (10 for
  # W=21). Reading center_idx placed the ch9 delta 4 positions
  # off-centre, onto a background base: measured on Ecoli_DM, column
  # center_idx*L shows A/C/G/T ~= 23/27/30/20% (background) while the
  # true centre shows 50% A / 50% T (the GATC 6mA sites).
            self.center_idx = int(hf.attrs.get(
                'half_window', self.window_positions // 2))
  # legacy_ch9_center: models trained BEFORE the ch9 fix learned ch9 at
  # the k-mer centre (attrs['center_idx']). To score such a model on
  # matched inputs, compute ch9 at that same (wrong) column so training
  # and inference agree. New models leave this False.
            self._kmer_center = int(hf.attrs.get('center_idx',
                                                 self.window_positions // 2))
            self.legacy_ch9_center = legacy_ch9_center
  # Optionally blank the base one-hots (ch 2-5) everywhere EXCEPT the
  # candidate position. The centre base is legitimate information
  # (6mA only occurs at A, 5mC only at C); the FLANKING bases are what
  # let a model memorise a recognition motif such as GATC and fire on
  # it regardless of the signal. See experiments/pipeline3.
            self.mask_flank_bases = mask_flank_bases
  # Optionally blank the base one-hots (ch 2-5) EVERYWHERE, including the
  # candidate position: the model then has NO nucleotide-identity input and
  # must detect a modification from signal alone (raw current, dwell, strand,
  # mapq, matches_ref, deltas). Strongest test of chemistry/context-agnostic
  # detection — a modification on an unseen base cannot be found via "which
  # base is this" if no base identity is available.
        self.mask_all_bases = mask_all_bases

  # RAWMOD_DROP_CH9=1 : append ONLY the window_delta channel, not the
  # center-base-only delta (ch9). Permutation importance shows ch9 is a net
  # negative for 5hmU (redundant with window_delta) and it is the sole
  # query-position-dependent input; dropping it yields a 10-channel model.
        self.drop_ch9 = os.environ.get('RAWMOD_DROP_CH9', '0') == '1'

        if preload:
            self._preload_to_memory()

    def _preload_to_memory(self) -> None:
        """
        Bulk-load every tensor for this split into one contiguous in-RAM array,
        pre-transposed to (C, H, W).  Featurized HDF5 tensors are gzip-chunked
        (64 imgs/chunk), so random per-item reads decompress a whole chunk per
        image (~59 img/s).  Reading each file once sequentially (~3,400 img/s)
        and serving __getitem__ from RAM removes that bottleneck entirely.

        Memory: ~0.234 MB/image (9-channel; delta channels are added per-item
        after augmentation, so they are NOT stored here).
        """
  # Group this split's global indices by source file, remembering the
  # position of each within self.indices so _mem stays aligned to `item`.
        n = len(self.indices)
        H = W = C = None
        mem = None
        for file_idx, path in enumerate(self.h5_paths):
            lo, hi = int(self.offsets[file_idx]), int(self.offsets[file_idx + 1])
            sel = np.nonzero((self.indices >= lo) & (self.indices < hi))[0]
            if len(sel) == 0:
                continue
            local = (self.indices[sel] - lo).astype(np.int64)
            with h5py.File(path, 'r') as hf:
                dset = hf['tensors']
                if mem is None:
                    H, W, C = dset.shape[1], dset.shape[2], dset.shape[3]
                    mem = np.empty((n, C, H, W), dtype=np.float32)
                    self._mem_y = np.empty((n,), dtype=np.float32)
                lbl = hf['labels'][:]
  # Read the file in large SEQUENTIAL chunks and gather the needed
  # rows from each, instead of materializing the whole file at once.
  # Sequential chunk reads keep the fast-decompress advantage of a
  # bulk read, but bound the transient copy to CHUNK images
  # (~0.234 MB/image) rather than the entire file — critical for
  # large pooled splits, where a whole-file read (up to ~117 GB for
  # a 500k-image file) on top of the ~0.234 MB/image `mem` array
  # would blow past node memory.
                CHUNK = 50000
                n_file = dset.shape[0]
                order = np.argsort(local)
                local_sorted = local[order]
                dest_sorted = sel[order]
                self._mem_y[sel] = lbl[local].astype(np.float32)
                j = 0
                for c0 in range(0, n_file, CHUNK):
                    c1 = min(c0 + CHUNK, n_file)
                    j2 = j
                    while j2 < len(local_sorted) and local_sorted[j2] < c1:
                        j2 += 1
                    if j2 > j:
                        chunk = dset[c0:c1]  # sequential read
                        sub = chunk[local_sorted[j:j2] - c0]  # (m, H, W, C)
                        mem[dest_sorted[j:j2]] = np.transpose(sub, (0, 3, 1, 2))
                        del chunk, sub
                    j = j2
                del lbl
        self._mem = mem

    def _resolve(self, global_idx: int) -> tuple:
        file_idx  = int(np.searchsorted(self.offsets[1:], global_idx, side='right'))
        local_idx = global_idx - int(self.offsets[file_idx])
        return file_idx, local_idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        if self._mem is not None:
  # In-RAM path: already (C, H, W); copy so augmentation cannot
  # corrupt the shared cache.
            x = self._mem[item].copy()
            y = float(self._mem_y[item])
        else:
            global_idx          = int(self.indices[item])
            file_idx, local_idx = self._resolve(global_idx)

  # Use per-process cached handle — safe for multiprocessing workers
            x, y = _h5_read(self.h5_paths[file_idx], local_idx)  # (H, W, C)

            x = np.transpose(x, (2, 0, 1))  # → (C, H, W)

  # Blank flanking base identity so a recognition motif (e.g. GATC) cannot
  # be read off the tensor, while keeping the candidate base itself.
  # Applied before augmentation/deltas so everything downstream is
  # consistent with what the model will actually see.
        if self.mask_flank_bases and self.samples_per_base > 0:
            L = self.samples_per_base
            cs, ce = self.center_idx * L, self.center_idx * L + L
            keep = x[2:6, :, cs:ce].copy()
            x[2:6, :, :] = 0.0
            x[2:6, :, cs:ce] = keep

  # Blank ALL base-identity channels (is_A/C/G/T) so the model sees no
  # nucleotide identity at all — signal-only detection.
        if self.mask_all_bases:
            x[2:6, :, :] = 0.0

        if self.augment:
            H = x.shape[1]
            if self.train_max_reads > 0:
                live = np.flatnonzero(np.abs(x[0, 1:]).sum(-1) > 1e-6) + 1
                if len(live) > self.train_max_reads:
                    x[:, self.rng.choice(live, len(live) - self.train_max_reads, replace=False), :] = 0.0
  # Zero out up to 30% of read rows (row 0 = reference, never masked)
            n_mask = int((H - 1) * 0.30 * self.rng.random())
            if n_mask > 0:
                rows = self.rng.choice(np.arange(1, H), size=n_mask, replace=False)
                x[:, rows, :] = 0.0
  # RAWMOD_DENSE_AUG=p: with probability p, a positive image has
  # each read's centre deviation (read minus reference level at the candidate base) replicated at every
  # other window base that carries the same reference base, scaled U(0.3, 1) per base, and the added
  # profile is re-centred per read (per-read normalisation would have removed a bulk shift). Teaches the
  # model that modification evidence can be present at all same-base positions, not only at the centre.
            if self.dense_aug > 0 and y > 0 and self.samples_per_base > 0 \
                    and self.rng.random() < self.dense_aug:
                L = self.samples_per_base
                W = x.shape[2] // L
                c = self.center_idx
                ref_b = x[2:6, 0, ::L].argmax(0)  # (W,) reference base per window position
                same = np.flatnonzero(ref_b == ref_b[c])
                same = same[same != c]
                if len(same):
                    cs, ce = c * L, c * L + L
                    valid = np.abs(x[0, 1:]).sum(-1) > 1e-6  # non-padding, non-masked reads
                    dev = x[0, 1:, cs:ce].mean(-1) - x[0, 0, cs:ce].mean()
                    add = np.zeros((H - 1, W), dtype=np.float32)
                    add[:, same] = dev[:, None] * self.rng.uniform(0.3, 1.0, size=(1, len(same)))
                    add -= add.mean(1, keepdims=True)
                    add[~valid] = 0.0
                    x[0, 1:] += np.repeat(add, L, axis=1)
  # RAWMOD_READ_JITTER: nuisance transform of the signal channel (see __init__). The reference row (0)
  # is the expected level and is never jittered, so the model must stay invariant to a read's residual
  # gain/offset rather than to the contrast between read and reference.
            jg_i, jg_r, jo_r = self.read_jitter
            if jg_i > 0 or jg_r > 0 or jo_r > 0:
                Hr = x.shape[1] - 1
                live = np.abs(x[0, 1:]).sum(-1) > 1e-6  # padded/masked rows must stay exactly zero
                g = np.ones(Hr, dtype=np.float32)
                if jg_i > 0:
                    g *= np.float32(1.0 + self.rng.normal(0.0, jg_i))
                if jg_r > 0:
                    g *= (1.0 + self.rng.normal(0.0, jg_r, size=Hr)).astype(np.float32)
                o = (self.rng.normal(0.0, jo_r, size=Hr).astype(np.float32)
                     if jo_r > 0 else np.zeros(Hr, dtype=np.float32))
                g[~live] = 1.0
                o[~live] = 0.0
                x[0, 1:] = x[0, 1:] * g[:, None] + o[:, None]
  # Additive noise only on raw signal.  Noisy one-hot/metadata
  # channels are not meaningful features.
            if self.signal_noise_std > 0:
                x[0] += (self.rng.standard_normal(x[0].shape).astype(np.float32)
                         * self.signal_noise_std)
            if self.rc_augment and self.rng.random() < 0.5:
                x = reverse_complement_tensor(
                    x, self.window_positions, self.samples_per_base)

  # Append delta channels after augmentation so they are consistent with
  # the (potentially noised) raw signal channel the model sees.
        if self.delta_channels and self.samples_per_base > 0:
            ci = self._kmer_center if self.legacy_ch9_center else self.center_idx
            x = add_delta_channels(x, ci, self.samples_per_base, drop_ch9=self.drop_ch9)

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


def reverse_complement_tensor(x: np.ndarray,
                              window_positions: int,
                              samples_per_base: int) -> np.ndarray:
    """
    Reverse-complement a channels-first pileup tensor.

    This reverses position/sample order, swaps A/T and C/G one-hot channels,
    and flips strand sign.  The reference-row expected-level channel is simply
    reversed; if the pore model is strand-specific, prefer leaving
    --rc-augment disabled.
    """
    c, h, total_w = x.shape
    y = x
    if window_positions > 0 and samples_per_base > 0:
        expected_w = window_positions * samples_per_base
        if expected_w == total_w:
            y = x.reshape(c, h, window_positions, samples_per_base)
            y = y[:, :, ::-1, ::-1].reshape(c, h, total_w).copy()
        else:
            y = x[:, :, ::-1].copy()
    else:
        y = x[:, :, ::-1].copy()

    if c >= 6:
        bases = y[2:6].copy()
        y[2] = bases[3]  # A <- T
        y[3] = bases[2]  # C <- G
        y[4] = bases[1]  # G <- C
        y[5] = bases[0]  # T <- A
    if c > 6:
        y[6] *= -1.0
    return y


class ProjectionHead(nn.Module):
    """
    Two-layer MLP projection head for Supervised Contrastive Learning.

    Maps the 512-dim backbone embedding to a lower-dimensional L2-normalised
    sphere where SupConLoss is computed. The head is only used during training;
    inference uses the backbone + classifier head unchanged.

    Parameters
    ----------
    in_dim     : backbone embedding dimension (512 for PileupInceptionV3).
    hidden_dim : intermediate width (default 256).
    out_dim    : output embedding dimension written to the unit hypersphere
                 via L2-normalisation (default 128).
    """
    def __init__(self, in_dim: int = 512, hidden_dim: int = 256,
                 out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


def make_loader_kwargs(batch_size: int,
                       num_workers: int,
                       device: torch.device,
                       worker_init_fn=None) -> dict:
    """
    Build DataLoader kwargs that are conservative on shared/HPC systems.

    Multiprocessing DataLoader workers require PyTorch shared-memory/resource
    sharing support.  Some batch/sandbox environments block that path, which
    makes training appear to hang before the first batch.  The default
    num_workers=0 avoids that failure mode; users can opt into workers when
    their runtime supports it.
    """
    kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
    )
    if num_workers > 0:
        kwargs['worker_init_fn'] = worker_init_fn
        kwargs['persistent_workers'] = True
    return kwargs


def binary_labels(labels: np.ndarray) -> np.ndarray:
    """Coerce any non-zero modification label to the binary modified class."""
    return (labels.astype(np.int64) > 0).astype(np.int64)


def decode_ref_names(values: np.ndarray) -> list[str]:
    names = []
    for value in values:
        if isinstance(value, bytes):
            names.append(value.decode('utf-8'))
        else:
            names.append(str(value))
    return names


def make_position_keys(ref_names: np.ndarray, ref_pos: np.ndarray) -> list[tuple]:
    names = decode_ref_names(ref_names)
    return list(zip(names, ref_pos.astype(np.int64).tolist()))


def _split_group_keys(group_to_label: dict,
                      val_frac: float,
                      test_frac: float,
                      seed: int):
    """Stratify a list of leakage-safe groups into train/val/test keys."""
    by_label = {
        0: [k for k, v in group_to_label.items() if v == 0],
        1: [k for k, v in group_to_label.items() if v == 1],
    }
    rng = np.random.default_rng(seed)

    def split_keys(keys: list):
        keys = list(keys)
        rng.shuffle(keys)
        n = len(keys)
        if n == 0:
            return [], [], []
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        if test_frac > 0 and n_test == 0 and n >= 3:
            n_test = 1
        if val_frac > 0 and n_val == 0 and n - n_test >= 2:
            n_val = 1
        while n_test + n_val >= n and n_test + n_val > 0:
            if n_val >= n_test and n_val > 0:
                n_val -= 1
            else:
                n_test -= 1
        test_keys = keys[:n_test]
        val_keys = keys[n_test:n_test + n_val]
        train_keys = keys[n_test + n_val:]
        return train_keys, val_keys, test_keys

    train_keys, val_keys, test_keys = [], [], []
    for label in (0, 1):
        tr, va, te = split_keys(by_label[label])
        train_keys.extend(tr)
        val_keys.extend(va)
        test_keys.extend(te)

    return train_keys, val_keys, test_keys


def split_position_groups(labels: np.ndarray,
                          position_keys: list[tuple],
                          val_frac: float,
                          test_frac: float,
                          seed: int):
    """
    Stratify train/val/test by unique genomic position, then expand to images.

    All images with the same (ref_name, ref_pos) stay in one split.  If the
    same coordinate appears in multiple input files, those images stay together
    too, which prevents sequence/context leakage across splits.
    """
    group_to_indices: dict[tuple, list[int]] = {}
    group_to_label: dict[tuple, int] = {}
    for i, key in enumerate(position_keys):
        group_to_indices.setdefault(key, []).append(i)
        group_to_label[key] = max(group_to_label.get(key, 0), int(labels[i] > 0))

    train_keys, val_keys, test_keys = _split_group_keys(
        group_to_label=group_to_label,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )

    def expand(keys: list[tuple]) -> np.ndarray:
        idx = []
        for key in keys:
            idx.extend(group_to_indices[key])
        return np.array(idx, dtype=np.int64)

    stats = {
        'train_positions': len(train_keys),
        'val_positions':   len(val_keys),
        'test_positions':  len(test_keys),
    }
    return expand(train_keys), expand(val_keys), expand(test_keys), stats


def source_position_keys(global_indices: np.ndarray,
                         position_keys: list[tuple],
                         file_sizes: np.ndarray):
    """
    Return source-aware base keys for image indices.

    Splitting uses bare (ref_name, ref_pos) keys so the same reference base is
    never shared across train/val/test.  Scoring uses (file_idx, ref_name,
    ref_pos), because the same reference coordinate can appear as unmodified in
    control and modified in treatment files.
    """
    offsets = np.concatenate([[0], np.cumsum(file_sizes)])
    file_idx = np.searchsorted(offsets[1:], global_indices, side='right')
    keys = [
        (int(fi), position_keys[int(i)][0], int(position_keys[int(i)][1]))
        for i, fi in zip(global_indices, file_idx)
    ]
    return keys, file_idx.astype(np.int64)


def aggregate_by_position(y_true: np.ndarray,
                          y_prob: np.ndarray,
                          group_keys: list[tuple]):
    """Average image probabilities into one probability per supplied group."""
    grouped: dict[tuple, list] = {}
    for yt, yp, key in zip(y_true, y_prob, group_keys):
        if key not in grouped:
            grouped[key] = [0, []]
        grouped[key][0] = max(grouped[key][0], int(yt > 0))
        grouped[key][1].append(float(yp))

    keys, labels, probs = [], [], []
    for key, (label, values) in grouped.items():
        keys.append(key)
        labels.append(label)
        probs.append(float(np.mean(values)))
    return (np.array(labels, dtype=np.int64),
            np.array(probs, dtype=np.float32),
            keys)


@torch.no_grad()
def run_inference(model: nn.Module, loader: DataLoader,
                  device: torch.device):
    model.eval()
    all_y, all_p = [], []
    for x, y in loader:
        x      = x.to(device, non_blocking=True)
        logits = model(x).squeeze(1)
        probs  = torch.sigmoid(logits).cpu().numpy()
        all_y.append(y.numpy())
        all_p.append(probs)
    return np.concatenate(all_y), np.concatenate(all_p)
