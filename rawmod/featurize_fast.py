"""Array-based pass 1 / pass 2 of featurization.py, producing the same HDF5 datasets bit for bit.

The original passes keep, for every (read, position), a dict of the 21 segments around the position, built by
scanning the read's whole segment dict (O(read length) per position, O(length^2) per read), and then assemble each
image base by base in Python. Here every read is stored once as arrays (resampled segments, dwell, base, and a dense
reference-position -> segment map); positions keep only the indices of their retained reads; images are gathered in
vectorized batches and written incrementally. The random draws (reservoir sampling of reads, site sampling, read
partitioning) are made with the same generator, in the same order, with the same arguments, and the per-base
resampling repeats numpy.interp's arithmetic, so the output equals that of featurization.py --legacy.

Requires --max-images-per-base (the bounded reservoir); featurization.py falls back to the legacy passes otherwise.
"""
import sys

import numpy as np
import pysam
import h5py
from tqdm import tqdm

# base codes: 0-3 = A, C, G, T; 4 = N; 5 = any other character (one-hot all zero, as BASE_ONEHOT.get(...) gives)
_CODE = np.full(256, 5, dtype=np.uint8)
for _i, _b in enumerate('ACGT'):
    _CODE[ord(_b)] = _i
    _CODE[ord(_b.lower())] = _i
_CODE[ord('N')] = 4
_CODE[ord('n')] = 4
_ONEHOT = np.zeros((6, 4), dtype=np.float32)
_ONEHOT[np.arange(4), np.arange(4)] = 1.0
_COMPC = np.array([3, 2, 1, 0, 4, 4], dtype=np.uint8)  # complement code; non-ACGT -> N (as _COMP_MAP.get(.., 'N'))


def _codes(s: str) -> np.ndarray:
    return _CODE[np.frombuffer(s.encode('latin-1'), dtype=np.uint8)]


_PLAN = {}


def _interp_plan(n, L):
    """numpy.interp(linspace(0,1,L), linspace(0,1,n), y) for every y of length n, as index arrays."""
    p = _PLAN.get((n, L))
    if p is None:
        dx = np.linspace(0, 1, n)
        x = np.linspace(0, 1, L)
        j = np.searchsorted(dx, x, side='right') - 1  # binary_search_with_guess: last dx[j] <= x
        jc = np.clip(j, 0, n - 1)
        exact = (j == n - 1) | (dx[jc] == x)  # interp returns dy[j] exactly in these cases
        jj = np.minimum(jc, n - 2)
        p = (jc, exact, jj, dx[jj + 1] - dx[jj], x - dx[jj])
        _PLAN[(n, L)] = p
    return p


def resample_read(signal, starts, ends, L):
    """Resample every segment [starts[i], ends[i]) of one read to L points, as resample_segment does per segment."""
    n = ends - starts
    out = np.empty((len(n), L), dtype=np.float32)
    for m in np.unique(n):
        sel = np.nonzero(n == m)[0]
        idx = starts[sel, None] + np.arange(m)
        Y = signal[idx]  # float64, (k, m)
        if m == L:
            out[sel] = Y.astype(np.float32)
        elif m == 1:
            out[sel] = np.repeat(Y.astype(np.float32), L, axis=1)
        else:
            jc, exact, jj, den, off = _interp_plan(int(m), L)
            slope = (Y[:, jj + 1] - Y[:, jj]) / den
            r = slope * off + Y[:, jj]
            out[sel] = np.where(exact, Y[:, jc], r).astype(np.float32)
    return out


def _level_table(kmer_levels, k):
    """kmer_levels dict -> array over base-4 k-mer index (NaN = absent)."""
    t = np.full(4 ** k, np.nan, dtype=np.float64)
    keys = [km.upper() for km in kmer_levels]
    ok = [len(km) == k for km in keys]
    keys = [km for km, o in zip(keys, ok) if o]
    vals = np.array([float(v) for v, o in zip(kmer_levels.values(), ok) if o], dtype=np.float64)
    if not keys:
        return t
    c = _CODE[np.frombuffer(''.join(keys).encode('latin-1'), dtype=np.uint8)].reshape(len(keys), k).astype(np.int64)
    good = (c < 4).all(1)
    t[c[good] @ (4 ** np.arange(k - 1, -1, -1))] = vals[good]
    return t


def _contig_levels(ctx, table, k, start_off, revcomp):
    """Expected level of the k-mer starting at p - start_off for every p (0 where any base is not ACGT or absent)."""
    n = len(ctx)
    lev = np.zeros(n, dtype=np.float64)
    if table is None:
        return lev
    s = np.arange(n) - start_off
    ok = (s >= 0) & (s + k <= n)
    idx = np.zeros(n, dtype=np.int64)
    good = ok.copy()
    for t in range(k):
        b = np.where(ok, ctx[np.clip(s + t, 0, n - 1)], 4).astype(np.int64)
        good &= b < 4
        if revcomp:  # reverse complement: base t of the + k-mer is complemented at digit t (least significant first)
            idx += (3 - np.minimum(b, 3)) * 4 ** t
        else:
            idx += np.minimum(b, 3) * 4 ** (k - 1 - t)
    v = table[idx]
    good &= ~np.isnan(v)
    lev[good] = v[good]
    return lev


class H5Sink:
    """The featurization.py HDF5 layout (tensors gzip-1, one image per chunk)."""
    def __init__(self, path):
        self.path = path

    def open(self, n_images, height, WL, n_channels, names, img_t, img_p, labels, n_reads, image_idx):
        print(f"\nWriting HDF5 to {self.path} ...", file=sys.stderr)
        self.hf = h5py.File(self.path, 'w')
        self.dset = self.hf.create_dataset('tensors', shape=(n_images, height, WL, n_channels), dtype=np.float32,
                                           compression='gzip', compression_opts=1, chunks=(1, height, WL, n_channels))
        self.hf.create_dataset('labels', data=labels)
        self.hf.create_dataset('ref_names', data=np.array([names[t].encode('utf-8') for t in img_t.tolist()],
                                                          dtype=h5py.special_dtype(vlen=bytes)))
        self.hf.create_dataset('ref_pos', data=img_p)
        self.hf.create_dataset('n_reads', data=n_reads)
        self.hf.create_dataset('image_idx', data=np.minimum(image_idx, 32767).astype(np.int16), compression='gzip',
                               compression_opts=1)

    def write(self, b0, b1, T):
        self.dset[b0:b1] = T

    def close(self):
        self.hf.close()


class ScoreSink:
    """Score the images as they are built, without writing them: the model input is formed exactly as
    model.PileupDataset.__getitem__ forms it from a stored image (no augmentation), and the output file is the one
    scripts/evaluate/score_genome.py writes (contig, pos, score, label, n_reads)."""
    def __init__(self, model, device, out_path, half_window, L, mask_all_bases=False, drop_ch9=False, gpu_batch=1024):
        import torch
        self.torch, self.model, self.device, self.out_path = torch, model, device, out_path
        self.ci, self.L, self.mask_all_bases, self.drop_ch9 = half_window, L, mask_all_bases, drop_ch9
        self.gpu_batch = gpu_batch

    def open(self, n_images, height, WL, n_channels, names, img_t, img_p, labels, n_reads, image_idx):
        print(f"\nScoring {n_images:,} images into {self.out_path} (no HDF5) ...", file=sys.stderr)
        self.meta = (names, img_t, img_p, labels, n_reads)
        self.scores = np.empty(n_images, dtype=np.float32)

    def model_input(self, T):
        x = np.ascontiguousarray(np.transpose(T, (0, 3, 1, 2)))  # (B, C, H, W)
        if self.mask_all_bases:
            x[:, 2:6] = 0.0
        ch10 = np.zeros((x.shape[0], 1, x.shape[2], x.shape[3]), dtype=np.float32)
        ch10[:, 0, 1:] = x[:, 0, 1:] - x[:, 0, 0:1]
        if self.drop_ch9:
            return np.concatenate([x, ch10], axis=1)
        cs, ce = self.ci * self.L, self.ci * self.L + self.L
        ref_ctr = np.array([float(x[b, 0, 0, cs:ce].mean()) for b in range(x.shape[0])], dtype=np.float32)
        read_ctr = np.stack([x[b, 0, 1:, cs:ce].mean(axis=1) for b in range(x.shape[0])])
        ch9 = np.zeros_like(ch10)
        ch9[:, 0, 1:, :] = (read_ctr - ref_ctr[:, None])[:, :, None]
        return np.concatenate([x, ch9, ch10], axis=1)

    def write(self, b0, b1, T):
        torch = self.torch
        with torch.no_grad():
            for s0 in range(0, b1 - b0, self.gpu_batch):
                xb = torch.from_numpy(self.model_input(T[s0:s0 + self.gpu_batch])).to(self.device, non_blocking=True)
                p = torch.sigmoid(self.model(xb).squeeze(-1)).float().cpu().numpy()
                self.scores[b0 + s0:b0 + s0 + len(p)] = p

    def close(self):
        import gzip
        names, img_t, img_p, labels, n_reads = self.meta
        with gzip.open(self.out_path, 'wt') as fh:
            fh.write("contig\tpos\tscore\tlabel\tn_reads\n")
            for t, p, sc, l, nr in zip(img_t.tolist(), img_p.tolist(), self.scores, labels.tolist(), n_reads.tolist()):
                fh.write(f"{names[t]}\t{int(p)}\t{sc:.6f}\t{int(l)}\t{int(nr)}\n")
        print(f"wrote {self.out_path}  ({len(self.scores):,} sites)  mean_score={self.scores.mean():.4f}", file=sys.stderr)


def run(args, *, read_reader_map, seg_borders, exclude_map, gt_set, candidate_set, kmer_levels, kmer_size,
        center_idx, read_select, target_bases, rng, max_retained, n_channels, read_overlaps_excluded,
        mad_normalize, zscore, get_ref_info_from_bam, get_haplotype_index, sink=None):
    R = int(max_retained)
    hw, L, MR = args.half_window, args.L, args.max_reads
    W = 2 * hw + 1
    bam_fh = pysam.AlignmentFile(args.bam, 'rb', check_sq=False)
    names = list(bam_fh.references)
    lens = list(bam_fh.lengths)
    PAD = hw + (kmer_size or 0) + 2
    ctx = [np.full(n + 2 * PAD, 4, dtype=np.uint8) for n in lens]  # pos_ref_context, offset by PAD
    counts = [np.zeros(n, dtype=np.int64) for n in lens]  # pos_read_counts
    slots = [None] * len(lens)  # pos_reads: (n, R) read index, -1 empty
    cand = None
    if candidate_set is not None:
        cand = [np.zeros(n, dtype=bool) for n in lens]
        t2i = {nm: i for i, nm in enumerate(names)}
        for nm, p in candidate_set:
            if nm in t2i and 0 <= p < lens[t2i[nm]]:
                cand[t2i[nm]][p] = True
    tcodes = np.array(sorted({int(_CODE[ord(b)]) for b in target_bases}), dtype=np.uint8) if target_bases else None

  # per-read storage
    R_samples, R_dwell, R_base, R_rmin, R_span, R_dense = [], [], [], [], [], []
    R_strand, R_mapq, R_key = [], [], []
    nseg = 0

    n_total = n_eval = n_skip = n_skip_excluded = 0
    BLOCK = 512

    def process_block(block):
        nonlocal n_eval, n_skip, nseg
  # one POD5 selection per reader for the whole block (the legacy loop asked for one read at a time)
        sig = {}
        by_reader = {}
        for br in block:
            by_reader.setdefault(id(read_reader_map[br.query_name]), (read_reader_map[br.query_name], []))[1].append(br.query_name)
        for rd, ids in by_reader.values():
            for r in rd.reads(selection=ids, missing_ok=True):
                sig[str(r.read_id)] = r.signal
        for br in block:
            read_id = br.query_name
            try:
                raw = sig[read_id].astype(np.float64)
                signal = mad_normalize(raw) if args.normalize else zscore(raw)
                peaks = seg_borders[read_id]
                ref_seq, ref_positions = get_ref_info_from_bam(br)
                if ref_seq is None:
                    n_skip += 1
                    continue
                strand = -1 if br.is_reverse else +1
                mapq = br.mapping_quality
                tid = br.reference_id
                n_use = min(len(peaks) - 1, len(ref_seq))
                st = peaks[:n_use].astype(np.int64)
                en = peaks[1:n_use + 1].astype(np.int64)
                ok = (st < en) & (st >= 0) & (en <= len(signal))
                iv = np.nonzero(ok)[0]
                rpos = np.asarray(ref_positions, dtype=np.int64)[iv]
                bcode = _codes(ref_seq[:n_use])[iv]
                samples = resample_read(signal, st[iv], en[iv], L)
                dwell = (en - st)[iv]
                if len(rpos) == 0:
                    n_eval += 1
                    continue
  # pos_ref_context: every covered position, before any filter
                ctx[tid][rpos + PAD] = bcode
                mask = np.ones(len(rpos), dtype=bool)
                if tcodes is not None:
                    mask &= np.isin(bcode, tcodes)
                if cand is not None:
                    mask &= cand[tid][rpos]
                if read_select is not None:
                    sel = read_select.get(read_id)
                    if sel is None or sel[0] != names[tid]:
                        mask[:] = False
                    else:
                        j = np.searchsorted(sel[1], rpos, side='right') - 1
                        mask &= (j >= 0) & (rpos < sel[2][np.clip(j, 0, None)])
                ridx = len(R_samples)
                lo, hi = int(rpos.min()), int(rpos.max())
                dense = np.full(hi - lo + 1, -1, dtype=np.int64)
                dense[rpos - lo] = nseg + np.arange(len(rpos))
                R_samples.append(samples); R_dwell.append(dwell); R_base.append(bcode)
                R_rmin.append(lo); R_span.append(hi - lo + 1); R_dense.append(dense)
                R_strand.append(strand); R_mapq.append(mapq)
                rs = br.reference_start
                R_key.append((get_haplotype_index(br), int(rs if rs is not None and rs >= 0 else 0), read_id))
                nseg += len(rpos)
                Q = rpos[mask]
                if len(Q):
                    if slots[tid] is None:
                        slots[tid] = np.full((lens[tid], R), -1, dtype=np.int32)
                    c = counts[tid][Q] + 1
                    counts[tid][Q] = c
                    keep = c <= R
                    slots[tid][Q[keep], c[keep] - 1] = ridx
                    for q, cc in zip(Q[~keep].tolist(), c[~keep].tolist()):
                        j = int(rng.integers(cc))  # reservoir draw, same order and argument as the legacy loop
                        if j < R:
                            slots[tid][q, j] = ridx
                n_eval += 1
            except Exception as e:
                n_skip += 1
                if n_skip <= 5:
                    print(f"  Warning: skipped {read_id}: {e}", file=sys.stderr)

    block = []
    for br in tqdm(bam_fh, desc="Processing reads", file=sys.stderr):
        n_total += 1
        if br.is_supplementary or br.is_secondary or br.is_unmapped or br.mapping_quality < args.min_mapq:
            n_skip += 1
            continue
        if exclude_map and read_overlaps_excluded(br.reference_name, br.reference_start, br.reference_end,
                                                  exclude_map):
            n_skip += 1
            n_skip_excluded += 1
            continue
        if br.query_name not in seg_borders or br.query_name not in read_reader_map:
            n_skip += 1
            continue
        strand = -1 if br.is_reverse else +1
        if args.strand != 'both' and strand != (+1 if args.strand == '+' else -1):
            n_skip += 1
            continue
        block.append(br)
        if len(block) >= BLOCK:
            process_block(block)
            block = []
    if block:
        process_block(block)
    bam_fh.close()
    print(f"\nReads evaluated: {n_eval:,}  skipped: {n_skip:,} "
          f"(of which {n_skip_excluded:,} overlapped the genomic holdout)", file=sys.stderr)

    if not R_samples:
        print("No eligible positions found — nothing to write.", file=sys.stderr)
        sys.exit(1)
    S_all = np.concatenate(R_samples); del R_samples
    D_all = np.concatenate(R_dwell); del R_dwell
    B_all = np.concatenate(R_base); del R_base
    rmin = np.array(R_rmin, dtype=np.int64)
    rspan = np.array(R_span, dtype=np.int64)
    rdoff = np.concatenate([[0], np.cumsum(rspan)[:-1]]).astype(np.int64)
    dense_all = np.concatenate(R_dense); del R_dense
    rstrand = np.array(R_strand, dtype=np.float32)
    rmapq = np.array([float(min(m, 60) / 60.0) for m in R_mapq], dtype=np.float32)
    order = sorted(range(len(R_key)), key=lambda i: R_key[i])  # sort_reads_for_pileup key
    rank = np.empty(len(R_key), dtype=np.int64)
    rank[np.array(order, dtype=np.int64)] = np.arange(len(order))

  # ── pass 2: eligible positions in (contig name, position) order ─────────────
    keys_t, keys_p = [], []
    for tid in sorted(range(len(names)), key=lambda i: names[i]):
        if slots[tid] is None:
            continue
        stored = np.minimum(counts[tid], R)
        e = (counts[tid] >= args.min_reads) & (stored >= args.min_reads)
        if tcodes is not None:
            e &= np.isin(ctx[tid][PAD:PAD + lens[tid]], tcodes)
        p = np.nonzero(e)[0]
        keys_t.append(np.full(len(p), tid, dtype=np.int64)); keys_p.append(p.astype(np.int64))
    kt = np.concatenate(keys_t) if keys_t else np.zeros(0, np.int64)
    kp = np.concatenate(keys_p) if keys_p else np.zeros(0, np.int64)
    n_elig = len(kt)
    print(f"Eligible positions: {n_elig:,}  (>= {args.min_reads} reads"
          + (f", ref_base in {''.join(sorted(target_bases))}" if target_bases else "") + ")", file=sys.stderr)
    print(f"Maximum retained reads per eligible position: {R:,}", file=sys.stderr)
    if n_elig == 0:
        print("No eligible positions found — nothing to write.", file=sys.stderr)
        sys.exit(1)

    if args.sample_n_sites is not None and n_elig > args.sample_n_sites:
        bw = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0} if args.uniform_sampling else {0: 3.0, 1: 3.0, 2: 1.0, 3: 1.0}
        base = np.array([ctx[t][p + PAD] for t, p in zip(kt.tolist(), kp.tolist())])
        weights = np.array([bw.get(int(b), 1.0) for b in base], dtype=np.float64)
        weights /= weights.sum()
        idx = rng.choice(n_elig, size=args.sample_n_sites, replace=False, p=weights)
        idx.sort()
        kt, kp = kt[idx], kp[idx]
        print(f"Sampled {len(kt):,} / {n_elig:,} eligible positions "
              f"({'uniform' if args.uniform_sampling else 'A/C bias 3:1 vs G/T'})", file=sys.stderr)

  # partition_reads: one image (all retained reads) unless a position holds more than max_reads
    img_t, img_p, img_rows, img_idx = [], [], [], []
    for t, p in zip(kt.tolist(), kp.tolist()):
        row = slots[t][p]
        n = int(min(counts[t][p], R))
        rl = row[:n]
        if n <= MR:
            chunks = [rl]
        else:
            ind = np.arange(n)
            rng.shuffle(ind)
            n_im = n // MR
            if args.max_images_per_base is not None:
                n_im = min(n_im, args.max_images_per_base)
            chunks = [rl[ind[i * MR:(i + 1) * MR]] for i in range(n_im)]
        for ci, ch in enumerate(chunks):
            ch = ch[np.argsort(rank[ch], kind='stable')]
            r = np.full(MR, -1, dtype=np.int64)
            r[:len(ch)] = ch
            img_t.append(t); img_p.append(p); img_rows.append(r); img_idx.append(ci)
    img_t = np.array(img_t, dtype=np.int64)
    img_p = np.array(img_p, dtype=np.int64)
    img_rows = np.stack(img_rows)
    img_idx = np.array(img_idx, dtype=np.int64)
    n_images, n_pos = len(img_t), len(kt)
    height = MR + 1
    print(f"\nPositions: {n_pos:,}  →  Images (rows): {n_images:,}  (avg {n_images / n_pos:.1f} images/position)",
          file=sys.stderr)
    print(f"Tensor shape per image: ({height}, {W * L}, {n_channels})", file=sys.stderr)

  # per-contig reference levels (plain and reverse-complement k-mers)
    table = _level_table(kmer_levels, kmer_size) if kmer_levels is not None else None
    lev = {}
    for t in set(img_t.tolist()):
        lev[t] = (_contig_levels(ctx[t], table, kmer_size, center_idx, False).astype(np.float32),
                  _contig_levels(ctx[t], table, kmer_size, kmer_size - 1 - center_idx, True).astype(np.float32)
                  if args.orient == 'read' else None)

    labels = np.zeros(n_images, dtype=np.int8)
    if gt_set:
        gts = {(names.index(r) if r in names else -1, p) for r, p in gt_set}
        labels = np.array([1 if (t, p) in gts else 0 for t, p in zip(img_t.tolist(), img_p.tolist())], dtype=np.int8)

    n_reads_img = np.minimum((img_rows >= 0).sum(1), 32767).astype(np.int16)
    if sink is None:
        sink = H5Sink(args.output)
    sink.open(n_images=n_images, height=height, WL=W * L, n_channels=n_channels, names=names, img_t=img_t, img_p=img_p,
              labels=labels, n_reads=n_reads_img, image_idx=img_idx)
    off = np.arange(-hw, hw + 1)
    BS = 2048
    for b0 in tqdm(range(0, n_images, BS), desc="Building tensors", file=sys.stderr):
        b1 = min(n_images, b0 + BS)
        T = np.zeros((b1 - b0, height, W, L, n_channels), dtype=np.float32)
        tt, pp = img_t[b0:b1], img_p[b0:b1]
        win = pp[:, None] + off  # (B, W)
        rc = np.stack([ctx[t][w + PAD] for t, w in zip(tt.tolist(), win)])  # reference base codes
        lv = np.stack([lev[t][0][w + PAD] for t, w in zip(tt.tolist(), win)])
        T[:, 0, :, :, 0] = lv[:, :, None]
        T[:, 0, :, :, 2:6] = _ONEHOT[rc][:, :, None, :]
        rr = img_rows[b0:b1]  # (B, MR)
        have = rr >= 0
        rrc = np.where(have, rr, 0)
        q = win[:, None, :] - rmin[rrc][:, :, None]  # (B, MR, W)
        valid = have[:, :, None] & (q >= 0) & (q < rspan[rrc][:, :, None])
        g = dense_all[np.where(valid, rdoff[rrc][:, :, None] + q, 0)]
        valid &= g >= 0
        gc = np.where(valid, g, 0)
        vf = valid[..., None]
        T[:, 1:, :, :, 0] = np.where(vf, S_all[gc], 0)
        T[:, 1:, :, :, 1] = np.where(valid, np.log1p(D_all[gc].astype(np.float64)).astype(np.float32), 0)[..., None]
        bc = B_all[gc]
        T[:, 1:, :, :, 2:6] = np.where(vf, _ONEHOT[bc], 0)[:, :, :, None, :]
        T[:, 1:, :, :, 6] = np.where(valid, rstrand[rrc][:, :, None], 0)[..., None]
        T[:, 1:, :, :, 7] = np.where(valid, rmapq[rrc][:, :, None], 0)[..., None]
        rcb = rc[:, None, :]
        match = ((rcb < 4) & (bc == rcb)) | ((rcb >= 4) & (bc == 5))
        T[:, 1:, :, :, 8] = np.where(valid & match, 1.0, 0.0).astype(np.float32)[..., None]
        T = T.reshape(b1 - b0, height, W * L, n_channels)
        if args.orient == 'read':
            T = T[:, :, ::-1, :].copy()
            T[..., [2, 3, 4, 5]] = T[..., [5, 4, 3, 2]]
            R0 = np.zeros((b1 - b0, W, L, n_channels), dtype=np.float32)
            lvr = np.stack([lev[t][1][w + PAD] for t, w in zip(tt.tolist(), win)])
            R0[:, :, :, 0] = lvr[:, :, None]
            R0[:, :, :, 2:6] = _ONEHOT[_COMPC[rc]][:, :, None, :]
            T[:, 0] = R0.reshape(b1 - b0, W * L, n_channels)[:, ::-1, :]
        sink.write(b0, b1, T)
    sink.close()
    return dict(n_pos=n_pos, n_images=n_images, height=height, W=W, labels=labels, center_idx=center_idx)


def extract_base_segments_fast(signal, peaks, ref_seq, ref_positions, L, mapq, strand):
    """Vectorised drop-in for featurization.extract_base_segments.

    Same filtering and the same per-segment resampling, but every segment of a
    read is interpolated in one grouped pass instead of one numpy.interp call
    per base. Output is bitwise identical.
    """
    n_usable = min(len(peaks) - 1, len(ref_seq))
    if n_usable <= 0:
        return {}
    starts = np.asarray(peaks[:n_usable], dtype=np.int64)
    ends = np.asarray(peaks[1:n_usable + 1], dtype=np.int64)
    keep = (starts < ends) & (starts >= 0) & (ends <= len(signal))
    sel = np.nonzero(keep)[0]
    if sel.size == 0:
        return {}
    res = resample_read(signal, starts[sel], ends[sel], L)
    dwell = (ends[sel] - starts[sel]).astype(np.int64)
    out = {}
    for j, i in enumerate(sel):
        out[ref_positions[i]] = {
            'samples': res[j],
            'dwell': int(dwell[j]),
            'base': ref_seq[i],
            'mapq': mapq,
            'strand': strand,
        }
    return out
