#!/usr/bin/env python3
"""Refine move tables using Remora's signal mapping refinement.

Takes a dorado BAM (aligned, with --emit-moves) and pod5 files, refines the
signal-to-base mapping using a k-mer level table, and outputs:
  1. moves_refined.tsv - move table compatible with rawhash2 --moves-file
  2. reads_refined.bam - BAM with refined mv/ts tags replacing the originals

With --ref-mapping, uses reference-anchored refinement instead of query-anchored.
This produces sample-level segmentation points (ref_to_signal) suitable for
rawhash2 --peaks-file.

Usage:
  # Query-anchored (default, moves output):
  python refine_moves_remora.py \
    --pod5 /path/to/pod5_dir_or_file \
    --bam /path/to/aligned_reads.bam \
    --level-table /path/to/9mer_levels_v1.txt \
    --output moves_refined.tsv \
    --output-bam reads_refined.bam

  # Reference-anchored (peaks output):
  python refine_moves_remora.py \
    --pod5 /path/to/pod5_dir_or_file \
    --bam /path/to/aligned_reads.bam \
    --level-table /path/to/9mer_levels_v1.txt \
    --output moves_refined.tsv \
    --output-peaks peaks_refined.tsv \
    --ref-mapping --min-mapq 20

Output formats:
  moves TSV (--output): read_id\tmv:B:c,STRIDE,0,1,...\tts:i:OFFSET
  peaks TSV (--output-peaks): read_id\tpeak1\tpeak2\tpeak3\t...
"""

import os
import sys
import array
import argparse
import pod5
import pysam
import numpy as np
from tqdm import tqdm
from remora import io, refine_signal_map


def get_pod5_readers(pod5_path):
    """Build a dict mapping read_id -> pod5.Reader."""
    read_reader_map = {}
    if pod5_path.endswith('.pod5'):
        reader = pod5.Reader(pod5_path)
        for rid in reader.read_ids:
            read_reader_map[str(rid)] = reader
        return read_reader_map
    for fname in os.listdir(pod5_path):
        if fname.endswith('.pod5'):
            reader = pod5.Reader(os.path.join(pod5_path, fname))
            for rid in reader.read_ids:
                read_reader_map[str(rid)] = reader
    return read_reader_map


def query_to_signal_to_mv_tag(query_to_signal, stride, original_ts=0):
    """Convert a query_to_signal array to mv:B:c tag string.

    query_to_signal: int array of length (seq_len + 1), where
        query_to_signal[i] is the signal index for the start of base i.
        Remora returns these in adapter-trimmed space (0-based).
    stride: the basecaller stride (e.g., 6 for dorado R10.4.1 SUP).
    original_ts: the original template_start from the BAM ts tag.
        Remora's q2s is in trimmed space; we add original_ts to convert
        back to raw signal coordinates that rawhash2 expects.

    Returns: "mv:B:c,STRIDE,move_values..." and ts:i offset.
    """
    # Remora q2s is 0-based (adapter-trimmed). Add original_ts to get
    # positions in raw signal space.
    ts_offset = int(query_to_signal[0]) + original_ts

    # Number of signal chunks from ts_offset to end
    # Each chunk is `stride` samples wide
    last_sig = int(query_to_signal[-1]) + original_ts
    n_chunks = (last_sig - ts_offset + stride - 1) // stride

    # Build move array: for each chunk, 1 if a new base starts, else 0
    moves = [0] * n_chunks
    for base_idx in range(len(query_to_signal) - 1):
        sig_start = int(query_to_signal[base_idx]) + original_ts
        chunk_idx = (sig_start - ts_offset) // stride
        if 0 <= chunk_idx < n_chunks:
            moves[chunk_idx] = 1

    # First move should always be 1 (first base starts at chunk 0)
    if n_chunks > 0:
        moves[0] = 1

    mv_values = ','.join(str(m) for m in moves)
    mv_tag = f"mv:B:c,{stride},{mv_values}"
    ts_tag = f"ts:i:{ts_offset}"
    return mv_tag, ts_tag


def parse_mv_moves(mv_tag_str):
    """Parse move values from an mv:B:c,STRIDE,0,1,... string.

    Returns list of int move values (without the stride prefix).
    """
    # mv_tag_str = "mv:B:c,6,1,0,0,1,..."
    parts = mv_tag_str.split(',')
    # parts[0] = "mv:B:c", parts[1] = stride, parts[2:] = move values
    return [int(x) for x in parts[2:]]


def main():
    parser = argparse.ArgumentParser(
        description='Refine move tables using Remora signal mapping refinement.')
    parser.add_argument('--pod5', required=True,
                        help='Path to pod5 file or directory of pod5 files')
    parser.add_argument('--bam', required=True,
                        help='Path to aligned BAM with move tables (dorado --emit-moves --reference)')
    parser.add_argument('--level-table', required=True,
                        help='Path to k-mer level table (e.g., 9mer_levels_v1.txt)')
    parser.add_argument('--output', default=None,
                        help='Output moves TSV file (rawhash2 --moves-file format)')
    parser.add_argument('--output-peaks', default=None,
                        help='Output peaks TSV file (rawhash2 --peaks-file format, sample-level)')
    parser.add_argument('--output-bam', default=None,
                        help='Output BAM file with refined mv/ts tags (optional)')
    parser.add_argument('--no-refine', action='store_true', default=False,
                        help='skip the level-table refinement: write the mapping derived from the raw move table '
                             'and the alignment (a control for the value of refinement)')
    parser.add_argument('--ref-mapping', action='store_true', default=False,
                        help='Use reference-anchored refinement (ref_to_signal) instead of '
                             'query-anchored (query_to_signal). Produces sample-level peaks.')
    parser.add_argument('--min-mapq', type=int, default=0,
                        help='Minimum mapping quality filter (default: 0)')
    parser.add_argument('--rough-rescale', action='store_true', default=True,
                        help='Enable rough rescaling (default: True)')
    parser.add_argument('--no-rough-rescale', action='store_false', dest='rough_rescale',
                        help='Disable rough rescaling')
    args = parser.parse_args()

    # Validate: at least one output must be specified
    if args.output is None and args.output_peaks is None:
        parser.error("At least one of --output or --output-peaks is required.")

    print(f"Loading pod5 readers from: {args.pod5}", file=sys.stderr)
    read_reader_map = get_pod5_readers(args.pod5)
    print(f"  Found {len(read_reader_map)} reads in pod5", file=sys.stderr)

    print(f"Opening BAM: {args.bam}", file=sys.stderr)
    bam_fh = pysam.AlignmentFile(args.bam, 'rb', check_sq=False)

    print(f"Initializing SigMapRefiner with: {args.level_table}", file=sys.stderr)
    print(f"  ref_mapping: {args.ref_mapping}", file=sys.stderr)
    print(f"  min_mapq: {args.min_mapq}", file=sys.stderr)
    print(f"  rough_rescale: {args.rough_rescale}", file=sys.stderr)
    sig_map_refiner = refine_signal_map.SigMapRefiner(
        kmer_model_filename=args.level_table,
        do_rough_rescale=args.rough_rescale,
        scale_iters=0,
        do_fix_guage=True,
    )

    n_refined = 0
    n_skipped = 0
    n_skipped_mapq = 0
    n_total = 0

    # Open output BAM if requested
    bam_out_fh = None
    if args.output_bam:
        bam_out_fh = pysam.AlignmentFile(args.output_bam, 'wb', header=bam_fh.header)
        print(f"Will write refined BAM to: {args.output_bam}", file=sys.stderr)

    # Open output file handles
    moves_fh = open(args.output, 'w') if args.output else None
    peaks_fh = open(args.output_peaks, 'w') if args.output_peaks else None

    try:
        for bam_read in tqdm(bam_fh, desc="Refining", file=sys.stderr):
            n_total += 1

            # Skip supplementary, secondary, unmapped
            if bam_read.is_supplementary or bam_read.is_secondary or bam_read.is_unmapped:
                n_skipped += 1
                continue

            # MAPQ filter
            if bam_read.mapping_quality < args.min_mapq:
                n_skipped_mapq += 1
                continue

            read_id = bam_read.query_name

            try:
                pod5_reader = read_reader_map[read_id]
                pod5_read = next(pod5_reader.reads(selection=[read_id]))
                io_read = io.Read.from_pod5_and_alignment(pod5_read, bam_read)

                # Get stride and original ts before refinement
                stride = io_read.stride
                original_ts = bam_read.get_tag('ts')

                # Refine signal mapping
                if not args.no_refine:
                    io_read.set_refine_signal_mapping(
                        sig_map_refiner, ref_mapping=args.ref_mapping)

                if args.ref_mapping:
                    # Reference-anchored: get ref_to_signal (sample-level, trimmed space)
                    r2s = io_read.ref_to_signal

                    # Write peaks file: convert from trimmed to raw signal space
                    if peaks_fh is not None:
                        peaks_raw = r2s.astype(int) + original_ts
                        peak_strs = '\t'.join(str(p) for p in peaks_raw)
                        peaks_fh.write(f"{read_id}\t{peak_strs}\n")

                    # Also write moves TSV if requested (convert r2s to mv tag)
                    if moves_fh is not None:
                        mv_tag, ts_tag = query_to_signal_to_mv_tag(
                            r2s, stride, original_ts)
                        moves_fh.write(f"{read_id}\t{mv_tag}\t{ts_tag}\n")
                else:
                    # Query-anchored (original behavior)
                    q2s = io_read.query_to_signal

                    if moves_fh is not None:
                        mv_tag, ts_tag = query_to_signal_to_mv_tag(
                            q2s, stride, original_ts)
                        moves_fh.write(f"{read_id}\t{mv_tag}\t{ts_tag}\n")

                    # Also write peaks from query-space if requested
                    if peaks_fh is not None:
                        peaks_raw = q2s.astype(int) + original_ts
                        peak_strs = '\t'.join(str(p) for p in peaks_raw)
                        peaks_fh.write(f"{read_id}\t{peak_strs}\n")

                # Write refined BAM record with updated mv/ts tags
                if bam_out_fh is not None:
                    if args.ref_mapping:
                        sig_map = io_read.ref_to_signal
                    else:
                        sig_map = io_read.query_to_signal
                    mv_tag_str, _ = query_to_signal_to_mv_tag(
                        sig_map, stride, original_ts)
                    ts_offset = original_ts
                    moves = parse_mv_moves(mv_tag_str)
                    mv_arr = array.array('b', [stride] + moves)
                    bam_read.set_tag('mv', mv_arr)
                    bam_read.set_tag('ts', ts_offset)
                    bam_out_fh.write(bam_read)

                n_refined += 1

            except Exception as e:
                n_skipped += 1
                if n_skipped <= 5:
                    print(f"  Warning: skipped {read_id}: {e}", file=sys.stderr)
                continue

    finally:
        if moves_fh is not None:
            moves_fh.close()
        if peaks_fh is not None:
            peaks_fh.close()
        if bam_out_fh is not None:
            bam_out_fh.close()

    print(f"\nDone.", file=sys.stderr)
    print(f"  Total reads in BAM: {n_total}", file=sys.stderr)
    print(f"  Refined: {n_refined}", file=sys.stderr)
    print(f"  Skipped (flags): {n_skipped}", file=sys.stderr)
    print(f"  Skipped (mapq<{args.min_mapq}): {n_skipped_mapq}", file=sys.stderr)
    if args.output:
        print(f"  Output moves TSV: {args.output}", file=sys.stderr)
    if args.output_peaks:
        print(f"  Output peaks TSV: {args.output_peaks}", file=sys.stderr)
    if args.output_bam:
        print(f"  Output BAM: {args.output_bam}", file=sys.stderr)


if __name__ == '__main__':
    main()