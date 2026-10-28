# Third-party files

| File | Origin | License |
|---|---|---|
| `refine_moves_remora.py` | RawHash2 (`test/scripts/refine_moves_remora.py`): refines Dorado move tables against a k-mer level table with Remora, writing reference-anchored signal boundaries (`peaks`) | GPL-3.0 |
| `uncalled_r1041_model_only_means.txt` | Expected current level of every 9-mer for R10.4.1 pores, as distributed with Uncalled4 (derived from Oxford Nanopore Technologies' public k-mer models) | see Uncalled4 and ONT kmer_models |

Both are included so that RawMod runs from this repository alone. They are used unchanged by `scripts/genome_map/` and `scripts/evaluate/`.

Local change to `refine_moves_remora.py`: a `--no-refine` option that skips the level-table refinement and writes the mapping Remora derives from the raw move table and the alignment (used by the benchmark's `BENCH_REFINE=none` control).
