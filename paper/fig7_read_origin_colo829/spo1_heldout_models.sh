#!/bin/bash
# SPO1 native-against-PCR typing with models that did not train on these reads (Supplementary Table S10).
# Same arm as the paper's SPO1 panel: per window, 200 random forward-strand T sites, the pc1 split + consensus of
# discover.py, coverage-matched scoring (coverage_matched.py, since the native and PCR reads differ in length).
#   released  checkpoints/rawmod.pt                    (all data; saw these reads)
#   region    checkpoints/region_holdout/mixed.pt      (never saw 52.6-72.5 kb of the genome or a read overlapping it)
#   loco5hmU  checkpoints/region_holdout/loco_5hmU.pt  (no 5hmU positive at all)
#   noSPO1    checkpoints/leaveout/spo1.pt             (no SPO1 read at all)
#
#   SPO1_DIR=... SPO1_REF=SPO1.fa bash spo1_heldout_models.sh [OUT_DIR]
#   python spo1_heldout_table.py collect OUT_DIR      # -> paper/data/work/read_origin/spo1_heldout.tsv
#
# SPO1_DIR holds one directory per barcode, bc01..bc05 (one PCR library each, against the native reads), with
# reads.pod5, reads_refined.bam (+ .bai) and peaks_refined.tsv (scripts/genome_map/submit.sh --until preprocess),
# windows.tsv and reads.tsv (window and read-group tables in the format of inputs/), and read_lengths.tsv.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
: "${SPO1_DIR:?set SPO1_DIR}" "${SPO1_REF:?set SPO1_REF}"
PY=${PYTHON:-python}; OUT=${1:-spo1_heldout}; mkdir -p "$OUT"
LEVELS=$REPO/rawmod/extern/uncalled_r1041_model_only_means.txt
export PILEUP_MASK_BASES=0
for bc in 01 02 03 04 05; do
  B=$SPO1_DIR/bc$bc
  for spec in "released checkpoints/rawmod.pt" "region checkpoints/region_holdout/mixed.pt" \
              "loco5hmU checkpoints/region_holdout/loco_5hmU.pt" "noSPO1 checkpoints/leaveout/spo1.pt"; do
    set -- $spec; D=$OUT/$1/bc$bc; mkdir -p "$D"
    [[ -s $D/emb.npz ]] || $PY "$HERE/emb_all_reads.py" --pod5 "$B/reads.pod5" --bam "$B/reads_refined.bam" \
        --peaks "$B/peaks_refined.tsv" --windows "$B/windows.tsv" --read-groups "$B/reads.tsv" --ref "$SPO1_REF" \
        --ckpt "$REPO/$2" --level-table "$LEVELS" --strand + --site-type t --max-sites 200 --min-reads 20 \
        --normalize --output "$D/emb.npz"
    [[ -s $D/covmatched.tsv ]] || {
      $PY "$HERE/discover.py" --npz "$D/emb.npz" --out-dir "$D" --per-window --split-method pc1 \
          --pos-group mod_native --read-meta "$B/reads.tsv"
      $PY "$HERE/coverage_matched.py" --npz "$D/emb.npz" --out "$D/covmatched.tsv" \
          --pos-group mod_native --read-lengths "$B/read_lengths.tsv"; }
  done
done
echo "done: $OUT"
