#!/bin/bash
# Host and GPU memory of RawMod's fused featurization + scoring against the number of parallel processes
# (Supplementary Section S10.4). The reads are basecalled once with Dorado (--emit-moves), refined with the bundled
# Remora refiner, split into N chunks of equal read coverage and scored on both strands with N processes sharing one GPU.
# Measurements as in tool_cost.sh (timing_lib.sh).
#
#   POD5=reads.pod5 REF=ref.fa bash memory_scaling.sh [OUT_DIR]
#   python make_memory_table.py OUT_DIR/timing.tsv     # the summary (default: the shipped measurement)
#
# Settings as in tool_cost.sh (DORADO, DORADO_MODELS, SAMTOOLS, RAWMOD_PYTHON, REFINE_PYTHON, CHECKPOINT), plus PROCS
# (process counts, default "1 2 4 8"). The paper ran 13,110 H. pylori 26695 reads on one RTX A5000 (24 GB) with 8 CPU
# cores; the result is shipped in paper/data/work/cost/memory_scaling.tsv.
set -uo pipefail
SUPP_COST_DIR=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$SUPP_COST_DIR/../.." && pwd)
SITE=$REPO/test/benchmark/config/site.env; [[ -f $SITE ]] && { set -a; source "$SITE"; set +a; }
: "${POD5:?set POD5}" "${REF:?set REF}" "${DORADO_MODELS:?set DORADO_MODELS}"
DORADO=${DORADO:-$(command -v dorado)}; SAM=${SAMTOOLS:-samtools}; PY=${RAWMOD_PYTHON:-python}; RPY=${REFINE_PYTHON:-$PY}
CK=${CHECKPOINT:-$REPO/checkpoints/rawmod.pt}; PROCS=${PROCS:-1 2 4 8}
LEVELS=$REPO/rawmod/extern/uncalled_r1041_model_only_means.txt; REFINE=$REPO/rawmod/extern/refine_moves_remora.py
OUT=${1:-memory_scaling}; mkdir -p "$OUT"; cd "$OUT"
source "$SUPP_COST_DIR/timing_lib.sh"
B=dna_r10.4.1_e8.2_400bps_sup@v5.0.0

run dorado_basecall "$DORADO basecaller $DORADO_MODELS/$B $POD5 --emit-moves --reference $REF --device cuda:0 > base.bam && \
  $SAM sort -@8 -o base.sorted.bam base.bam && $SAM index base.sorted.bam && rm base.bam"
rawmod_refine 8
# cost of an idle torch process with CUDA (the per-process overhead)
run torch_cuda_baseline "$PY -c 'import torch,time; x=torch.zeros(1).cuda(); time.sleep(8)'"

FEAT="$PY $REPO/rawmod/featurization.py --pod5 $POD5 --level-table $LEVELS --max-reads 15 --half-window 10 --L 10 \
  --min-reads 10 --min-mapq 20 --normalize --max-images-per-base 1"
for N in $PROCS; do
  split_chunks "$N"
  for s in plus minus; do sd=$([ $s = plus ] && echo + || echo -); o=$([ $s = minus ] && echo "--orient read" || true)
    run fused_${s}_x${N}_gpu "seq 0 $((N-1)) | PILEUP_MASK_BASES=0 xargs -P $N -I{} $FEAT --bam chunk{}.bam --peaks peaks.tsv \
      --candidate-bed chunk{}.cand --strand '$sd' $o --score-checkpoint $CK --score-batch 1024 --scores-out s_${s}_x${N}_{}.tsv.gz"
  done
  rm -f s_*_x${N}_*.tsv.gz
done
rm -rf chunk* refined.bam* peaks.tsv base.sorted.bam*
echo "done: $OUT/timing.tsv"
