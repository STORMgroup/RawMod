#!/bin/bash
# Cost of every benchmarked caller on the same reads (Supplementary Table S27): Dorado's basecall with move tables
# (shared by every tool), Dorado's four modification models (one pass each, and the fewest passes: 6mA + 4mC_5mC
# share one), DeepMod2, Rockfish, the released UniMeth 5mC and 6mA models (sharing one signal index), and RawMod
# (refinement in 8 read shards, then featurization fused with scoring in 8 genome chunks per strand, scoring on the
# GPU and on the CPU only). Each tool runs with the benchmark's own command (test/benchmark/bench/steps.py). A
# baseline whose setting is empty is skipped.
#
#   POD5=reads.pod5 REF=ref.fa bash tool_cost.sh [OUT_DIR]
#   python make_cost_table.py OUT_DIR/timing.tsv      # the table (default: the shipped measurement)
#
# Settings use the benchmark's names (test/benchmark/config/site.example.env; a site.env there is read if present):
# DORADO, DORADO_MODELS, SAMTOOLS, RAWMOD_PYTHON, REFINE_PYTHON, DEEPMOD2 (+ DEEPMOD2_PYTHON), ROCKFISH, ROCKFISH_MODEL,
# UNIMETH, UNIMETH_CHECKPOINTS; CHECKPOINT (default checkpoints/rawmod.pt). The paper ran 13,110 H. pylori 26695
# reads on one node with 8 CPU cores, one RTX A5000 (24 GB) and 64 GB of RAM; the result is shipped in
# paper/data/work/cost/tool_cost.tsv.
set -uo pipefail
SUPP_COST_DIR=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$SUPP_COST_DIR/../.." && pwd)
SITE=$REPO/test/benchmark/config/site.env; [[ -f $SITE ]] && { set -a; source "$SITE"; set +a; }
: "${POD5:?set POD5}" "${REF:?set REF}" "${DORADO_MODELS:?set DORADO_MODELS}"
DORADO=${DORADO:-$(command -v dorado)}; SAM=${SAMTOOLS:-samtools}; PY=${RAWMOD_PYTHON:-python}; RPY=${REFINE_PYTHON:-$PY}
CK=${CHECKPOINT:-$REPO/checkpoints/rawmod.pt}; N=8
LEVELS=$REPO/rawmod/extern/uncalled_r1041_model_only_means.txt; REFINE=$REPO/rawmod/extern/refine_moves_remora.py
OUT=${1:-tool_cost}; mkdir -p "$OUT"; cd "$OUT"
source "$SUPP_COST_DIR/timing_lib.sh"
M=$DORADO_MODELS; B=dna_r10.4.1_e8.2_400bps_sup@v5.0.0

run dorado_basecall "$DORADO basecaller $M/$B $POD5 --emit-moves --reference $REF --device cuda:0 > base.bam && \
  $SAM sort -@8 -o base.sorted.bam base.bam && $SAM index base.sorted.bam && rm base.bam"
for mm in 6mA@v3 4mC_5mC@v3 5mC_5hmC@v3 5mCG_5hmCG@v3; do
  run dorado_$mm "$DORADO basecaller $M/$B $POD5 --modified-bases-models $M/${B}_$mm --reference $REF --device cuda:0 > mods_$mm.bam"
done
# Dorado's best case: models on different bases share one pass; the three cytosine models cannot, so four models
# need at least three passes. Timed here: 6mA + 4mC_5mC in one pass.
run dorado_6mA+4mC_5mC@v3 "$DORADO basecaller $M/$B $POD5 --modified-bases-models $M/${B}_6mA@v3,$M/${B}_4mC_5mC@v3 --reference $REF --device cuda:0 > mods_6mA+4mC_5mC@v3.bam"
[[ -n ${DEEPMOD2:-} ]] && run deepmod2 "${DEEPMOD2_PYTHON:-$PY} $DEEPMOD2 detect --bam base.sorted.bam --input $POD5 --file_type pod5 \
  --model bilstm_r10.4.1_5khz_v5.0 --ref $REF --seq_type dna --threads 8 --bam_threads 4 --batch_size 1024 --device cuda \
  --output deepmod2 --prefix output --skip_unmapped"
[[ -n ${ROCKFISH:-} ]] && run rockfish "export TMPDIR=\$(mktemp -d); $ROCKFISH inference -i $POD5 --bam_path base.sorted.bam \
  --model_path $ROCKFISH_MODEL -t 8 -b 4096 -d 0 -o rockfish_predictions.tsv"
if [[ -n ${UNIMETH:-} ]]; then
  run unimeth_5mC "PYTHONNOUSERSITE=1 $UNIMETH infer --pod5 $POD5 --bam base.sorted.bam --model $UNIMETH_CHECKPOINTS/unimeth_r10.4.1_5kHz_5mC.pt \
    --5mCpG 1 --5mCHG 1 --5mCHH 1 --pore_type R10.4.1 --frequency 5khz --output_format tsv --out unimeth_5mC.tsv --batch_size 128 --signal_index signal-index.sqlite"
  run unimeth_6mA "PYTHONNOUSERSITE=1 $UNIMETH infer --pod5 $POD5 --bam base.sorted.bam --model $UNIMETH_CHECKPOINTS/unimeth_r10.4.1_5kHz_6mA.pt \
    --6mA 1 --pore_type R10.4.1 --frequency 5khz --output_format tsv --out unimeth_6mA.tsv --batch_size 128 --signal_index signal-index.sqlite"
fi

# RawMod after the shared basecall
rawmod_refine $N
split_chunks $N
FEAT="$PY $REPO/rawmod/featurization.py --pod5 $POD5 --level-table $LEVELS --max-reads 15 --half-window 10 --L 10 \
  --min-reads 10 --min-mapq 20 --normalize --max-images-per-base 1"
for dev in gpu cpu; do env=$([ $dev = cpu ] && echo "CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1" || true)
  for s in plus minus; do sd=$([ $s = plus ] && echo + || echo -); o=$([ $s = minus ] && echo "--orient read" || true)
    run rawmod_fused_${s}_x${N}_$dev "seq 0 $((N-1)) | $env PILEUP_MASK_BASES=0 xargs -P $N -I{} $FEAT --bam chunk{}.bam --peaks peaks.tsv \
      --candidate-bed chunk{}.cand --strand '$sd' $o --score-checkpoint $CK --scores-out rawmod_${s}_${dev}_{}.tsv.gz"
  done
done
# same sites and reads: the CPU and GPU scores must agree to float tolerance
$PY - <<'PYEOF' | tee rawmod_cpu_vs_gpu.txt
import glob, numpy as np, pandas as pd
for s in ('plus', 'minus'):
    g = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(glob.glob(f'rawmod_{s}_gpu_*.tsv.gz'))])
    c = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(glob.glob(f'rawmod_{s}_cpu_*.tsv.gz'))])
    m = g.merge(c, on=['contig', 'pos'], suffixes=('_g', '_c'))
    print(f'{s}: gpu {len(g)} cpu {len(c)} matched {len(m)} max|gpu-cpu| {np.abs(m.score_g - m.score_c).max():.2e}')
PYEOF
rm -rf mods_*.bam deepmod2 rockfish_predictions.tsv unimeth_*.tsv rawmod_*.tsv.gz chunk* refined.bam* peaks.tsv base.sorted.bam*
echo "done: $OUT/timing.tsv"
