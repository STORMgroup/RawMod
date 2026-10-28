#!/bin/bash
# UniMeth-Agn: one modification-agnostic UniMeth model fine-tuned on every training image of the released RawMod model,
# the UniMeth counterpart of RawMod's single model. Each RawMod training image (library, position, strand, label) labels
# the reads of that library's benchmark read set at that base (bench.unimeth_agn_labels), whatever the modification:
# C, A and T bases carry the labels under one code per base, and UniMeth asks the same question at every C ([CpG],
# [CHG] or [CHH]), A ([m6A]) and T ([5hmU]). Every other base is masked, so the benchmark's held-out positions stay
# unseen. Base model: the released 5 kHz 5mC model. Loss: UniMeth's class- and token-balanced loss (--plant 1), so that
# each question token and each label contributes equally. --frequency 4khz in the patched 0.3.1 applies the same signal
# normalization as the released 0.3.3 at 5 kHz for Dorado >= 0.9.2 basecalls.
# UNIMETH_AGN_MOD_TOKEN=1 trains the [MOD] variant instead: one general question token [MOD] at every C, A and T (the
# vocabulary row starts as the mean of the released question tokens), run with the source in UNIMETH_MODTOK_PYTHONPATH
# (the fine-tuning source plus the mod-token patch); same labels and validation reads; output in run_mod/.
# UNIMETH_AGN_REGION=1 builds the region-holdout version: the same labels, but every read overlapping the library's
# held-out region (config/holdout_region.tsv) is dropped (labels --exclude-region), as for the region UniMeth-FT. Its
# labels, validation reads and runs go to $UNIMETH_AGN_REGION_CHECKPOINTS (default $BENCH_WORK/unimeth_agn_region).
# UNIMETH_AGN_LEAVEOUT=NAME:lib1,lib2 trains the same model without
# those libraries (no labels, no validation reads from them), as RawMod's leave-out models withhold the row's
# organism or library; output in <run dir>_lo_NAME/. The labels of the other libraries are reused unchanged.
#
# Requires: the benchmark basecalls of the 16 libraries (rawmod_bench.py run --steps subset,basecall), UNIMETH,
# UNIMETH_CHECKPOINTS and UNIMETH_FT_PYTHONPATH (UniMeth 11215d4 + patches/unimeth-11215d4-finetune.patch), one GPU.
#   bash test/evaluation/train_unimeth_agn.sh [max_steps]       (default 12000)
# Output: $UNIMETH_AGN_CHECKPOINTS/final.pt (default $BENCH_WORK/unimeth_agn).
set -euo pipefail
HERE=$(cd "$(dirname "$0")/../benchmark" && pwd)
cd "$HERE"
STEPS=${1:-12000}
PY=$(python3 -c "from bench import settings as S; print(S.get('RAWMOD_PYTHON') or 'python')")
setting() { $PY -c "from bench import settings as S; print(S.need('$1', 'UniMeth-Agn training'))"; }
sample() { $PY -c "from bench import common as C; print(C.samples()['$1']['$2'])"; }
UNIMETH=$(setting UNIMETH); CK=$(setting UNIMETH_CHECKPOINTS); FT=$(setting UNIMETH_FT_PYTHONPATH)
RUNDIR=run; TOKENS=""
if [ "${UNIMETH_AGN_MOD_TOKEN:-0}" = 1 ]; then FT=$(setting UNIMETH_MODTOK_PYTHONPATH); RUNDIR=run_mod; TOKENS="--mod_token 1"; fi
UPY=$(dirname "$UNIMETH")/python
if [ "${UNIMETH_AGN_REGION:-0}" = 1 ]; then OKEY=UNIMETH_AGN_REGION_CHECKPOINTS; ODEF=unimeth_agn_region; LARGS=--exclude-region
else OKEY=UNIMETH_AGN_CHECKPOINTS; ODEF=unimeth_agn; LARGS=""; fi
OUT=$($PY -c "from bench import settings as S; import os; print(S.get('$OKEY') or os.path.join(S.get('BENCH_WORK'), '$ODEF'))")
mkdir -p "$OUT/bam" "$OUT/pod5" "$OUT/$RUNDIR"
LIBS=(syn_5mC syn_5hmC syn_6mA syn_control hp26695 hp26695_wga spo1_bc07 spo1_bc01 ecoli_wt_w1m ecoli_dm_w1m
      ecoli_mssi_w1m anabaena hpj99 tdenticola arabidopsis_chr1 hg002)
if [ -n "${UNIMETH_AGN_LEAVEOUT:-}" ]; then
  LO_NAME=${UNIMETH_AGN_LEAVEOUT%%:*}; IFS=, read -ra LO <<< "${UNIMETH_AGN_LEAVEOUT#*:}"
  KEEP=(); for L in "${LIBS[@]}"; do [[ " ${LO[*]} " == *" $L "* ]] || KEEP+=("$L"); done
  [ ${#KEEP[@]} -eq $(( ${#LIBS[@]} - ${#LO[@]} )) ] || { echo "unknown library in $UNIMETH_AGN_LEAVEOUT" >&2; exit 1; }
  LIBS=("${KEEP[@]}"); RUNDIR=${RUNDIR}_lo_$LO_NAME; mkdir -p "$OUT/$RUNDIR"
  echo "leave-out $LO_NAME: training on ${#LIBS[@]} libraries: ${LIBS[*]}"
fi

BAMS=(); TRAIN=(); VAL=()
for L in "${LIBS[@]}"; do
  # 1. labels at RawMod training images of this library
  [ -s "$OUT/bam/$L.bam" ] || BENCH_HOLDOUT=positions $PY -m bench.unimeth_agn_labels --sample $L --out "$OUT/bam/$L.bam" $LARGS
  # 2. validation reads: the first 200 labelled reads of the library
  if [ ! -s "$OUT/pod5/${L}_val.pod5" ]; then
    $UPY -c "import pysam; f=pysam.AlignmentFile('$OUT/bam/$L.bam'); ids=[]
for r in f:
    if r.has_tag('MM') and r.query_name not in ids: ids.append(r.query_name)
    if len(ids) >= 200: break
open('$OUT/pod5/${L}_val_ids.txt','w').write('\n'.join(ids)+'\n')"
    $(dirname "$UNIMETH")/pod5 filter --ids "$OUT/pod5/${L}_val_ids.txt" --missing-ok \
      --output "$OUT/pod5/${L}_val.pod5" "$(sample $L sub_pod5)"
  fi
  BAMS+=("$OUT/bam/$L.bam"); TRAIN+=("$(sample $L sub_pod5)"); VAL+=("$OUT/pod5/${L}_val.pod5")
done
join() { local IFS=,; echo "$*"; }

# 3. fine-tuning
RUN=$OUT/$RUNDIR
if [ -s "$RUN/final.pt" ]; then echo "$RUN/final.pt exists"; exit 0; fi
for B in "${BAMS[@]}"; do  # UniMeth builds its read index before the data-loader workers start
  PYTHONNOUSERSITE=1 PYTHONPATH=$FT $UPY -c "from unimeth.ioutils.reader import BamReader; BamReader('$B')" >/dev/null
done
(cd "$RUN" && PYTHONNOUSERSITE=1 PYTHONPATH=$FT PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True UNIMETH_OUT_DIR=$RUN \
   UNIMETH_RESUME=0 UNIMETH_LOG_STEPS=50 UNIMETH_EVAL_STEPS=1000 UNIMETH_SAVE_STEPS=1000 UNIMETH_DL_WORKERS=6 \
   $UPY -m unimeth.training --mode finetune --bam_dir "$(join "${BAMS[@]}")" --train_pod5_dir "$(join "${TRAIN[@]}")" \
     --val_pod5_dir "$(join "${VAL[@]}")" --model_dir "$CK/unimeth_r10.4.1_5kHz_5mC.pt" --pore_type R10.4.1 \
     --frequency 4khz --cpg 1 --chg 1 --chh 1 --m6A 1 --hmU 1 --agnostic 1 $TOKENS --plant 1 --dorado_version 1.4 \
     --max_steps "$STEPS" --batch_size 32 --run_name agnostic_$RUNDIR > "$RUN/train.log" 2>&1)
echo "$RUN/final.pt"
