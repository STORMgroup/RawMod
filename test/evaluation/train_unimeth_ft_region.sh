#!/bin/bash
# UniMeth-FT for the REGION-HOLDOUT benchmark: as train_unimeth_ft.sh, but every read that overlaps the held-out test
# region of the organism (data/holdout/regions/*.bed, the regions the region-holdout RawMod model never saw) is removed
# from the fine-tuning data, so UniMeth-FT, like RawMod, is scored on reads and genomic regions it never trained on. One model per modification RawMod covers and UniMeth does not:
#   5hmC  ONT oligonucleotides, syn_5hmC vs. syn_control     base: released 5 kHz 5mC model
#   4mC   H. pylori 26695, native vs. WGA                    base: released 5 kHz 5mC model
#   5hmU  SPO1, native vs. PCR                               base: released 5 kHz 6mA model
# Labels (bench.unimeth_ft_labels) are written into copies of the benchmark basecalls: label 255 / 0 only at RawMod
# training positions of the organism; every other base is masked. Fine-tuning: --frequency 4khz, 3,000 steps, batch 32,
# 800 validation reads per library.
#
# Requires: the benchmark basecalls of the six samples (rawmod_bench.py run --steps subset,basecall), UNIMETH,
# UNIMETH_CHECKPOINTS and UNIMETH_FT_PYTHONPATH (UniMeth 11215d4 + patches/unimeth-11215d4-finetune.patch), one GPU.
#   bash test/evaluation/train_unimeth_ft.sh [5hmC|4mC|5hmU ...]        (default: all three)
# Output: $UNIMETH_FT_REGION_CHECKPOINTS/<mod>/final.pt (default $BENCH_WORK/unimeth_ft_region).
set -euo pipefail
HERE=$(cd "$(dirname "$0")/../benchmark" && pwd)
cd "$HERE"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"  # the helpers below run from inside the run directory
PY=$(python3 -c "from bench import settings as S; print(S.get('RAWMOD_PYTHON') or 'python')")
setting() { $PY -c "from bench import settings as S; import sys; print(S.need('$1', 'UniMeth-FT training'))"; }
sample() { $PY -c "from bench import common as C; print(C.samples()['$1']['$2'])"; }
UNIMETH=$(setting UNIMETH); CK=$(setting UNIMETH_CHECKPOINTS); FT=$(setting UNIMETH_FT_PYTHONPATH)
UPY=$(dirname "$UNIMETH")/python
OUT=$($PY -c "from bench import settings as S; import os; print(S.get('UNIMETH_FT_REGION_CHECKPOINTS') or os.path.join(S.get('BENCH_WORK'), 'unimeth_ft_region'))")
REG=$HERE/../data/holdout/regions
mkdir -p "$OUT/bam" "$OUT/pod5"
GT4=$HERE/../data/unimeth_ft/hp26695_4mC_labels.bed.gz

MODS=("$@"); [ ${#MODS[@]} -gt 0 ] || MODS=(5hmC 4mC 5hmU)
for MOD in "${MODS[@]}"; do
  case $MOD in
    5hmC) POS=syn_5hmC;  NEG=syn_control; BASE=C; CODE=h;     ORG='^ONT::';  BASEMODEL=$CK/unimeth_r10.4.1_5kHz_5mC.pt
          POSARGS=(--pos-files '^ONT::5hmC'); EX=$REG/ont_oligos.bed;                FLAGS="--cpg 1 --chg 1 --chh 1 --hmC 1" ;;
    4mC)  POS=hp26695;   NEG=hp26695_wga; BASE=C; CODE=21839; ORG='^HP::';   BASEMODEL=$CK/unimeth_r10.4.1_5kHz_5mC.pt
          POSARGS=(--pos-files '^HP::WT' --gt "$GT4"); EX=$REG/hpylori_26695.bed;       FLAGS="--cpg 1 --chg 1 --chh 1 --m4C 1" ;;
    5hmU) POS=spo1_bc07; NEG=spo1_bc01;   BASE=T; CODE=g;     ORG='^SPO1::'; BASEMODEL=$CK/unimeth_r10.4.1_5kHz_6mA.pt
          POSARGS=(--all-modified); EX=$REG/spo1.bed;                          FLAGS="--hmU 1" ;;
    *) echo "unknown modification $MOD" >&2; exit 1 ;;
  esac
  # 1. labels at RawMod training positions (bench.holdout source table)
  [ -s "$OUT/bam/${MOD}_pos.bam" ] || BENCH_HOLDOUT=positions $PY -m bench.unimeth_ft_labels --sample $POS --state modified \
      --base $BASE --code $CODE --org-files "$ORG" "${POSARGS[@]}" --exclude-bed "$EX" --out "$OUT/bam/${MOD}_pos.bam"
  [ -s "$OUT/bam/${MOD}_neg.bam" ] || BENCH_HOLDOUT=positions $PY -m bench.unimeth_ft_labels --sample $NEG --state unmodified \
      --base $BASE --code $CODE --org-files "$ORG" --exclude-bed "$EX" --out "$OUT/bam/${MOD}_neg.bam"
  # 2. validation reads: the first 800 labelled reads of each library
  for S in pos neg; do
    smp=$([ $S = pos ] && echo $POS || echo $NEG)
    if [ ! -s "$OUT/pod5/${MOD}_${S}_val.pod5" ]; then
      $UPY -c "import pysam; f=pysam.AlignmentFile('$OUT/bam/${MOD}_${S}.bam'); ids=[]
for r in f:
    if r.has_tag('MM') and r.query_name not in ids: ids.append(r.query_name)
    if len(ids) >= 800: break
open('$OUT/pod5/${MOD}_${S}_val_ids.txt','w').write('\n'.join(ids)+'\n')"
      $(dirname "$UNIMETH")/pod5 filter --ids "$OUT/pod5/${MOD}_${S}_val_ids.txt" --missing-ok \
        --output "$OUT/pod5/${MOD}_${S}_val.pod5" "$(sample $smp sub_pod5)"
    fi
  done
  # 3. fine-tuning
  RUN=$OUT/$MOD; mkdir -p "$RUN"
  if [ -s "$RUN/final.pt" ]; then echo "$MOD: $RUN/final.pt exists"; continue; fi
  for S in pos neg; do  # UniMeth builds its read index before the data-loader workers start
    PYTHONNOUSERSITE=1 PYTHONPATH=$FT $UPY -c "from unimeth.ioutils.reader import BamReader; BamReader('$OUT/bam/${MOD}_${S}.bam')" >/dev/null
  done
  # A run that dies (out of GPU memory, or UniMeth's zip_longest padding one library with None once the other is
  # exhausted) is continued from its latest checkpoint for the remaining steps, as the 5hmC parity run was
  # (UNIMETH_INIT_WEIGHTS; optimizer and LR schedule restart). Each continuation is a sub-directory from<step>.
  SEG=$RUN; DONE=0; INIT=
  for TRY in 1 2 3 4 5; do
    if [ ! -s "$SEG/train.log" ]; then
      (cd "$SEG" && PYTHONNOUSERSITE=1 PYTHONPATH=$FT PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True UNIMETH_OUT_DIR=$SEG \
         UNIMETH_RESUME=0 UNIMETH_INIT_WEIGHTS=$INIT UNIMETH_LOG_STEPS=50 UNIMETH_EVAL_STEPS=500 UNIMETH_SAVE_STEPS=500 \
         UNIMETH_DL_WORKERS=6 \
         $UPY -m unimeth.training --mode finetune --bam_dir "$OUT/bam/${MOD}_pos.bam,$OUT/bam/${MOD}_neg.bam" \
           --train_pod5_dir "$(sample $POS sub_pod5),$(sample $NEG sub_pod5)" \
           --val_pod5_dir "$OUT/pod5/${MOD}_pos_val.pod5,$OUT/pod5/${MOD}_neg_val.pod5" \
           --model_dir "$BASEMODEL" --pore_type R10.4.1 --frequency 4khz $FLAGS --dorado_version 1.4 \
           --max_steps $((3000 - DONE)) --batch_size 32 --run_name ${MOD}_ft_region > "$SEG/train.log" 2>&1) || true
    fi
    [ -s "$SEG/final.pt" ] && break
    N=$(ls -d "$SEG"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
    [ -n "$N" ] || { echo "$MOD: $SEG failed before its first checkpoint" >&2; exit 1; }
    CKP=$SEG/checkpoint-$N; DONE=$((DONE + N)); INIT=$CKP/pytorch_model.bin
    echo "continued from $CKP (step $DONE of 3000)" >> "$RUN/PROVENANCE.txt"
    SEG=$RUN/from$DONE; mkdir -p "$SEG"
  done
  [ -s "$SEG/final.pt" ] || { echo "$MOD: no final.pt after 5 attempts" >&2; exit 1; }
  [ "$SEG" = "$RUN" ] || ln -sfn "$SEG/final.pt" "$RUN/final.pt"
  echo "$MOD: $RUN/final.pt"
done
