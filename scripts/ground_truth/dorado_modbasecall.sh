#!/bin/bash
# Dorado modification basecalling for a single dataset -> modkit pileup.
#
# Produces per-site modification fractions so build_screened_candidates.py can
# define "confident unmodified" negatives (a non-motif A/C position that ALL
# active Dorado models call unmodified) rather than assuming every non-motif
# base is unmethylated. Runs the 6mA (A) + 5mC_5hmC (C) specialist models over
# v5.0.0 sup basecalls, aligned to the reference.
#
# Required env: DATASET, POD5_DIR, REF, OUTDIR
# Optional env: MODMODELS (space-sep --modified-bases shorthands, default
#               "6mA 5mC_5hmC")
#
#SBATCH --job-name=dorado_mod
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out

set -a; source "${RAWMOD_PATHS_FILE:-$HOME/.config/rawmod/paths.env}" 2>/dev/null || true; set +a  # site paths; see paths.env.example
set -euo pipefail

: "${DATASET:?}" ; : "${POD5_DIR:?}" ; : "${REF:?}" ; : "${OUTDIR:?}"

# Tools from the site configuration (paths.env.example). The modified-base models are
# the directories Dorado downloads with `dorado download --model <name>`.
DORADO=${RAWMOD_DORADO:-$(command -v dorado || true)}; : "${DORADO:?RAWMOD_DORADO must be set}"
SIMPLEX=${RAWMOD_DORADO_MODEL:?RAWMOD_DORADO_MODEL (dna_r10.4.1_e8.2_400bps_sup@v5.0.0) must be set}
MODELS_DIR=${RAWMOD_DORADO_MODELS:-$(dirname "${SIMPLEX}")}
# Pre-downloaded exact model dirs (A-context 6mA + C-context 5mC/5hmC all-context,
# NOT the CpG-only 5mCG variant — E. coli 5mC is at CCWGG/CpG, much of it non-CpG).
MOD_6MA=${MODELS_DIR}/dna_r10.4.1_e8.2_400bps_sup@v5.0.0_6mA@v1
MOD_5MC=${MODELS_DIR}/dna_r10.4.1_e8.2_400bps_sup@v5.0.0_5mC_5hmC@v2.0.1
# --modified-bases-models takes ONE comma-separated arg (paths ok).
MODMODELS=${MODMODELS:-"${MOD_6MA},${MOD_5MC}"}
SAMTOOLS=${SAMTOOLS:-$(command -v samtools || true)}; : "${SAMTOOLS:?samtools not found; set SAMTOOLS}"
MODKIT=${MODKIT:-${RAWMOD_ENV:+${RAWMOD_ENV}/bin/modkit}}; MODKIT=${MODKIT:-$(command -v modkit || true)}; : "${MODKIT:?modkit not found; set MODKIT}"

mkdir -p "${OUTDIR}"
MODBAM_UNS="${OUTDIR}/${DATASET}_mod.unsorted.bam"
MODBAM="${OUTDIR}/${DATASET}_mod.bam"
PILEUP="${OUTDIR}/${DATASET}_modkit_pileup.bed"

echo "=== [1/3] dorado basecaller (mods: ${MODMODELS}) — $(date) ==="
"${DORADO}" basecaller "${SIMPLEX}" "${POD5_DIR}" \
    --modified-bases-models "${MODMODELS}" \
    --reference "${REF}" \
    --device cuda:0 \
    > "${MODBAM_UNS}"

echo "=== [2/3] sort + index — $(date) ==="
"${SAMTOOLS}" sort -@ 8 -o "${MODBAM}" "${MODBAM_UNS}"
"${SAMTOOLS}" index "${MODBAM}"
rm -f "${MODBAM_UNS}"

echo "=== [3/3] modkit pileup — $(date) ==="
"${MODKIT}" pileup "${MODBAM}" "${PILEUP}" \
    --ref "${REF}" \
    --threads 8

echo "=== done ${DATASET} — $(date) ==="
wc -l "${PILEUP}"
ls -lh "${OUTDIR}/"
