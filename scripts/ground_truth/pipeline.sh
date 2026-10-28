#!/bin/bash
# Nanopore modification detection pipeline: Dorado → Remora → Featurization
#
# Required env vars (set by submit_all.sh or --export):
#   DATASET       - short name for this dataset (used as output subdir)
#   POD5_DIR      - path to directory of pod5 files (may contain subdirs)
#   REF           - path to reference FASTA (may be .fa.gz or .fna.bgz)
#   GT_BED        - path to ground-truth BED (tab: ref_name, 0-based pos).
#                   Must be from an orthogonal method (emseq/bisulfite/motif).
#                   Featurization is skipped if this is missing or empty.
#   CANDIDATE_BED - (optional) BED of high-confidence +/- candidate sites.
#                   When set, only these positions are featurized, preventing
#                   ambiguous sites from being mislabeled as negative.
#   RECURSIVE     - set to "true" to pass --recursive to Dorado (for nested pod5 dirs)

#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -a; source "${RAWMOD_PATHS_FILE:-$HOME/.config/rawmod/paths.env}" 2>/dev/null || true; set +a  # site paths; see paths.env.example
set -euo pipefail

# ── Tools (site configuration: paths.env.example) ──────────────────────────────
REPO=${RAWMOD_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
DORADO_BIN=${RAWMOD_DORADO:-$(command -v dorado || true)}
MODEL=${RAWMOD_DORADO_MODEL:?RAWMOD_DORADO_MODEL (the Dorado basecalling model) must be set}
: "${DORADO_BIN:?RAWMOD_DORADO (the dorado executable) must be set}"
REFINE_PY=${REPO}/rawmod/extern/refine_moves_remora.py
REFINE_ENV=${RAWMOD_REFINE_ENV:-remora}
LEVEL_TABLE=${RAWMOD_LEVEL_TABLE:-${REPO}/rawmod/extern/uncalled_r1041_model_only_means.txt}
FEATURIZE=${REPO}/rawmod/featurization.py
FEATURIZE_ENV=${RAWMOD_ENV:-}
# conda is used only to switch between the Remora and the RawMod environments; without
# CONDA_SH the steps run in the environment this script was started from.
activate() { if [[ -n "${CONDA_SH:-}" && -n "$1" ]]; then set +u; source "${CONDA_SH}"; conda activate "$1"; set -u; fi; }

# ── Validate required env vars ─────────────────────────────────────────────────
: "${DATASET:?DATASET env var is required}"
: "${POD5_DIR:?POD5_DIR env var is required}"
: "${REF:?REF env var is required}"

OUTDIR=${RAWMOD_RESULTS}/benchmark_results/${DATASET}
mkdir -p "${OUTDIR}"

echo "================================================================"
echo " Dataset  : ${DATASET}"
echo " Pod5 dir : ${POD5_DIR}"
echo " Reference: ${REF}"
echo " GT BED   : ${GT_BED:-<none>}"
echo " Recursive: ${RECURSIVE:-false}"
echo " Output   : ${OUTDIR}"
echo " Host     : $(hostname)"
echo " Date     : $(date)"
echo "================================================================"

# ── Step 1: Dorado basecalling ─────────────────────────────────────────────────
echo ""
echo "=== [1/3] Dorado basecalling ==="
if [[ -s "${OUTDIR}/reads.bam" ]]; then
    echo "  [skip] reads.bam already exists ($(du -h "${OUTDIR}/reads.bam" | cut -f1))"
else
    [[ -f "${OUTDIR}/reads.bam" ]] && rm -f "${OUTDIR}/reads.bam"
    RECURSIVE_ARG=""
    [[ "${RECURSIVE:-false}" == "true" ]] && RECURSIVE_ARG="--recursive"

    # Move tables are required: they link the raw signal to the called bases.
    "${DORADO_BIN}" basecaller --emit-moves --reference "${REF}" ${RECURSIVE_ARG} \
        "${MODEL}" "${POD5_DIR}" > "${OUTDIR}/reads.bam.tmp" 2> "${OUTDIR}/dorado.log"
    mv "${OUTDIR}/reads.bam.tmp" "${OUTDIR}/reads.bam"
fi

# ── Step 2: Remora refinement (reference-anchored, produces peaks_refined.tsv) ─
echo ""
echo "=== [2/3] Remora refinement ==="
if [[ -s "${OUTDIR}/peaks_refined.tsv" ]]; then
    echo "  [skip] peaks_refined.tsv already exists ($(du -h "${OUTDIR}/peaks_refined.tsv" | cut -f1))"
else
    activate "${REFINE_ENV}"
    python "${REFINE_PY}" --pod5 "${POD5_DIR}" --bam "${OUTDIR}/reads.bam" \
        --level-table "${LEVEL_TABLE}" --ref-mapping --min-mapq 20 \
        --output "${OUTDIR}/moves_refined.tsv" --output-peaks "${OUTDIR}/peaks_refined.tsv" \
        --output-bam "${OUTDIR}/reads_refined.bam"
    samtools index "${OUTDIR}/reads_refined.bam"
fi

# ── Step 3: Featurization ──────────────────────────────────────────────────────
echo ""
echo "=== [3/3] Featurization ==="
if [[ -s "${OUTDIR}/features.h5" ]]; then
    echo "  [skip] features.h5 already exists ($(du -h "${OUTDIR}/features.h5" | cut -f1))"
else
  # GT is required for supervised training. Skip featurization if not yet available
  # so Dorado/Remora results are preserved and featurization can be rerun later.
    if [[ -z "${GT_BED:-}" || ! -f "${GT_BED}" ]]; then
        echo "  [skip] GT_BED not set or file missing — skipping featurization."
        echo "         Set GT_BED and rerun this job to produce features.h5."
        exit 0
    fi

    activate "${FEATURIZE_ENV}"

    CANDIDATE_ARG=""
    if [[ -n "${CANDIDATE_BED:-}" && -f "${CANDIDATE_BED}" ]]; then
        CANDIDATE_ARG="--candidate-bed ${CANDIDATE_BED}"
    fi

  # MIN_READS: 25 for bacteria/plants (20x+ coverage), 5 for mouse/human (5-7x)
    : "${MIN_READS:=25}"
  # SAMPLE_N_SITES: leave unset to use every eligible site (fine for small
  # bacterial genomes); set explicitly for large genomes with huge candidate
  # pools (arabidopsis/osativa/human) to stay within the storage budget.
  # MAX_IMAGES_PER_BASE stays at 1 — breadth of distinct sites is worth more
  # than extra images stacked on the same site.
    : "${MAX_IMAGES_PER_BASE:=1}"

    SAMPLE_ARG=""
    [[ -n "${SAMPLE_N_SITES:-}" ]] && SAMPLE_ARG="--sample-n-sites ${SAMPLE_N_SITES}"

    python "${FEATURIZE}" \
        --pod5               "${POD5_DIR}" \
        --bam                "${OUTDIR}/reads_refined.bam" \
        --peaks              "${OUTDIR}/peaks_refined.tsv" \
        --output             "${OUTDIR}/features.h5" \
        --level-table        "${LEVEL_TABLE}" \
        --gt                 "${GT_BED}" \
        ${CANDIDATE_ARG} \
        --half-window        10 \
        --L                  10 \
        --max-reads          30 \
        --min-reads          "${MIN_READS}" \
        --min-mapq           60 \
        --target-bases       AC \
        ${SAMPLE_ARG} \
        --uniform-sampling \
        --max-images-per-base "${MAX_IMAGES_PER_BASE}"
fi

echo ""
echo "================================================================"
echo " Pipeline complete: ${DATASET}"
echo " $(date)"
echo " Output: ${OUTDIR}"
ls -lh "${OUTDIR}/" 2>/dev/null
echo "================================================================"
