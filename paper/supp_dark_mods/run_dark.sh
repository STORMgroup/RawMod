#!/bin/bash
# The A. carterae panel of the dark-modification figure: Dorado's all-context calls on contig_3762, a set of + strand
# sites chosen by what Dorado says about them, their RawMod pileups and the released model's embeddings.
#   bash run_dark.sh WORK_DIR CONTIG.fasta POD5_DIR REFINED.bam PEAKS.tsv
# Needs DORADO and DORADO_MODELS (basecaller and the modification models, as in ~/.config/rawmod/paths.env),
# SAMTOOLS and MODKIT; REFINED.bam and PEAKS.tsv are the refined alignment of the contig's reads
# (scripts/genome_map/run_local.sh writes both). Writes WORK_DIR/embedding_dark_acart.npz for make_dark_mods_fig.py.
set -euo pipefail
D=$1; REF=$2; POD5=$3; BAM=$4; PEAKS=$5
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
[[ -f ~/.config/rawmod/paths.env ]] && set -a && source ~/.config/rawmod/paths.env && set +a
: "${DORADO:?set DORADO (dorado binary)}" "${DORADO_MODELS:?set DORADO_MODELS (directory of basecalling and modification models)}"
SAMTOOLS=${SAMTOOLS:-samtools}; MODKIT=${MODKIT:-modkit}; PY=${RAWMOD_PYTHON:-python}
BC=dna_r10.4.1_e8.2_400bps_sup@v5.0.0
declare -A MODS=( [passA]="$DORADO_MODELS/${BC}_6mA@v3,$DORADO_MODELS/${BC}_5mC_5hmC@v3" [passB]="$DORADO_MODELS/${BC}_4mC_5mC@v3" )
mkdir -p "$D/bc"
for pass in passA passB; do
  [[ -s $D/bc/$pass.bed ]] && continue
  for P in "$POD5"/*.pod5; do
    n=$(basename "$P" .pod5); [[ -s $D/bc/${pass}_$n.sorted.bam ]] && continue
    "$DORADO" basecaller --reference "$REF" --modified-bases-models "${MODS[$pass]}" "$DORADO_MODELS/$BC" "$P" \
      2> "$D/bc/${pass}_$n.err" | "$SAMTOOLS" sort -@ 8 -o "$D/bc/${pass}_$n.sorted.bam" -
  done
  "$SAMTOOLS" merge -f -@ 8 "$D/bc/$pass.bam" "$D"/bc/${pass}_*.sorted.bam && "$SAMTOOLS" index "$D/bc/$pass.bam"
  "$MODKIT" pileup "$D/bc/$pass.bam" "$D/bc/$pass.bed" --ref "$REF" -t 16 --log-filepath "$D/bc/$pass.modkit.log"
done
"$PY" "$HERE/select_sites.py" "$D" "$REF"
[[ -s $D/features.h5 ]] || "$PY" "$REPO/rawmod/featurization.py" --pod5 "$POD5" --bam "$BAM" --peaks "$PEAKS" \
  --level-table "$REPO/rawmod/extern/uncalled_r1041_model_only_means.txt" --target-bases ACGT --strand + \
  --candidate-bed "$D/candidates.tsv" --max-reads 15 --half-window 10 --L 10 --min-reads 10 --min-mapq 20 --normalize \
  --max-images-per-base 1 --output "$D/features.h5"
"$PY" "$HERE/embed_sites.py" "$D"
