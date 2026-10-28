#!/bin/bash
# Analysis of the A. carterae contig_3762 5hmU map: coding vs intergenic, repeats, intergenic regions and strand-switch
# junctions, strand concordance, GC-matched nulls, and the comparison with the Li et al. (2024) pull-down profiles.
# Inputs: $RAWMOD_WORK/dinoflagellate/contig_3762_strand{+,-}_scores.tsv.gz (from scripts/genome_map/submit.sh --base T,
# forward and reverse strand) and $RAWMOD_DATA/acarterae_contig_3762 (contig FASTA, GFF3 and RepeatMasker output).
set -euo pipefail
: "${RAWMOD_WORK:?set RAWMOD_WORK}" "${RAWMOD_DATA:?set RAWMOD_DATA}"
HERE=$(cd "$(dirname "$0")" && pwd); W=$RAWMOD_WORK/dinoflagellate; PY=${PYTHON:-python}
export MPLBACKEND=Agg PYTHONPATH=$HERE/../common:${PYTHONPATH:-}
mkdir -p "$W/out" "$W/strand"
cd "$W/out"
for s in dino_gaps dino_analysis dino_repeats dino_alignment_confound; do $PY "$HERE/$s.py" > $s.log 2>&1; done
$PY "$HERE/dino_literature.py" "$W/contig_3762_strand+_scores.tsv.gz" dino_literature > dino_literature.log 2>&1
$PY "$HERE/make_dino_figures.py" > make_figures.log 2>&1
$PY "$HERE/make_dino_tables.py" "$W/out" > make_tables.log 2>&1
cd "$W/strand"
for s in strand_concordance gc_matched_null genestrand_and_context array_blocks; do $PY "$HERE/$s.py" > $s.log 2>&1; done
# pull-down and bisulfite profiles of Li et al. (2024), lifted onto the contig ($RAWMOD_DATA/acarterae_li2024)
cd "$RAWMOD_DATA/acarterae_li2024"
$PY "$HERE/li2024/lift.py" > lift.log 2>&1
MODEL=all $PY "$HERE/li2024/dip_analysis.py" > dip_analysis.out 2>&1
MODEL=all $PY "$HERE/li2024/correlate.py" > correlate.out 2>&1
echo "done $(date)"
