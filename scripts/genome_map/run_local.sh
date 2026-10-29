#!/bin/bash
# Build a RawMod genome map on this machine, without a job scheduler.
#
# Same pipeline and same settings as submit.sh (basecall -> refine -> featurize -> score); every step that submit.sh
# sends to the cluster as an array job is run here as a loop. Use submit.sh on a cluster: a genome-scale map is hours
# of GPU work, and this runner has no parallelism beyond what each tool does itself.
#
#   bash scripts/genome_map/run_local.sh --pod5 DIR --ref genome.fasta --out DIR [options]
#
# Options are the ones of submit.sh (--bam, --base, --sites, --strands, --contigs, --checkpoint, --name, --from,
# --until, --dry-run) and are documented there.
set -euo pipefail
GM_DIR=$(cd "$(dirname "$0")" && pwd)
export GM_DIR RAWMOD_LOCAL=1
exec "$BASH" "$GM_DIR/submit.sh" --local "$@"
