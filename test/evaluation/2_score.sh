#!/bin/bash
# Tables and figures of the benchmark. Three test sets, each in its own output directory (see test/README.md, "Outputs"):
#   --holdout region     inside the genomic regions the region-holdout models never trained on: Table 1 and Figure 3
#   --holdout positions  at the positions the released RawMod model never trained on (supplementary version)
#   (none)               every site
#   bash test/evaluation/2_score.sh
set -euo pipefail
B=$(cd "$(dirname "$0")/../benchmark" && pwd)
python "$B/rawmod_bench.py" score --holdout region "$@"
python "$B/rawmod_bench.py" score --holdout positions "$@"
python "$B/rawmod_bench.py" score "$@"
