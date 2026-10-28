# Shared by tool_cost.sh and memory_scaling.sh (source it). `run STAGE COMMAND` runs COMMAND in its own session and
# appends one row to timing.tsv: wall time, CPU time (rusage of every waited descendant), peak summed PSS and RSS over
# every process of the stage (sampled once a second), peak GPU memory, GPU-busy seconds (utilization x time, sampled
# twice a second) and GPU energy (power x time). Needs GNU time (/usr/bin/time -v) and nvidia-smi.
TIMING_HEADER="stage\twall_s\tcpu_s\tmax_rss_kb\tpeak_pss_sum_kb\tpeak_rss_sum_kb\tgpu_peak_mib\tgpu_busy_s\tgpu_energy_j"
[[ -s timing.tsv ]] || echo -e "$TIMING_HEADER" > timing.tsv

mem_sampler() {   # mem_sampler SID: "<summed PSS kB> <summed RSS kB>" over the session, once a second
  while kill -0 "$1" 2>/dev/null; do
    p=0; r=0; for q in $(ps -o pid= -s "$1" 2>/dev/null); do
      read -r a b < <(awk '/^Pss:/{p=$2} /^Rss:/{r=$2} END{print p+0, r+0}' /proc/$q/smaps_rollup 2>/dev/null) || continue
      p=$((p + ${a:-0})); r=$((r + ${b:-0})); done
    echo "$p $r"; sleep 1; done
}

run() {   # run STAGE COMMAND
  local st=$1; shift
  nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits -lms 500 > gpu_$st.txt 2>/dev/null & local gp=$!
  setsid /usr/bin/time -v -o time_$st.txt bash -c "$*" > log_$st.txt 2>&1 & local sid=$!
  mem_sampler $sid > mem_$st.txt & local mp=$!
  wait $sid; local rc=$?; kill $gp 2>/dev/null; wait $gp $mp 2>/dev/null
  local wall=$(awk -F': ' '/Elapsed \(wall clock\)/{n=split($2,a,":"); s=0; for(i=1;i<=n;i++) s=s*60+a[i]; print s}' time_$st.txt)
  local cpu=$(awk -F': ' '/User time/{u=$2} /System time/{s=$2} END{print u+s}' time_$st.txt)
  local rss=$(awk -F': ' '/Maximum resident/{print $2}' time_$st.txt)
  local pss=$(sort -n mem_$st.txt | tail -1 | cut -d' ' -f1); local rsum=$(sort -k2,2n mem_$st.txt | tail -1 | cut -d' ' -f2)
  local gpu=$(cut -d, -f1 gpu_$st.txt | sort -n | tail -1)
  local busy=$(awk -F', *' '{b+=$2/100*0.5} END{printf "%.1f", b}' gpu_$st.txt); local joule=$(awk -F', *' '{e+=$3*0.5} END{printf "%.0f", e}' gpu_$st.txt)
  echo -e "$st\t$wall\t$cpu\t$rss\t${pss:-0}\t${rsum:-0}\t${gpu:-0}\t$busy\t$joule" | tee -a timing.tsv
  [ $rc = 0 ] || echo "### $st exited $rc (see log_$st.txt)"
}

rawmod_refine() {   # rawmod_refine N: refine base.sorted.bam in N read shards -> refined.bam, peaks.tsv
  local N=$1
  run rawmod_refine_x$N "mkdir -p shards refine && $SAM view -F 0x900 base.sorted.bam | cut -f1 | sort -u | awk -v n=$N -v d=shards '{print > (d \"/names_\" (NR%n) \".txt\")}' && \
    for i in \$(seq 0 $((N-1))); do $SAM view -b -N shards/names_\$i.txt -o shards/shard_\$i.bam base.sorted.bam & done; wait; \
    seq 0 $((N-1)) | xargs -P $N -I{} $RPY $REFINE --pod5 $POD5 --bam shards/shard_{}.bam --level-table $LEVELS --ref-mapping --min-mapq 20 \
      --output refine/moves_{}.tsv --output-peaks refine/peaks_{}.tsv --output-bam refine/reads_refined_{}.bam && \
    cat refine/peaks_*.tsv > peaks.tsv && $SAM cat -o refined.unsorted.bam refine/reads_refined_*.bam && \
    $SAM sort -@ $N -o refined.bam refined.unsorted.bam && $SAM index refined.bam && rm -f refined.unsorted.bam"
  rm -rf shards refine
}

split_chunks() {   # split_chunks N: chunk<i>.bam (+ .cand, every position) of equal read coverage from refined.bam
  local N=$1; rm -f chunk*
  $PY "$SUPP_COST_DIR/split_chunks.py" refined.bam "$N"
  for i in $(seq 0 $((N-1))); do $SAM view -b -o chunk$i.bam refined.bam $(cat chunk$i.reg) && $SAM index chunk$i.bam; done
}
