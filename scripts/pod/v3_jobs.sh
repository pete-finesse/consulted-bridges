#!/usr/bin/env bash
# ON THE POD: a worker for GPU run 3 (Qwen3.6-27B confirmation, results/v2_run3_qwen36_27b/frozen.md). Same queue as
# v2_jobs.sh: each worker takes the next line of $Q/jobs.txt whose lock it can create (mkdir is atomic), runs it,
# marks it done. One worker per 27B slot (one per 80 GB card, two per H200).
#   ROOT=<code copy> nohup bash scripts/pod/v3_jobs.sh <worker-name> > $Q/w-<worker-name>.log 2>&1 &
# Job line:
#   q <frame> <question>   for one frozen question and one wording, chained:
#       conformity (J++, own part, hidden / hidden_sub / told_copy, wrong rule free)  -> results/conformity_v3[_defer]_qwen36_27b_jpp/<q>
#       block, 2x2, dose-response, patch on its rows                                   -> results/conformity_block_v3[_defer]_qwen36_27b_jpp/<q>
#       swap toward another bridge (conformity_swap) on its onsets                     -> results/conformity_swap_v3[_defer]_qwen36_27b_jpp/<q>
# The block and swap parts wait for $OUT/LEAK_PASS ("USE_HUB_KERNELS=0" inside is exported) and stop at $OUT/LEAK_FAIL.
set -uo pipefail
source /workspace/env.sh
ROOT=${ROOT:-/workspace/v2run3}; cd "$ROOT" || exit 1
PY=${PY:-/workspace/consulted-bridges/.venv/bin/python}
export PYTHONPATH=$ROOT/src HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CB_WRONG_RULE=free
CFG=qwen36_27b_jpp
OUT=results/v2_run3_qwen36_27b; Q=$OUT/queue; L=$OUT/logs; mkdir -p $Q/lock $Q/done $L
me=${1:?worker name}
t() { echo "== $(date -u +%T) [$me] $*"; }

run_job() {
  set -- $1
  case $1 in
    q)
      local frame=$2 q=$3 sfx=""
      [ $frame = defer ] && sfx="_defer"
      local w=results/conformity_v3${sfx}_${CFG}/$q o=results/conformity_block_v3${sfx}_${CFG}/$q s=results/conformity_swap_v3${sfx}_${CFG}/$q
      $PY -m consulted_bridges.conformity --config $CFG --questions $q --frame $frame --own-part --wrong-rule free \
        --variants hidden hidden_sub told_copy --out $w > $L/conf-$frame-$q.log 2>&1 || return 1
      while [ ! -e $OUT/LEAK_PASS ] && [ ! -e $OUT/LEAK_FAIL ]; do sleep 15; done
      [ -e $OUT/LEAK_FAIL ] && { t "leak test failed: block and swap skipped for $frame $q"; return 0; }
      grep -q "USE_HUB_KERNELS=0" $OUT/LEAK_PASS && export USE_HUB_KERNELS=0
      $PY -m consulted_bridges.conformity_block --config $CFG --frame $frame --windows $w/shard0 --questions $q \
        --patch --wrong-rule free --out $o > $L/block-$frame-$q.log 2>&1 || return 1
      $PY -m consulted_bridges.conformity_swap --config $CFG --frame $frame --windows $w/shard0 --questions $q \
        --wrong-rule free --out $s > $L/swap-$frame-$q.log 2>&1 ;;
    *) t "unknown job: $*"; return 1 ;;
  esac
}

t "worker up on $(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
while true; do
  [ -e $Q/STOP ] && { t "STOP file: exiting"; break; }
  job=""
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    key=$(echo "$line" | tr ' /' '__')
    if mkdir $Q/lock/$key 2>/dev/null; then job=$line; break; fi
  done < $Q/jobs.txt
  [ -z "$job" ] && { t "queue empty: WORKER DONE"; break; }
  t "start: $job"; s=$(date +%s)
  if run_job "$job"; then echo "$me $(( $(date +%s) - s ))s" > $Q/done/$key; t "done ($(( $(date +%s) - s ))s): $job"
  else t "FAILED ($(( $(date +%s) - s ))s): $job"; echo "$me failed" > $Q/done/$key.failed; fi
done
