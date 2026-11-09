#!/usr/bin/env bash
# ON THE POD: a worker for GPU run 2 (Qwen3.6-27B), the same steps as v2_run.sh but spread over many pods through a
# job queue on the shared volume. Each worker takes the next line of $Q/jobs.txt whose lock it can create (mkdir is
# atomic on the volume), runs it, and marks it done; so pods that come up late simply join in. One worker per 27B
# slot: one on an A100 80 GB or RTX PRO 6000, two on an H200 (run 1 hit CUDA out-of-memory with eight on one H200).
#   nohup bash scripts/pod/v2_jobs.sh <worker-name> > $Q/w-<worker-name>.log 2>&1 &
# Job lines (run from ROOT=/workspace/v2run27, the code copied there from the Mac):
#   screen <cfg> <i> <N>     gate screen, shard i/N of the 171 candidates (trimmed solo scorecard) under <cfg>
#   chain <frame> <i> <N>    conformity shard i/N on the development questions (J++, every B1 variant, own part), then
#                            the block validation, 2x2, dose-response and patch on that shard's own rows
# The block part waits for $OUT/LEAK_PASS (written by hand after the leak test passes; if it holds
# "USE_HUB_KERNELS=0", that is exported first) and is skipped if $OUT/LEAK_FAIL exists.
set -uo pipefail
source /workspace/env.sh
ROOT=${ROOT:-/workspace/v2run27}; cd "$ROOT" || exit 1
PY=/workspace/consulted-bridges/.venv/bin/python
export PYTHONPATH=$ROOT/src HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL=qwen36_27b; CFG=${MODEL}_jpp
OUT=results/v2_run2_qwen36_27b; Q=$OUT/queue; L=$OUT/logs; mkdir -p $Q/lock $Q/done $L
me=${1:?worker name}
t() { echo "== $(date -u +%T) [$me] $*"; }
VARIANTS="hidden hidden_sub disagree_g told_copy told_ignore peer_bridge peer_bridge_wrong"

run_job() {
  set -- $1
  case $1 in
    screen)
      local c=$2 i=$3 n=$4
      specs=$($PY -c "from consulted_bridges.conformity import CANDIDATE; print(' '.join(CANDIDATE))")
      $PY -m consulted_bridges.scorecard --config $c --sets candidates --trim --spec $specs --shard "$i/$n" \
        --out results/solo_${c}_candidates > $L/screen-$c-$i.log 2>&1 ;;
    chain)
      local frame=$2 i=$3 n=$4
      qs=$($PY -c "from consulted_bridges.conformity import QUESTIONS; print(' '.join(QUESTIONS['$MODEL']))")
      w=results/conformity_v2_${CFG}; o=results/conformity_block_${CFG}
      [ $frame = defer ] && { w=results/conformity_v2_defer_${CFG}; o=results/conformity_block_defer_${CFG}; }
      $PY -m consulted_bridges.conformity --config $CFG --questions $qs --frame $frame --own-part --variants $VARIANTS \
        --shard "$i/$n" --out $w > $L/conf-$frame-$i.log 2>&1 || return 1
      while [ ! -e $OUT/LEAK_PASS ] && [ ! -e $OUT/LEAK_FAIL ]; do sleep 20; done
      [ -e $OUT/LEAK_FAIL ] && { t "leak test failed: block skipped for $frame $i"; return 0; }
      grep -q "USE_HUB_KERNELS=0" $OUT/LEAK_PASS && export USE_HUB_KERNELS=0
      $PY -m consulted_bridges.conformity_block --config $CFG --frame $frame --windows $w/shard$i --questions $qs \
        --patch --shard "$i/$n" --out $o > $L/block-$frame-$i.log 2>&1 ;;
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
