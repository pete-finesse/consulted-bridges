#!/usr/bin/env bash
# ON THE POD: GPU run 1 of the conformity plan (docs/conformity/plan.md Phase C, §7 order) on Qwen3.5-4B, one pod.
# Runs from its own copy of the code (/workspace/v2run, pushed there from the Mac), using
# /workspace/consulted-bridges/.venv read-only, so it never touches another job's code or results (as jpp.sh).
#   a  leak test of the peer block on the real model; on failure retried with USE_HUB_KERNELS=0, which then holds
#      for every later step; a second failure stops the run          -> results/v2_run1_qwen35_4b/leak_test*.json
#   b  gate screen of the 171 candidates, trimmed solo scorecard, J-lens and J++ in parallel shards
#                                                                     -> results/solo_<cfg>_candidates/shard<i>
#      then the selection rule (need >= 0.5, comes first)             -> results/conformity_select_<cfg>_candidates.csv
#      and the funnel per family and lens                             -> results/v2_run1_qwen35_4b/funnel.csv
#   c+d conformity on the 24 development questions (conformity.QUESTIONS["qwen35_4b"]), J++, every B1 variant and
#      the own-part removal, neutral and group-aware wording         -> results/conformity_v2[_defer]_qwen35_4b_jpp
#   e+f block validation, the 2x2, dose-response and the patch, both wordings
#                                                                     -> results/conformity_block[_defer]_qwen35_4b_jpp
# Logs in results/v2_run1_qwen35_4b/logs/; "V2RUN DONE" at the end of v2.log.
#   nohup bash scripts/pod/v2_run.sh > /workspace/v2run/results/v2_run1_qwen35_4b/logs/v2.log 2>&1 &
# Local smoke test on the tiny model (CPU, every gate skipped, a few candidates):
#   TOY=1 ROOT=<scratch dir with src/ linked> bash scripts/pod/v2_run.sh
# Knobs: NS (screen shards per lens, 3), NC (conformity shards per wording, 3), NB (block shards per wording, 2),
# STEPS (default "a b c e"; any subset, in that order). Step s (with N and SHARDS) is c and e chained per shard, for
# spreading the group steps over several pods: N=12 SHARDS="0 1" STEPS=s bash scripts/pod/v2_run.sh
set -uo pipefail
TOY=${TOY:-0}
if [ "$TOY" = 1 ]; then
  ROOT=${ROOT:?ROOT}; PY=${PY:-python}; export CARD2_SKIP_GATES=1
  CFGS="tiny"; CFG=tiny; QCFG=tiny
else
  source /workspace/env.sh
  ROOT=${ROOT:-/workspace/v2run}; PY=/workspace/consulted-bridges/.venv/bin/python
  export HF_HUB_OFFLINE=1
  CFGS="qwen35_4b qwen35_4b_jpp"; CFG=qwen35_4b_jpp; QCFG=qwen35_4b
fi
cd "$ROOT" || exit 1
export PYTHONPATH=$ROOT/src
NS=${NS:-3}; NC=${NC:-3}; NB=${NB:-2}; STEPS=${STEPS:-a b c e}
OUT=results/v2_run1_qwen35_4b; L=$OUT/logs; mkdir -p $L
has() { [[ " $STEPS " == *" $1 "* ]]; }
t() { echo "== $(date -u +%T) $*"; }
gpu() { command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader; }

if has a; then
  t "a: leak test ($CFG)"
  $PY -m consulted_bridges.leaktest --config $CFG --out $OUT > $L/leak.log 2>&1; rc=$?
  grep -v "Loading weights" $L/leak.log | grep "PASS\|FAIL\|Error" | tail -12
  if [ $rc = 2 ]; then
    t "a: leak with hub kernels; retry with USE_HUB_KERNELS=0 (kept for every later step)"
    export USE_HUB_KERNELS=0
    $PY -m consulted_bridges.leaktest --config $CFG --out $OUT > $L/leak_nohub.log 2>&1; rc=$?
    grep "PASS\|FAIL\|Error" $L/leak_nohub.log | tail -12
  fi
  if [ $rc != 0 ]; then t "LEAK TEST FAILED (rc $rc): stopping before anything else"; echo "V2RUN STOPPED"; exit 3; fi
fi

if has b; then
  t "b: gate screen, $CFGS, $NS shards each"
  if [ "$TOY" = 1 ]; then
    specs=$($PY -c "from consulted_bridges.conformity import CANDIDATE; print(' '.join(CANDIDATE[::9]))")
  else
    specs=$($PY -c "from consulted_bridges.conformity import CANDIDATE; print(' '.join(CANDIDATE))")
  fi
  for c in $CFGS; do
    for ((i = 0; i < NS; i++)); do
      $PY -m consulted_bridges.scorecard --config $c --sets candidates --trim --spec $specs --shard "$i/$NS" \
        --out results/solo_${c}_candidates > $L/screen-$c-$i.log 2>&1 &
    done
  done
  sleep 60; gpu; wait
  grep -h "ALLDONE" $L/screen-*.log | sort | uniq -c; cat $L/screen-*.log | grep -c "failed:" | sed "s/^/failed cards: /"
  for c in $CFGS; do
    if [ "$TOY" = 1 ]; then echo "(toy: select needs all 171 cards; skipped)"; else
      $PY -m consulted_bridges.conformity_select --config $c --set candidates 2>&1 | tail -3; fi
  done
  $PY -m consulted_bridges.conformity_funnel --configs $CFGS --out $OUT > $L/funnel.log 2>&1; grep " all " $L/funnel.log
fi

if has c; then
  qs=$($PY -c "from consulted_bridges.conformity import QUESTIONS; print(' '.join(QUESTIONS['$QCFG']))")
  t "c+d: conformity v2 on $(echo $qs | wc -w) questions, $CFG, both wordings, $NC shards each"
  for frame in neutral defer; do
    o=results/conformity_v2_${CFG}; [ $frame = defer ] && o=results/conformity_v2_defer_${CFG}
    for ((i = 0; i < NC; i++)); do
      $PY -m consulted_bridges.conformity --config $CFG --questions $qs --frame $frame --own-part \
        --variants hidden hidden_sub disagree_g told_copy told_ignore peer_bridge peer_bridge_wrong \
        --shard "$i/$NC" --out $o > $L/conf-$frame-$i.log 2>&1 &
    done
  done
  sleep 60; gpu; wait
  grep -h "ALLDONE" $L/conf-*.log | sort | uniq -c; grep -h "failed:" $L/conf-*.log | head
  for o in results/conformity_v2_${CFG} results/conformity_v2_defer_${CFG}; do
    $PY -m consulted_bridges.conformity_report $o > $L/report-$(basename $o).log 2>&1 && echo "merged $o"
  done
fi

if has e; then
  qs=$($PY -c "from consulted_bridges.conformity import QUESTIONS; print(' '.join(QUESTIONS['$QCFG']))")
  t "e+f: block, 2x2, dose, patch; both wordings, $NB shards each"
  for frame in neutral defer; do
    w=results/conformity_v2_${CFG}; o=results/conformity_block_${CFG}
    [ $frame = defer ] && { w=results/conformity_v2_defer_${CFG}; o=results/conformity_block_defer_${CFG}; }
    for ((i = 0; i < NB; i++)); do
      $PY -m consulted_bridges.conformity_block --config $CFG --frame $frame --windows $w --questions $qs --patch \
        --shard "$i/$NB" --out $o > $L/block-$frame-$i.log 2>&1 &
    done
  done
  wait
  grep -h "ALLDONE" $L/block-*.log | sort | uniq -c; grep -h "!!" $L/block-*.log | head
fi
if has s; then   # chained shards: conformity shard i/N, then the block on that shard's own rows, both wordings at once
  qs=$($PY -c "from consulted_bridges.conformity import QUESTIONS; print(' '.join(QUESTIONS['$QCFG']))")
  N=${N:?N}; SHARDS=${SHARDS:?SHARDS}
  t "s: chained conformity + block, shards $SHARDS of $N, both wordings"
  for i in $SHARDS; do
    for frame in neutral defer; do
      w=results/conformity_v2_${CFG}; o=results/conformity_block_${CFG}
      [ $frame = defer ] && { w=results/conformity_v2_defer_${CFG}; o=results/conformity_block_defer_${CFG}; }
      ( $PY -m consulted_bridges.conformity --config $CFG --questions $qs --frame $frame --own-part \
          --variants hidden hidden_sub disagree_g told_copy told_ignore peer_bridge peer_bridge_wrong \
          --shard "$i/$N" --out $w > $L/conf-$frame-$i.log 2>&1 &&
        $PY -m consulted_bridges.conformity_block --config $CFG --frame $frame --windows $w/shard$i --questions $qs \
          --patch --shard "$i/$N" --out $o > $L/block-$frame-$i.log 2>&1 ) &
    done
  done
  sleep 60; gpu; wait
  grep -h "ALLDONE" $L/conf-*.log $L/block-*.log | sort | uniq -c; grep -h "failed:\|!!" $L/conf-*.log $L/block-*.log | head
fi
t "V2RUN DONE"
