#!/usr/bin/env bash
# ON THE POD (from /workspace/v2run): re-run whole questions that lost prompts to CUDA OOM in GPU run 1: v2_redo.sh neutral:Q defer:Q ...
source /workspace/env.sh; cd /workspace/v2run; export PYTHONPATH=/workspace/v2run/src HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/consulted-bridges/.venv/bin/python; L=results/v2_run1_qwen35_4b/logs; CFG=qwen35_4b_jpp
for item in "$@"; do
  frame=${item%%:*}; q=${item#*:}
  w=results/conformity_v2_redo_${CFG}/${frame}_$q; o=results/conformity_block_redo_${CFG}/${frame}_$q
  ( $PY -m consulted_bridges.conformity --config $CFG --questions $q --frame $frame --own-part \
      --variants hidden hidden_sub disagree_g told_copy told_ignore peer_bridge peer_bridge_wrong \
      --out $w > $L/redo-conf-${frame}-$q.log 2>&1 &&
    $PY -m consulted_bridges.conformity_block --config $CFG --frame $frame --windows $w/shard0 --questions $q \
      --patch --out $o > $L/redo-block-${frame}-$q.log 2>&1 ) &
done
wait
grep -h "ALLDONE" $L/redo-*.log | sort | uniq -c; grep -h "failed:\|!!" $L/redo-*.log | head
echo "REDO DONE"
