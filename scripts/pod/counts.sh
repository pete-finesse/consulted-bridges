#!/usr/bin/env bash
# ON THE POD: screen the count questions (questions/proposed/counts.py) with the scorecard, one shard.
# Runs from its own copy of the code (/workspace/counts) so it never touches /workspace/consulted-bridges,
# using that checkout's .venv read-only. Results: /workspace/counts/results/counts_<config>/shard<i>.
#   bash scripts/pod/counts.sh qwen36_27b 0/2
set -uo pipefail
cfg=${1:?config}; shard=${2:-0/1}
source /workspace/env.sh
export HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/counts/src
cd /workspace/counts
ANCHORS="T_place_states_austria T_place_states_australia V_T_basketball"; [ "${ROUND:-1}" = 2 ] && ANCHORS="T_place_states_austria N_BP_jackman"
NEW=$(/workspace/consulted-bridges/.venv/bin/python -c "from consulted_bridges.questions.proposed import counts as c; print(' '.join(c.ROUND2 if '${ROUND:-1}' == '2' else c.SPECS))")
/workspace/consulted-bridges/.venv/bin/python -m consulted_bridges.scorecard --config "$cfg" --sets counts,trend,proposed \
  --trim --spec $NEW $ANCHORS --shard "$shard" --out "results/counts${ROUND:+_r$ROUND}_$cfg"
echo "SHARD DONE $cfg $shard"
