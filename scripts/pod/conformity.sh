#!/usr/bin/env bash
# ON THE POD: one shard of the conformity experiment (src/consulted_bridges/conformity.py).
# Each shard writes only its own folder (results/conformity_<config>/shard<i>).
#   bash scripts/pod/conformity.sh qwen36_27b 0/5 [neutral|defer]
set -uo pipefail
cfg=${1:?config}; shard=${2:-0/1}; frame=${3:-neutral}
source /workspace/env.sh
export HF_HUB_OFFLINE=1
cd /workspace/consulted-bridges
.venv/bin/python -m consulted_bridges.conformity --config "$cfg" --shard "$shard" --frame "$frame"
echo "SHARD DONE $cfg $shard $frame"
