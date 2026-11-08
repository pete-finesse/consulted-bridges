#!/usr/bin/env bash
# ON THE POD: one shard of the conformity experiment on the count questions (conformity.py --set counts).
# Runs from its own copy of the code (/workspace/counts), using /workspace/consulted-bridges/.venv read-only.
#   bash scripts/pod/conformity_counts.sh qwen36_27b 0/2 [neutral|defer]
set -uo pipefail
cfg=${1:?config}; shard=${2:-0/1}; frame=${3:-neutral}
source /workspace/env.sh
export HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/counts/src
cd /workspace/counts
/workspace/consulted-bridges/.venv/bin/python -m consulted_bridges.conformity --config "$cfg" --set counts --shard "$shard" --frame "$frame"
echo "SHARD DONE $cfg $shard $frame"
