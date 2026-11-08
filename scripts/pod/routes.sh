#!/usr/bin/env bash
# ON THE POD: one shard of the conformity follow-up (src/consulted_bridges/conformity_routes.py), both wordings.
#   bash scripts/pod/routes.sh qwen36_27b 0/3 [swap|copy]
set -uo pipefail
cfg=${1:?config}; shard=${2:-0/1}; test=${3:-swap}
source /workspace/env.sh
export HF_HUB_OFFLINE=1
cd /workspace/consulted-bridges
for frame in neutral defer; do
  .venv/bin/python -m consulted_bridges.conformity_routes --config "$cfg" --shard "$shard" --frame "$frame" --test "$test"
done
echo "SHARD DONE $cfg $shard"
