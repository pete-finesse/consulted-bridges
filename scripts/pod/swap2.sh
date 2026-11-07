#!/usr/bin/env bash
# ON THE POD: one shard of the swap re-run (docs/swap_test_rules.md). Runs, one after the other:
#   1. the swap test alone on the saved needed-and-first questions (results/swap2_jobs/<config>.csv)
#   2. the trimmed scorecard plus the swap test on the trend questions
# Each shard writes only its own folders (results/swap2_<config>/shard<i>, results/trend_<config>/shard<i>).
#   bash scripts/pod/swap2.sh qwen36_27b 0/4
set -uo pipefail
cfg=${1:?config}; shard=${2:-0/1}
source /workspace/env.sh
export HF_HUB_OFFLINE=1
uv run python -m consulted_bridges.scorecard --config "$cfg" --swaps-only "results/swap2_jobs/$cfg.csv" \
  --shard "$shard" --out "results/swap2_$cfg"
uv run python -m consulted_bridges.scorecard --config "$cfg" --sets trend --trim --swaps \
  --shard "$shard" --out "results/trend_$cfg"
echo "SHARD DONE $cfg $shard"
