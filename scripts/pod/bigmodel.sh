#!/usr/bin/env bash
# ON THE POD: every conformity experiment for a model that fits only once on the pod (one process, e.g.
# DeepSeek-V4-Flash, ~160 GB): the letter pipeline (newmodel.sh with one process), then the count questions:
#   1. letters: solo scorecard on the 37 letter questions, selection, conformity in both wordings (newmodel.sh)
#   2. counts:  trimmed scorecard on all count questions (both rounds) and their anchors -> results/counts_<cfg>
#   3. counts:  conformity on the ones the screen calls consulted (conformity.counts_for), both wordings
# Logs in /workspace/logs/<cfg>-*.log; "ALL DONE" at the end.
#   nohup bash scripts/pod/bigmodel.sh deepseek_v4_flash > /workspace/logs/bigmodel.log 2>&1 &
set -uo pipefail
cfg=${1:?config}
source /workspace/env.sh
cd /workspace/consulted-bridges
L=/workspace/logs; mkdir -p $L
PY=.venv/bin/python
export HF_HUB_OFFLINE=0   # the FP8 kernel (kernels-community/finegrained-fp8) resolves on the Hub even when cached; load() still refuses model downloads

bash scripts/pod/newmodel.sh "$cfg" 1

echo "== $(date -u +%T) count screen"
specs=$($PY -c "
from consulted_bridges.questions.proposed import counts as c
print(' '.join(dict.fromkeys(list(c.SPECS) + list(c.ROUND2) + ['T_place_states_austria', 'T_place_states_australia', 'V_T_basketball'])))")
$PY -m consulted_bridges.scorecard --config "$cfg" --sets counts,trend,proposed --trim --spec $specs \
  --shard 0/1 --out "results/counts_$cfg" > $L/$cfg-counts.log 2>&1
grep -h "ALLDONE\|failed" $L/$cfg-counts.log

echo "== $(date -u +%T) count conformity"
for frame in neutral defer; do
  $PY -m consulted_bridges.conformity --config "$cfg" --set counts --shard 0/1 --frame $frame > $L/$cfg-conf-counts-$frame.log 2>&1
  grep -h "ALLDONE\|failed" $L/$cfg-conf-counts-$frame.log
done
echo "== $(date -u +%T) ALL DONE $cfg"
