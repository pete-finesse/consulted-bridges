#!/usr/bin/env bash
# ON THE POD: the whole conformity pipeline for a model that has no saved solo scorecard.
#   1. fit its J-lens if the config asks for one (scripts/pod/fit_lens.py; needs the Hub, for wikitext)
#   2. the trimmed solo scorecard on all 37 letter questions          -> results/solo_<cfg>/shard<i>
#   3. pick the questions with the Qwen rule (needed >= 0.5, comes first) -> results/conformity_select_<cfg>.csv
#   4. the conformity run, neutral and deferential wording           -> results/conformity[_defer]_<cfg>/shard<i>
# Each step runs N processes on the one GPU. Logs in /workspace/logs/<cfg>-*.log; DONE marker at the end.
#   bash scripts/pod/newmodel.sh gemma4_e4b_it 2
set -uo pipefail
cfg=${1:?config}; n=${2:-2}
source /workspace/env.sh
cd /workspace/consulted-bridges
L=/workspace/logs; mkdir -p $L
PY=.venv/bin/python

echo "== $(date -u +%T) lens"
$PY scripts/pod/fit_lens.py --config "$cfg" > $L/$cfg-fit.log 2>&1; tail -2 $L/$cfg-fit.log
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}   # 0 for models whose kernels transformers fetches at run time (DeepSeek-V4 FP8)

if [ "${RESUME:-0}" = 1 ] && [ -s "results/conformity_select_$cfg.csv" ]; then
  echo "== RESUME: solo scorecard and selection already done (results/conformity_select_$cfg.csv)"
else
echo "== $(date -u +%T) solo scorecard"
specs=$($PY -c "from consulted_bridges.conformity import LETTER; print(' '.join(('v3_' + q) if q.startswith('L_') else q for q in LETTER))")
for ((i = 0; i < n; i++)); do
  $PY -m consulted_bridges.scorecard --config "$cfg" --sets card3,proposed --spec $specs --trim \
    --shard "$i/$n" --out "results/solo_$cfg" > $L/$cfg-solo-$i.log 2>&1 &
done
wait
grep -h "ALLDONE\|failed" $L/$cfg-solo-*.log

echo "== $(date -u +%T) select"
$PY -m consulted_bridges.conformity_select --config "$cfg" || { echo "SELECT FAILED"; exit 1; }
fi

echo "== $(date -u +%T) conformity"
for frame in neutral defer; do
  for ((i = 0; i < n; i++)); do
    $PY -m consulted_bridges.conformity --config "$cfg" --shard "$i/$n" --frame $frame > $L/$cfg-conf-$frame-$i.log 2>&1 &
    [ "$n" = 1 ] && wait     # one process per pod (a model that fits once): the two wordings in turn
  done
done
wait
grep -h "ALLDONE\|failed" $L/$cfg-conf-*.log
echo "== $(date -u +%T) DONE $cfg"
