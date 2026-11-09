#!/usr/bin/env bash
# ON THE POD: the conformity pipeline with Neuronpedia's J++ lens (NEXT_EXPERIMENTS 5), one step at a time so
# several pods can share it. Runs from its own copy of the code (/workspace/jpp, pushed there from the Mac),
# using /workspace/consulted-bridges/.venv read-only, so it never touches another job's code or results.
#   bash scripts/pod/jpp.sh lens                         download both J++ files into the volume's HF cache (one pod)
#   bash scripts/pod/jpp.sh smoke  qwen36_27b_jpp        load, answer and lens ranks for the Colosseum question
#   bash scripts/pod/jpp.sh solo   qwen36_27b_jpp 0/6    trimmed solo scorecard on the 37 letter questions, one shard
#   bash scripts/pod/jpp.sh counts qwen36_27b_jpp 0/2    trimmed scorecard on the 53 count questions + 3 anchors, one shard
#   bash scripts/pod/jpp.sh card   qwen36_27b_jpp 0/2 Q..  full scorecard with the swap test on the named questions, one shard
#   bash scripts/pod/jpp.sh cconf  qwen36_27b_jpp 0/2 Q..  conformity on the named count questions, both wordings, one shard
#   bash scripts/pod/jpp.sh select qwen36_27b_jpp        same rule as before (need >= 0.5, comes first)
#   bash scripts/pod/jpp.sh conf   qwen36_27b_jpp 0/6    conformity, neutral then deferential wording, one shard
# Results under /workspace/jpp/results/; each shard writes only its own folder.
set -uo pipefail
step=${1:?step}; cfg=${2:-}; shard=${3:-0/1}
source /workspace/env.sh
cd /workspace/jpp
export PYTHONPATH=/workspace/jpp/src
PY=/workspace/consulted-bridges/.venv/bin/python
[[ $step == lens ]] || export HF_HUB_OFFLINE=1

case $step in
  lens)
    for f in qwen3.5-4b/jpp/Salesforce-wikitext/Qwen3.5-4B_jpp_lens.pt qwen3.6-27b/jpp/Salesforce-wikitext/Qwen3.6-27B_jpp_lens.pt; do
      $PY -c "from huggingface_hub import hf_hub_download as d; print(d('neuronpedia/jacobian-lens', '$f', revision='main'))" || exit 1
    done ;;
  smoke)
    $PY scripts/pod/smoke.py "$cfg" ;;
  solo)
    specs=$($PY -c "from consulted_bridges.conformity import LETTER; print(' '.join(('v3_' + q) if q.startswith('L_') else q for q in LETTER))")
    $PY -m consulted_bridges.scorecard --config "$cfg" --sets card3,proposed --spec $specs --trim \
      --shard "$shard" --out "results/solo_$cfg" ;;
  counts)   # the count-question screen (all 53 count questions, both rounds, plus the 3 anchors), trimmed scorecard
    new=$($PY -c "from consulted_bridges.questions.proposed import counts as c; print(' '.join(c.SPECS))")
    $PY -m consulted_bridges.scorecard --config "$cfg" --sets counts,trend,proposed --trim \
      --spec $new T_place_states_austria T_place_states_australia V_T_basketball --shard "$shard" --out "results/counts_$cfg" ;;
  card)     # the full scorecard with the swap test on named questions: jpp.sh card CFG i/n Q1 Q2 ...
    shift 3
    $PY -m consulted_bridges.scorecard --config "$cfg" --sets counts,trend,proposed --swaps \
      --spec "$@" --shard "$shard" --out "results/fullcard_counts_$cfg" ;;
  cconf)    # conformity on named count questions, neutral then deferential: jpp.sh cconf CFG i/n Q1 Q2 ...
    shift 3
    for frame in neutral defer; do
      $PY -m consulted_bridges.conformity --config "$cfg" --set counts --questions "$@" --shard "$shard" --frame $frame
    done ;;
  select)
    $PY -m consulted_bridges.conformity_select --config "$cfg" ;;
  conf)
    for frame in neutral defer; do
      $PY -m consulted_bridges.conformity --config "$cfg" --shard "$shard" --frame $frame
    done ;;
  *) sed -n 2,10p "$0"; exit 1 ;;
esac
echo "STEP DONE $step $cfg $shard"
