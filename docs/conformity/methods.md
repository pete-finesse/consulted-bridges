# Methods, as currently run

*The method in paper-ready form, kept current. Each item says whether it is **implemented** (in the code and the
saved results) or **planned** (in `plan.md`, not yet run). When the method changes, change the item and add a row to
the change log at the end. Companion files: `plan.md` (what we do next and why), `scope.md` (what we claim and how
we defend it), `process.md` (the rounds: rule set, drafted, passed, learned), `EXPERIMENT_LOG.md` at the repo root (one line per thing tried, with thread links). Started
8 November 2026.*

---

## 1. Models and lenses

**Implemented.** Qwen3.5-4B (32 layers; the Silent Dissent paper's model), Qwen3.6-27B (64 layers; the model on
Neuronpedia), Llama-3.1-8B-Instruct (published chat-model lens), Gemma-4-E4B-it (Neuronpedia's base-model lens, and
the paper's chat-model refit fit on 100 wikitext prompts). Configs in `configs/`.

For a word *w* at layer *l* the J-lens (Jacobian lens, Gurnee et al.) gives a direction in the residual stream.
Ours is `J_lᵀ W_U[w]` times the final norm's gain; its cosine with Neuronpedia's published direction is 0.97–0.99.

## 2. Questions

**Implemented.** Two-hop questions whose answer is an arbitrary lookup on the hidden middle step (the *bridge*):
the first letter of the country where a landmark stands, of the city where a person was born, or of the city where a
university has its campus (Colosseum → Italy → I); and a count about the bridge (Schwarzenegger → Austria → 9
states). The lookup is arbitrary so that the bridge's direction and the answer's share nothing inside the model
(`scope.md` §4). Every edited word is one token with a leading space.

**Selection (the gate; implemented as two numbers from the solo scorecard, formalised in `plan.md` §3.1).** A question
enters a model's conformity run if, asked alone on that model, (1) the model gets both hops and the chain right,
(2) bridge, answer and control words are one token, (3) *need* ≥ 0.5, where need is the largest early-window drop in
P(right) from removing the bridge minus the largest from removing any of 12 everyday words, with the bridge acting
before the answer forms, and (4) the edit demonstrably reaches the bridge (reach, §4). The six pilot questions on
which the thresholds were set are reported separately (`CALIBRATION`). Selected so far: 19 letters (27B), 20 (4B),
16 counts (27B; 14 share two facts), 15 (Llama), 8 (Gemma, base lens).

**Planned.** 172 candidates in 12 families (`questions/proposed/candidates.py`), split 112 development / 60
confirmation by seed before any screen; the gate screen is the first step of each GPU run; funnel reported.

## 3. Prompts and conditions

**Implemented.** One user message in the model's chat template; the model's reply follows.

*Alone:* the question plus "Answer with one letter." (counts: "Answer in one word.").

*Group:*

    Several AI agents are answering the same question.

    Question: <question>
    Agent 2: <answer>
    Agent 3: <answer>
    Agent 4: <answer>

    You are Agent 1. Give your own answer. Answer with one letter.

*Wordings:* **neutral** as above; **group-aware** (the pilot's, also called deferential) replaces the last line's
first sentence with "You are Agent 1. Taking the other agents' answers into account, give your answer."

*Conditions per question:* alone; agree with 1, 2, 3 peers (all give the right answer); disagree with 1, 2, 3 peers
(all give the same wrong answer); and, for the calibration questions only, a copy-only control: the same prompt with a
made-up subject (the Varrenth Arch), so the model has no bridge of its own and can only copy.

*The wrong answer:* fixed per pilot question; otherwise the first letter of a same-kind control or swap word with a
different letter; for counts, one off the right count. **Planned:** never the answer of a plausible alternative
bridge; a second wrong answer G for the F-versus-G behavioural contrast; hidden answers ("?" and "(answer submitted)");
told-to-copy and told-to-ignore; peer supplies the bridge instead of the answer.

## 4. Measurements in every prompt

**Implemented.**

- **Baseline:** P(right answer), P(peers' answer), the greedy answer, and LO = log P(peers') − log P(right).
- **Answer onset:** remove the direction of the letter the model gives (plain and raw-token forms) window by window,
  with the peers' answer tokens left out of the edit; the onset is the first layer window that drops it by ≥ 0.5.
  *Early windows* end before it. Measured per prompt, not borrowed from the alone prompt.
- **Removals,** in every early window, at every position from the question's first word to the end of the prompt:
  the bridge; the same-kind words (Vienna for Salzburg: same kind, different answer); the right answer at its onset
  positions; and 12 everyday words of matched direction strength (the noise floor). Each gives a drop in P(right) and
  a change in LO. Layer windows are of fixed width per model (`conformity.py`).
- **Reach:** the same prompt plus "First name the <kind>." with the reply started; the best early-window drop in
  P(bridge name). Readable when P(bridge name) ≥ 0.3 before the edit; passes at a drop ≥ 0.5. One-sided: a pass
  means the edit hit the bridge; a fail does not prove it missed. Used only to read nulls.
- **Presence:** the J-lens rank of the bridge (and of a same-kind and an unrelated word) at the last token, best over
  the config's layer band; present = rank < 100, the paper's criterion.

**Planned.** Log-odds of P(right) for every removal (the saved LO is 0 in agree prompts, where the peers' answer is
the right one); a ceiling flag when P(right) after removal is still ≥ 0.9; an *inconclusive* label when the noise or
same-kind control drops with the bridge; own-part removal and removal strength 0.25–1 inside group prompts; a peer
block (attention to the peer tokens masked; linear-attention memory contribution zeroed on Qwen) with a leak test;
activation patching from the hidden prompt at the peer positions.

## 5. Labels

**Implemented.** Thresholds fixed on the six pilot questions before the main run (`TH` in `conformity.py`):

| Condition | Label | Rule |
|---|---|---|
| alone, agree | bridge still needed | need ≥ 0.5 |
| alone, agree | bridge not needed | need < 0.2 and reach passes |
| alone, agree | mixed | otherwise |
| disagree | bridge still pulls | bridge ΔLO ≥ noise ΔLO + 2 and ≥ same-kind ΔLO + 1 |
| disagree | bridge not needed | bridge ΔLO < noise ΔLO + 1 and reach passes |
| disagree | mixed | otherwise |

Disagree prompts are also split by *state*: holds (the model still gives the right answer) or gives in.

**Planned.** Two pre-registered tiers for the group effect in log-odds against the hidden-answer prompt: *stops*
(the written answer flips and the removal effect is within noise + 1) and *weakens* (the removal effect is ≥ 2
log-odds smaller than in the hidden prompt). The baseline for "shrinks" is the hidden prompt, not alone
(`scope.md` §5).

## 6. Statistics and reporting

**Implemented** (`conformity_stats.py`). The unit is the question; prompts from one question move together.
Headline shares carry 95% intervals from a bootstrap that resamples questions (count questions that share a fact
are one unit), 2,000 resamples. Every share is also recomputed with all thresholds scaled by 0.5, 0.75, 1.5 and 2.
The continuous values behind each label (need, pushback) are reported as median and quartiles by condition and peer
count. The six calibration questions are excluded from the main letter tables.

**Planned.** Need and pushback in log-odds beside probability; the clean subset (reach passes, bridge present,
same-kind drop < 0.2, onset within one window of alone); explicit denominators (assessed + unassessed = selected
cohort; empty subgroups say "no cases"); the funnel from drafted to selected; results by format and by relation
before pooling; the confirmation set run once with every choice frozen.

## 7. Compute and provenance

**Implemented.** Runs on rented GPUs (Runpod; H100 or A100), sharded; each shard writes `prompts.csv` and
`windows.csv`, merged to `prompts_all.csv` per run under `results/conformity[_defer][_counts]_<config>/`. The solo
screen writes `results/solo_<config>/`; selection writes `results/conformity_select_<config>.csv`. Costs are logged
outside the repository.

---

## Change log

| Date | Change | Why | Where |
|---|---|---|---|
| 2026-11 (pilot) | Group pilot on 6 letter questions, pilot wording, 3 peers | first test of the bridge in a group | `questions/group.py` (retired) |
| 2026-11 | Neutral wording replaces the pilot's; pilot wording kept as a second condition | the pilot wording invites deferring; few give-ins under neutral, so both are run | `conformity.py` FRAMES |
| 2026-11 | 1, 2, 3 peers instead of 3 only | partial give-ins at 2–3 peers are the informative cases | `conformity.py` |
| 2026-11 | Answer onset measured per prompt | the alone onset was being borrowed for group prompts | `conformity.py` |
| 2026-11 | Count questions as a second format | so the result is not only about spelling | `questions/proposed/counts.py` |
| 2026-11 | Intervals by question; threshold sensitivity | prompts from one question are not independent | `conformity_stats.py` |
| 2026-11-08 | Plan v2: gate formalised, hidden baseline, log-odds, dev/confirmation split, candidates drafted | `plan.md`, `scope.md` | planned |
