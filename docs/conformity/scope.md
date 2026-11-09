# Scope, claims and justifications

*A living reference for write-ups. Each entry records a decision, the reason, and what it costs. Add to it when a
scope or method question is settled; date the entry. Started 8 November 2026 from the planning discussion. The plan
itself is `plan.md`, the method as run is `methods.md`, the rounds are in `process.md`, and the candidate questions are
`src/consulted_bridges/questions/proposed/candidates.py`.*

---

## 1. The claim, in words that stay honest

**We claim:** on two-hop questions where the model demonstrably uses an intermediate step (the bridge) when asked
alone, we can measure whether that step still shapes the answer when the model sits in a group, by removing the
bridge's direction and reading the answer's probability in the same prompt.

**The wording to use:** "the answer depends on the model's own intermediate step", reported "among the sampled
cases where the measurement was validated", with the number of unresolved cases alongside.

**We do not claim:** that the model "thinks for itself" in any broad sense; that the bridge *is* Italy rather than
a related signal; that the bridge is the only route; anything about free-form or multi-turn answers; anything about
questions where the peers dispute the bridge itself. Those need other tools or are Phase D.

## 2. Presence is not use (the gap we fill)

Silent Dissent reads the bridge with the J-lens after the agent gives in and finds it still ranks in the top 100.
That shows the bridge is in the model's state, not that it does anything. Our own data: the bridge is present
(rank < 100) in 100% of alone and agree prompts on both Qwen models, whether or not removal changes the answer.
So presence cannot sort prompts into "used" and "unused"; removal can. "Computed and overridden" and "lit up but
unused" both leave the bridge readable; only removing it separates them. (8 Nov 2026)

## 3. Why the gate is defensible

The removal probe only answers questions where the bridge is demonstrably used alone. A question that fails that
gate cannot support the *null* reading in groups ("peers made the bridge unnecessary"), because "the tool never
worked here" predicts the same nothing. It can still support a positive reading (the group recruits the bridge), and
it enters every behavioural measure (F versus G peers) without any gate.

**The gate is on the tool, not the outcome.** It asks whether the model uses the bridge alone. The hypothesis is
about what happens when peers appear, and a gated question is free to go either way: 79% still need the bridge
with one agreeing peer, 26% with three (Qwen3.6-27B, neutral wording). So the gate does not manufacture the result.

**What it costs: population.** The findings are about questions where the model has a demonstrable intermediate
step. That is also the paper's population: an agent cannot dissent silently if it never reasoned. Report the funnel
(drafted, passing each check, selected) so the reader sees how narrow the population is.

**The gate is minimal.** Four checks: knows both hops alone; bridge, answer, peer answers and control words are one
token with no answer-changing alias; need (bridge drop minus everyday-word drop) ≥ 0.5 with the same-kind drop below
the bridge drop; reach (the edit demonstrably hits the bridge in this prompt). Not used for selection: isolation,
specificity, quality, swap. They served the identity claim, which we do not make, and the swap mostly fails (10 of
240). Call it "the gate", not "the scorecard". (8 Nov 2026)

## 4. Why the format is narrow, and how the peer side generalises

The bridge and the answer must share nothing inside the model, or removing the bridge's direction also removes part
of the answer's (Mozart → Salzburg → Austria is the paper's own example, and on it removal cannot separate the two).
Natural second hops (capital, language, demonym) fail this. So the removal arm uses arbitrary lookups on the bridge:
first letter, a count, a yes/no property. The claim is scoped to those.

The peer block and patching have no such constraint. The "peers visible versus blocked" half of the 2×2 can run on
natural two-hop questions. That is how the peer-side findings generalise beyond the format even where the removal
side cannot follow. (8 Nov 2026)

## 5. Selection bias and the baseline

Selecting on solo need picks up some questions that passed by luck. Re-measured in any new prompt they look weaker
on average even if peers do nothing, so "alone versus agree" overstates the shrinkage. Unselected questions have the
opposite problem: where the probe never worked, removal shows nothing and reads as "bridge never needed".

**Rule: gate on one prompt, compare two others.** Gate on the solo prompt. Measure the effect in the hidden-answer
group prompt (peers write "?") and in the agreeing-peers group prompt, and compare those two. Both are
re-measurements the selection never saw. "Shrinks" means agree against hidden, not against alone. For the
confirmation set, the gate uses one paraphrase and the comparison another. (8 Nov 2026)

## 6. Development and confirmation

The split guards against **tuning**, not selection. On the development questions every threshold, window, onset
rule, control and label rule was chosen while looking at results; enough such choices can make noise look like a
finding. Confirmation freezes every choice and runs new questions once. The gate applies to both sets identically.
Development questions stay in the write-up in full as the main result; nothing is given up. (8 Nov 2026)

## 7. Effect sizes and sample size

**Unit:** the question, never the prompt. Prompts from one question are not independent, and count questions that
share a fact are one unit (14 of the 16 count questions rest on two facts, so that set is effectively two
questions). Intervals come from a bootstrap that resamples questions.

**Two tiers, chosen where the words of the claim change, fixed before the runs:**
- *stops*: after removal the right answer is no longer the model's top choice (the written answer flips), and the
  removal effect is within the noise floor plus 1 log-odds;
- *weakens*: the removal effect is at least 2 log-odds smaller than in the hidden prompt. Two is about a third of
  the full alone effect (which averages about 6 log-odds with three agreeing peers). It is a judgment; the defence
  is that it is prespecified and the continuous values are reported alongside.

**Ceiling.** Probability saturates; log-odds do not. On the 27B half of the "not needed" verdicts leave P(right)
above 0.9 while the log-odds still fall by about 2. Report both, and flag ceiling rows. (8 Nov 2026)

**How many.** With a spread (SD) of about 3 log-odds across questions, a 2 log-odds difference at 80% power needs
about 20 questions per contrast; three formats read separately need about 60 gated questions per model; with a pass
rate of about two thirds, about 90 to 100 candidates per model. (8 Nov 2026)

**Re-analysis thresholds (A1), fixed 8 Nov 2026 before computing.** All on the saved runs, no new data.
- *Log-odds fall* of a removal = logit P(right) before − logit P(right) after, P clipped to [1e-6, 1 − 1e-6].
  *Need in log-odds* = the largest early-window bridge fall − the largest early-window everyday-word fall.
- Labels in log-odds (alone and agree): **needed** if need_LO ≥ 2; **not needed** if need_LO < 1 and reach passes;
  **mixed** otherwise. Disagree prompts keep the existing pushback rule.
- **Ceiling flag:** P(right) after the best bridge removal still ≥ 0.9.
- **Inconclusive** replaces any label when the controls also move: alone/agree if the everyday-word drop ≥ 0.2 or the
  same-kind drop ≥ 0.2; disagree if the everyday-word ΔLO ≥ 2.
- **Clean subset:** reach passes, bridge present (rank < 100), same-kind drop < 0.2, and the prompt's answer onset
  within one window step of the alone prompt's.
- **Unassessed** = prompts with no early window or no need value; reported with the assessed so the two add to the
  selected cohort; an empty subgroup is written "no cases".

**Stop rule, fixed before the screens.** "Enough" means at least 20 gated questions per format for at least two
formats on the main model (Qwen3.6-27B). Past pass rates forecast about 65 of 99 first-letter candidates and about
10 of 32 number candidates on the 27B, and almost no number questions on the 4B. If a format passes below one in
five it is out for the removal arm on that model and the claim is scoped to the formats that pass; this is a finding
about the probe, reported in the funnel. If a family passes at a decent rate but too few made it, more are drafted
in that family, at most two further drafting rounds; drafting from screen results is allowed because the screen is
solo-only and cannot bias the group comparison. If even letters fail on a model, that model leaves the removal arm
and stays in the behavioural and peer-block arms, which need no bridge. (8 Nov 2026)

## 8. What was tried and did not work (so it is not retried by accident)

- **Last letter of the hidden word:** untestable in the ideas run (the model cannot do the second hop reliably).
- **Even/odd letter count:** expected to fail the same way; four candidates kept so the screen can say so.
- **Swap (replace the bridge's direction with another word's):** 10 of 240 follow on the 27B; parked.
- **Currency as a lookup:** every currency name is two tokens on the Qwen tokenizer; dropped before any run.
- **Vowel/consonant as the answer word:** " Vowel" and " Consonant" are two tokens on Qwen; recast as "starts with
  a vowel, yes/no".
- **Two-digit answers:** Qwen splits digits, so 10+ shares a first token with a small number; answers are 1–9.
- **Shapes to sides:** the shape's name carries the number (octagon, pentagon).
- **Capital / language / demonym as the second hop:** overlap with the bridge inside the model.
- **Presence as a discriminator:** 100% present in alone and agree prompts; uninformative. (all 8 Nov 2026)

## 9. Open, to be defended later

- Whether a 2 log-odds "weakens" threshold survives review; alternatives are a relative threshold (a third of the
  alone effect) or reporting only continuous values.
- Whether the hidden-answer prompt ("?") is itself neutral; the model may react to the "?" content. "(answer
  submitted)" is the second version, and agreement between them is the check.
- Whether the attention block misses information already copied elsewhere in the model. Patching is the second
  method; agreement between block and patch is the check.
- Why presence fails on Llama with the group-aware wording (the diagnostic says the wording fills the last token).

## 10. Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-11-08 | Re-analysis in log-odds comes before any GPU run | half the 27B "not needed" rows are at the ceiling |
| 2026-11-08 | Baseline for "shrinks" is the hidden prompt, not alone | selection bias (§5) |
| 2026-11-08 | Gate is four checks, not the scorecard | §3 |
| 2026-11-08 | Candidates drafted and split before any screen; digest committed | §6 |
| 2026-11-08 | Currency family and 28 multi-token bridges dropped | token check, §8 |
| 2026-11-08 | A1 thresholds: need_LO ≥ 2 / < 1, ceiling 0.9, inconclusive at control drop 0.2, clean-subset rule | §7, fixed before computing |
| 2026-11-08 | Finding 1 reworded per model and wording: "weakens" (27B neutral), "stops" (27B group-aware, 4B) | A1 result, process.md round 13 |
| 2026-11-08 | Stop rule: ≥ 20 gated per format, ≥ 2 formats, on the 27B; at most two more drafting rounds | §7 |
