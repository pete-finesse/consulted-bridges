# Presentation updates owed

*Changes the results now require in `presentation.html`, `outline.html` and `script-3min.md`, recorded here instead of
being edited in, so the pages change once, deliberately. Each item names its source. Tick when applied. Started
8 November 2026.*

## From round 13 (the log-odds re-analysis, `results/conformity_stats_v2/`)

- [ ] **Finding 1 is per model and wording, not one sentence.** Now: "When the peers agree, the model stops using its
  bridge, though the answer stays right." Owed: on Qwen3.6-27B with the neutral wording agreeing peers *weaken* the
  bridge's effect (need in log-odds 8.4 alone → 5.9 / 3.0 / 2.2 with 1 / 2 / 3 peers; 10 of 19 still needed at three
  peers, 9 of 15 on the clean subset); under the group-aware wording, and on Qwen3.5-4B under both, removal does
  nothing (27B group-aware: 2 of 19 needed, 14 at the ceiling; 4B group-aware: 0 of 20 needed).
- [ ] **Say why probability and log-odds disagree.** Half the 27B "not needed" verdicts leave P(right) above 0.9
  while the log-odds still fall by about 2. Show both numbers wherever a "not needed" share appears; add the ceiling
  count beside it.
- [ ] **Finding 2 (present but not needed) stands, and gets sharper.** The bridge is present in 100% of alone and agree
  prompts on both Qwen models, so presence cannot sort prompts at all; say "cannot", not "overstates".
- [ ] **Qwen3.5-4B caveat in the tests section.** The same-kind control drops 0.38 on median even in alone prompts, so
  the 4B probe is nonspecific on most of its questions; its group results are reported with that limit.
- [ ] **Llama section.** 9 of 90 agree prompts have no early window (the answer forms in the first layer window once
  peers are present) and its controls start moving when peers appear; its group numbers wait on that check. Keep the
  deferential-wording readout diagnostic (the wording fills the last token).
- [ ] **A new reported category:** "no early window", with counts per model and condition (4B disagree 11 of 120).
- [ ] **Script, 2:35:** "When the peers agree, the model stops using its bridge" → "…uses its bridge less, and under
  the group-aware wording stops using it".

## From the plan review (plan.md, scope.md)

- [ ] **Scope sentence** in the hero or section 3: the findings are about questions where the model demonstrably uses
  an intermediate step alone; that is also the paper's population (no silent dissent without reasoning).
- [ ] **Funnel** in section 5: candidates drafted → passing each gate check → selected, per model; 37 → 25 (27B),
  37 → 24 (4B), 37 → 15 (Llama), 37 → 8 (Gemma base lens), 37 → 1 (Gemma refit).
- [ ] **Baseline for "shrinks"** becomes the hidden-answer group prompt once it has been run (Phase C step 5); until
  then say that the alone-versus-agree comparison is biased toward shrinkage by selection, and by how much is open.
- [ ] **Counts caveat** wherever the 16 count questions appear: 14 rest on two facts.
- [ ] **Format scope** in section 4: the removal arm is limited to arbitrary lookups on the bridge; natural second
  hops (capital, language) overlap with the bridge; the peer-side tests will not have that limit.

## Pending results that will add items

- Candidate screen pass rates (Phase C step 4b); the hidden-answer tool check (step 5); the 2×2 (step 6).
