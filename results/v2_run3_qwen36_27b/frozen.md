# GPU run 3: the frozen confirmation set (Qwen3.6-27B, J++)

*Written and committed 9 Nov 2026 before any run-3 GPU job, with the code the run uses. Rules: `docs/conformity/scope.md`
§6 (development / confirmation), §11 (pass marks and the run-3 readings). Nothing here changes after results.*

## How the set was picked (fixed rule, run 2 §8 item 2)

1. The confirmation half of the candidates: `candidates.split()["confirm"]`, 55 active questions.
2. Selected on the 27B by the J++ trimmed gate screen of run 2 (`results/conformity_select_qwen36_27b_jpp_candidates.csv`):
   need ≥ 0.5 and the bridge comes first. 24 of 55.
3. Reach ≥ 0.5 in that screen (the scorecard's reach, q→end positions as screened). 20 of 24.
4. Families that select nothing on the 27B (continent, vowel class, even/odd) are out; none of the 20 is in them. Months
   and driving side (1 selected each, both in the development half) have none here either.
5. Of the candidates whose same-kind controls were replaced at the fact pass (commit 2180ed4), one is in the set:
   C_LU_case (Detroit, Pittsburgh). Re-screened with the current controls on one A100 before the freeze
   (`results/v2_run3_qwen36_27b/rescreen/`): need 0.95, reach 0.60, comes first. It stays.

**Frozen: 20 questions** (16 first-letter, 4 number lookups):

| Question | Bridge | Right | Wrong peer (free rule, run 3) | Old rule (runs 1–2) | Swap target | need | reach |
|---|---|---|---|---|---|---|---|
| C_EL_helium | helium | 2 | 3 | 1 | hydrogen → 1 | 0.78 | 0.78 |
| C_EL_oxygen | oxygen | 8 | 9 | 7 | nitrogen → 7 | 0.60 | 0.85 |
| C_LM_alhambra | Spain | S | L | P | Portugal → P | 0.90 | 0.98 |
| C_LM_bluelagoon | Iceland | I | R | N | Norway → N | 0.76 | 0.78 |
| C_LM_chichen | Mexico | M | T | G | Guatemala → G | 0.98 | 0.94 |
| C_LM_galapagos | Ecuador | E | N | P | Peru → P | 1.00 | 1.00 |
| C_LM_moai | Chile | C | D | P | Peru → P | 1.00 | 1.00 |
| C_LM_tikal | Guatemala | G | T | M | Mexico → M | 0.99 | 0.99 |
| C_LM_timbuktu | Mali | M | L | N | Niger → N | 0.82 | 0.99 |
| C_LM_vasa | Sweden | S | T | N | Norway → N | 1.00 | 1.00 |
| C_LP_dante | Florence | F | A | R | Rome → R | 0.97 | 0.99 |
| C_LP_kanye | Atlanta | A | V | M | Memphis → M | 0.90 | 1.00 |
| C_LU_bohr | Copenhagen | C | N | S | Stockholm → S | 0.87 | 0.54 |
| C_LU_case | Cleveland | C | I | D | Detroit → D | 0.95 | 0.60 |
| C_LU_emory | Atlanta | A | S | N | Nashville → N | 0.99 | 1.00 |
| C_LU_goethe | Frankfurt | F | V | M | Munich → M | 0.99 | 0.99 |
| C_LU_mcgill | Montreal | M | H | T | Toronto → T | 1.00 | 1.00 |
| C_LU_unsw | Sydney | S | U | M | Melbourne → M | 0.96 | 0.97 |
| C_PL_saturn | Saturn | 6 | 7 | 7 | Jupiter → 5 | 0.96 | 0.51 |
| C_SP_gretzky | hockey | 6 | 7 | 7 | baseball → 9 | 0.98 | 0.99 |

Ids, space-separated (the run's `--questions`):

    C_EL_helium C_EL_oxygen C_LM_alhambra C_LM_bluelagoon C_LM_chichen C_LM_galapagos C_LM_moai C_LM_tikal C_LM_timbuktu C_LM_vasa C_LP_dante C_LP_kanye C_LU_bohr C_LU_case C_LU_emory C_LU_goethe C_LU_mcgill C_LU_unsw C_PL_saturn C_SP_gretzky

- sha256 of the id list (json, sorted): `d868873468bd14b383170e15284578fce64953799fe1d6bf40a8619ce4dce832`
- `candidates.digest()` (candidate table and split): `f35aa82d26e4fc6b6a071b1e6aa6ededabd70ad7b63440de52177d8058202424`

## What runs on them, once, with every choice frozen

Config `qwen36_27b_jpp` (J++ lens), both wordings (neutral "Give your own answer"; group-aware "Taking the other
agents' answers into account"):

- **Conformity** (`conformity`, `--own-part`, `--variants hidden hidden_sub told_copy`, wrong rule `free`): alone;
  agree 1–3; disagree 1–3; hidden "?" 1–3; hidden "(answer submitted)" 1–3; told to copy 3. Every early-window removal
  (bridge, same-kind, answer, 12 everyday words, the bridge's own part), reach on shared positions (q→end recorded
  beside it), presence, labels.
- **Block, 2×2, dose-response, patch** (`conformity_block --patch`): A B C D (and C_late D_late) on alone, agree 1–3,
  disagree 1–3, hidden "?" 1–3, "(answer submitted)" 3, told to copy 3; P(right), P(peers'), P(same-kind answer);
  strength 0.25/0.5/0.75 in alone, hidden 3, agree 3; the hidden-"?" patch on agree 3, every layer and from layer L up.
- **Swap toward another bridge** (`conformity_swap`, plan.md step 7b): the move, bridge → swap target above, question
  sentence tokens, layers 0 to onset − 1; alone, agree 3, disagree 3, visible and under the peer block; the everyday
  word control.
- No copy-only (made-up subject) prompts: they exist only for the calibration questions, none of which is here.

Not done, by choice: the second paraphrase for the comparison (scope.md §5) does not exist for these candidates; the
gate and the group prompts use the same wording. Llama and Gemma stay out.
