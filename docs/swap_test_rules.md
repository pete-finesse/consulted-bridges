# Swap test: rules fixed before the run

*Written and committed before the run that uses them (NEXT_EXPERIMENTS 1f). If a rule changes after results are seen, the change is logged here with the reason, and both versions are reported.*

**What the test asks.** Removal shows the bridge's direction is *needed*. The swap asks whether it carries the bridge's *content*: if we move the bridge's component onto another word of its kind, does the answer become that word's answer?

## Which questions

- **Swaps run only where the bridge is needed and comes first:** beats noise ≥ 0.5 and comes first, from the v4.1 scorecard.
  - Question bank: 12 on Qwen3.5-4B, 12 on Qwen3.6-27B.
  - Close variants: 24 on the 4B, 20 on the 27B.
  - Trend questions: they get a trimmed scorecard first (below), and swaps only if they pass the same rule.
- **Answer onset:** the question's saved v4.1 onset. The trend questions use the onset from their own new card.

## The edit

- **Neuronpedia's move:** h ← h + (h·ŝ)(t̂ − ŝ), where ŝ is the bridge's unit J-lens direction and t̂ the target's. The directions are ours (with the final norm's gain), as for removal.
- **Layer sets:**
  - each scorecard window that ends before the answer onset;
  - all layers from 0 to onset − 1 (the **whole pre-onset range**).
- **Positions:**
  - **Rule A:** the question's own words only, from the question's first word up to, not including, `<|im_end|>`;
  - **Rule B:** from the question's first word to the end of the prompt (q→end), the same tokens as removal.

## Words swapped in

- **Targets (4):** taken in order from the question's swap list. A word qualifies if:
  - it is a single token;
  - the model's own answer to the question's one-hop check about it ("What letter does the word Spain start with?") is a known answer;
  - that answer differs from the right answer and from the answers of the targets already chosen.

  If fewer than 4 qualify, the test uses what qualifies.
- **Everyday words (3):** the first 3 of table, river, garden, lamp, bottle, window that are single tokens and whose own answer, asked the same way, isn't the right answer.

## Scoring

- **The model's answer:** its greedy reply (up to 6 tokens), sorted into the question's answers as the scorecard does.
- **Fallback:** the answer when the bridge is *removed* at the same layers and positions. A move contains a removal, so a target whose own answer equals the fallback can't show anything in that setting. It is marked *uninformative* there and left out of the count.
- **Follows:** a target follows if the model's answer becomes the target's own answer.
- **Gets through:** an everyday word gets through if the model's answer becomes that word's own answer. This can only be read if the word has a known answer that isn't the fallback. If not, the word is *unreadable* for that question.
- **A setting counts** (one question, one rule, one layer set) if at least one everyday word is readable and none gets through. If none is readable, the setting is *not checked*: reported, but not counted as passing. The same check applies to Rule A and Rule B.

## What is reported

- **Headline, per question and rule:** the share of informative targets that follow over the **whole pre-onset range**, with whether that range counts. Single windows are reported as detail; the best window is never the headline.
- **Source controls (reported, not used as a gate):** moving a word the model isn't using onto the first target should change nothing. Two sources:
  - "table";
  - the first unused word of the question's swap list.

  The second shares part of its direction with the bridge (same kind), so it may move some of the bridge too.
- **Placement:** both rules go **next to** quality in the question bank, not inside it. Quality and verdicts stay as they are.

## Trimmed scorecard (trend questions)

- **What runs:**
  - Gates 1–4;
  - tests 1 Needed early, 2 Beats noise, 3 Comes first, 6 Spread out, 7 Own part, 7b Bridge-only vs its kind;
  - quality.
- **Noise words:** the 12 everyday words of matched strength run only in windows that end before the onset, the only windows tests 1–2 read.
- **Dropped:** the whole-answer onset, 4 Beats own kind, 5 Collateral, and the old exchange swaps 8–9.
- **Verdicts:** from quality's factors, as in the question bank. Without test 4, the old verdict reads "consulted (kind not checked)", which the bank then sorts by isolation and specificity like any other.
