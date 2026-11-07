# Silent Dissent harness

Run the Neuronpedia checks from `PLAYGROUND.md` and `DESIGN.md` (both in [swarm-explore](https://github.com/pete-finesse/swarm-explore/tree/main/projects/silent-dissent), written before the hackathon) in code, on one prompt or on hundreds of TwoHopFact facts. It reuses the authors' code (a pinned dependency, MIT) for loading the lens, reading ranks, building prompts and splitting words into pieces.

## Terms

- **Position:** one token of the prompt. "Last" is the final token, e.g. "of", just before the answer.
- **Layer:** one of the model's processing steps (32 in Qwen3.5-4B). The last layer is the model's ordinary next-word list.
- **Rank:** where a word sits in a lens's ranked vocabulary list (0 = top). There is no top-100 cutoff.
- **Piece:** a word can be split into tokens, e.g. " Aj" + "accio". As in the paper, a word's rank is the best rank over the first piece of each of its content words. A piece shared by two words counts for neither.
- **Band:** the layers that summaries average over. For Qwen3.5-4B these are 23–30, the paper's pre-registered layers.
- **Score:** log10(rank + 1), averaged over the band. Lower means the word ranks higher.
- **Roles** (for a fact like Mozart → Salzburg → Austria):
  - bridge: Salzburg;
  - same_country: another bridge city of Austria in the dataset;
  - capital: Vienna;
  - random: a city in another country;
  - answer: Austria;
  - peer_answer: the wrong country the peers push.

## Setup

```bash
uv sync --extra dev
```

The authors' code is installed as a pinned dependency. `uv run pytest` runs 15 tests in about 10 seconds on CPU.

## Models (`--config`)

| Config | What | Download |
|---|---|---|
| `tiny` (default) | A random 4-layer Qwen3.5, built offline. **Its numbers mean nothing**; it only tests the code. | none |
| `qwen35_4b` | The paper's model, with the Neuronpedia J-lens (`qwen-n1000`) | ~9.3 GB of weights plus the lens |
| `qwen36_27b` | The model on Neuronpedia's site | ~54 GB; needs a GPU |

The harness won't download anything unless you add `--allow-download`. Device: `--device cpu|mps|cuda` (default: GPU if available). On a Mac, two caveats:
- Qwen3.5's linear-attention layers fall back to slower PyTorch code (that's what the "falling back" warnings mean).
- bf16 can give tied ranks; `dtype: float32` in the config avoids that, using ~18 GB of memory.

## On the RunPod GPU

Run these from the repository root on the Mac:
1. Set the SSH command RunPod shows: `export POD_SSH="ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519"`.
2. `./pod.sh push`
   - copies the code and TwoHopFact to `/workspace`;
   - runs `setup_pod.sh` there: uv, packages, the speed kernels and the model and lens downloads (into `/workspace/hf_cache`, so they survive a pod stop);
   - then runs the tests and one real readout as a check.
3. `./pod.sh run run1 batch --limit 200 --out results/run1` starts any `cb` command in the background.
4. `./pod.sh status run1` shows the GPU and whether jobs are running. "none: pod idle" means it's safe to stop the pod.
5. `./pod.sh pull` copies the results to `data/pod-results/` (git-ignored).

After editing code here, `./pod.sh push --code` updates the pod without rerunning the setup.

The pod's driver must be recent. The setup falls back to CUDA 12.8 / 12.6 builds if the driver is older than CUDA 13 needs.

## The experiments

Put `uv run cb --config qwen35_4b` before each command (or leave out `--config` to use `tiny`).

**1. Worked example: Peter's Mozart checks.**

```bash
uv run cb mozart
```

- **Readouts:** PLAYGROUND prompts 1–8, for Salzburg, Graz, Vienna, Tokyo, Austria and Budapest.
- **Free replies:** what the model answers to prompts 7–8 without a prefill.
- **Interventions:** the swaps (Salzburg / Vienna / Austria → Budapest) and the removal test (ablate Salzburg vs Graz / Vienna / Tokyo), with P(Austria / France / Hungary).
- **Output:** `results/mozart_<config>/`.

**2. Readout of any prompt.**

```bash
uv run cb readout --user "Complete the statement: The city where Wolfgang Amadeus Mozart was born is in the country of ..." --prefill "The city where Wolfgang Amadeus Mozart was born is in the country of" --words Salzburg,Graz,Vienna,Tokyo --start "Complete" --out mozart.csv
```

- **Printed:** each word's rank per layer at the last position, plus a summary. The summary has the best rank, the band score, the final-layer rank, and the best rank over all positions from `--start` on.
- **`--out`:** the full table, for every position × layer × word × lens.
- **Plain-text prompt:** use `--text` instead of `--user`.

**3. Interventions and answer probabilities.**

```bash
uv run cb intervene --user "..." --prefill "..." --candidates Austria,France,Hungary --edit "ablate words=Salzburg layers=23-30 'pos=from:Complete the statement'" --edit "ablate words=Vienna layers=23-30 'pos=from:Complete the statement'" --edit "swap words=Salzburg,Budapest layers=23-30 pos=last s=1" --edit "steer words=Salzburg alpha=-0.5 layers=26" --generate
```

- **ablate:** removes the word's J-lens direction and adds nothing back, so the lens's score for that word drops to about 0.
- **steer:** adds `alpha` × (size of the internal state) × the direction. A negative alpha pushes away from the word.
- **swap:** exchanges the two words' components. `s=1` is an exact swap.
- **Options:**
  - `pos=`: `last`, `all`, `-3`, `from:<text>` or `span:<text>`;
  - `layers=`: `23-30`, `23,26`, `band` or `all`;
  - `lens=logit`: use logit-lens directions instead.
- **Reports:**
  - `p_first`: the probability of the word's first piece;
  - `p_full`: the probability of the whole word;
  - `d_p_full`: the change from the unedited run;
  - the top tokens; with `--generate`, also the model's greedy answer.
- **Caution:** Neuronpedia may define ablate and swap differently. Ours are written out in `src/consulted_bridges/interventions.py`.

**4. Batch readouts over many facts.**

```bash
uv run cb batch --limit 200 --per-bridge 3 --out results/run1
```

- **Facts:** TwoHopFact's two city → country categories, the paper's.
- **Per bridge:** at most `--per-bridge` facts, so a few cities can't dominate.
- **Controls:** picked automatically. `--same-country frequent` uses a well-known city instead of a random one.
- **Dropped:**
  - facts where the bridge and the answer share a piece (Luxembourg, Luxembourg);
  - facts whose subject names the bridge or the answer ("University of Zurich" → Zürich). `--keep-leaky` keeps them.
- **Conditions** (`--conditions`), as in `conditions.py`:
  - `q_alone`, `country_alone`, `subject_alone`, `answer_written`, `filler`;
  - `np_agree`, `np_pressure`, `np_hidden`: Neuronpedia-style chat;
  - `paper_agree`, `paper_pressure`, `paper_hidden`, `paper_absent`, `paper_solo`: the paper's exact prompts.
- **Output files:**
  - `items.csv`: each fact's words and pieces;
  - `ranks.csv`: rank at the last position, and the best rank over the question's positions, per condition × lens × layer × role;
  - `probs.csv`: P(answer) and P(peer answer);
  - `summary_*.csv`: the summaries below.

**5. Batch removal test.**

```bash
uv run cb ablate --condition q_alone --layers band --where question --limit 200 --out results/run1
```

For each fact, it ablates each role's word in turn and records how much log P(answer) drops.

**6. Known facts and summaries.**

- **Known facts:** `uv run cb known --out known.csv` runs the paper's 3-prompt screen. Then pass `--known known.csv` to `batch` / `ablate`.
- **Summaries:** `uv run cb summarize results/run1 --band 23-30` recomputes all the tables.

## Reading the summaries

- **Same questions:** every table uses only facts that have all the conditions and roles being compared.
- **Two averages:** each table reports the mean over facts, and the mean with **each bridge counted once**, with a 95% interval that resamples bridges.
- **`summary_contrasts.csv`, `same_country - bridge`:** how much better the bridge ranks than another city of the same country. This is DESIGN.md's key measure (> 0 favours reasoning).
- **`summary_conditions.csv`, `answer_written - filler`:** how much writing "Austria" lifts each city, against text of the same length. If the bridge and the same-country city rise together, that points to association.
- **`summary_ablate.csv`:** whether ablating the bridge hurts the answer more than ablating Vienna or a random city does.

## Not checked yet

- Nothing has run on the real Qwen weights. Only the Qwen3.5-4B tokenizer was checked: " Salzburg" is one piece, Ajaccio's first piece is the generic " Aj", and the chat prompt is right.
- The agent's own earlier answer is taken to be the true answer, as the paper keeps only facts the model gets right in round 0.
- The `known` screen matches answers against the dataset name plus a few short forms, which is stricter than the paper.
