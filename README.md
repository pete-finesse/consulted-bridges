# Consulted bridges

**Does a language model actually use the hidden middle step of a two-hop question?**

Take "What letter does the name of the country where the Colosseum is located start with?" To answer "I", the model must pass through **Italy**, a word that is never written. We call that word the **bridge**. The J-lens (Jacobian lens, Gurnee et al.; on Neuronpedia) gives a direction in the model's residual stream for each word at each layer. Editing the residual along a word's direction tests causally whether the model **consults** the bridge, not just whether the bridge can be read out.

This repository holds:
- **The bridge scorecard:** a set of gates and removal and swap tests that sorts two-hop questions into *consulted bridge* and its alternatives (answer echo, category effect, spread out, bystander, untestable);
- **Results** on Qwen3.5-4B and Qwen3.6-27B;
- **An application to *Silent Dissent*** (arXiv 2610.02702): when an agent gives in to its peers, does its own hidden reasoning still push back?

## The scorecard in brief

- **Gates.** The model knows both hops. The edited words are single tokens. The answer's own direction has an onset layer.
- **Reach check** (was gate 4; not a gate). The edit stops the model naming the bridge in the same prompt. It is read only when removing the bridge changes nothing: pass means bystander, fail means untestable. A positive result stands either way.
- **Quality** = need × specificity × isolation, set to 0 unless the bridge acts before the answer:
  - **need:** how much removing the bridge, before the answer forms, drops P(answer), beyond a noise floor of 12 everyday words of matched strength;
  - **specificity:** the share of that drop still there when only the bridge's own part, not shared with words of the same kind, is removed;
  - **isolation:** the share still there when the part shared with the answer is kept.
- **Swaps** test whether the bridge's direction carries its *content*. The current swap exchanges the two words' components. Neuronpedia's swap instead moves the source word's component onto the target, h ← h + (h·ŝ)(t̂ − ŝ), and is the planned replacement. See [`docs/swap_replication.md`](docs/swap_replication.md).

Full method and results: [`docs/scorecard/v3_v4.md`](docs/scorecard/v3_v4.md). What's next: [`NEXT_EXPERIMENTS.md`](NEXT_EXPERIMENTS.md). What was tried: [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).

## Setup

```bash
uv sync --extra dev
uv run pytest        # 15 tests on a tiny random model, about 30 s on CPU
```

- **TwoHopFact** (only for the Silent Dissent paper replication) goes in `data/twohopfact/twohopfact_slim.csv`.
- **Models:** `configs/qwen35_4b.yaml` (~10 GB) and `configs/qwen36_27b.yaml` (~54 GB, needs a GPU). Nothing downloads unless you pass `--allow-download`.
- **GPU runs** use RunPod with a shared network volume: [`docs/runpod.md`](docs/runpod.md).

```bash
# score questions (question sets: card3, proposed, ideas, card2)
uv run python -m consulted_bridges.scorecard --config qwen35_4b --sets card3 --out results/my_run
# rebuild the question bank tables from saved results
uv run python -m consulted_bridges.bank OUTDIR
```

## Layout

```
src/consulted_bridges/
  load.py text.py interventions.py readout.py tiny.py cli.py   model + J-lens loading, positions, edits (ablate, steer, swap, ...)
  scorecard.py   the current bridge scorecard (v4.1)
  bank.py        builds the question bank tables from results/
  questions/     question sets: core (v3), v2, ideas/, proposed/, group
  silent_dissent/  paper replication (conditions, facts, batch, mozart, ...) and the give-up experiments
  legacy/        earlier scorecard versions and one-off tests; some saved results come from them
scripts/         one-off runs (swap replication, 27B onset rescoring, give-up runs), pod/
configs/         model configs: tiny, qwen35_4b, qwen36_27b
docs/            method, reviews, proposals, harness and RunPod guides
results/         saved runs (CSV), named <run>_<model>
tests/
```

## Prior work

The Silent Dissent experiment harness and the paper notes were written before the hackathon, in [swarm-explore](https://github.com/pete-finesse/swarm-explore). The following files are extended for this work from swarm-explore:

- `src/consulted_bridges/`: `load.py`, `text.py`, `interventions.py`, `readout.py`, `tiny.py`, `cli.py`
- `src/consulted_bridges/silent_dissent/`: `conditions.py`, `facts.py`, `batch.py`, `known.py`, `mozart.py`, `summarize.py`
- `tests/`
- `configs/`
- `scripts/pod/pod.sh`, `scripts/pod/setup_pod.sh`
- `docs/harness.md`
- `pyproject.toml`, `uv.lock`

### What was built during the sprint weekend

Everything not listed above was written during the sprint (7 to 9 November 2026): the bridge scorecard and its
gates (`scorecard.py`, `legacy/bridge_card*.py`, `questions/`), the question bank (`bank.py`), the conformity
experiment (`conformity.py`, `conformity_stats.py`, `conformity_select.py`, `conformity_funnel.py`,
`conformity_block.py`, `conformity_v2_report.py`), the peer block and patching (`peerblock.py`), the candidate
questions (`questions/proposed/candidates.py`, `status.py`), the pod runners under `scripts/pod/`, and every result
under `results/`. The write-ups under `docs/conformity/` and the experiment log record the order in which it happened.
If the sprint's official start was later than 6 November, the 6 November scorecard commits are pre-existing too.

## Credits

- **Tools:** much of the code and writing was done with Claude Code agents, directed and reviewed by Peter Lakner.
- **Third-party code:** the Silent Dissent authors' code (MIT), [github.com/ziangni-sys/silent-dissent](https://github.com/ziangni-sys/silent-dissent), is a pinned dependency.
