# Running on RunPod: use the shared network volume

*Everything needed is already on the volume, so a new pod is ready in about 30 seconds and runs a scorecard question in 2–4 minutes, with no downloads.*

## The volume

| | |
|---|---|
| Name / id | `silent-dissent-models` / `urkqhujhdk` |
| Region | **CA-MTL-3** (every pod that uses it must run there) |
| Size | 250 GB (74 GB used) |
| Contents | Qwen3.5-4B and Qwen3.6-27B weights, both J-lenses (`/workspace/hf_cache`), uv and its cache (`/workspace/uv`), `/workspace/env.sh`. The old `swarm-explore` layout and its `.venv` are still there too |
| Cost | ~$17.50/month whether or not a pod is running |

## Starting a pod

GPUs that fit both models in CA-MTL-3: RTX PRO 6000 (96 GB), A100 80 GB PCIe, H200. Any of them works; try them in that order.

```bash
set -a && . ./.env && set +a      # from the repo root: loads RUNPOD_API_KEY
runpodctl pod create --name sd-<job> --gpu-id "NVIDIA A100 80GB PCIe" \
  --image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 --container-disk-in-gb 40 \
  --network-volume-id urkqhujhdk --data-center-ids CA-MTL-3 --min-cuda-version 12.8 --ports "22/tcp"
runpodctl pod get <pod-id>         # the "ssh" field has the IP and port
```

Then, from the repository root:

```bash
export POD_SSH="ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519"
scripts/pod/pod.sh push --code     # copy your latest code to /workspace/consulted-bridges
scripts/pod/pod.sh ssh "uv run python -m consulted_bridges.scorecard --config qwen36_27b ..."
```

Don't run `scripts/pod/pod.sh push` without `--code`: the models are already there. The first time this repository is used on the volume, run `bash scripts/pod/setup_pod.sh --no-download` once on the pod to build its `.venv` (packages come from the uv cache on the volume).

## When you're done

- Copy results to the Mac (`scripts/pod/pod.sh pull`, or `tar` over ssh).
- **Delete the pod** (`runpodctl pod delete <pod-id>`). Don't just stop it: the volume keeps everything, and a stopped pod's own disk keeps billing.
- Log the hours in your cost log.

## Rules

- **One writer at a time.** Two pods writing the same files on the volume can corrupt them. Several pods *reading* the models is fine.
- **Write results under a folder named for your job** (`results/<job>/`), not over someone else's.
- **If CA-MTL-3 has no GPU free,** retry for a few minutes. Failing that, create a pod in any region *without* the volume and run the full `scripts/pod/pod.sh push` setup (10–20 min).

## Tested

On a fresh A100 pod with Hugging Face downloads switched off:
- SSH ready **23 s** after creating the pod;
- spider legs on Qwen3.5-4B: **consulted bridge**, 123 s including model load;
- Colosseum letter on Qwen3.6-27B: **consulted bridge**, 212 s including model load;
- **6 min 14 s** from creating the pod to both results.
