#!/usr/bin/env bash
# One-shot setup ON THE POD (RunPod, Linux + NVIDIA GPU). Safe to rerun: each step skips work done.
# Everything lives under /workspace, which survives a pod stop:
#   /workspace/consulted-bridges  (this repository, pushed by scripts/pod/pod.sh; data/twohopfact inside it)
#   /workspace/hf_cache       model + lens downloads        /workspace/uv  uv, its Python and package cache
# Usage (pod.sh push does this for you):  bash setup_pod.sh [--no-download]
set -euo pipefail

W=/workspace
H=$W/consulted-bridges
export HF_HOME=$W/hf_cache UV_CACHE_DIR=$W/uv/cache UV_PYTHON_INSTALL_DIR=$W/uv/python
export PATH=$W/uv/bin:$PATH
mkdir -p "$HF_HOME" "$W/uv/bin" "$W/logs"

# Environment for later shells (pod.sh run sources it)
cat > $W/env.sh <<EOF
export HF_HOME=$HF_HOME UV_CACHE_DIR=$UV_CACHE_DIR UV_PYTHON_INSTALL_DIR=$UV_PYTHON_INSTALL_DIR
export PATH=$W/uv/bin:\$PATH
export TOKENIZERS_PARALLELISM=false
export UV_NO_SYNC=1      # keep the extra kernel packages installed below (uv sync would remove them)
cd $H
EOF

echo "== GPU"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
df -h $W | tail -1

echo "== uv"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$W/uv/bin INSTALLER_NO_MODIFY_PATH=1 sh
uv --version

echo "== packages (the repository's uv.lock, Linux CUDA wheels)"
cd $H
uv sync --extra dev --python 3.12
export UV_NO_SYNC=1
# The lock's Linux torch uses CUDA 13 wheels, which need an NVIDIA driver >= 580. On older
# drivers, swap in the same torch version built for CUDA 12.8 / 12.6.
cuda_ok() { uv run python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; }
if ! cuda_ok; then
  TV=$(uv run python -c "import torch; print(torch.__version__.split('+')[0])")
  for cu in cu128 cu126; do
    echo "torch $TV sees no GPU with this driver; trying the $cu build"
    uv pip install --reinstall "torch==$TV" --index-url https://download.pytorch.org/whl/$cu && cuda_ok && break
  done
fi
uv run python -c "import torch; assert torch.cuda.is_available(), 'torch sees no GPU'; print('torch', torch.__version__, 'CUDA', torch.version.cuda, torch.cuda.get_device_name())"

echo "== Qwen3.5 speed kernels (optional: without them the model runs, only slower)"
# flash-linear-attention: pure Triton, no compile. kernels: lets transformers fetch prebuilt
# causal-conv1d / gated-delta-rule kernels from the HF Hub (kernels-community) instead of compiling.
uv pip install flash-linear-attention kernels || echo "WARN: flash-linear-attention / kernels not installed"
# causal-conv1d: only a prebuilt wheel (a source build needs nvcc and can take 20+ min)
uv pip install --only-binary=:all: causal-conv1d 2>/dev/null || echo "note: no prebuilt causal-conv1d wheel; transformers uses the Hub kernel or the PyTorch fallback"

if [[ "${1:-}" != "--no-download" ]]; then
  echo "== downloads into $HF_HOME for config ${SETUP_CONFIG:-qwen35_4b}"
  uv run python - <<'PY'
import yaml
from huggingface_hub import hf_hub_download, snapshot_download
import os
cfg = yaml.safe_load(open(f"configs/{os.environ.get('SETUP_CONFIG', 'qwen35_4b')}.yaml"))
p = snapshot_download(cfg["model"]["name"], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
print("model:", p)
k = cfg["lens"]["kwargs"]
print("lens:", hf_hub_download(k["repo"], k["filename"], revision=k["revision"]))
PY
fi

echo "== tests (tiny model, CPU)"
uv run pytest -q -p no:cacheprovider -W ignore

echo "== smoke test on the real model (one readout)"
uv run cb --config ${SETUP_CONFIG:-qwen35_4b} readout \
  --user "Complete the statement: The city where Wolfgang Amadeus Mozart was born is in the country of ..." \
  --prefill "The city where Wolfgang Amadeus Mozart was born is in the country of" \
  --words Salzburg,Graz,Vienna,Tokyo > $W/logs/smoke.log 2>&1
grep -v "Loading weights\|falling back" $W/logs/smoke.log | tail -25
n=$(grep -c "falling back" $W/logs/smoke.log || true)
echo "== speed kernels: $([[ $n == 0 ]] && echo 'fast path in use' || echo "$n slow PyTorch fallbacks (works, slower)")"
echo "setup done. Later shells: source $W/env.sh"
