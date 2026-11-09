"""Load a model, its tokenizer and the two lenses (J-lens and logit lens) from a YAML config.

Reuses the authors' loaders (silent_dissent.model.load_model, silent_dissent.lenses.build_lens).
Downloads are refused unless `allow_download=True` (CLI: --allow-download), so nothing large
arrives without Peter saying yes. `model.name: tiny` builds a tiny random Qwen3.5 offline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml

from silent_dissent.lenses import LogitLens, build_lens
from silent_dissent.model import get_layers

HERE = Path(__file__).resolve().parents[2]  # repository root
CONFIGS = HERE / "configs"


class DownloadNeeded(RuntimeError):
    pass


@dataclass
class Harness:
    model: torch.nn.Module
    tok: object
    lenses: dict                      # {"jlens": JLens, "logit": LogitLens}
    cfg: dict = field(default_factory=dict)

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def out_device(self):
        """Where the unembedding (and so the lenses' maps) live; captured residuals go here."""
        return self.model.get_output_embeddings().weight.device

    @property
    def n_layers(self) -> int:
        return len(get_layers(self.model))

    @property
    def band(self) -> list[int]:
        """Layers averaged in summaries: the config's band, else the paper-like default
        (roughly the last quarter, without the final layer)."""
        b = self.cfg.get("band")
        if b:
            return list(range(b[0], b[1] + 1)) if len(b) == 2 else list(b)
        L = self.n_layers
        return list(range(max(0, int(0.7 * L)), L - 1)) or [L - 1]

    def encode(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]


def read_config(path_or_name: str | dict) -> dict:
    if isinstance(path_or_name, dict):
        return path_or_name
    p = Path(path_or_name)
    if not p.exists():
        p = CONFIGS / f"{path_or_name}.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


def _pick_device(want: str | None) -> str:
    if want and want != "auto":
        return want
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _cached(repo: str, filename: str, revision: str | None = None) -> bool:
    from huggingface_hub import try_to_load_from_cache

    p = try_to_load_from_cache(repo, filename, revision=revision)
    return isinstance(p, str)


def _check_downloads(cfg: dict) -> list[str]:
    """What would have to be downloaded for this config (empty = everything is local)."""
    need = []
    name = cfg["model"]["name"]
    if name != "tiny" and not Path(name).exists():
        if not (_cached(name, "config.json") and (_cached(name, "model.safetensors.index.json")
                                                  or _cached(name, "model.safetensors"))):
            need.append(f"model {name} ({cfg['model'].get('size_note', 'several GB')})")
    lk = cfg.get("lens", {}).get("kwargs", {})
    if "repo" in lk and not _cached(lk["repo"], lk["filename"], lk.get("revision")):
        need.append(f"lens {lk['repo']}@{lk.get('revision')}:{lk['filename']} "
                    f"({cfg['lens'].get('size_note', 'hundreds of MB')})")
    return need


def load(config: str | dict = "tiny", allow_download: bool = False, device: str | None = None) -> Harness:
    cfg = dict(read_config(config))
    m = cfg["model"]
    if m["name"] == "tiny":
        from .tiny import build_tiny, tiny_device

        dev = tiny_device(device or m.get("device", "cpu"))
        model, tok, lens_path = build_tiny(**m.get("tiny", {}), device=dev)
        lenses = {"jlens": build_lens(model, "jlens", path=str(lens_path)), "logit": LogitLens(model)}
        return Harness(model, tok, lenses, cfg)

    need = _check_downloads(cfg)
    if need and not allow_download:
        raise DownloadNeeded("Not downloaded yet: " + "; ".join(need) +
                             ". Ask Peter first, then rerun with --allow-download (allow_download=True).")
    from silent_dissent.model import load_model

    if m.get("device_map"):     # too big for one GPU: let accelerate spread the blocks over all of them
        model, tok = load_model(m["name"], m.get("dtype", "bfloat16"), m["device_map"], m.get("chat_template_kwargs"))
    else:
        dev = _pick_device(device or m.get("device"))
        model, tok = load_model(m["name"], m.get("dtype", "bfloat16"), None, m.get("chat_template_kwargs"))
        model = model.to(dev)
    if m.get("chat_template"):  # a model released without one (DeepSeek-V4): a file in configs/
        tok.chat_template = (CONFIGS / m["chat_template"]).read_text()
    lc = cfg.get("lens", {"name": "jlens", "kwargs": {}})
    lenses = {"jlens": build_lens(model, lc.get("name", "jlens"), **lc.get("kwargs", {})), "logit": LogitLens(model)}
    return Harness(model, tok, lenses, cfg)
