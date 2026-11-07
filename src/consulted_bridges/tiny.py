"""A tiny randomly initialised Qwen3.5 (same architecture family as Qwen3.5-4B) for offline tests.

Its numbers are meaningless: it knows no facts. It exists so every code path (chat template,
word pieces, J-lens fitting, hooks on Qwen3.5's mixed linear/full attention layers) runs on a
laptop CPU in seconds. The tokenizer is a small byte-level BPE trained on the harness's own
prompts, so rare names split into several pieces the way " Salz" + "burg" does in Qwen.
"""
from __future__ import annotations

from pathlib import Path

import torch

from .load import HERE

CACHE = HERE / ".tiny_cache"
TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def _corpus() -> list[str]:
    from silent_dissent.entity_prompts import AGAIN, CHAT_SYSTEM, PLAIN_HEADER, PLAIN_SHOTS

    from .silent_dissent.conditions import FILLER, NP_AGAIN

    base = [PLAIN_HEADER, PLAIN_SHOTS, CHAT_SYSTEM, AGAIN, NP_AGAIN, FILLER,
            "Complete the statement: The city where Wolfgang Amadeus Mozart was born is in the country of ...",
            "The answer is Austria. Tell me about Austria. Austria is Wolfgang Amadeus Mozart was",
            "Question: Agent 1: Agent 2: Agent 3: Agent 4: Moderator: (answer submitted)",
            "The university where the city is located is in the country of has its headquarters in"]
    common = ("the of and in is was a to city country born where where answer statement complete "
              "agent agents moderator taking other into account again tell me about please briefly "
              "university located headquarters capital which").split()
    places = ("Salzburg Graz Vienna Tokyo Osaka Budapest Austria France Hungary Japan Paris Lyon Ajaccio "
              "Madrid Spain Italy Rome Germany Berlin Napoleon Mishima Yukio").split()
    return base + [" ".join(common)] * 30 + [" ".join(places)] * 2


def build_tokenizer(vocab_size: int = 600):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tk.train_from_iterator(_corpus(), trainer)
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                  additional_special_tokens=SPECIAL[1:])
    tok.chat_template = TEMPLATE
    return tok


def build_model(vocab_size: int, n_layers: int = 4, d: int = 64, seed: int = 0):
    from transformers.models.qwen3_5 import configuration_qwen3_5 as c, modeling_qwen3_5 as mm

    torch.manual_seed(seed)
    cfg = c.Qwen3_5TextConfig(vocab_size=vocab_size, hidden_size=d, intermediate_size=2 * d,
                              num_hidden_layers=n_layers, num_attention_heads=4, num_key_value_heads=2,
                              head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
                              linear_key_head_dim=16, linear_value_head_dim=16, tie_word_embeddings=False,
                              layer_types=["linear_attention", "full_attention"] * (n_layers // 2)
                              + ["linear_attention"] * (n_layers % 2))
    return mm.Qwen3_5ForCausalLM(cfg).eval()


def tiny_device(want: str | None = "cpu") -> str:
    """CPU, except on a CUDA machine: once flash-linear-attention is installed, Qwen3.5's
    linear-attention layers run Triton kernels, which need GPU tensors."""
    import torch

    return "cuda" if (want in (None, "cpu", "auto") and torch.cuda.is_available()) else (want or "cpu")


def build_tiny(n_layers: int = 4, seed: int = 0, rebuild: bool = False, device: str = "cpu"):
    """(model, tokenizer, J-lens path), cached in .tiny_cache. The model is on `device`."""
    from transformers import AutoTokenizer

    from silent_dissent.jlens import fit

    out = CACHE / f"L{n_layers}_s{seed}"
    lens = out / "jlens.pt"
    if rebuild or not lens.exists():
        out.mkdir(parents=True, exist_ok=True)
        tok = build_tokenizer()
        tok.save_pretrained(out)
        model = build_model(len(tok), n_layers, seed=seed)
        model.save_pretrained(out)
        model = model.to(device)
        text = " ".join(_corpus()[:6])
        prompts = [text[i : i + 400] for i in range(0, max(1, len(text) - 200), 150)]
        fit(model, tok, prompts, str(lens), dim_batch=16, max_seq_len=96, skip_first=2, checkpoint_every=0)
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    tok = AutoTokenizer.from_pretrained(out)
    tok.chat_template = TEMPLATE
    tok.sd_template_kwargs = {}
    tok.padding_side = "left"
    model = Qwen3_5ForCausalLM.from_pretrained(out).eval().to(device)
    return model, tok, lens
