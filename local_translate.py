"""Local neural machine translation (no cloud API)."""

from __future__ import annotations

import os
import threading
from typing import Any

DEFAULT_LOCAL_MODEL = os.environ.get(
    "DUB_TRANSLATE_MODEL", "facebook/nllb-200-distilled-600M"
)

# NLLB-200 language codes for Japanese source.
NLLB_SOURCE = "jpn_Jpan"
NLLB_TARGETS: dict[str, str] = {
    "hi": "hin_Deva",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "zh": "zho_Hans",
    "ko": "kor_Hang",
    "ja": "jpn_Jpan",
}

_lock = threading.Lock()
_engine: dict[str, Any] = {}


def supports_local_target(language: str) -> bool:
    return language.split("-", 1)[0].lower() in NLLB_TARGETS


def _device() -> str:
    import torch

    if os.environ.get("DUB_TRANSLATE_DEVICE"):
        return os.environ["DUB_TRANSLATE_DEVICE"]
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_engine(model_name: str) -> tuple[Any, Any]:
    if model_name in _engine:
        return _engine[model_name]

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = _device()
    print(f"Loading local translation model: {model_name} ({device})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if device == "cuda":
        model = model.half().to(device)
    else:
        model = model.to(device)
    model.eval()
    _engine[model_name] = (tokenizer, model, device)
    print(f"Local translator ready on {device}.")
    return _engine[model_name]


def local_translate(
    text: str,
    target_language: str,
    *,
    model_name: str | None = None,
    source_language: str = "ja",
) -> str:
    """Translate text with a local NLLB model (default: distilled-600M)."""
    text = text.strip()
    if not text:
        return ""

    lang = target_language.split("-", 1)[0].lower()
    if lang == source_language.split("-", 1)[0].lower():
        return text

    tgt_code = NLLB_TARGETS.get(lang)
    if not tgt_code:
        raise ValueError(f"Local translator has no NLLB code for target '{target_language}'")

    model_name = model_name or DEFAULT_LOCAL_MODEL
    tokenizer, model, device = _load_engine(model_name)

    import torch

    tokenizer.src_lang = NLLB_SOURCE
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    forced_bos = tokenizer.convert_tokens_to_ids(tgt_code)

    with _lock, torch.no_grad():
        output_ids = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos,
            max_new_tokens=256,
            num_beams=4,
            early_stopping=True,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
