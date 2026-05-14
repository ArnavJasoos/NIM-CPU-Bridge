"""
nim-cpu-bridge: backends/onnx_gen.py

ONNX Runtime text-generation helper.
Called lazily by router.py when NCB_BACKEND=onnxruntime.

Requires:
  - onnxruntime>=1.18      (pip install onnxruntime)
  - transformers>=4.41     (for tokenizer)

Apache 2.0 — see LICENSE
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    import onnxruntime as ort

logger = logging.getLogger("ncb.onnx_gen")


def _load_tokenizer(model_path: str):
    """
    Load a HuggingFace tokenizer from the model directory.
    Falls back to a generic GPT-2 tokenizer if model_path has no tokenizer.
    """
    from transformers import AutoTokenizer  # type: ignore

    try:
        tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        logger.info("Tokenizer loaded from %s", model_path)
        return tok
    except Exception:
        # Fallback: use gpt2 tokenizer (reasonable for most LLM families)
        logger.warning(
            "Could not load tokenizer from %s — falling back to gpt2 tokenizer",
            model_path,
        )
        return AutoTokenizer.from_pretrained("gpt2")


def generate_onnx(
    session: "ort.InferenceSession",
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: list[str] | None,
) -> Iterator[str]:
    """
    Generate text from an ONNX Runtime InferenceSession.

    The session is expected to have been created by router.py with
    CPUExecutionProvider and a model exported via optimum
    (task="text-generation-with-past").

    Yields:
        The full decoded output string as a single chunk (ONNX Runtime
        does not natively stream; async streaming is planned for v0.2).
    """
    import numpy as np  # type: ignore

    # Derive model directory from the session model path
    try:
        model_dir = os.path.dirname(session._model_path)  # type: ignore[attr-defined]
    except AttributeError:
        model_dir = os.getenv("NCB_MODEL_PATH", ".")

    tokenizer = _load_tokenizer(model_dir)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids: np.ndarray = inputs["input_ids"]          # shape (1, seq_len)
    attention_mask: np.ndarray = inputs.get(
        "attention_mask",
        np.ones_like(input_ids),
    )

    generated_ids: list[int] = []
    stop_sequences = stop or []

    for _ in range(max_tokens):
        feed = {
            "input_ids":      input_ids.astype(np.int64),
            "attention_mask": attention_mask.astype(np.int64),
        }
        # Filter feed dict to only keys the session actually accepts
        input_names = {inp.name for inp in session.get_inputs()}
        feed = {k: v for k, v in feed.items() if k in input_names}

        outputs = session.run(None, feed)

        # Logits shape: (1, seq_len, vocab_size)
        logits = outputs[0]
        next_token_logits = logits[0, -1, :]  # (vocab_size,)

        # Temperature + top-p sampling
        next_token_id = _sample(next_token_logits, temperature, top_p)
        generated_ids.append(int(next_token_id))

        # Extend input for next step
        next_token_arr = np.array([[next_token_id]], dtype=np.int64)
        input_ids = np.concatenate([input_ids, next_token_arr], axis=1)
        attention_mask = np.concatenate(
            [attention_mask, np.ones((1, 1), dtype=np.int64)], axis=1
        )

        # Check stop sequences
        decoded_so_far = tokenizer.decode(generated_ids, skip_special_tokens=True)
        if any(s in decoded_so_far for s in stop_sequences):
            break

        # Check EOS
        if next_token_id == tokenizer.eos_token_id:
            break

    yield tokenizer.decode(generated_ids, skip_special_tokens=True)


# ── Sampling helpers ──────────────────────────────────────────────────────────

def _sample(logits: "np.ndarray", temperature: float, top_p: float) -> int:
    """Greedy / temperature / top-p nucleus sampling."""
    import numpy as np  # type: ignore

    if temperature <= 0.0 or temperature == 1.0 and top_p >= 1.0:
        # Greedy
        return int(np.argmax(logits))

    # Apply temperature
    logits = logits / max(temperature, 1e-8)

    # Softmax
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()

    # Top-p nucleus
    if 0.0 < top_p < 1.0:
        sorted_idx = np.argsort(probs)[::-1]
        cumsum = np.cumsum(probs[sorted_idx])
        cutoff_mask = cumsum - probs[sorted_idx] >= top_p
        probs[sorted_idx[cutoff_mask]] = 0.0
        probs /= probs.sum()

    return int(np.random.choice(len(probs), p=probs))
