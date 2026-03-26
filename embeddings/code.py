"""
Omnex — Code Embeddings (CodeBERT)
Produces 768-dim semantic embeddings for code chunks using microsoft/codebert-base.

CodeBERT is trained on code+NL pairs across 6 languages (Python, Java, JS, PHP, Ruby, Go)
and generalises well to other languages. Embeddings capture semantic meaning of code
beyond syntax — two functions that do the same thing in different languages are close.

Usage:
    from embeddings.code import embed, embed_batch
"""

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

_tokenizer = None
_model = None
_device: str | None = None


def _get_model():
    global _tokenizer, _model, _device
    if _model is not None:
        return _tokenizer, _model, _device

    import torch
    from transformers import AutoTokenizer, AutoModel

    model_name = os.getenv("CODEBERT_MODEL", "microsoft/codebert-base")
    cache_dir   = os.getenv("MODEL_CACHE_DIR", "models/cache")

    logger.info(f"Loading CodeBERT: {model_name}")
    _tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    _model     = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

    _device = "cuda" if (os.getenv("USE_GPU", "0") == "1" and torch.cuda.is_available()) else "cpu"
    _model = _model.to(_device)
    _model.eval()

    logger.info(f"CodeBERT loaded on {_device}")
    return _tokenizer, _model, _device


def embed(text: str) -> np.ndarray:
    """
    Embed a single code chunk. Returns a 768-dim L2-normalised numpy vector.
    Truncates to 512 tokens (CodeBERT's context window).
    """
    return embed_batch([text])[0]


def embed_batch(texts: list[str]) -> np.ndarray:
    """
    Embed a batch of code chunks. Returns np.ndarray of shape (N, 768), L2-normalised.
    """
    import torch

    tokenizer, model, device = _get_model()

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # CLS token as chunk representation
    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    # L2 normalise
    norms = np.linalg.norm(cls_embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return cls_embeddings / norms
