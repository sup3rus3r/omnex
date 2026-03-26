"""
Omnex — Text Embeddings
MiniLM (all-MiniLM-L6-v2) sentence-transformer wrapper.
384-dimensional embeddings, cosine similarity.
Lazy-loaded on first use. Cached for the process lifetime.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

_model = None
_model_lock = threading.Lock()
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                import sys, time
                print(f"[embeddings.text] loading SentenceTransformer...", flush=True, file=sys.stderr)
                t0 = time.time()
                from sentence_transformers import SentenceTransformer
                # MiniLM-L6-v2 is fast enough on CPU and avoids torch CUDA initialisation
                # in the main uvicorn process (CUDA init + jemalloc + onnxruntime-gpu
                # in the same process causes heap corruption / SIGABRT).
                _model = SentenceTransformer(MODEL_ID, device="cpu")
                print(f"[embeddings.text] model ready ({time.time()-t0:.1f}s)", flush=True, file=sys.stderr)
    return _model


def embed(text: str) -> np.ndarray:
    return embed_batch([text])[0]


def embed_batch(texts: list[str], batch_size: int = 64) -> np.ndarray:
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.astype(np.float32)


def _gpu_available() -> bool:
    if os.getenv("GPU_ENABLED", "false").lower() != "true":
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
