"""
Omnex — LEANN Vector Index
File-based vector index using LEANN — 97% storage savings vs Qdrant.
No server process required. Indexes are .leann files on the destination drive.

Indexes:
    text_chunks.leann   — 384-dim MiniLM, cosine
    image_chunks.leann  — 512-dim CLIP, cosine  (Phase 2)
    video_frames.leann  — 512-dim CLIP, cosine  (Phase 5)
    audio_chunks.leann  — 384-dim MiniLM, cosine (Phase 5)
    code_chunks.leann   — 768-dim CodeBERT, cosine (Phase 6)
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

try:
    import leann
    _LEANN_AVAILABLE = True
except ImportError:
    _LEANN_AVAILABLE = False


class IndexName(str, Enum):
    TEXT   = "text_chunks"
    IMAGE  = "image_chunks"
    VIDEO  = "video_frames"
    AUDIO  = "audio_chunks"
    CODE   = "code_chunks"


INDEX_DIMS = {
    IndexName.TEXT:  384,
    IndexName.IMAGE: 512,
    IndexName.VIDEO: 512,
    IndexName.AUDIO: 384,
    IndexName.CODE:  768,
}

_indexes: dict[str, Any] = {}


def _index_dir() -> Path:
    root = Path(os.getenv("OMNEX_DATA_PATH", "./data"))
    d = root / "leann"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_index(name: IndexName):
    """Load or create a LEANN index."""
    if not _LEANN_AVAILABLE:
        raise RuntimeError(
            "LEANN is not installed. Run: pip install leann"
        )

    key = name.value
    if key not in _indexes:
        index_path = str(_index_dir() / f"{key}.leann")
        dim = INDEX_DIMS[name]

        if Path(index_path).exists():
            _indexes[key] = leann.load(index_path)
        else:
            _indexes[key] = leann.Index(
                dim=dim,
                metric="cosine",
                path=index_path,
            )

    return _indexes[key]


def add_vector(
    index_name: IndexName,
    chunk_id: str,
    vector: list[float] | np.ndarray,
    metadata: dict | None = None,
) -> str:
    """
    Add a vector to the index.

    Returns:
        LEANN internal node ID (stored on the chunk document as leann_id)
    """
    idx = _get_index(index_name)
    vec = np.array(vector, dtype=np.float32)
    node_id = idx.add(vec, payload={"chunk_id": chunk_id, **(metadata or {})})
    idx.save()
    return str(node_id)


def search(
    index_name: IndexName,
    query_vector: list[float] | np.ndarray,
    top_k: int = 20,
    filters: dict | None = None,
) -> list[dict]:
    """
    Search the index for nearest neighbours.

    Returns:
        List of dicts: [{chunk_id, score, payload}, ...]
        Sorted by score descending (most similar first).
    """
    idx = _get_index(index_name)
    vec = np.array(query_vector, dtype=np.float32)

    results = idx.search(vec, k=top_k, filter=filters)

    return [
        {
            "chunk_id": r.payload.get("chunk_id"),
            "score":    float(r.score),
            "payload":  r.payload,
            "leann_id": str(r.id),
        }
        for r in results
    ]


def delete_vector(index_name: IndexName, leann_id: str) -> None:
    """Remove a vector from the index by its LEANN node ID."""
    idx = _get_index(index_name)
    idx.delete(leann_id)
    idx.save()


def index_size(index_name: IndexName) -> int:
    """Return the number of vectors in the index."""
    try:
        idx = _get_index(index_name)
        return len(idx)
    except Exception:
        return 0


def index_path(index_name: IndexName) -> Path:
    return _index_dir() / f"{index_name.value}.leann"
