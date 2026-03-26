"""
Omnex — Vector Index (USearch)
File-based compressed vector index using USearch — i8 quantization gives ~4x
storage compression vs float32. No server process required.

Indexes:
    text_chunks.usearch   — 384-dim MiniLM, cosine, i8
    image_chunks.usearch  — 512-dim CLIP, cosine, i8
    video_frames.usearch  — 512-dim CLIP, cosine, i8
    audio_chunks.usearch  — 384-dim MiniLM, cosine, i8
    code_chunks.usearch   — 768-dim CodeBERT, cosine, i8
"""

from __future__ import annotations

import json
import os
import threading
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from usearch.index import Index


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

_indexes:  dict[str, Index] = {}
_payloads: dict[str, dict[int, dict]] = {}   # key → payload
_counters: dict[str, int] = {}               # next key
_locks:    dict[str, threading.Lock] = {}


def _index_dir() -> Path:
    root = Path(os.getenv("OMNEX_DATA_PATH", "./data"))
    d = root / "leann"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _index_file(name: IndexName) -> Path:
    return _index_dir() / f"{name.value}.usearch"


def _payload_file(name: IndexName) -> Path:
    return _index_dir() / f"{name.value}.payload.json"


def _get_lock(name: IndexName) -> threading.Lock:
    key = name.value
    if key not in _locks:
        _locks[key] = threading.Lock()
    return _locks[key]


def _get_index(name: IndexName) -> tuple[Index, dict[int, dict], int]:
    key = name.value
    if key not in _indexes:
        with _get_lock(name):
            if key not in _indexes:
                dim = INDEX_DIMS[name]
                idx = Index(ndim=dim, metric="cos", dtype="i8")
                ip  = _index_file(name)
                pp  = _payload_file(name)

                if ip.exists():
                    try:
                        idx.load(str(ip))
                        payload = json.loads(pp.read_text()) if pp.exists() else {}
                        payload = {int(k): v for k, v in payload.items()}
                        counter = max(payload.keys(), default=-1) + 1
                    except Exception:
                        import logging
                        logging.getLogger("omnex.leann").warning(
                            f"Corrupt index file {ip} — deleting and starting fresh"
                        )
                        ip.unlink(missing_ok=True)
                        if pp.exists():
                            pp.unlink(missing_ok=True)
                        idx = Index(ndim=dim, metric="cos", dtype="i8")
                        payload = {}
                        counter = 0
                else:
                    payload = {}
                    counter = 0

                _indexes[key]  = idx
                _payloads[key] = payload
                _counters[key] = counter

    return _indexes[key], _payloads[key], _counters[key]


def _save(name: IndexName) -> None:
    key = name.value
    _indexes[key].save(str(_index_file(name)))
    _payload_file(name).write_text(json.dumps(_payloads[key]))


def add_vector(
    index_name: IndexName,
    chunk_id: str,
    vector: list[float] | np.ndarray,
    metadata: dict | None = None,
) -> str:
    _get_index(index_name)  # ensure loaded before acquiring lock
    with _get_lock(index_name):
        key     = index_name.value
        idx     = _indexes[key]
        payload = _payloads[key]
        counter = _counters[key]

        vec = np.array(vector, dtype=np.float32)
        idx.add(counter, vec)
        payload[counter] = {"chunk_id": chunk_id, **(metadata or {})}
        _counters[key] = counter + 1

        _save(index_name)
        return str(counter)


def search(
    index_name: IndexName,
    query_vector: list[float] | np.ndarray,
    top_k: int = 20,
    filters: dict | None = None,
) -> list[dict]:
    idx, payload, _ = _get_index(index_name)

    if len(idx) == 0:
        return []

    k = min(top_k, len(idx))
    vec = np.array(query_vector, dtype=np.float32)
    matches = idx.search(vec, k)

    results = []
    for key, distance in zip(matches.keys, matches.distances):
        p = payload.get(int(key), {})
        # usearch cosine distance = 1 - cosine_similarity
        score = float(1.0 - distance)
        results.append({
            "chunk_id": p.get("chunk_id"),
            "score":    score,
            "payload":  p,
            "leann_id": str(key),
        })

    return results


def delete_vector(index_name: IndexName, leann_id: str) -> None:
    _get_index(index_name)  # ensure loaded before acquiring lock
    with _get_lock(index_name):
        key = index_name.value
        iid = int(leann_id)
        _indexes[key].remove(iid)
        _payloads[key].pop(iid, None)
        _save(index_name)


def delete_vectors(chunk_ids: list[str]) -> int:
    """Remove all vectors matching the given chunk_ids across all indexes."""
    id_set = set(chunk_ids)
    deleted = 0
    for name in IndexName:
        try:
            # _get_index acquires the lock internally — call it first,
            # then re-acquire for the mutation to avoid deadlock.
            _get_index(name)  # ensure loaded
            with _get_lock(name):
                key = name.value
                idx     = _indexes[key]
                payload = _payloads[key]
                to_remove = [iid for iid, p in payload.items() if p.get("chunk_id") in id_set]
                for iid in to_remove:
                    try:
                        idx.remove(iid)
                        payload.pop(iid, None)
                        deleted += 1
                    except Exception:
                        pass
                if to_remove:
                    _save(name)
        except Exception:
            pass
    return deleted


def index_size(index_name: IndexName) -> int:
    try:
        idx, _, _ = _get_index(index_name)
        return len(idx)
    except Exception:
        return 0


def index_path(index_name: IndexName) -> Path:
    return _index_file(index_name)
