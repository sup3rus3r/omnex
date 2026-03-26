"""
Omnex — Binary Chunk Store
GridFS-inspired content-addressed storage for raw binary data.
Files are split into 4MB chunks stored as flat files in a
content-addressed directory structure on the destination drive.

Layout:
    {OMNEX_DATA_PATH}/chunks/{ab}/{ab3f92...chunk0.bin}
    {OMNEX_DATA_PATH}/thumbnails/{chunk_id}.jpg
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

CHUNK_SIZE = 4 * 1024 * 1024  # 4MB


def _store_root() -> Path:
    root = Path(os.getenv("OMNEX_DATA_PATH", "./data"))
    return root


def _chunks_dir() -> Path:
    d = _store_root() / "chunks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _thumbnails_dir() -> Path:
    d = _store_root() / "thumbnails"
    d.mkdir(parents=True, exist_ok=True)
    return d


def store_file(source_path: Path, content_hash: str) -> list[str]:
    """
    Store a binary file in the chunk store.
    Files > 4MB are split into multiple chunks.

    Returns:
        List of data_ref strings for each stored chunk
        e.g. ["ab/ab3f92...chunk0.bin", "ab/ab3f92...chunk1.bin"]
    """
    prefix = content_hash[:2]
    chunk_dir = _chunks_dir() / prefix
    chunk_dir.mkdir(exist_ok=True)

    refs: list[str] = []
    chunk_index = 0

    with open(source_path, "rb") as f:
        while True:
            data = f.read(CHUNK_SIZE)
            if not data:
                break
            filename = f"{content_hash}_chunk{chunk_index}.bin"
            dest = chunk_dir / filename
            if not dest.exists():
                dest.write_bytes(data)
            refs.append(f"{prefix}/{filename}")
            chunk_index += 1

    return refs


def read_chunk(data_ref: str) -> bytes:
    """Read a stored binary chunk by its data_ref."""
    path = _chunks_dir() / data_ref
    if not path.exists():
        raise FileNotFoundError(f"Chunk not found: {data_ref}")
    return path.read_bytes()


def store_thumbnail(chunk_id: str, image_data: bytes) -> str:
    """
    Store a thumbnail image for a chunk.

    Returns:
        data_ref string e.g. "thumbnails/{chunk_id}.jpg"
    """
    dest = _thumbnails_dir() / f"{chunk_id}.jpg"
    dest.write_bytes(image_data)
    return f"thumbnails/{chunk_id}.jpg"


def read_thumbnail(chunk_id: str) -> bytes | None:
    path = _thumbnails_dir() / f"{chunk_id}.jpg"
    if not path.exists():
        return None
    return path.read_bytes()


def delete_file_chunks(content_hash: str) -> int:
    """Delete all stored chunks for a given content hash. Returns count deleted."""
    prefix = content_hash[:2]
    chunk_dir = _chunks_dir() / prefix
    deleted = 0
    if chunk_dir.exists():
        for f in chunk_dir.glob(f"{content_hash}_chunk*.bin"):
            f.unlink()
            deleted += 1
    return deleted


def chunk_exists(content_hash: str) -> bool:
    """Returns True if at least one chunk for this hash is already stored."""
    prefix = content_hash[:2]
    chunk_dir = _chunks_dir() / prefix
    if not chunk_dir.exists():
        return False
    return any(chunk_dir.glob(f"{content_hash}_chunk*.bin"))
