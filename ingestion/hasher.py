"""
Omnex — Content Hasher
Generates content hashes for deduplication.
Uses xxhash (fast, non-cryptographic) for change detection.
Files with identical hashes are never re-processed.
"""

from pathlib import Path

import xxhash

CHUNK_SIZE = 65536  # 64KB read buffer


def hash_file(path: Path) -> str:
    """
    Compute xxHash3-128 of file contents.
    Fast enough for large files — reads in 64KB chunks.

    Returns:
        Hex digest string (32 chars)
    """
    h = xxhash.xxh3_128()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


def hash_text(text: str) -> str:
    """Hash a string — used for chunk-level deduplication."""
    return xxhash.xxh3_128(text.encode("utf-8")).hexdigest()


def hash_bytes(data: bytes) -> str:
    """Hash raw bytes."""
    return xxhash.xxh3_128(data).hexdigest()
