"""
Omnex — Chunker
Splits extracted text into indexable chunks.
Strategy varies by file type to preserve semantic boundaries.

Chunk types:
- Text/Documents : semantic chunking — sentence/paragraph boundaries
- Code           : AST-aware (function/class level), line fallback
- Binary         : not handled here — binary_store.py manages raw splits
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ingestion.detector import FileType

# ── Config ────────────────────────────────────────────────────────────────────

TEXT_CHUNK_SIZE    = 512   # tokens (approximate — we use chars * 0.75)
TEXT_CHUNK_OVERLAP = 64
CHARS_PER_TOKEN    = 4     # rough approximation


@dataclass
class Chunk:
    text: str
    chunk_index: int
    chunk_total: int        # filled in after all chunks are created
    metadata: dict = field(default_factory=dict)


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_text(text: str, file_type: FileType, source_path: Path | None = None) -> list[Chunk]:
    """
    Split text into chunks appropriate for the given file type.

    Args:
        text        : extracted text content
        file_type   : used to select chunking strategy
        source_path : optional, used for code AST chunking

    Returns:
        List of Chunk objects with index set. chunk_total is set on all items.
    """
    if not text or not text.strip():
        return []

    if file_type == FileType.CODE:
        raw_chunks = _chunk_code(text, source_path)
    else:
        raw_chunks = _chunk_semantic(text)

    chunks = [
        Chunk(text=c, chunk_index=i, chunk_total=len(raw_chunks))
        for i, c in enumerate(raw_chunks)
    ]
    return chunks


# ── Semantic chunker (documents, audio transcripts, etc.) ─────────────────────

def _chunk_semantic(text: str) -> list[str]:
    """
    Chunk text at sentence/paragraph boundaries.
    Target: ~512 tokens per chunk with 64-token overlap.
    """
    max_chars    = TEXT_CHUNK_SIZE * CHARS_PER_TOKEN
    overlap_chars = TEXT_CHUNK_OVERLAP * CHARS_PER_TOKEN

    # Split into sentences — handles . ! ? followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        s_len = len(sentence)

        if current_len + s_len > max_chars and current:
            chunks.append(" ".join(current))
            # Overlap: keep last N chars worth of sentences
            overlap: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) > overlap_chars:
                    break
                overlap.insert(0, s)
                overlap_len += len(s)
            current = overlap
            current_len = overlap_len

        current.append(sentence)
        current_len += s_len

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c.strip()]


# ── Code chunker ──────────────────────────────────────────────────────────────

def _chunk_code(text: str, source_path: Path | None = None) -> list[str]:
    """
    Chunk code at function/class boundaries using regex heuristics.
    Falls back to line-based chunking if no boundaries detected.

    Supports: Python, JS/TS, Go, Java, Rust, C/C++, C#
    """
    suffix = source_path.suffix.lower() if source_path else ""

    if suffix == ".py":
        chunks = _chunk_python(text)
    elif suffix in {".js", ".ts", ".jsx", ".tsx"}:
        chunks = _chunk_js_ts(text)
    elif suffix == ".go":
        chunks = _chunk_go(text)
    else:
        chunks = _chunk_generic_code(text)

    if not chunks:
        chunks = _chunk_lines(text)

    return chunks


def _chunk_python(text: str) -> list[str]:
    """Split Python at top-level def/class boundaries."""
    pattern = re.compile(r'(?=^(?:def |class |async def )\S)', re.MULTILINE)
    return _split_by_pattern(text, pattern)


def _chunk_js_ts(text: str) -> list[str]:
    """Split JS/TS at function/class/arrow function boundaries."""
    pattern = re.compile(
        r'(?=^(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function|class)\s)',
        re.MULTILINE
    )
    return _split_by_pattern(text, pattern)


def _chunk_go(text: str) -> list[str]:
    """Split Go at func boundaries."""
    pattern = re.compile(r'(?=^func\s)', re.MULTILINE)
    return _split_by_pattern(text, pattern)


def _chunk_generic_code(text: str) -> list[str]:
    """Generic: split at blank lines between blocks."""
    blocks = re.split(r'\n{2,}', text)
    return _merge_small_blocks(blocks)


def _split_by_pattern(text: str, pattern: re.Pattern) -> list[str]:
    """Split text at regex match positions and merge small fragments."""
    positions = [m.start() for m in pattern.finditer(text)]
    if not positions:
        return []
    if positions[0] != 0:
        positions.insert(0, 0)
    parts = [text[positions[i]:positions[i+1]] for i in range(len(positions) - 1)]
    parts.append(text[positions[-1]:])
    return _merge_small_blocks([p.strip() for p in parts if p.strip()])


def _merge_small_blocks(blocks: list[str], min_chars: int = 100) -> list[str]:
    """Merge very small blocks with the next block to avoid tiny chunks."""
    merged: list[str] = []
    buffer = ""
    for block in blocks:
        if len(buffer) < min_chars:
            buffer = (buffer + "\n\n" + block).strip() if buffer else block
        else:
            merged.append(buffer)
            buffer = block
    if buffer:
        merged.append(buffer)
    return merged


def _chunk_lines(text: str, lines_per_chunk: int = 80, overlap_lines: int = 10) -> list[str]:
    """Line-based fallback chunker."""
    lines = text.splitlines()
    chunks: list[str] = []
    i = 0
    while i < len(lines):
        chunk_lines = lines[i:i + lines_per_chunk]
        chunks.append("\n".join(chunk_lines))
        i += lines_per_chunk - overlap_lines
    return chunks
