"""
Omnex — FUSE Virtual Filesystem
Mounts the Omnex index as a real directory tree.

Structure:
  /omnex/
    documents/        — all indexed documents
    images/           — all indexed images
    audio/            — all indexed audio
    video/            — all indexed video
    code/             — all indexed code files
    by_date/
      YYYY/
        MM/
          filename    — files organised by ingestion date
    search/           — magic directory: name a file to query
      what is X       — reading this file returns search results as text

Usage:
    python -m fuse.omnex_fs --mount /mnt/omnex
"""

from __future__ import annotations

import errno
import logging
import os
import stat
import sys
import time
from pathlib import Path

import importlib.util as _ilu, site as _site, os as _os
# fusepy installs as fuse.py in site-packages; our local fuse/ package shadows it.
# Load fusepy directly from site-packages by file path.
_fusepy = None
for _sp in _site.getsitepackages():
    _candidate = _os.path.join(_sp, "fuse.py")
    if _os.path.exists(_candidate):
        _spec = _ilu.spec_from_file_location("_fusepy_lib", _candidate)
        _fusepy = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_fusepy)
        break
if _fusepy is None:
    raise ImportError("fusepy (fuse.py) not found in site-packages")
FUSE = _fusepy.FUSE
FuseOSError = _fusepy.FuseOSError
Operations = _fusepy.Operations

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger("omnex.fuse")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_ts() -> int:
    return int(time.time())


def _dir_stat() -> dict:
    t = _now_ts()
    return dict(
        st_mode=stat.S_IFDIR | 0o555,
        st_nlink=2,
        st_size=0,
        st_atime=t, st_mtime=t, st_ctime=t,
        st_uid=os.getuid(), st_gid=os.getgid(),
    )


def _file_stat(size: int, mtime: int | None = None) -> dict:
    t = mtime or _now_ts()
    return dict(
        st_mode=stat.S_IFREG | 0o444,
        st_nlink=1,
        st_size=size,
        st_atime=t, st_mtime=t, st_ctime=t,
        st_uid=os.getuid(), st_gid=os.getgid(),
    )


# ── MongoDB helpers ───────────────────────────────────────────────────────────

def _get_db():
    from pymongo import MongoClient
    uri  = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    name = os.getenv("MONGO_DB", "omnex")
    return MongoClient(uri, serverSelectionTimeoutMS=3000)[name]


def _chunks_by_type(file_type: str) -> list[dict]:
    db = _get_db()
    return list(db["chunks"].find(
        {"file_type": file_type},
        {"_id": 1, "source_path": 1, "data_ref": 1, "text_content": 1,
         "mime_type": 1, "created_at": 1, "chunk_index": 1}
    ).limit(2000))


def _chunk_by_id(chunk_id: str) -> dict | None:
    from bson import ObjectId
    db = _get_db()
    return db["chunks"].find_one({"_id": ObjectId(chunk_id)})


def _all_chunks() -> list[dict]:
    db = _get_db()
    return list(db["chunks"].find(
        {},
        {"_id": 1, "source_path": 1, "file_type": 1, "created_at": 1,
         "data_ref": 1, "text_content": 1, "chunk_index": 1}
    ).limit(5000))


# ── Name helpers ──────────────────────────────────────────────────────────────

def _safe_name(path: str, chunk_id: str, chunk_index: int) -> str:
    """Derive a unique filename from source path."""
    base = Path(path).name or "file"
    # Make unique if multiple chunks from same file
    if chunk_index > 0:
        stem = Path(base).stem
        ext  = Path(base).suffix
        base = f"{stem}_chunk{chunk_index}{ext}"
    # Sanitise
    base = base.replace("/", "_").replace("\x00", "")
    return base or f"chunk_{chunk_id[:8]}"


# ── FUSE implementation ───────────────────────────────────────────────────────

class OmnexFS(Operations):
    """
    Read-only FUSE filesystem over the Omnex index.
    All reads hit MongoDB + binary store on demand — no in-memory cache needed
    for correctness (FUSE caches at the kernel level).
    """

    # Top-level directories
    _TOP = {"documents", "images", "audio", "video", "code", "by_date", "search"}

    # Map top-level dir → MongoDB file_type
    _TYPE_MAP = {
        "documents": "document",
        "images":    "image",
        "audio":     "audio",
        "video":     "video",
        "code":      "code",
    }

    def __init__(self):
        self._search_cache: dict[str, bytes] = {}  # query → result bytes

    # ── Filesystem metadata ────────────────────────────────────────────────────

    def getattr(self, path: str, fh=None) -> dict:
        parts = [p for p in path.split("/") if p]

        # Root
        if not parts:
            return _dir_stat()

        top = parts[0]

        # Top-level dirs
        if len(parts) == 1:
            if top in self._TOP:
                return _dir_stat()
            raise FuseOSError(errno.ENOENT)

        # Type directories (documents/, images/, etc.)
        if top in self._TYPE_MAP:
            if len(parts) == 2:
                # File in type dir — look up by name
                chunks = _chunks_by_type(self._TYPE_MAP[top])
                for c in chunks:
                    name = _safe_name(c["source_path"], str(c["_id"]), c.get("chunk_index", 0))
                    if name == parts[1]:
                        size = self._chunk_size(c)
                        mtime = int(c["created_at"].timestamp()) if c.get("created_at") else None
                        return _file_stat(size, mtime)
            raise FuseOSError(errno.ENOENT)

        # by_date/YYYY/MM/filename
        if top == "by_date":
            if len(parts) <= 3:
                return _dir_stat()
            if len(parts) == 4:
                chunks = self._chunks_for_date(parts[1], parts[2])
                for c in chunks:
                    name = _safe_name(c["source_path"], str(c["_id"]), c.get("chunk_index", 0))
                    if name == parts[3]:
                        return _file_stat(self._chunk_size(c))
            raise FuseOSError(errno.ENOENT)

        # search/<query>
        if top == "search":
            if len(parts) == 2:
                return _file_stat(len(self._run_search(parts[1])))
            raise FuseOSError(errno.ENOENT)

        raise FuseOSError(errno.ENOENT)

    def readdir(self, path: str, fh) -> list[str]:
        parts = [p for p in path.split("/") if p]
        entries = [".", ".."]

        # Root
        if not parts:
            return entries + list(self._TOP)

        top = parts[0]

        # Type dir listing
        if top in self._TYPE_MAP and len(parts) == 1:
            seen: set[str] = set()
            for c in _chunks_by_type(self._TYPE_MAP[top]):
                name = _safe_name(c["source_path"], str(c["_id"]), c.get("chunk_index", 0))
                if name not in seen:
                    entries.append(name)
                    seen.add(name)
            return entries

        # by_date/
        if top == "by_date":
            if len(parts) == 1:
                # List years
                years = set()
                for c in _all_chunks():
                    if c.get("created_at"):
                        years.add(str(c["created_at"].year))
                return entries + sorted(years)

            if len(parts) == 2:
                # List months for year
                year = parts[1]
                months = set()
                for c in _all_chunks():
                    if c.get("created_at") and str(c["created_at"].year) == year:
                        months.add(f"{c['created_at'].month:02d}")
                return entries + sorted(months)

            if len(parts) == 3:
                # List files for year/month
                seen = set()
                for c in self._chunks_for_date(parts[1], parts[2]):
                    name = _safe_name(c["source_path"], str(c["_id"]), c.get("chunk_index", 0))
                    if name not in seen:
                        entries.append(name)
                        seen.add(name)
                return entries

        # search/ — show cached queries as files
        if top == "search" and len(parts) == 1:
            return entries + list(self._search_cache.keys())

        return entries

    # ── File reads ─────────────────────────────────────────────────────────────

    def read(self, path: str, size: int, offset: int, fh) -> bytes:
        parts = [p for p in path.split("/") if p]
        if len(parts) < 2:
            raise FuseOSError(errno.EISDIR)

        top = parts[0]

        # search/<query> — run search and return text
        if top == "search":
            data = self._run_search(parts[1])
            return data[offset:offset + size]

        # Type or by_date file — return binary or text
        chunk = self._find_chunk(parts)
        if chunk is None:
            raise FuseOSError(errno.ENOENT)

        data = self._read_chunk_data(chunk)
        return data[offset:offset + size]

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _chunk_size(self, c: dict) -> int:
        if c.get("data_ref"):
            try:
                from storage.binary_store import chunk_size
                return chunk_size(c["data_ref"])
            except Exception:
                pass
        if c.get("text_content"):
            return len(c["text_content"].encode())
        return 0

    def _read_chunk_data(self, c: dict) -> bytes:
        if c.get("data_ref"):
            try:
                from storage.binary_store import read_chunk
                return read_chunk(c["data_ref"])
            except Exception:
                pass
        if c.get("text_content"):
            return c["text_content"].encode()
        return b""

    def _find_chunk(self, parts: list[str]) -> dict | None:
        top = parts[0]
        filename = parts[-1]

        if top in self._TYPE_MAP:
            for c in _chunks_by_type(self._TYPE_MAP[top]):
                name = _safe_name(c["source_path"], str(c["_id"]), c.get("chunk_index", 0))
                if name == filename:
                    return c

        if top == "by_date" and len(parts) == 4:
            for c in self._chunks_for_date(parts[1], parts[2]):
                name = _safe_name(c["source_path"], str(c["_id"]), c.get("chunk_index", 0))
                if name == filename:
                    return c

        return None

    def _chunks_for_date(self, year: str, month: str) -> list[dict]:
        db = _get_db()
        from datetime import datetime, timezone
        try:
            y, m = int(year), int(month)
            start = datetime(y, m, 1, tzinfo=timezone.utc)
            end   = datetime(y, m + 1, 1, tzinfo=timezone.utc) if m < 12 else datetime(y + 1, 1, 1, tzinfo=timezone.utc)
        except ValueError:
            return []
        return list(db["chunks"].find(
            {"created_at": {"$gte": start, "$lt": end}},
            {"_id": 1, "source_path": 1, "data_ref": 1, "text_content": 1,
             "created_at": 1, "chunk_index": 1}
        ).limit(1000))

    def _run_search(self, query: str) -> bytes:
        if query in self._search_cache:
            return self._search_cache[query]

        try:
            import asyncio
            from api.query_engine import search
            response = asyncio.run(search(query=query, top_k=10))
            lines = [f"Omnex search: {query}", f"Results: {response.total}", ""]
            for i, r in enumerate(response.results, 1):
                lines.append(f"[{i}] {r.file_type.upper()} — {r.source_path}")
                if r.text:
                    lines.append(f"    {r.text[:300]}")
                lines.append("")
            if response.llm_response:
                lines += ["---", response.llm_response]
            data = "\n".join(lines).encode()
        except Exception as e:
            data = f"Search error: {e}\n".encode()

        self._search_cache[query] = data
        return data


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Omnex FUSE filesystem")
    parser.add_argument("--mount", default="/mnt/omnex", help="Mount point")
    parser.add_argument("--foreground", action="store_true", default=True)
    args = parser.parse_args()

    mount = args.mount
    Path(mount).mkdir(parents=True, exist_ok=True)
    log.info(f"Mounting Omnex at {mount}")

    FUSE(
        OmnexFS(),
        mount,
        nothreads=True,
        foreground=args.foreground,
        allow_other=True,
        ro=True,
    )


if __name__ == "__main__":
    main()
