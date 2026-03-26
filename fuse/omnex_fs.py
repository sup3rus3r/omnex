"""
Omnex — FUSE Virtual Filesystem
Mounts the Omnex index as a real directory tree.

Structure:
  /omnex/
    documents/        — all indexed documents (read + write)
    images/           — all indexed images (read + write)
    audio/            — all indexed audio (read + write)
    video/            — all indexed video (read + write)
    code/             — all indexed code files (read + write)
    drop/             — write any file here → immediately ingested
    by_date/
      YYYY/
        MM/
          filename    — files organised by ingestion date (read-only)
    search/           — magic directory: name a file to run a query (read-only)

Write behaviour:
  - Writing a file to documents/, images/, audio/, video/, code/, or drop/
    → file is saved to a staging area, then submitted to the ingestion pipeline
  - Deleting a file from any type directory
    → chunk is removed from MongoDB and the vector index

Usage:
    python -m fuse.omnex_fs --mount /mnt/omnex
"""

from __future__ import annotations

import errno
import logging
import os
import stat
import sys
import tempfile
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

# Staging directory for incoming writes
_STAGE_DIR = Path(os.getenv("OMNEX_DATA_PATH", "/data")) / "fuse_stage"
_STAGE_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_ts() -> int:
    return int(time.time())


def _dir_stat(writable: bool = False) -> dict:
    t = _now_ts()
    mode = stat.S_IFDIR | (0o755 if writable else 0o555)
    return dict(
        st_mode=mode, st_nlink=2, st_size=0,
        st_atime=t, st_mtime=t, st_ctime=t,
        st_uid=os.getuid(), st_gid=os.getgid(),
    )


def _file_stat(size: int, mtime: int | None = None, writable: bool = False) -> dict:
    t = mtime or _now_ts()
    mode = stat.S_IFREG | (0o644 if writable else 0o444)
    return dict(
        st_mode=mode, st_nlink=1, st_size=size,
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


def _delete_chunks_for_source(source_path: str) -> int:
    """Remove all chunks for a source path from MongoDB and vector indexes."""
    db = _get_db()
    docs = list(db["chunks"].find({"source_path": source_path}, {"_id": 1, "data_ref": 1}))
    if not docs:
        return 0

    chunk_ids = [str(d["_id"]) for d in docs]

    # Remove from vector indexes
    try:
        from storage.leann_store import delete_vectors
        delete_vectors(chunk_ids)
    except Exception as e:
        log.warning(f"Vector delete failed: {e}")

    # Remove binary blobs
    for d in docs:
        if d.get("data_ref"):
            try:
                from storage.binary_store import delete_chunk
                delete_chunk(d["data_ref"])
            except Exception:
                pass

    # Remove from MongoDB
    db["chunks"].delete_many({"source_path": source_path})
    log.info(f"Deleted {len(docs)} chunks for {source_path}")
    return len(docs)


# ── Name helpers ──────────────────────────────────────────────────────────────

def _safe_name(path: str, chunk_id: str, chunk_index: int) -> str:
    base = Path(path).name or "file"
    if chunk_index > 0:
        stem = Path(base).stem
        ext  = Path(base).suffix
        base = f"{stem}_chunk{chunk_index}{ext}"
    base = base.replace("/", "_").replace("\x00", "")
    return base or f"chunk_{chunk_id[:8]}"


# ── Ingestion trigger ─────────────────────────────────────────────────────────

def _trigger_ingest(file_path: Path) -> None:
    """Submit a file to the Omnex ingestion pipeline via the API."""
    import threading

    def _run():
        try:
            import httpx
            api = os.getenv("OMNEX_API_URL", "http://omnex-api:8000")
            with open(file_path, "rb") as f:
                resp = httpx.post(
                    f"{api}/ingest/upload",
                    files={"file": (file_path.name, f)},
                    timeout=120,
                )
            if resp.status_code == 200:
                log.info(f"FUSE ingestion submitted: {file_path.name}")
            else:
                log.warning(f"FUSE ingestion failed ({resp.status_code}): {resp.text[:200]}")
        except Exception as e:
            log.error(f"FUSE ingestion error: {e}")

    threading.Thread(target=_run, daemon=True).start()


# ── FUSE implementation ───────────────────────────────────────────────────────

class OmnexFS(Operations):
    """
    Read-write FUSE filesystem over the Omnex index.
    - Read: serves indexed files from MongoDB + binary store
    - Write: buffers incoming files, triggers ingestion on close
    - Delete: removes chunks from MongoDB + vector index
    """

    _TOP = {"documents", "images", "audio", "video", "code", "drop", "by_date", "search"}
    _WRITABLE = {"documents", "images", "audio", "video", "code", "drop"}

    _TYPE_MAP = {
        "documents": "document",
        "images":    "image",
        "audio":     "audio",
        "video":     "video",
        "code":      "code",
    }

    def __init__(self):
        self._search_cache: dict[str, bytes] = {}
        # fh → (Path, bytearray) for in-progress writes
        self._write_buffers: dict[int, tuple[Path, bytearray]] = {}
        self._next_fh = 1

    # ── Filesystem metadata ────────────────────────────────────────────────────

    def getattr(self, path: str, fh=None) -> dict:
        parts = [p for p in path.split("/") if p]

        if not parts:
            return _dir_stat()

        top = parts[0]

        if len(parts) == 1:
            if top in self._TOP:
                return _dir_stat(writable=top in self._WRITABLE)
            raise FuseOSError(errno.ENOENT)

        if top in self._TYPE_MAP:
            if len(parts) == 2:
                chunks = _chunks_by_type(self._TYPE_MAP[top])
                for c in chunks:
                    name = _safe_name(c["source_path"], str(c["_id"]), c.get("chunk_index", 0))
                    if name == parts[1]:
                        size  = self._chunk_size(c)
                        mtime = int(c["created_at"].timestamp()) if c.get("created_at") else None
                        return _file_stat(size, mtime)
            raise FuseOSError(errno.ENOENT)

        if top == "drop":
            if len(parts) == 2:
                # Staged files appear here while being written
                stage = _STAGE_DIR / parts[1]
                if stage.exists():
                    return _file_stat(stage.stat().st_size, writable=True)
            raise FuseOSError(errno.ENOENT)

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

        if top == "search":
            if len(parts) == 2:
                return _file_stat(len(self._run_search(parts[1])))
            raise FuseOSError(errno.ENOENT)

        raise FuseOSError(errno.ENOENT)

    def readdir(self, path: str, fh) -> list[str]:
        parts = [p for p in path.split("/") if p]
        entries = [".", ".."]

        if not parts:
            return entries + list(self._TOP)

        top = parts[0]

        if top in self._TYPE_MAP and len(parts) == 1:
            seen: set[str] = set()
            for c in _chunks_by_type(self._TYPE_MAP[top]):
                name = _safe_name(c["source_path"], str(c["_id"]), c.get("chunk_index", 0))
                if name not in seen:
                    entries.append(name)
                    seen.add(name)
            return entries

        if top == "drop" and len(parts) == 1:
            return entries + [f.name for f in _STAGE_DIR.iterdir() if f.is_file()]

        if top == "by_date":
            if len(parts) == 1:
                years = set()
                for c in _all_chunks():
                    if c.get("created_at"):
                        years.add(str(c["created_at"].year))
                return entries + sorted(years)
            if len(parts) == 2:
                year = parts[1]
                months = set()
                for c in _all_chunks():
                    if c.get("created_at") and str(c["created_at"].year) == year:
                        months.add(f"{c['created_at'].month:02d}")
                return entries + sorted(months)
            if len(parts) == 3:
                seen = set()
                for c in self._chunks_for_date(parts[1], parts[2]):
                    name = _safe_name(c["source_path"], str(c["_id"]), c.get("chunk_index", 0))
                    if name not in seen:
                        entries.append(name)
                        seen.add(name)
                return entries

        if top == "search" and len(parts) == 1:
            return entries + list(self._search_cache.keys())

        return entries

    def access(self, path: str, mode: int) -> None:
        # Allow all access checks — permissions handled by stat mode bits
        return

    # ── Read ──────────────────────────────────────────────────────────────────

    def read(self, path: str, size: int, offset: int, fh) -> bytes:
        parts = [p for p in path.split("/") if p]
        if len(parts) < 2:
            raise FuseOSError(errno.EISDIR)

        top = parts[0]

        if top == "search":
            data = self._run_search(parts[1])
            return data[offset:offset + size]

        if top == "drop":
            stage = _STAGE_DIR / parts[1]
            if stage.exists():
                with open(stage, "rb") as f:
                    f.seek(offset)
                    return f.read(size)
            raise FuseOSError(errno.ENOENT)

        chunk = self._find_chunk(parts)
        if chunk is None:
            raise FuseOSError(errno.ENOENT)

        data = self._read_chunk_data(chunk)
        return data[offset:offset + size]

    # ── Write ─────────────────────────────────────────────────────────────────

    def create(self, path: str, mode: int, fi=None) -> int:
        parts = [p for p in path.split("/") if p]
        if not parts or parts[0] not in self._WRITABLE:
            raise FuseOSError(errno.EACCES)
        if len(parts) != 2:
            raise FuseOSError(errno.EACCES)

        filename = parts[1]
        stage_path = _STAGE_DIR / filename
        stage_path.touch()

        fh = self._next_fh
        self._next_fh += 1
        self._write_buffers[fh] = (stage_path, bytearray())
        return fh

    def write(self, path: str, data: bytes, offset: int, fh: int) -> int:
        if fh not in self._write_buffers:
            raise FuseOSError(errno.EBADF)
        stage_path, buf = self._write_buffers[fh]
        # Extend buffer if needed
        if offset + len(data) > len(buf):
            buf.extend(b"\x00" * (offset + len(data) - len(buf)))
        buf[offset:offset + len(data)] = data
        return len(data)

    def release(self, path: str, fh: int) -> int:
        if fh not in self._write_buffers:
            return 0
        stage_path, buf = self._write_buffers.pop(fh)
        # Flush buffer to disk
        with open(stage_path, "wb") as f:
            f.write(bytes(buf))
        log.info(f"FUSE write complete: {stage_path} ({len(buf)} bytes) — triggering ingestion")
        _trigger_ingest(stage_path)
        return 0

    def truncate(self, path: str, length: int, fh: int = None) -> None:
        # Called before write — just accept it
        if fh and fh in self._write_buffers:
            stage_path, buf = self._write_buffers[fh]
            del buf[length:]
        return

    def utimens(self, path: str, times=None) -> None:
        return  # Accept timestamp updates

    def chmod(self, path: str, mode: int) -> None:
        return  # Accept chmod calls

    def chown(self, path: str, uid: int, gid: int) -> None:
        return  # Accept chown calls

    def mkdir(self, path: str, mode: int) -> None:
        return  # Virtual dirs — always succeed

    def rmdir(self, path: str) -> None:
        return  # Virtual dirs — always succeed

    # ── Delete ────────────────────────────────────────────────────────────────

    def unlink(self, path: str) -> None:
        parts = [p for p in path.split("/") if p]
        if not parts:
            raise FuseOSError(errno.EACCES)

        top = parts[0]

        # Delete from drop/ staging area
        if top == "drop" and len(parts) == 2:
            stage = _STAGE_DIR / parts[1]
            if stage.exists():
                stage.unlink()
            return

        # Delete from type directories — remove from index
        if top in self._TYPE_MAP and len(parts) == 2:
            filename = parts[1]
            chunks = _chunks_by_type(self._TYPE_MAP[top])
            for c in chunks:
                name = _safe_name(c["source_path"], str(c["_id"]), c.get("chunk_index", 0))
                if name == filename:
                    _delete_chunks_for_source(c["source_path"])
                    return
            raise FuseOSError(errno.ENOENT)

        raise FuseOSError(errno.EACCES)

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
    log.info(f"Mounting Omnex at {mount} (read-write)")

    FUSE(
        OmnexFS(),
        mount,
        nothreads=True,
        foreground=args.foreground,
        allow_other=True,
        ro=False,
    )


if __name__ == "__main__":
    main()
