"""
Omnex — Ingestion Routes
POST /ingest/trigger  — Trigger ingestion for a server-side path
POST /ingest/upload   — Upload files directly from the browser, then ingest
GET  /ingest/status   — Current ingestion progress
"""

from __future__ import annotations

import tempfile
import shutil
import threading
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi import Form
from typing import List
from pydantic import BaseModel

from api.auth import require_api_key

router = APIRouter()

# Track active ingestion so it can be cancelled
_active_cancel = threading.Event()
_active_path: str | None = None


class IngestRequest(BaseModel):
    path:    str
    workers: int = 4


@router.post("/upload", dependencies=[Depends(require_api_key)])
async def upload_and_ingest(
    files: List[UploadFile] = File(...),
    workers: int = Form(default=4),
):
    """
    Accept file uploads from the browser and ingest them.
    Files are saved to a temp directory, ingested, then cleaned up.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="omnex_upload_"))
    saved: list[Path] = []

    try:
        for upload in files:
            rel = upload.filename or "file"
            dest = tmp_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as f:
                shutil.copyfileobj(upload.file, f)
            saved.append(dest)
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    ingest_path = saved[0] if len(saved) == 1 else tmp_dir

    # Use threading.Thread directly — BackgroundTasks silently drops exceptions
    t = threading.Thread(
        target=_run_ingestion_and_cleanup,
        args=(str(ingest_path), workers, tmp_dir),
        daemon=True,
        name="omnex-ingest-upload",
    )
    t.start()

    return {
        "status":  "started",
        "path":    str(ingest_path),
        "files":   len(saved),
        "workers": workers,
    }


@router.post("/trigger", dependencies=[Depends(require_api_key)])
async def trigger_ingest(req: IngestRequest):
    """
    Trigger ingestion for a file, folder, or drive path.
    Runs in the background — returns immediately.
    Poll /ingest/status for progress.
    """
    source = Path(req.path)
    if not source.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {req.path}")

    threading.Thread(
        target=_run_ingestion,
        args=(req.path, req.workers),
        daemon=True,
        name="omnex-ingest-trigger",
    ).start()

    return {
        "status":  "started",
        "path":    req.path,
        "workers": req.workers,
        "message": f"Ingestion started. Poll /ingest/status?path={req.path} for progress.",
    }


@router.get("/status")
async def ingest_status(path: str | None = None):
    """
    Return ingestion progress.
    If path is provided, return status for that specific path.
    Otherwise return all active/completed ingestion records.
    Includes eta_seconds and files_per_minute for running jobs.
    """
    from storage.mongo import get_db
    from datetime import datetime, timezone
    db = get_db()

    query = {"source_path": path} if path else {}
    records = list(db["ingestion_state"].find(query, {"_id": 0}))

    now = datetime.now(timezone.utc)

    for r in records:
        # Compute ETA for running jobs
        if r.get("status") == "running":
            started = r.get("started_at")
            processed = r.get("processed", 0)
            total = r.get("total_files", 0)
            if started and processed > 0 and total > processed:
                elapsed = (now - started).total_seconds()
                rate = processed / elapsed          # files/sec
                remaining = total - processed
                r["eta_seconds"]      = round(remaining / rate)
                r["files_per_minute"] = round(rate * 60, 1)
            else:
                r["eta_seconds"]      = None
                r["files_per_minute"] = None

        # Convert datetime objects to ISO strings for JSON serialisation
        for k, v in r.items():
            if hasattr(v, "isoformat"):
                r[k] = v.isoformat()

    if path:
        if not records:
            return {"status": "not_started", "path": path, "total_files": 0, "processed": 0, "indexed": 0, "skipped": 0, "errors": 0}
        return records[0]

    return {"ingestion": records}


@router.delete("/source", dependencies=[Depends(require_api_key)])
async def delete_source(source_path: str):
    """
    Remove all indexed data for a given source path.
    Deletes chunks from MongoDB, vector indexes, and binary store.
    """
    from storage.mongo import get_db
    from storage.leann_store import delete_vectors

    db = get_db()
    docs = list(db["chunks"].find({"source_path": source_path}, {"_id": 1, "data_ref": 1}))
    if not docs:
        raise HTTPException(status_code=404, detail="No indexed data found for that path")

    chunk_ids = [str(d["_id"]) for d in docs]

    # Remove from vector indexes
    try:
        delete_vectors(chunk_ids)
    except Exception as e:
        pass  # Log but don't fail

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
    db["ingestion_state"].delete_one({"source_path": source_path})

    return {"deleted": len(docs), "source_path": source_path}


@router.delete("/chunk/{chunk_id}", dependencies=[Depends(require_api_key)])
async def delete_chunk_by_id(chunk_id: str):
    """Remove a single indexed chunk by ID."""
    from storage.mongo import get_db
    from bson import ObjectId

    db = get_db()
    try:
        oid = ObjectId(chunk_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid chunk ID")

    doc = db["chunks"].find_one({"_id": oid})
    if not doc:
        raise HTTPException(status_code=404, detail="Chunk not found")

    # Remove from vector index
    try:
        from storage.leann_store import delete_vectors
        delete_vectors([chunk_id])
    except Exception:
        pass

    # Remove binary blob
    if doc.get("data_ref"):
        try:
            from storage.binary_store import delete_chunk
            delete_chunk(doc["data_ref"])
        except Exception:
            pass

    db["chunks"].delete_one({"_id": oid})
    return {"deleted": 1, "chunk_id": chunk_id}


@router.post("/cancel")
async def cancel_ingest():
    """Signal the active ingestion run to stop after its current file."""
    global _active_path
    if _active_cancel.is_set() or _active_path is None:
        return {"status": "no_active_ingestion"}
    _active_cancel.set()
    return {"status": "cancel_requested", "path": _active_path}


def _run_ingestion(path: str, workers: int) -> None:
    """Run the ingestion pipeline in a background thread."""
    import logging, traceback
    log = logging.getLogger("omnex.ingest.bg")
    global _active_path
    _active_cancel.clear()
    _active_path = path
    try:
        from ingestion.__main__ import run
        run(Path(path), workers=workers, cancel_event=_active_cancel)
    except Exception:
        log.error(f"Ingestion failed for {path}:\n{traceback.format_exc()}")
    finally:
        _active_path = None


def _run_ingestion_and_cleanup(path: str, workers: int, tmp_dir: Path) -> None:
    """Run ingestion on uploaded files, then remove the temp directory."""
    import logging, traceback, sys
    log = logging.getLogger("omnex.ingest.bg")
    global _active_path
    _active_cancel.clear()
    _active_path = path
    print(f"[ingest] background thread started for {path}", flush=True, file=sys.stderr)
    try:
        from ingestion.__main__ import run
        print(f"[ingest] calling run()", flush=True, file=sys.stderr)
        run(Path(path), workers=workers, cancel_event=_active_cancel)
        print(f"[ingest] run() completed", flush=True, file=sys.stderr)
    except Exception:
        msg = traceback.format_exc()
        print(f"[ingest] FAILED:\n{msg}", flush=True, file=sys.stderr)
        log.error(f"Ingestion failed for {path}:\n{msg}")
    finally:
        _active_path = None
        shutil.rmtree(tmp_dir, ignore_errors=True)
