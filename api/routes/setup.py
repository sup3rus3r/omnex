"""
Omnex — Setup Routes
GET  /setup/status    — Returns presence status of all required models
POST /setup/download  — Triggers model downloads, streams progress via SSE
"""

from __future__ import annotations

import json
import threading
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter()

# Track if a download is already running
_download_thread: threading.Thread | None = None
_download_lock = threading.Lock()


@router.get("/status")
async def setup_status():
    """Return current model presence status."""
    from models.manager import status_snapshot, all_ready
    return {
        "ready": all_ready(),
        "models": status_snapshot(),
    }


@router.post("/download")
async def start_download():
    """
    Start downloading all missing models in the background.
    Returns SSE stream of progress events.

    Event format: data: {"id": "minilm", "progress": 45.2, "status": "downloading"}
    """
    from models.manager import download_all, all_ready

    if all_ready():
        async def already_done():
            yield "data: " + json.dumps({"type": "complete"}) + "\n\n"
        return StreamingResponse(already_done(), media_type="text/event-stream")

    # Queue of progress events written by the download thread, read by SSE
    import queue
    progress_queue: queue.Queue = queue.Queue()
    done_event = threading.Event()

    def on_progress(model_id: str, percent: float, status: str):
        progress_queue.put({"id": model_id, "progress": round(percent, 1), "status": status})

    def run_downloads():
        try:
            download_all(on_progress=on_progress)
        except Exception as e:
            progress_queue.put({"type": "error", "message": str(e)})
        finally:
            progress_queue.put({"type": "complete"})
            done_event.set()

    global _download_thread
    with _download_lock:
        if _download_thread is None or not _download_thread.is_alive():
            _download_thread = threading.Thread(target=run_downloads, daemon=True, name="omnex-setup")
            _download_thread.start()

    async def stream():
        import asyncio
        while True:
            try:
                # Non-blocking check with async sleep to not block the event loop
                event = progress_queue.get_nowait()
                yield "data: " + json.dumps(event) + "\n\n"
                if event.get("type") in ("complete", "error"):
                    break
            except Exception:
                await asyncio.sleep(0.3)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/tunnel")
async def tunnel_status():
    """Return ngrok tunnel URL and auth status."""
    from api.tunnel import get_status
    return get_status()


@router.get("/config")
async def public_config():
    """Return public configuration the UI needs to know about."""
    from api.auth import api_key_enabled
    return {
        "auth_enabled": api_key_enabled(),
    }


@router.get("/fuse")
async def fuse_status():
    """Return FUSE mount status and path."""
    import os
    from pathlib import Path
    mount = os.getenv("OMNEX_FUSE_MOUNT", "./omnex-mount")
    mount_path = Path(mount)
    # Check if mount is active by looking for known subdirs
    is_mounted = (mount_path / "documents").exists() or (mount_path / "images").exists()
    return {
        "mount_path": str(mount_path.resolve()),
        "mounted":    is_mounted,
        "dirs":       ["documents", "images", "audio", "video", "code", "by_date", "search"],
    }
