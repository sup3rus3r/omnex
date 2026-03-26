"""
Omnex — File Watcher (Phase 8)
Watches one or more directories for new/modified/deleted files and
keeps the Omnex index in sync automatically.

Usage (standalone):
    python -m ingestion.watcher --path /path/to/folder
    python -m ingestion.watcher --path C:\\Users\\MKhan --workers 4

Usage (embedded):
    from ingestion.watcher import start_watcher, stop_watcher
    watcher = start_watcher(Path("/some/dir"))
    ...
    stop_watcher(watcher)
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

if TYPE_CHECKING:
    from watchdog.observers.api import BaseObserver

logger = logging.getLogger("omnex.watcher")


# ── Event handler ─────────────────────────────────────────────────────────────

class OmnexEventHandler(FileSystemEventHandler):
    """
    Enqueues file-system events for background processing.
    Uses a debounce map to avoid re-indexing files that are still
    being written (e.g. large video files).
    """

    DEBOUNCE_SECONDS = 3.0  # wait this long after last event before processing

    def __init__(self, work_queue: queue.Queue) -> None:
        super().__init__()
        self._queue = work_queue
        self._pending: dict[str, float] = {}  # path → last-seen timestamp
        self._lock = threading.Lock()

    # watchdog callbacks ───────────────────────────────────────────────────────

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._touch(event.src_path, "upsert")

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._touch(event.src_path, "upsert")

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            # Old path no longer valid — delete it from index
            self._queue.put(("delete", event.src_path))
            # New path should be indexed
            self._touch(event.dest_path, "upsert")

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            with self._lock:
                self._pending.pop(event.src_path, None)
            self._queue.put(("delete", event.src_path))

    # debounce ─────────────────────────────────────────────────────────────────

    def _touch(self, path: str, action: str) -> None:
        with self._lock:
            self._pending[path] = time.monotonic()

    def flush_debounced(self) -> None:
        """Called by the debounce thread — promotes ready events to the queue."""
        now = time.monotonic()
        with self._lock:
            ready = [
                p for p, ts in self._pending.items()
                if now - ts >= self.DEBOUNCE_SECONDS
            ]
            for p in ready:
                del self._pending[p]
        for p in ready:
            self._queue.put(("upsert", p))


# ── Worker thread ─────────────────────────────────────────────────────────────

def _worker(work_queue: queue.Queue, workers: int, stop_event: threading.Event) -> None:
    """
    Drain the work queue and run ingestion / deletion on each event.
    Runs in its own thread; uses a small internal thread pool for heavy work.
    """
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as pool:
        while not stop_event.is_set():
            try:
                action, path_str = work_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            path = Path(path_str)

            if action == "upsert":
                if not path.exists():
                    work_queue.task_done()
                    continue
                pool.submit(_ingest_one, path)

            elif action == "delete":
                pool.submit(_delete_one, path_str)

            work_queue.task_done()


def _ingest_one(path: Path) -> None:
    from ingestion.detector import is_indexable
    from ingestion.__main__ import ingest_file

    if not is_indexable(path):
        return
    try:
        result = ingest_file(path)
        status = result.get("status", "?")
        if status == "indexed":
            logger.info(f"[watcher] indexed  {path}")
        elif status == "already_indexed":
            logger.debug(f"[watcher] unchanged {path}")
        elif status == "error":
            logger.warning(f"[watcher] error    {path} — {result.get('error', '')}")
    except Exception as e:
        logger.error(f"[watcher] failed   {path} — {e}")


def _delete_one(source_path: str) -> None:
    """Remove all chunks belonging to the deleted file from Mongo + LEANN."""
    try:
        from storage.mongo import get_db
        from storage.leann_store import IndexName, remove_vector

        db = get_db()
        chunks = list(db["chunks"].find({"source_path": source_path}, {"_id": 1, "leann_id": 1, "file_type": 1}))

        if not chunks:
            return

        # Map file_type → LEANN index
        _index_map = {
            "image":    IndexName.IMAGE,
            "audio":    IndexName.AUDIO,
            "video":    IndexName.VIDEO,
            "code":     IndexName.CODE,
            "text":     IndexName.TEXT,
            "document": IndexName.TEXT,
        }

        for chunk in chunks:
            leann_id = chunk.get("leann_id")
            file_type = chunk.get("file_type", "text")
            if leann_id is not None:
                index = _index_map.get(file_type, IndexName.TEXT)
                try:
                    remove_vector(index, leann_id)
                except Exception:
                    pass  # vector may already be gone

        chunk_ids = [c["_id"] for c in chunks]
        db["chunks"].delete_many({"_id": {"$in": chunk_ids}})
        logger.info(f"[watcher] deleted  {source_path} ({len(chunks)} chunks removed)")

    except Exception as e:
        logger.error(f"[watcher] delete failed {source_path} — {e}")


# ── Debounce thread ───────────────────────────────────────────────────────────

def _debounce_loop(handler: OmnexEventHandler, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        time.sleep(1.0)
        handler.flush_debounced()


# ── Public API ────────────────────────────────────────────────────────────────

class WatcherHandle:
    """Returned by start_watcher — call .stop() to shut down cleanly."""

    def __init__(
        self,
        observer: "BaseObserver",
        stop_event: threading.Event,
        worker_thread: threading.Thread,
        debounce_thread: threading.Thread,
    ) -> None:
        self._observer = observer
        self._stop_event = stop_event
        self._worker_thread = worker_thread
        self._debounce_thread = debounce_thread

    def stop(self) -> None:
        self._stop_event.set()
        self._observer.stop()
        self._observer.join()
        self._debounce_thread.join(timeout=5)
        self._worker_thread.join(timeout=10)
        logger.info("[watcher] stopped")

    def is_alive(self) -> bool:
        return self._observer.is_alive()


def start_watcher(path: Path, workers: int = 2) -> WatcherHandle:
    """
    Start watching *path* recursively.
    Returns a WatcherHandle — call .stop() to shut down.
    """
    work_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    event_handler = OmnexEventHandler(work_queue)

    observer = Observer()
    observer.schedule(event_handler, str(path), recursive=True)
    observer.start()

    debounce_thread = threading.Thread(
        target=_debounce_loop, args=(event_handler, stop_event),
        daemon=True, name="omnex-debounce",
    )
    debounce_thread.start()

    worker_thread = threading.Thread(
        target=_worker, args=(work_queue, workers, stop_event),
        daemon=True, name="omnex-worker",
    )
    worker_thread.start()

    logger.info(f"[watcher] watching {path}  (workers={workers})")
    return WatcherHandle(observer, stop_event, worker_thread, debounce_thread)


def stop_watcher(handle: WatcherHandle) -> None:
    handle.stop()


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import signal

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Omnex file watcher")
    parser.add_argument("--path", required=True, help="Directory to watch")
    parser.add_argument("--workers", type=int, default=2, help="Ingestion workers (default: 2)")
    args = parser.parse_args()

    watch_path = Path(args.path)
    if not watch_path.is_dir():
        print(f"Not a directory: {watch_path}")
        raise SystemExit(1)

    handle = start_watcher(watch_path, workers=args.workers)

    def _shutdown(sig, frame):
        logger.info("[watcher] shutting down…")
        handle.stop()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info(f"[watcher] running — Ctrl+C to stop")
    while handle.is_alive():
        time.sleep(1)
