"""
Omnex — Ingestion CLI
Entry point for the ingestion pipeline.

Usage:
    python -m ingestion --path /path/to/folder
    python -m ingestion --path /path/to/file.pdf
    python -m ingestion --path D:\\               # Full drive
    python -m ingestion --path /mnt/drive --workers 8
    python -m ingestion --status                  # Show ingestion progress
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from ingestion.detector import FileType, detect, is_indexable
from ingestion.hasher import hash_file
from ingestion.router import route
from ingestion.chunker import chunk_text
from embeddings.text import embed_batch
from storage.mongo import (
    build_chunk_doc,
    chunk_exists,
    insert_chunk,
    update_chunk_leann_id,
    upsert_ingestion_state,
    get_ingestion_state,
)
from storage.leann_store import IndexName, add_vector
from storage.binary_store import store_file, chunk_exists as binary_exists

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("omnex.ingest")


def collect_files(path: Path) -> list[Path]:
    """Collect all indexable files under a path (file, folder, or drive root)."""
    if path.is_file():
        return [path] if is_indexable(path) else []
    files = []
    for f in path.rglob("*"):
        if f.is_file() and is_indexable(f):
            files.append(f)
    return files


def ingest_file(path: Path) -> dict:
    """
    Full ingestion pipeline for a single file.
    Returns a status dict.
    """
    result = {"path": str(path), "status": "skipped", "chunks": 0}

    try:
        content_hash = hash_file(path)

        # Skip if already indexed
        if chunk_exists(content_hash):
            result["status"] = "already_indexed"
            return result

        text, file_type, mime = route(path)

        # Binary files — store raw, no text embedding yet
        if file_type in (FileType.IMAGE, FileType.VIDEO, FileType.AUDIO):
            refs = store_file(path, content_hash)
            for i, ref in enumerate(refs):
                doc = build_chunk_doc(
                    source_path=str(path),
                    source_hash=content_hash,
                    chunk_index=i,
                    chunk_total=len(refs),
                    file_type=file_type.value,
                    mime_type=mime,
                    text_content=None,
                    data_ref=ref,
                    tags=_auto_tags(path, file_type),
                    metadata=_extract_metadata(path),
                    embedding_model="pending",
                )
                insert_chunk(doc)
            result["status"] = "stored_binary"
            result["chunks"] = len(refs)
            return result

        # Text/code/document — chunk and embed
        if not text:
            result["status"] = "no_text"
            return result

        chunks = chunk_text(text, file_type, source_path=path)
        if not chunks:
            result["status"] = "no_chunks"
            return result

        # Batch embed all chunks
        texts = [c.text for c in chunks]
        embeddings = embed_batch(texts)

        index_name = IndexName.CODE if file_type == FileType.CODE else IndexName.TEXT

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            doc = build_chunk_doc(
                source_path=str(path),
                source_hash=content_hash,
                chunk_index=chunk.chunk_index,
                chunk_total=chunk.chunk_total,
                file_type=file_type.value,
                mime_type=mime,
                text_content=chunk.text,
                data_ref=None,
                tags=_auto_tags(path, file_type),
                metadata=_extract_metadata(path),
                embedding_model="all-MiniLM-L6-v2",
            )
            chunk_id = insert_chunk(doc)
            leann_id = add_vector(
                index_name,
                chunk_id,
                embedding.tolist(),
                metadata={"file_type": file_type.value, "source_path": str(path)},
            )
            update_chunk_leann_id(chunk_id, leann_id)

        result["status"] = "indexed"
        result["chunks"] = len(chunks)

    except Exception as e:
        logger.error(f"Failed to ingest {path}: {e}")
        result["status"] = "error"
        result["error"] = str(e)

    return result


def run(source_path: Path, workers: int = 4) -> None:
    logger.info(f"Omnex ingestion starting — source: {source_path}")

    files = collect_files(source_path)
    total = len(files)

    if total == 0:
        logger.info("No indexable files found.")
        return

    logger.info(f"Found {total:,} files to process")

    upsert_ingestion_state(str(source_path), {
        "source_path": str(source_path),
        "total_files": total,
        "processed": 0,
        "indexed": 0,
        "skipped": 0,
        "errors": 0,
        "started_at": datetime.now(timezone.utc),
        "status": "running",
    })

    processed = indexed = skipped = errors = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(ingest_file, f): f for f in files}

        for future in as_completed(futures):
            res = future.result()
            processed += 1

            if res["status"] == "indexed":
                indexed += 1
            elif res["status"] in ("skipped", "already_indexed", "no_text", "no_chunks", "stored_binary"):
                skipped += 1
            elif res["status"] == "error":
                errors += 1
                logger.warning(f"Error: {res['path']} — {res.get('error', '')}")

            if processed % 100 == 0 or processed == total:
                pct = (processed / total) * 100
                logger.info(
                    f"Remembering {processed:,}/{total:,} ({pct:.1f}%) — "
                    f"indexed: {indexed:,}  skipped: {skipped:,}  errors: {errors}"
                )
                upsert_ingestion_state(str(source_path), {
                    "processed": processed,
                    "indexed": indexed,
                    "skipped": skipped,
                    "errors": errors,
                })

    upsert_ingestion_state(str(source_path), {
        "status": "complete",
        "completed_at": datetime.now(timezone.utc),
    })

    logger.info(
        f"\nIngestion complete.\n"
        f"  Total files : {total:,}\n"
        f"  Indexed     : {indexed:,}\n"
        f"  Skipped     : {skipped:,}\n"
        f"  Errors      : {errors}\n"
    )


def _auto_tags(path: Path, file_type: FileType) -> list[str]:
    tags = [file_type.value]
    suffix = path.suffix.lower().lstrip(".")
    if suffix:
        tags.append(suffix)
    return tags


def _extract_metadata(path: Path) -> dict:
    try:
        stat = path.stat()
        return {
            "created_at": datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
            "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            "size_bytes": stat.st_size,
        }
    except Exception:
        return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Omnex ingestion pipeline")
    parser.add_argument("--path", type=str, help="File, folder, or drive to ingest")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default: 4)")
    parser.add_argument("--status", action="store_true", help="Show ingestion status for a path")
    args = parser.parse_args()

    if args.status:
        if not args.path:
            print("--path required with --status")
            sys.exit(1)
        state = get_ingestion_state(args.path)
        if state:
            state.pop("_id", None)
            for k, v in state.items():
                print(f"  {k}: {v}")
        else:
            print("No ingestion record found for that path.")
        sys.exit(0)

    if not args.path:
        parser.print_help()
        sys.exit(1)

    source = Path(args.path)
    if not source.exists():
        print(f"Path does not exist: {source}")
        sys.exit(1)

    run(source, workers=args.workers)
