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


def _ingest_file_safe(path: Path) -> dict:
    import sys, traceback
    try:
        return ingest_file(path)
    except Exception:
        msg = traceback.format_exc()
        print(f"[ingest_file] CRASH on {path}:\n{msg}", flush=True, file=sys.stderr)
        return {"path": str(path), "status": "error", "error": msg[:200]}


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

        # Images — CLIP embed + EXIF + thumbnail + store binary
        if file_type == FileType.IMAGE:
            from ingestion.processors.image import process as process_image
            from storage.binary_store import store_thumbnail
            img_result = process_image(path)
            if img_result is None:
                result["status"] = "error"
                return result

            refs = store_file(path, content_hash)
            file_meta = {**_extract_metadata(path), **img_result.metadata}
            doc = build_chunk_doc(
                source_path=str(path),
                source_hash=content_hash,
                chunk_index=0,
                chunk_total=1,
                file_type=file_type.value,
                mime_type=mime,
                text_content=None,
                data_ref=refs[0] if refs else None,
                tags=_auto_tags(path, file_type, metadata=file_meta, image_embedding=img_result.embedding),
                metadata=file_meta,
                embedding_model="clip-vit-base-patch32",
            )
            chunk_id = insert_chunk(doc)
            store_thumbnail(chunk_id, img_result.thumbnail)
            leann_id = add_vector(
                IndexName.IMAGE,
                chunk_id,
                img_result.embedding,
                metadata={"file_type": "image", "source_path": str(path)},
            )
            update_chunk_leann_id(chunk_id, leann_id)

            # Store face crops and embeddings for Phase 4 clustering
            if img_result.detected_faces:
                _store_face_data(chunk_id, img_result.detected_faces)

            result["status"] = "indexed"
            result["chunks"] = 1
            return result

        # Audio — Whisper transcription → MiniLM embed → LEANN AUDIO index
        if file_type == FileType.AUDIO:
            from ingestion.processors.audio import process as process_audio
            audio_result = process_audio(path)
            if audio_result is None:
                result["status"] = "error"
                return result

            refs = store_file(path, content_hash)
            base_meta = {**_extract_metadata(path), **audio_result.metadata}
            segments = audio_result.segments
            if not segments:
                result["status"] = "no_text"
                return result

            texts = [s["text"] for s in segments]
            embeddings = embed_batch(texts)

            for i, (seg, embedding) in enumerate(zip(segments, embeddings)):
                seg_meta = {
                    **base_meta,
                    "start_seconds": seg["start"],
                    "end_seconds": seg["end"],
                    "language": seg["language"],
                }
                doc = build_chunk_doc(
                    source_path=str(path),
                    source_hash=content_hash,
                    chunk_index=i,
                    chunk_total=len(segments),
                    file_type=file_type.value,
                    mime_type=mime,
                    text_content=seg["text"],
                    data_ref=refs[0] if refs else None,
                    tags=_auto_tags(path, file_type, metadata=seg_meta, text_content=seg["text"]),
                    metadata=seg_meta,
                    embedding_model="whisper+all-MiniLM-L6-v2",
                )
                chunk_id = insert_chunk(doc)
                leann_id = add_vector(
                    IndexName.AUDIO,
                    chunk_id,
                    embedding.tolist(),
                    metadata={"file_type": "audio", "source_path": str(path)},
                )
                update_chunk_leann_id(chunk_id, leann_id)

            result["status"] = "indexed"
            result["chunks"] = len(segments)
            return result

        # Video — Whisper transcript + CLIP keyframes → LEANN AUDIO + VIDEO indexes
        if file_type == FileType.VIDEO:
            from ingestion.processors.video import process as process_video
            from storage.binary_store import store_thumbnail
            video_result = process_video(path)
            if video_result is None:
                result["status"] = "error"
                return result

            refs = store_file(path, content_hash)
            base_meta = {**_extract_metadata(path), **video_result.metadata}
            chunk_count = 0

            # Transcript chunks → MiniLM → LEANN AUDIO index
            if video_result.transcript_segments:
                texts = [s["text"] for s in video_result.transcript_segments]
                text_embeddings = embed_batch(texts)

                for i, (seg, embedding) in enumerate(zip(video_result.transcript_segments, text_embeddings)):
                    seg_meta = {
                        **base_meta,
                        "start_seconds": seg["start"],
                        "end_seconds": seg["end"],
                        "language": seg["language"],
                    }
                    doc = build_chunk_doc(
                        source_path=str(path),
                        source_hash=content_hash,
                        chunk_index=i,
                        chunk_total=len(video_result.transcript_segments),
                        file_type=file_type.value,
                        mime_type=mime,
                        text_content=seg["text"],
                        data_ref=refs[0] if refs else None,
                        tags=_auto_tags(path, file_type, metadata=seg_meta, text_content=seg["text"]) + ["transcript"],
                        metadata=seg_meta,
                        embedding_model="whisper+all-MiniLM-L6-v2",
                    )
                    chunk_id = insert_chunk(doc)
                    leann_id = add_vector(
                        IndexName.AUDIO,
                        chunk_id,
                        embedding.tolist(),
                        metadata={"file_type": "video_transcript", "source_path": str(path)},
                    )
                    update_chunk_leann_id(chunk_id, leann_id)
                    chunk_count += 1

            # Keyframes → CLIP → LEANN VIDEO index
            for i, frame in enumerate(video_result.frames):
                frame_meta = {**base_meta, "timestamp_seconds": frame.timestamp}
                doc = build_chunk_doc(
                    source_path=str(path),
                    source_hash=f"{content_hash}_frame{i}",
                    chunk_index=i,
                    chunk_total=len(video_result.frames),
                    file_type=file_type.value,
                    mime_type=mime,
                    text_content=None,
                    data_ref=refs[0] if refs else None,
                    tags=_auto_tags(path, file_type, metadata=frame_meta, image_embedding=frame.embedding) + ["keyframe"],
                    metadata=frame_meta,
                    embedding_model="clip-vit-base-patch32",
                )
                chunk_id = insert_chunk(doc)
                leann_id = add_vector(
                    IndexName.VIDEO,
                    chunk_id,
                    frame.embedding,
                    metadata={"file_type": "video_frame", "source_path": str(path),
                               "timestamp": frame.timestamp},
                )
                update_chunk_leann_id(chunk_id, leann_id)
                chunk_count += 1

            # Store video thumbnail (first good frame)
            if video_result.thumbnail and refs:
                # Use the source hash as a stable thumbnail key
                store_thumbnail(f"{content_hash}_thumb", video_result.thumbnail)

            result["status"] = "indexed"
            result["chunks"] = chunk_count
            return result

        # Text/code/document — chunk and embed
        if not text:
            result["status"] = "no_text"
            return result

        chunks = chunk_text(text, file_type, source_path=path)
        if not chunks:
            result["status"] = "no_chunks"
            return result

        chunk_texts = [c.text for c in chunks]

        if file_type == FileType.CODE:
            # CodeBERT embeddings for code
            from embeddings.code import embed_batch as code_embed_batch
            from ingestion.processors.code import process as process_code
            embeddings    = code_embed_batch(chunk_texts)
            code_result   = process_code(path, chunk_texts)
            index_name    = IndexName.CODE
            embedding_model = "codebert-base"
        else:
            embeddings      = embed_batch(chunk_texts)
            code_result     = None
            index_name      = IndexName.TEXT
            embedding_model = "all-MiniLM-L6-v2"

        base_meta = _extract_metadata(path)

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_meta = dict(base_meta)

            if code_result and i < len(code_result.chunk_metas):
                cm = code_result.chunk_metas[i]
                chunk_meta.update({
                    "language":    cm.language,
                    "symbol_name": cm.symbol_name,
                    "symbol_type": cm.symbol_type,
                    "start_line":  cm.start_line,
                    "end_line":    cm.end_line,
                })

            tags = _auto_tags(path, file_type, metadata=chunk_meta, text_content=chunk.text)
            if code_result:
                tags.append(code_result.language)

            doc = build_chunk_doc(
                source_path=str(path),
                source_hash=content_hash,
                chunk_index=chunk.chunk_index,
                chunk_total=chunk.chunk_total,
                file_type=file_type.value,
                mime_type=mime,
                text_content=chunk.text,
                data_ref=None,
                tags=tags,
                metadata=chunk_meta,
                embedding_model=embedding_model,
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


def run(source_path: Path, workers: int = 4, cancel_event=None) -> None:
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
        futures = {executor.submit(_ingest_file_safe, f): f for f in files}

        for future in as_completed(futures):
            if cancel_event and cancel_event.is_set():
                logger.info("Ingestion cancelled by user request.")
                for f in futures:
                    f.cancel()
                break
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

    # Run face clustering after ingestion completes
    _run_face_clustering()


def _store_face_data(chunk_id: str, detected_faces: list) -> None:
    """Persist face embeddings and crops linked to a chunk."""
    from storage.mongo import get_db
    from storage.binary_store import store_thumbnail

    db = get_db()
    for i, face in enumerate(detected_faces):
        face_id = f"{chunk_id}_face{i}"
        # Store crop as a thumbnail-style binary
        store_thumbnail(face_id, face.crop_bytes)
        # Store embedding in faces collection for later clustering
        db["face_embeddings"].update_one(
            {"face_id": face_id},
            {"$set": {
                "face_id":    face_id,
                "chunk_id":   chunk_id,
                "embedding":  face.embedding,
                "bbox":       face.bbox,
                "confidence": face.confidence,
                "cluster_id": None,  # filled in after clustering
            }},
            upsert=True,
        )


def _run_face_clustering() -> None:
    """
    Cluster all stored face embeddings into identity groups.
    Runs after ingestion — can also be triggered independently.
    """
    from storage.mongo import get_db
    from embeddings.faces import cluster_embeddings, cluster_centroid

    db = get_db()
    unclustered = list(db["face_embeddings"].find({"cluster_id": None}))

    if not unclustered:
        return

    logger.info(f"Clustering {len(unclustered)} face embeddings...")

    embeddings = [f["embedding"] for f in unclustered]
    chunk_ids  = [f["chunk_id"]  for f in unclustered]

    clusters = cluster_embeddings(embeddings, chunk_ids)

    if not clusters:
        logger.info("No face clusters formed.")
        return

    for cluster in clusters:
        centroid = cluster_centroid(cluster.embeddings)
        db["identities"].update_one(
            {"cluster_id": cluster.cluster_id},
            {"$set": {
                "cluster_id":      cluster.cluster_id,
                "face_count":      cluster.face_count,
                "face_embeddings": cluster.embeddings,
                "centroid":        centroid,
                "label":           None,
            }},
            upsert=True,
        )

    logger.info(f"Face clustering complete — {len(clusters)} identities found. "
                f"Open the UI to name them.")


def _auto_tags(
    path: Path,
    file_type: FileType,
    metadata: dict | None = None,
    text_content: str | None = None,
    image_embedding=None,
) -> list[str]:
    from embeddings.tagger import tag_chunk
    return tag_chunk(
        path=path,
        file_type=file_type.value,
        metadata=metadata or {},
        text_content=text_content,
        image_embedding=image_embedding,
    )


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
