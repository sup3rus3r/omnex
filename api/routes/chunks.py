"""
Omnex — Chunk Routes
GET  /chunk/{id}           — Full chunk metadata
GET  /chunk/{id}/raw       — Stream raw binary (image/video/audio)
GET  /chunk/{id}/thumbnail — Serve thumbnail image
DELETE /chunk/{id}         — Remove chunk from index (does not delete source)
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse

router = APIRouter()


@router.get("/{chunk_id}")
async def get_chunk(chunk_id: str):
    """Return full chunk document with metadata."""
    from storage.mongo import get_chunk_by_id
    doc = get_chunk_by_id(chunk_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Chunk not found")
    doc["_id"] = str(doc["_id"])
    return doc


@router.get("/{chunk_id}/raw")
async def get_chunk_raw(chunk_id: str):
    """
    Stream raw binary data for a chunk.
    Used by the UI to serve images, audio, video.
    """
    from storage.mongo import get_chunk_by_id
    from storage.binary_store import read_chunk

    doc = get_chunk_by_id(chunk_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Chunk not found")

    data_ref = doc.get("data_ref")
    if not data_ref:
        raise HTTPException(status_code=404, detail="No binary data for this chunk")

    try:
        data = read_chunk(data_ref)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Binary data not found on disk")

    mime = doc.get("mime_type", "application/octet-stream")
    return Response(content=data, media_type=mime)


@router.get("/{chunk_id}/thumbnail")
async def get_thumbnail(chunk_id: str):
    """
    Serve the thumbnail for an image or video chunk.
    Images: stored by chunk_id.
    Videos: stored by {source_hash}_thumb — fall back to that if direct lookup fails.
    """
    from storage.mongo import get_chunk_by_id
    from storage.binary_store import read_thumbnail

    data = read_thumbnail(chunk_id)
    if not data:
        # Try video thumbnail key: {source_hash}_thumb
        doc = get_chunk_by_id(chunk_id)
        if doc:
            source_hash = doc.get("source_hash")
            if source_hash:
                data = read_thumbnail(f"{source_hash}_thumb")

    if not data:
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    return Response(content=data, media_type="image/jpeg")


@router.delete("/{chunk_id}")
async def delete_chunk(chunk_id: str):
    """
    Remove a chunk from the Omnex index.
    Does NOT delete the source file — only removes from index.
    """
    from storage.mongo import get_chunk_by_id, get_db
    from storage.leann_store import delete_vector, IndexName
    from bson import ObjectId

    doc = get_chunk_by_id(chunk_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Chunk not found")

    # Remove from LEANN index
    leann_id = doc.get("leann_id")
    if leann_id:
        file_type = doc.get("file_type", "document")
        index_map = {
            "image":    IndexName.IMAGE,
            "video":    IndexName.VIDEO,
            "audio":    IndexName.AUDIO,
            "code":     IndexName.CODE,
        }
        index = index_map.get(file_type, IndexName.TEXT)
        try:
            delete_vector(index, leann_id)
        except Exception:
            pass  # Best effort

    # Remove from MongoDB
    get_db()["chunks"].delete_one({"_id": ObjectId(chunk_id)})

    return {"deleted": chunk_id}
