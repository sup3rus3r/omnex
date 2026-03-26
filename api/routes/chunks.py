"""
Omnex — Chunk Routes
GET    /chunk/{id}           — Full chunk metadata
GET    /chunk/{id}/raw       — Stream raw binary (image/video/audio)
GET    /chunk/{id}/thumbnail — Serve thumbnail image
GET    /chunk/{id}/sign      — Get signed URLs for raw + thumbnail
DELETE /chunk/{id}           — Remove chunk from index
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from api.signing import sign_url, verify_signed_request

router = APIRouter()


@router.get("/{chunk_id}")
async def get_chunk(chunk_id: str):
    from storage.mongo import get_chunk_by_id
    doc = get_chunk_by_id(chunk_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Chunk not found")
    doc["_id"] = str(doc["_id"])
    return doc


@router.get("/{chunk_id}/sign")
async def sign_chunk_urls(chunk_id: str, expires_in: int = Query(default=3600, ge=60, le=86400)):
    """Return signed URLs for raw binary and thumbnail. Valid for expires_in seconds."""
    from storage.mongo import get_chunk_by_id
    doc = get_chunk_by_id(chunk_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return {
        "chunk_id":      chunk_id,
        "raw_url":       sign_url(f"/chunk/{chunk_id}/raw",       expires_in),
        "thumbnail_url": sign_url(f"/chunk/{chunk_id}/thumbnail", expires_in),
        "expires_in":    expires_in,
        "signing":       True,
    }


@router.get("/{chunk_id}/raw")
async def get_chunk_raw(
    chunk_id: str,
    token:   str | None = Query(default=None),
    expires: int | None = Query(default=None),
):
    verify_signed_request(token, expires, f"/chunk/{chunk_id}/raw")

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
async def get_thumbnail(
    chunk_id: str,
    token:   str | None = Query(default=None),
    expires: int | None = Query(default=None),
):
    verify_signed_request(token, expires, f"/chunk/{chunk_id}/thumbnail")

    from storage.mongo import get_chunk_by_id
    from storage.binary_store import read_thumbnail

    data = read_thumbnail(chunk_id)
    if not data:
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
    from storage.mongo import get_chunk_by_id, get_db
    from storage.leann_store import delete_vector, IndexName
    from bson import ObjectId

    doc = get_chunk_by_id(chunk_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Chunk not found")

    leann_id = doc.get("leann_id")
    if leann_id:
        file_type = doc.get("file_type", "document")
        index_map = {
            "image": IndexName.IMAGE,
            "video": IndexName.VIDEO,
            "audio": IndexName.AUDIO,
            "code":  IndexName.CODE,
        }
        index = index_map.get(file_type, IndexName.TEXT)
        try:
            delete_vector(index, leann_id)
        except Exception:
            pass

    get_db()["chunks"].delete_one({"_id": ObjectId(chunk_id)})
    return {"deleted": chunk_id}
