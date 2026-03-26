"""
Omnex — Identity Routes (Phase 4 — Face Clustering)
POST /identity/label             — Assign name to a face cluster
GET  /identity/pending           — Fetch clusters awaiting user labelling
GET  /identity/all               — All named identities
GET  /identity/clusters          — All clusters with photo counts (for People UI)
GET  /identity/photos/{cluster}  — Photo chunks belonging to a cluster
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class LabelRequest(BaseModel):
    cluster_id: str
    label:      str


@router.post("/label")
async def label_identity(req: LabelRequest):
    """Assign a human name to a face cluster."""
    from storage.mongo import label_identity as _label
    _label(req.cluster_id, req.label)
    return {"cluster_id": req.cluster_id, "label": req.label, "status": "labelled"}


@router.get("/pending")
async def pending_identities(limit: int = 20):
    """Return face clusters that have not yet been named by the user."""
    from storage.mongo import get_unlabelled_identities
    records = get_unlabelled_identities(limit=limit)
    for r in records:
        r["_id"] = str(r["_id"])
    return {"pending": records, "count": len(records)}


@router.get("/all")
async def all_identities():
    """Return all named identities."""
    from storage.mongo import get_db
    records = list(
        get_db()["identities"].find(
            {"label": {"$exists": True}},
            {"_id": 1, "label": 1, "cluster_id": 1, "thumbnail_ref": 1}
        )
    )
    for r in records:
        r["_id"] = str(r["_id"])
    return {"identities": records}


@router.get("/clusters")
async def identity_clusters():
    """
    Return all face clusters with photo counts and optional label.
    Used by the People UI to list detected individuals.
    """
    from storage.mongo import get_db
    db = get_db()

    # Aggregate face_embeddings by cluster_id to get counts
    pipeline = [
        {"$group": {
            "_id": "$cluster_id",
            "photo_count": {"$sum": 1},
            "sample_chunk_id": {"$first": "$chunk_id"},
        }},
        {"$sort": {"photo_count": -1}},
    ]
    clusters = list(db["face_embeddings"].aggregate(pipeline))

    # Look up labels from identities collection
    cluster_ids = [c["_id"] for c in clusters]
    identity_map: dict = {}
    for doc in db["identities"].find({"cluster_id": {"$in": cluster_ids}}):
        identity_map[doc["cluster_id"]] = doc.get("label")

    results = []
    for c in clusters:
        cid = c["_id"]
        # Get a thumbnail URL from the sample chunk
        sample = c.get("sample_chunk_id")
        thumbnail_url = f"/chunk/{sample}/thumbnail" if sample else None
        results.append({
            "cluster_id":    cid,
            "label":         identity_map.get(cid),
            "photo_count":   c["photo_count"],
            "thumbnail_url": thumbnail_url,
        })

    return results


@router.get("/photos/{cluster_id}")
async def cluster_photos(cluster_id: str, limit: int = 50):
    """
    Return photo chunks belonging to a face cluster.
    Used by the People UI to show all photos of a person.
    """
    from storage.mongo import get_db
    db = get_db()

    # Get chunk_ids for this cluster
    face_docs = list(db["face_embeddings"].find(
        {"cluster_id": cluster_id},
        {"chunk_id": 1},
    ).limit(limit))

    chunk_ids = [d["chunk_id"] for d in face_docs]
    if not chunk_ids:
        return {"photos": [], "cluster_id": cluster_id}

    # Fetch chunk metadata
    from bson import ObjectId
    oids = []
    for cid in chunk_ids:
        try:
            oids.append(ObjectId(cid))
        except Exception:
            pass

    docs = list(db["chunks"].find(
        {"_id": {"$in": oids}},
        {"_id": 1, "source_path": 1, "file_type": 1, "created_at": 1},
    ))

    photos = []
    for d in docs:
        cid_str = str(d["_id"])
        photos.append({
            "chunk_id":      cid_str,
            "source_path":   d.get("source_path", ""),
            "file_type":     d.get("file_type", "image"),
            "created_at":    d["created_at"].isoformat() if d.get("created_at") else None,
            "thumbnail_url": f"/chunk/{cid_str}/thumbnail",
        })

    return {"photos": photos, "cluster_id": cluster_id, "total": len(photos)}
