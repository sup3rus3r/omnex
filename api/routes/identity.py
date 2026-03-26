"""
Omnex — Identity Routes (Phase 4 — Face Clustering)
POST /identity/label    — Assign name to a face cluster
GET  /identity/pending  — Fetch clusters awaiting user labelling
GET  /identity/all      — All named identities
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
