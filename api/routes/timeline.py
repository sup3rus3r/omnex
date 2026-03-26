"""
Omnex — Timeline Routes
GET /timeline           — Paginated chunks grouped by date
GET /timeline/years     — Available years
GET /timeline/months    — Available months for a year
"""

from __future__ import annotations

from fastapi import APIRouter, Query

router = APIRouter()


@router.get("/years")
async def timeline_years():
    """Return list of years that have indexed data."""
    from storage.mongo import get_db
    db = get_db()
    pipeline = [
        {"$match": {"created_at": {"$exists": True}}},
        {"$group": {"_id": {"$year": "$created_at"}, "count": {"$sum": 1}}},
        {"$sort": {"_id": -1}},
    ]
    return [{"year": d["_id"], "count": d["count"]} for d in db["chunks"].aggregate(pipeline)]


@router.get("/months")
async def timeline_months(year: int = Query(...)):
    """Return months with data for a given year."""
    from storage.mongo import get_db
    from datetime import datetime, timezone
    db = get_db()
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end   = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    pipeline = [
        {"$match": {"created_at": {"$gte": start, "$lt": end}}},
        {"$group": {
            "_id": {"$month": "$created_at"},
            "count": {"$sum": 1},
            "types": {"$addToSet": "$file_type"},
        }},
        {"$sort": {"_id": -1}},
    ]
    return [{"month": d["_id"], "count": d["count"], "types": d["types"]} for d in db["chunks"].aggregate(pipeline)]


@router.get("")
async def timeline(
    year:  int       = Query(...),
    month: int       = Query(...),
    page:  int       = Query(default=1, ge=1),
    limit: int       = Query(default=40, ge=1, le=100),
    file_type: str | None = Query(default=None),
):
    """Return paginated chunks for a given year/month."""
    from storage.mongo import get_db
    from datetime import datetime, timezone
    from bson import ObjectId

    db = get_db()
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    end   = datetime(year, month + 1, 1, tzinfo=timezone.utc) if month < 12 else datetime(year + 1, 1, 1, tzinfo=timezone.utc)

    query: dict = {"created_at": {"$gte": start, "$lt": end}}
    if file_type:
        query["file_type"] = file_type

    total = db["chunks"].count_documents(query)
    skip  = (page - 1) * limit

    docs = list(db["chunks"].find(
        query,
        {"_id": 1, "source_path": 1, "file_type": 1, "created_at": 1,
         "text_content": 1, "metadata": 1, "chunk_index": 1},
    ).sort("created_at", -1).skip(skip).limit(limit))

    results = []
    for d in docs:
        chunk_id = str(d["_id"])
        results.append({
            "chunk_id":    chunk_id,
            "source_path": d.get("source_path", ""),
            "file_type":   d.get("file_type", "unknown"),
            "created_at":  d["created_at"].isoformat() if d.get("created_at") else None,
            "text":        (d.get("text_content") or "")[:200],
            "metadata":    d.get("metadata", {}),
            "chunk_index": d.get("chunk_index", 0),
            "thumbnail_url": f"/chunk/{chunk_id}/thumbnail" if d.get("file_type") == "image" else None,
        })

    return {"total": total, "page": page, "limit": limit, "results": results}
