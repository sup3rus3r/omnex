"""
Omnex — Timeline Routes
GET /timeline           — Paginated source files grouped by date
GET /timeline/years     — Available years
GET /timeline/months    — Available months for a year
"""

from __future__ import annotations

from fastapi import APIRouter, Query

router = APIRouter()


@router.get("/years")
async def timeline_years():
    """Return list of years that have indexed data (one entry per source file)."""
    from storage.mongo import get_db
    db = get_db()
    pipeline = [
        {"$match": {"created_at": {"$exists": True}}},
        {"$group": {"_id": "$source_path", "created_at": {"$min": "$created_at"}}},
        {"$group": {"_id": {"$year": "$created_at"}, "count": {"$sum": 1}}},
        {"$sort": {"_id": -1}},
    ]
    return [{"year": d["_id"], "count": d["count"]} for d in db["chunks"].aggregate(pipeline)]


@router.get("/months")
async def timeline_months(year: int = Query(...)):
    """Return months with data for a given year (counts distinct source files)."""
    from storage.mongo import get_db
    from datetime import datetime, timezone
    db = get_db()
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end   = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    pipeline = [
        {"$match": {"created_at": {"$gte": start, "$lt": end}}},
        {"$group": {
            "_id": "$source_path",
            "month": {"$first": {"$month": "$created_at"}},
            "file_type": {"$first": "$file_type"},
        }},
        {"$group": {
            "_id": "$month",
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
    """Return paginated source files for a given year/month (one entry per source file)."""
    from storage.mongo import get_db
    from datetime import datetime, timezone

    db = get_db()
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    end   = datetime(year, month + 1, 1, tzinfo=timezone.utc) if month < 12 else datetime(year + 1, 1, 1, tzinfo=timezone.utc)

    match: dict = {"created_at": {"$gte": start, "$lt": end}}
    if file_type:
        match["file_type"] = file_type

    skip = (page - 1) * limit

    # Count distinct source files
    count_pipeline = [
        {"$match": match},
        {"$group": {"_id": "$source_path"}},
        {"$count": "total"},
    ]
    count_result = list(db["chunks"].aggregate(count_pipeline))
    total = count_result[0]["total"] if count_result else 0

    # Group by source_path — pick the chunk with chunk_index=0 as representative
    pipeline = [
        {"$match": match},
        {"$sort": {"chunk_index": 1}},
        {"$group": {
            "_id": "$source_path",
            "chunk_id":    {"$first": {"$toString": "$_id"}},
            "file_type":   {"$first": "$file_type"},
            "created_at":  {"$min": "$created_at"},
            "chunk_count": {"$sum": 1},
            "metadata":    {"$first": "$metadata"},
            "text":        {"$first": "$text_content"},
        }},
        {"$sort": {"created_at": -1}},
        {"$skip": skip},
        {"$limit": limit},
    ]

    docs = list(db["chunks"].aggregate(pipeline))

    results = []
    for d in docs:
        chunk_id  = d["chunk_id"]
        ft        = d.get("file_type", "unknown")
        src       = d["_id"]  # source_path is the group key
        results.append({
            "chunk_id":      chunk_id,
            "source_path":   src,
            "file_type":     ft,
            "created_at":    d["created_at"].isoformat() if d.get("created_at") else None,
            "text":          (d.get("text") or "")[:200],
            "metadata":      d.get("metadata", {}),
            "chunk_index":   0,
            "chunk_count":   d.get("chunk_count", 1),
            "thumbnail_url": f"/chunk/{chunk_id}/thumbnail" if ft == "image" else None,
        })

    return {"total": total, "page": page, "limit": limit, "results": results}
