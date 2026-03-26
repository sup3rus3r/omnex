"""
Omnex — Query Routes
POST /query         — Natural language search
POST /query/refine  — Narrow previous results (session-aware)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.query_engine import search, QueryResponse

router = APIRouter()


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:            str          = Field(..., min_length=1, max_length=2000)
    top_k:            int          = Field(default=20, ge=1, le=100)
    file_type:        str | None   = Field(default=None)
    date_from:        datetime | None = Field(default=None)
    date_to:          datetime | None = Field(default=None)
    session_id:       str | None   = Field(default=None)


class RefineRequest(BaseModel):
    query:        str        = Field(..., min_length=1)
    session_id:   str        = Field(..., description="Session ID from previous query")
    top_k:        int        = Field(default=20, ge=1, le=100)
    file_type:    str | None = None
    anchor_chunk_id: str | None = Field(
        default=None,
        description="Find results similar to this specific chunk"
    )


class QueryResultOut(BaseModel):
    chunk_id:      str
    score:         float
    file_type:     str
    source_path:   str
    text:          str | None
    metadata:      dict
    thumbnail_url: str | None


class QueryResponseOut(BaseModel):
    query:                  str
    results:                list[QueryResultOut]
    total:                  int
    llm_response:           str | None
    suggested_refinements:  list[str]
    session_id:             str | None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("", response_model=QueryResponseOut)
async def query(req: QueryRequest):
    """
    Natural language search across all indexed data.

    Automatically detects query intent (visual, temporal, code, location)
    and routes to the appropriate LEANN indexes.
    """
    try:
        response = await search(
            query=req.query,
            top_k=req.top_k,
            file_type_filter=req.file_type,
            date_from=req.date_from,
            date_to=req.date_to,
            session_id=req.session_id,
        )
        return _to_response_out(response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refine", response_model=QueryResponseOut)
async def refine(req: RefineRequest):
    """
    Refine a previous query — narrows results within a session context.
    Supports 'more like this' via anchor_chunk_id.
    """
    try:
        if req.anchor_chunk_id:
            # Similarity search from a specific result
            refined_query = await _build_similarity_query(req.anchor_chunk_id, req.query)
        else:
            refined_query = req.query

        response = await search(
            query=refined_query,
            top_k=req.top_k,
            file_type_filter=req.file_type,
            session_id=req.session_id,
        )
        return _to_response_out(response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_response_out(r: QueryResponse) -> QueryResponseOut:
    return QueryResponseOut(
        query=r.query,
        results=[
            QueryResultOut(
                chunk_id=res.chunk_id,
                score=res.score,
                file_type=res.file_type,
                source_path=res.source_path,
                text=res.text,
                metadata=res.metadata,
                thumbnail_url=res.thumbnail_url,
            )
            for res in r.results
        ],
        total=r.total,
        llm_response=r.llm_response,
        suggested_refinements=r.suggested_refinements,
        session_id=r.session_id,
    )


async def _build_similarity_query(chunk_id: str, hint: str) -> str:
    """Build a similarity-enhanced query using the anchor chunk's text."""
    from storage.mongo import get_chunk_by_id
    doc = get_chunk_by_id(chunk_id)
    if doc and doc.get("text_content"):
        snippet = doc["text_content"][:200]
        return f"{hint} {snippet}"
    return hint
