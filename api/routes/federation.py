"""
Omnex — Federation Routes

Manages peer Omnex instances and federated search across them.

POST   /federation/peers          — Register a peer instance
GET    /federation/peers          — List all registered peers
DELETE /federation/peers/{peer_id} — Remove a peer
GET    /federation/peers/{peer_id}/ping — Test connectivity to a peer

GET    /federation/search         — Federated search across all active peers
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.auth import require_api_key

router = APIRouter()
log    = logging.getLogger("omnex.federation")


# ── Models ────────────────────────────────────────────────────────────────────

class PeerRequest(BaseModel):
    url:       str   = Field(..., description="Base URL of the peer Omnex instance, e.g. https://abc.ngrok.io")
    api_key:   str   = Field(..., description="API key for authenticating with the peer")
    label:     str   = Field(..., description="Human-readable name for this peer")
    trust:     str   = Field(default="read", description="Trust level: read | read-write")


class PeerOut(BaseModel):
    peer_id:    str
    url:        str
    label:      str
    trust:      str
    active:     bool
    added_at:   str
    last_seen:  str | None


class FederatedSearchRequest(BaseModel):
    query:     str  = Field(..., min_length=1)
    top_k:     int  = Field(default=10, ge=1, le=50)
    file_type: str | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_peers_collection():
    from storage.mongo import get_db
    return get_db()["federation_peers"]


def _peer_to_out(doc: dict) -> PeerOut:
    return PeerOut(
        peer_id=str(doc["peer_id"]),
        url=doc["url"],
        label=doc["label"],
        trust=doc.get("trust", "read"),
        active=doc.get("active", True),
        added_at=doc["added_at"].isoformat() if hasattr(doc["added_at"], "isoformat") else str(doc["added_at"]),
        last_seen=doc["last_seen"].isoformat() if doc.get("last_seen") and hasattr(doc["last_seen"], "isoformat") else doc.get("last_seen"),
    )


async def _ping_peer(url: str, api_key: str, timeout: float = 5.0) -> bool:
    """Test connectivity to a peer instance."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(
                f"{url.rstrip('/')}/health",
                headers={"X-API-Key": api_key},
            )
            return resp.status_code == 200
    except Exception:
        return False


async def _search_peer(
    peer: dict,
    query: str,
    top_k: int,
    file_type: str | None,
    timeout: float = 10.0,
) -> list[dict]:
    """Run a search against a single peer. Returns annotated results."""
    try:
        payload: dict[str, Any] = {"query": query, "top_k": top_k}
        if file_type:
            payload["file_type"] = file_type

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{peer['url'].rstrip('/')}/query",
                json=payload,
                headers={
                    "X-API-Key": peer["api_key"],
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results", [])
        # Annotate each result with origin
        for r in results:
            r["origin_instance"] = peer["label"]
            r["origin_url"]      = peer["url"]
            r["origin_peer_id"]  = str(peer["peer_id"])

        # Update last_seen
        _get_peers_collection().update_one(
            {"peer_id": peer["peer_id"]},
            {"$set": {"last_seen": datetime.now(timezone.utc)}},
        )

        return results

    except Exception as e:
        log.warning(f"[federation] peer {peer['label']} search failed: {e}")
        return []


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/peers", dependencies=[Depends(require_api_key)])
async def register_peer(req: PeerRequest):
    """Register a new peer Omnex instance."""
    import uuid
    col = _get_peers_collection()

    # Check if already registered
    existing = col.find_one({"url": req.url})
    if existing:
        raise HTTPException(status_code=409, detail=f"Peer already registered: {req.url}")

    # Test connectivity before saving
    reachable = await _ping_peer(req.url, req.api_key)

    peer_id = str(uuid.uuid4())
    doc = {
        "peer_id":  peer_id,
        "url":      req.url.rstrip("/"),
        "api_key":  req.api_key,
        "label":    req.label,
        "trust":    req.trust,
        "active":   True,
        "added_at": datetime.now(timezone.utc),
        "last_seen": datetime.now(timezone.utc) if reachable else None,
    }
    col.insert_one(doc)

    return {
        "peer_id":   peer_id,
        "label":     req.label,
        "url":       req.url,
        "reachable": reachable,
        "message":   "Peer registered." if reachable else "Peer registered but could not reach it — check URL and API key.",
    }


@router.get("/peers", dependencies=[Depends(require_api_key)])
async def list_peers():
    """List all registered peers."""
    col  = _get_peers_collection()
    docs = list(col.find({}, {"api_key": 0}))  # never expose api_key
    return {"peers": [_peer_to_out(d) for d in docs]}


@router.delete("/peers/{peer_id}", dependencies=[Depends(require_api_key)])
async def remove_peer(peer_id: str):
    """Remove a registered peer."""
    col    = _get_peers_collection()
    result = col.delete_one({"peer_id": peer_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail=f"Peer not found: {peer_id}")
    return {"deleted": peer_id}


@router.get("/peers/{peer_id}/ping", dependencies=[Depends(require_api_key)])
async def ping_peer(peer_id: str):
    """Test connectivity to a specific peer."""
    col  = _get_peers_collection()
    peer = col.find_one({"peer_id": peer_id})
    if not peer:
        raise HTTPException(status_code=404, detail=f"Peer not found: {peer_id}")

    reachable = await _ping_peer(peer["url"], peer["api_key"])

    if reachable:
        col.update_one(
            {"peer_id": peer_id},
            {"$set": {"last_seen": datetime.now(timezone.utc)}},
        )

    return {"peer_id": peer_id, "label": peer["label"], "reachable": reachable}


@router.post("/search", dependencies=[Depends(require_api_key)])
async def federated_search(req: FederatedSearchRequest):
    """
    Fan out a search to all active peers and merge results by score.
    Also includes local results from this instance.
    """
    col   = _get_peers_collection()
    peers = list(col.find({"active": True}, {"_id": 0}))

    # Local search
    from api.query_engine import search as local_search
    local_response = await local_search(
        query=req.query,
        top_k=req.top_k,
        file_type_filter=req.file_type,
    )

    local_results = [
        {
            "chunk_id":         r.chunk_id,
            "score":            r.score,
            "file_type":        r.file_type,
            "source_path":      r.source_path,
            "text":             r.text,
            "metadata":         r.metadata,
            "thumbnail_url":    r.thumbnail_url,
            "origin_instance":  "local",
            "origin_url":       None,
            "origin_peer_id":   None,
        }
        for r in local_response.results
    ]

    # Fan out to peers concurrently
    peer_tasks = [
        _search_peer(peer, req.query, req.top_k, req.file_type)
        for peer in peers
    ]
    peer_results_lists = await asyncio.gather(*peer_tasks, return_exceptions=True)

    # Flatten peer results
    all_peer_results: list[dict] = []
    for res in peer_results_lists:
        if isinstance(res, list):
            all_peer_results.extend(res)

    # Merge and rank by score
    all_results = local_results + all_peer_results
    all_results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
    merged = all_results[: req.top_k * (1 + len(peers))]  # keep proportional top results

    return {
        "query":        req.query,
        "total":        len(merged),
        "peers_queried": len(peers),
        "results":      merged,
        "llm_response": local_response.llm_response,
    }
