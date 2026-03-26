"""
Omnex — Agent Routes (Phase 16 — Multi-Agent Write API)
POST /agents              — Register a named agent identity, returns API key
GET  /agents              — List all registered agents
DELETE /agents/{agent_id} — Remove an agent
POST /ingest/observation  — Agent pushes a text memory into the index
"""

from __future__ import annotations

import os
import secrets
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth import require_api_key

router = APIRouter()
log = logging.getLogger("omnex.agents")


# ── Models ────────────────────────────────────────────────────────────────────

class RegisterAgentRequest(BaseModel):
    name:        str
    description: str = ""


class ObservationRequest(BaseModel):
    text:        str
    source:      str = "agent"        # logical source label (e.g. "claude", "cursor")
    agent_id:    str                   # must match a registered agent _id
    metadata:    dict = {}


# ── Agent registration ────────────────────────────────────────────────────────

@router.post("", dependencies=[Depends(require_api_key)])
async def register_agent(req: RegisterAgentRequest):
    """Register a named agent and return its API key."""
    from storage.mongo import get_db
    db = get_db()

    api_key = "omnex-agent-" + secrets.token_urlsafe(24)
    doc = {
        "name":        req.name,
        "description": req.description,
        "api_key":     api_key,
        "created_at":  datetime.now(timezone.utc),
        "observation_count": 0,
    }
    result = db["agents"].insert_one(doc)
    agent_id = str(result.inserted_id)

    log.info(f"Registered agent: {req.name} ({agent_id})")
    return {
        "agent_id": agent_id,
        "name":     req.name,
        "api_key":  api_key,
        "message":  "Store this API key — it will not be shown again.",
    }


@router.get("", dependencies=[Depends(require_api_key)])
async def list_agents():
    """List all registered agents (without API keys)."""
    from storage.mongo import get_db
    db = get_db()
    docs = list(db["agents"].find({}, {"api_key": 0}))
    for d in docs:
        d["agent_id"] = str(d.pop("_id"))
        if "created_at" in d:
            d["created_at"] = d["created_at"].isoformat()
    return {"agents": docs, "total": len(docs)}


@router.delete("/{agent_id}", dependencies=[Depends(require_api_key)])
async def delete_agent(agent_id: str):
    """Remove a registered agent."""
    from storage.mongo import get_db
    from bson import ObjectId
    db = get_db()
    try:
        oid = ObjectId(agent_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid agent_id")
    result = db["agents"].delete_one({"_id": oid})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"status": "deleted", "agent_id": agent_id}


# ── Observation ingestion ─────────────────────────────────────────────────────

@router.post("/observe", dependencies=[Depends(require_api_key)])
async def store_observation(req: ObservationRequest):
    """
    Store a text observation from an agent directly into the Omnex index.
    The text is embedded with MiniLM and stored as a chunk with agent metadata.
    No file required — pure text memory injection.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")

    from storage.mongo import get_db
    from storage.leann_store import add_vector, IndexName
    from embeddings.text import embed as embed_text
    from bson import ObjectId
    import asyncio

    db = get_db()

    # Verify agent exists
    try:
        agent_oid = ObjectId(req.agent_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid agent_id")

    agent = db["agents"].find_one({"_id": agent_oid})
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found — register via POST /agents first")

    now = datetime.now(timezone.utc)

    # Chunk doc
    chunk_doc = {
        "source_path":  f"agent://{agent['name']}/{req.source}",
        "source_hash":  "",
        "file_type":    "observation",
        "chunk_index":  0,
        "text":         req.text.strip(),
        "tags":         ["agent", "observation", agent["name"]],
        "created_at":   now,
        "metadata": {
            "agent_id":    req.agent_id,
            "agent_name":  agent["name"],
            "source":      req.source,
            "created_at":  now,
            **req.metadata,
        },
    }

    result = db["chunks"].insert_one(chunk_doc)
    chunk_id = str(result.inserted_id)

    # Embed and index
    def _embed_and_index():
        vec = embed_text(req.text)
        add_vector(IndexName.TEXT, chunk_id, vec)

    await asyncio.get_event_loop().run_in_executor(None, _embed_and_index)

    # Increment agent observation count
    db["agents"].update_one({"_id": agent_oid}, {"$inc": {"observation_count": 1}})

    log.info(f"Agent {agent['name']} stored observation: {req.text[:80]}")
    return {
        "chunk_id":   chunk_id,
        "agent_id":   req.agent_id,
        "agent_name": agent["name"],
        "status":     "indexed",
    }
