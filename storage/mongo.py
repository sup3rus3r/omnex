"""
Omnex — MongoDB Client & Schema
Manages all metadata storage: chunks, identities, ingestion state.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.database import Database


# ── Connection ────────────────────────────────────────────────────────────────

_client: MongoClient | None = None
_db: Database | None = None


def get_db() -> Database:
    global _client, _db
    if _db is None:
        uri  = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        name = os.getenv("MONGO_DB", "omnex")
        _client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        _db = _client[name]
        _ensure_indexes(_db)
    return _db


def get_collection(name: str) -> Collection:
    return get_db()[name]


# ── Indexes ───────────────────────────────────────────────────────────────────

def _ensure_indexes(db: Database) -> None:
    chunks = db["chunks"]
    chunks.create_index([("source_hash", ASCENDING)], unique=False)
    chunks.create_index([("source_path", ASCENDING)])
    chunks.create_index([("file_type", ASCENDING)])
    chunks.create_index([("tags", ASCENDING)])
    chunks.create_index([("metadata.created_at", DESCENDING)])
    chunks.create_index([("leann_id", ASCENDING)], sparse=True)

    identities = db["identities"]
    identities.create_index([("cluster_id", ASCENDING)], unique=True)
    identities.create_index([("label", ASCENDING)], sparse=True)

    ingestion = db["ingestion_state"]
    ingestion.create_index([("source_path", ASCENDING)], unique=True)


# ── Chunk operations ──────────────────────────────────────────────────────────

def insert_chunk(chunk_doc: dict) -> str:
    """Insert a chunk document. Returns the inserted _id as string."""
    result = get_collection("chunks").insert_one(chunk_doc)
    return str(result.inserted_id)


def update_chunk_leann_id(chunk_id: str, leann_id: str) -> None:
    """Set the LEANN vector index ID on a chunk after embedding."""
    from bson import ObjectId
    get_collection("chunks").update_one(
        {"_id": ObjectId(chunk_id)},
        {"$set": {"leann_id": leann_id, "updated_at": _now()}}
    )


def get_chunk_by_hash(source_hash: str, chunk_index: int) -> dict | None:
    return get_collection("chunks").find_one({
        "source_hash": source_hash,
        "chunk_index": chunk_index,
    })


def chunk_exists(source_hash: str) -> bool:
    """Returns True if any chunk with this file hash already exists."""
    return get_collection("chunks").find_one({"source_hash": source_hash}) is not None


def get_chunk_by_id(chunk_id: str) -> dict | None:
    from bson import ObjectId
    return get_collection("chunks").find_one({"_id": ObjectId(chunk_id)})


def delete_chunks_by_path(source_path: str) -> int:
    result = get_collection("chunks").delete_many({"source_path": source_path})
    return result.deleted_count


# ── Ingestion state ───────────────────────────────────────────────────────────

def upsert_ingestion_state(source_path: str, state: dict) -> None:
    """Track ingestion progress for a source path."""
    get_collection("ingestion_state").update_one(
        {"source_path": source_path},
        {"$set": {**state, "updated_at": _now()}},
        upsert=True,
    )


def get_ingestion_state(source_path: str) -> dict | None:
    return get_collection("ingestion_state").find_one({"source_path": source_path})


# ── Identity operations ───────────────────────────────────────────────────────

def upsert_identity(cluster_id: str, data: dict) -> None:
    get_collection("identities").update_one(
        {"cluster_id": cluster_id},
        {"$set": {**data, "updated_at": _now()}},
        upsert=True,
    )


def get_unlabelled_identities(limit: int = 20) -> list[dict]:
    return list(
        get_collection("identities").find(
            {"label": {"$exists": False}},
            limit=limit
        )
    )


def label_identity(cluster_id: str, label: str) -> None:
    get_collection("identities").update_one(
        {"cluster_id": cluster_id},
        {"$set": {"label": label, "updated_at": _now()}}
    )


# ── Schema helpers ────────────────────────────────────────────────────────────

def build_chunk_doc(
    source_path: str,
    source_hash: str,
    chunk_index: int,
    chunk_total: int,
    file_type: str,
    mime_type: str,
    text_content: str | None,
    data_ref: str | None,
    tags: list[str],
    metadata: dict,
    embedding_model: str,
) -> dict:
    return {
        "source_path":     source_path,
        "source_hash":     source_hash,
        "chunk_index":     chunk_index,
        "chunk_total":     chunk_total,
        "file_type":       file_type,
        "mime_type":       mime_type,
        "text_content":    text_content,
        "data_ref":        data_ref,
        "leann_id":        None,
        "tags":            tags,
        "metadata":        metadata,
        "embedding_model": embedding_model,
        "created_at":      _now(),
        "updated_at":      _now(),
    }


def _now() -> datetime:
    return datetime.now(timezone.utc)
