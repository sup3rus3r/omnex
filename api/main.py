"""
Omnex — FastAPI Application
Local API server — serves both the Next.js UI and external agents.
Runs on 127.0.0.1:8000 by default.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import query, chunks, ingest, identity, setup


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — warm up DB connection
    from storage.mongo import get_db
    get_db()  # Establishes connection + creates indexes
    yield
    # Shutdown — nothing to clean up (LEANN saves on each write)


app = FastAPI(
    title="Omnex",
    description="The AI OS memory layer — everything, indexed.",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow Next.js dev server (any port) and local agent calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(query.router,    prefix="/query",    tags=["Query"])
app.include_router(chunks.router,   prefix="/chunk",    tags=["Chunks"])
app.include_router(ingest.router,   prefix="/ingest",   tags=["Ingestion"])
app.include_router(identity.router, prefix="/identity", tags=["Identity"])
app.include_router(setup.router,    prefix="/setup",    tags=["Setup"])


@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "service": "Omnex", "version": "0.1.0"}


@app.get("/health", tags=["Health"])
async def health():
    from storage.mongo import get_db
    from storage.leann_store import IndexName, index_size
    try:
        db = get_db()
        db.command("ping")
        mongo_ok = True
    except Exception:
        mongo_ok = False

    return {
        "status":       "ok" if mongo_ok else "degraded",
        "mongo":        mongo_ok,
        "index_sizes": {
            "text":  index_size(IndexName.TEXT),
            "image": index_size(IndexName.IMAGE),
            "code":  index_size(IndexName.CODE),
        },
    }


@app.get("/stats", tags=["Health"])
async def stats():
    from storage.mongo import get_db
    db = get_db()
    chunks_count = db["chunks"].count_documents({})
    by_type = db["chunks"].aggregate([
        {"$group": {"_id": "$file_type", "count": {"$sum": 1}}}
    ])
    ingestion = list(db["ingestion_state"].find({}, {"_id": 0}))
    return {
        "total_chunks": chunks_count,
        "by_type":      {d["_id"]: d["count"] for d in by_type},
        "ingestion":    ingestion,
    }
