"""
Omnex — MCP (Model Context Protocol) Server
Exposes Omnex search as an MCP tool so Claude and other agents can call it directly.

Implements MCP JSON-RPC 2.0 over HTTP POST /mcp
Supported methods:
  tools/list    — enumerate available tools
  tools/call    — execute a tool
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from typing import Any

from api.auth import require_api_key

router = APIRouter()

# ── MCP tool definitions ──────────────────────────────────────────────────────

_TOOLS = [
    {
        "name": "omnex_ingest",
        "description": (
            "Ingest a file or folder into the Omnex index so it can be searched. "
            "Pass a server-side path. Returns immediately — poll omnex_ingest_status for progress."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute server-side path to a file or folder to ingest",
                },
                "workers": {
                    "type": "integer",
                    "description": "Parallel ingestion workers (default 4)",
                    "default": 4,
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "omnex_ingest_status",
        "description": "Check the status of an ingestion job. Pass the same path used in omnex_ingest.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to check ingestion status for. Omit to get all active jobs.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "omnex_delete_source",
        "description": "Remove all indexed data for a given source path from the Omnex index.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_path": {
                    "type": "string",
                    "description": "The source path to delete all indexed chunks for",
                },
            },
            "required": ["source_path"],
        },
    },
    {
        "name": "omnex_list_indexed",
        "description": "List all source paths that have been indexed, with chunk counts and status.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_type": {
                    "type": "string",
                    "description": "Filter by file type: document, image, video, audio, code",
                    "enum": ["document", "image", "video", "audio", "code"],
                },
            },
            "required": [],
        },
    },
    {
        "name": "omnex_remember",
        "description": (
            "Store a text observation or memory directly into the Omnex index. "
            "Use this to persist information you want to recall later — facts, "
            "decisions, summaries, user preferences, or anything worth remembering. "
            "Requires OMNEX_AGENT_ID to be set in the MCP config headers."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The observation or memory to store",
                },
                "source": {
                    "type": "string",
                    "description": "Label for where this came from (e.g. 'claude', 'cursor', 'research')",
                    "default": "agent",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional key-value metadata to attach",
                },
                "broadcast": {
                    "type": "boolean",
                    "description": "If true, replicate this memory to all active federation peers",
                    "default": False,
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "omnex_search",
        "description": (
            "Search across all indexed personal data — documents, photos, videos, "
            "audio, and code. Returns relevant chunks with scores and metadata."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum results to return (default 10)",
                    "default": 10,
                },
                "file_type": {
                    "type": "string",
                    "description": "Filter by type: document, image, video, audio, code",
                    "enum": ["document", "image", "video", "audio", "code"],
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "omnex_stats",
        "description": "Get statistics about the Omnex index — total chunks, counts by type.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "omnex_search_federated",
        "description": (
            "Search across this Omnex instance AND all registered peer instances simultaneously. "
            "Results from all nodes are merged and ranked by score. "
            "Each result is annotated with origin_instance so you know which node it came from. "
            "Use this when you want to search the full federated memory network."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum results to return per node (default 10)",
                    "default": 10,
                },
                "file_type": {
                    "type": "string",
                    "description": "Filter by type: document, image, video, audio, code",
                    "enum": ["document", "image", "video", "audio", "code"],
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "omnex_list_peers",
        "description": "List all registered peer Omnex instances in the federation.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


# ── Request / Response ────────────────────────────────────────────────────────

class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    id:      Any = None
    method:  str
    params:  dict | None = None


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("", dependencies=[Depends(require_api_key)])
async def mcp_handler(req: MCPRequest, request: Request):
    """MCP JSON-RPC 2.0 endpoint."""

    if req.method == "tools/list":
        return _ok(req.id, {"tools": _TOOLS})

    if req.method == "tools/call":
        return await _handle_tool_call(req, request)

    # MCP initialize handshake
    if req.method == "initialize":
        return _ok(req.id, {
            "protocolVersion": "2024-11-05",
            "capabilities":    {"tools": {}},
            "serverInfo":      {"name": "omnex", "version": "0.1.0"},
        })

    if req.method == "notifications/initialized":
        return _ok(req.id, {})

    return _error(req.id, -32601, f"Method not found: {req.method}")


async def _handle_tool_call(req: MCPRequest, request: Request) -> dict:
    params    = req.params or {}
    tool_name = params.get("name")
    arguments = params.get("arguments", {})

    if tool_name == "omnex_ingest":
        return await _tool_ingest(req.id, arguments)

    if tool_name == "omnex_ingest_status":
        return await _tool_ingest_status(req.id, arguments)

    if tool_name == "omnex_delete_source":
        return await _tool_delete_source(req.id, arguments)

    if tool_name == "omnex_list_indexed":
        return await _tool_list_indexed(req.id, arguments)

    if tool_name == "omnex_remember":
        agent_id = request.headers.get("X-Agent-ID", "")
        return await _tool_remember(req.id, arguments, agent_id)

    if tool_name == "omnex_search":
        return await _tool_search(req.id, arguments)

    if tool_name == "omnex_stats":
        return await _tool_stats(req.id)

    if tool_name == "omnex_search_federated":
        return await _tool_search_federated(req.id, arguments)

    if tool_name == "omnex_list_peers":
        return await _tool_list_peers(req.id)

    return _error(req.id, -32602, f"Unknown tool: {tool_name}")


async def _tool_ingest(req_id: Any, args: dict) -> dict:
    from api.routes.ingest import _run_ingestion
    import threading, os
    path    = args.get("path", "").strip()
    workers = int(args.get("workers", 4))
    if not path:
        return _error(req_id, -32602, "path is required")
    if not os.path.exists(path):
        return _error(req_id, -32602, f"Path does not exist: {path}")
    threading.Thread(target=_run_ingestion, args=(path, workers), daemon=True).start()
    return _ok(req_id, {
        "content": [{"type": "text", "text": f"Ingestion started for: {path}\nPoll omnex_ingest_status to track progress."}],
        "isError": False,
    })


async def _tool_ingest_status(req_id: Any, args: dict) -> dict:
    from storage.mongo import get_db
    db   = get_db()
    path = args.get("path")
    query = {"source_path": path} if path else {}
    records = list(db["ingestion_state"].find(query, {"_id": 0}))
    for r in records:
        for k, v in r.items():
            if hasattr(v, "isoformat"):
                r[k] = v.isoformat()
    if not records:
        text = f"No ingestion record found for: {path}" if path else "No ingestion records found."
    else:
        lines = []
        for r in records:
            lines.append(
                f"{r.get('source_path')} — {r.get('status')} "
                f"({r.get('processed', 0)}/{r.get('total_files', 0)} files, "
                f"{r.get('errors', 0)} errors)"
            )
        text = "\n".join(lines)
    return _ok(req_id, {"content": [{"type": "text", "text": text}], "isError": False})


async def _tool_delete_source(req_id: Any, args: dict) -> dict:
    from storage.mongo import get_db
    from storage.leann_store import delete_vectors
    db          = get_db()
    source_path = args.get("source_path", "").strip()
    if not source_path:
        return _error(req_id, -32602, "source_path is required")
    docs = list(db["chunks"].find({"source_path": source_path}, {"_id": 1}))
    if not docs:
        return _error(req_id, -32602, f"No indexed data found for: {source_path}")
    chunk_ids = [str(d["_id"]) for d in docs]
    try:
        delete_vectors(chunk_ids)
    except Exception:
        pass
    db["chunks"].delete_many({"source_path": source_path})
    db["ingestion_state"].delete_one({"source_path": source_path})
    return _ok(req_id, {
        "content": [{"type": "text", "text": f"Deleted {len(chunk_ids)} chunks for: {source_path}"}],
        "isError": False,
    })


async def _tool_list_indexed(req_id: Any, args: dict) -> dict:
    from storage.mongo import get_db
    db        = get_db()
    file_type = args.get("file_type")
    match     = {"file_type": file_type} if file_type else {}
    pipeline  = [
        {"$match": match},
        {"$group": {"_id": "$source_path", "chunks": {"$sum": 1}, "file_type": {"$first": "$file_type"}}},
        {"$sort": {"chunks": -1}},
        {"$limit": 50},
    ]
    rows  = list(db["chunks"].aggregate(pipeline))
    if not rows:
        text = "No indexed sources found."
    else:
        lines = [f"{r['file_type'].upper():10} {r['chunks']:>5} chunks — {r['_id']}" for r in rows]
        text  = f"{len(rows)} indexed sources:\n" + "\n".join(lines)
    return _ok(req_id, {"content": [{"type": "text", "text": text}], "isError": False})


async def _tool_remember(req_id: Any, args: dict, agent_id: str) -> dict:
    from api.routes.agents import store_observation, ObservationRequest

    text      = args.get("text", "").strip()
    source    = args.get("source", "agent")
    meta      = args.get("metadata", {})
    broadcast = bool(args.get("broadcast", False))

    if not text:
        return _error(req_id, -32602, "text is required")

    if not agent_id:
        return _error(req_id, -32602,
            "X-Agent-ID header is required. Register an agent via POST /agents "
            "and add X-Agent-ID to your MCP config headers.")

    try:
        result = await store_observation(ObservationRequest(
            text=text,
            source=source,
            agent_id=agent_id,
            metadata=meta,
            broadcast=broadcast,
        ))
    except Exception as e:
        return _error(req_id, -32603, str(e))

    broadcast_note = ""
    if broadcast and result.get("broadcast_peers"):
        broadcast_note = f" Replicated to: {', '.join(result['broadcast_peers'])}."

    return _ok(req_id, {
        "content": [{"type": "text", "text": f"Stored: {text[:120]}{'…' if len(text) > 120 else ''}{broadcast_note}"}],
        "isError": False,
    })


async def _tool_search(req_id: Any, args: dict) -> dict:
    from api.query_engine import search

    query     = args.get("query", "")
    top_k     = int(args.get("top_k", 10))
    file_type = args.get("file_type")

    if not query:
        return _error(req_id, -32602, "query is required")

    try:
        response = await search(query=query, top_k=top_k, file_type_filter=file_type)
    except Exception as e:
        return _error(req_id, -32603, str(e))

    results = [
        {
            "chunk_id":    r.chunk_id,
            "score":       round(r.score, 4),
            "file_type":   r.file_type,
            "source_path": r.source_path,
            "text":        r.text or "",
        }
        for r in response.results
    ]

    content = {
        "query":        query,
        "total":        response.total,
        "results":      results,
        "llm_response": response.llm_response,
    }

    return _ok(req_id, {
        "content": [{"type": "text", "text": _format_results(content)}],
        "isError": False,
    })


async def _tool_stats(req_id: Any) -> dict:
    from storage.mongo import get_db
    db = get_db()
    total = db["chunks"].count_documents({})
    by_type = {d["_id"]: d["count"] for d in db["chunks"].aggregate([
        {"$group": {"_id": "$file_type", "count": {"$sum": 1}}}
    ])}
    text = f"Omnex index: {total:,} total chunks\n" + "\n".join(
        f"  {k}: {v:,}" for k, v in sorted(by_type.items())
    )
    return _ok(req_id, {
        "content": [{"type": "text", "text": text}],
        "isError": False,
    })


async def _tool_search_federated(req_id: Any, args: dict) -> dict:
    from api.routes.federation import federated_search, FederatedSearchRequest
    query     = args.get("query", "")
    top_k     = int(args.get("top_k", 10))
    file_type = args.get("file_type")
    if not query:
        return _error(req_id, -32602, "query is required")
    try:
        response = await federated_search(FederatedSearchRequest(
            query=query, top_k=top_k, file_type=file_type,
        ))
    except Exception as e:
        return _error(req_id, -32603, str(e))

    results = [
        {
            "chunk_id":        r.get("chunk_id", ""),
            "score":           round(r.get("score", 0.0), 4),
            "file_type":       r.get("file_type", ""),
            "source_path":     r.get("source_path", ""),
            "text":            r.get("text") or "",
            "origin_instance": r.get("origin_instance", "local"),
        }
        for r in response["results"]
    ]
    content = {
        "query":         query,
        "total":         response["total"],
        "peers_queried": response["peers_queried"],
        "results":       results,
        "llm_response":  response.get("llm_response"),
    }
    return _ok(req_id, {
        "content": [{"type": "text", "text": _format_results(content)}],
        "isError": False,
    })


async def _tool_list_peers(req_id: Any) -> dict:
    from storage.mongo import get_db
    db   = get_db()
    docs = list(db["federation_peers"].find({}, {"_id": 0, "api_key": 0}))
    if not docs:
        text = "No peers registered. Use POST /federation/peers to add one."
    else:
        lines = []
        for p in docs:
            status = "active" if p.get("active") else "inactive"
            last   = p.get("last_seen", "never")
            lines.append(f"[{status}] {p['label']} — {p['url']} (last seen: {last})")
        text = f"{len(docs)} peer(s):\n" + "\n".join(lines)
    return _ok(req_id, {"content": [{"type": "text", "text": text}], "isError": False})


def _format_results(content: dict) -> str:
    lines = [f"Query: {content['query']}", f"Results: {content['total']}", ""]
    for i, r in enumerate(content["results"], 1):
        lines.append(f"[{i}] {r['file_type'].upper()} (score: {r['score']}) — {r['source_path']}")
        if r["text"]:
            lines.append(f"    {r['text']}")
    if content.get("llm_response"):
        lines += ["", "Summary:", content["llm_response"]]
    return "\n".join(lines)


def _ok(req_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _error(req_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}
