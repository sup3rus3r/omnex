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

    if tool_name == "omnex_remember":
        agent_id = request.headers.get("X-Agent-ID", "")
        return await _tool_remember(req.id, arguments, agent_id)

    if tool_name == "omnex_search":
        return await _tool_search(req.id, arguments)

    if tool_name == "omnex_stats":
        return await _tool_stats(req.id)

    return _error(req.id, -32602, f"Unknown tool: {tool_name}")


async def _tool_remember(req_id: Any, args: dict, agent_id: str) -> dict:
    from api.routes.agents import store_observation, ObservationRequest

    text   = args.get("text", "").strip()
    source = args.get("source", "agent")
    meta   = args.get("metadata", {})

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
        ))
    except Exception as e:
        return _error(req_id, -32603, str(e))

    return _ok(req_id, {
        "content": [{"type": "text", "text": f"Stored: {text[:120]}{'…' if len(text) > 120 else ''}"}],
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
            "text":        (r.text or "")[:500],
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


def _format_results(content: dict) -> str:
    lines = [f"Query: {content['query']}", f"Results: {content['total']}", ""]
    for i, r in enumerate(content["results"], 1):
        lines.append(f"[{i}] {r['file_type'].upper()} (score: {r['score']}) — {r['source_path']}")
        if r["text"]:
            lines.append(f"    {r['text'][:200]}")
    if content.get("llm_response"):
        lines += ["", "Summary:", content["llm_response"]]
    return "\n".join(lines)


def _ok(req_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _error(req_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}
