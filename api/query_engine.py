"""
Omnex — Query Engine (LangGraph)

Graph:
  START → search_index → score_route → expand_doc → answer → END
                                     ↘ chat ──────────────────↗

Node responsibilities:
  search_index — always runs: vector search + mongo fallback + filters
  score_route  — pure function: if best score >= threshold → answer, else → chat
  expand_doc   — if single source, fetch all chunks for full context
  answer       — LLM answers using retrieved context, returns structured filters
  chat         — LLM answers conversationally (no index access)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    chunk_id:    str
    score:       float
    file_type:   str
    source_path: str
    text:        str | None
    metadata:    dict
    thumbnail_url: str | None = None


@dataclass
class ApplicableFilter:
    """A filter the user can apply — only emitted when the LLM knows it's valid."""
    label:      str   # Display text, e.g. "Only documents"
    query:      str   # The follow-up query to send when clicked
    file_type:  str | None = None
    date_from:  str | None = None
    date_to:    str | None = None


@dataclass
class QueryResponse:
    query:       str
    results:     list[QueryResult]
    total:       int
    llm_response: str | None = None
    applicable_filters: list[ApplicableFilter] = field(default_factory=list)
    session_id:  str | None = None


# ── Graph state ───────────────────────────────────────────────────────────────

from typing import TypedDict

class OmnexState(TypedDict, total=False):
    # Inputs
    query:            str
    top_k:            int
    file_type_filter: str | None
    date_from:        datetime | None
    date_to:          datetime | None
    session_id:       str | None
    history:          list[dict]

    # Routing
    route:            Literal["answer", "chat"]

    # Search
    intent:           dict
    raw_results:      list[dict]
    results:          list[QueryResult]
    expansion_ids:    set[str]

    # Output
    llm_response:     str | None
    final_results:    list[QueryResult]
    applicable_filters: list[ApplicableFilter]


# ── Intent detection ──────────────────────────────────────────────────────────

VISUAL_KEYWORDS = {
    "photo", "photos", "picture", "pictures", "image", "images",
    "screenshot", "selfie", "video", "videos", "clip", "footage",
    "show me", "find photos", "find images",
}

TEMPORAL_PATTERNS = [
    r"\b(\d{4})\b",
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
    r"\b(last|this|next)\s+(week|month|year)\b",
    r"\b(\d+)\s+(days?|weeks?|months?|years?)\s+ago\b",
    r"\brecently\b|\blast\s+\w+\b",
]

CODE_KEYWORDS = {"code", "function", "class", "script", "module", "implementation", "def", "import"}

_DEVICE_PATTERNS = re.compile(
    r'\b(iphone|ipad|samsung|galaxy|pixel|huawei|sony|canon|nikon|fuji(?:film)?|gopro|oneplus|xiaomi|android)\b',
    re.IGNORECASE,
)

_EXT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\bpdf[s]?\b', re.I),             "document"),
    (re.compile(r'\bspreadsheet[s]?\b', re.I),      "document"),
    (re.compile(r'\bword\s+doc[s]?\b', re.I),       "document"),
    (re.compile(r'\bpresentation[s]?\b', re.I),     "document"),
    (re.compile(r'\bmp3[s]?\b', re.I),              "audio"),
    (re.compile(r'\bpodcast[s]?\b', re.I),          "audio"),
    (re.compile(r'\bvideo[s]?\b|\bclip[s]?\b', re.I), "video"),
    (re.compile(r'\bphoto[s]?\b|\bpicture[s]?\b|\bimage[s]?\b', re.I), "image"),
    (re.compile(r'\bscreenshot[s]?\b', re.I),       "image"),
    (re.compile(r'\bcode\b|\bscript[s]?\b', re.I), "code"),
]


def detect_intent(query: str) -> dict:
    q = query.lower()
    words = set(q.split())
    is_visual   = bool(words & VISUAL_KEYWORDS)
    is_temporal = any(re.search(p, q) for p in TEMPORAL_PATTERNS)
    is_code     = bool(words & CODE_KEYWORDS)
    is_location = bool({"in", "at", "from", "near", "around"} & words)
    person_name = _extract_person_name(q)
    device_match = _DEVICE_PATTERNS.search(query)
    device_hint  = device_match.group(1).lower() if device_match else None
    file_type_hint = None
    for pattern, ft in _EXT_PATTERNS:
        if pattern.search(query):
            file_type_hint = ft
            break
    return {
        "visual": is_visual, "temporal": is_temporal, "code": is_code,
        "location": is_location, "person_name": person_name,
        "device_hint": device_hint, "file_type_hint": file_type_hint,
    }


def _extract_person_name(query: str) -> str | None:
    patterns = [
        r'\bwith\s+([A-Z][a-z]+)\b',
        r'\bof\s+(?:my\s+)?([A-Z][a-z]+)\b',
        r'\bshow\s+(?:me\s+)?([A-Z][a-z]+)\b',
        r'\bfind\s+([A-Z][a-z]+)\b',
    ]
    skip = {"photos", "pictures", "images", "video", "files", "documents", "code", "all", "my", "the"}
    for p in patterns:
        m = re.search(p, query, re.IGNORECASE)
        if m and m.group(1).lower() not in skip:
            return m.group(1)
    return None


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _get_provider() -> str:
    import os
    return os.getenv("LLM_PROVIDER", "local").lower()


async def _llm_call(messages: list[dict], system: str, max_tokens: int = 1024) -> str:
    """Unified LLM call — routes to Anthropic, OpenAI, or local Ollama."""
    import os, httpx
    provider = _get_provider()

    if provider == "anthropic":
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        msg = await client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return msg.content[0].text.strip()

    if provider == "openai":
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = await client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}] + messages,
        )
        return resp.choices[0].message.content.strip()

    # Local Ollama
    host  = os.getenv("LOCAL_LLM_HOST", "http://localhost:11434")
    model = os.getenv("LOCAL_LLM_MODEL", "phi3:mini")
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "system", "content": system}] + messages,
                "stream": False,
            },
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()


def _dedup_history(history: list[dict]) -> list[dict]:
    messages: list[dict] = []
    for turn in history:
        role    = turn.get("role", "user")
        content = (turn.get("content") or "").strip()
        if not content or role not in ("user", "assistant"):
            continue
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] += "\n" + content
        else:
            messages.append({"role": role, "content": content})
    return messages


# ── Graph nodes ───────────────────────────────────────────────────────────────

# Score threshold: results above this are considered relevant
_SCORE_THRESHOLD = 0.20


async def _node_search_index(state: OmnexState) -> OmnexState:
    from storage.leann_store import IndexName, search as leann_search
    from storage.mongo import get_chunk_by_id
    from embeddings.tagger import extract_tag_filters

    query            = state["query"]
    top_k            = state.get("top_k", 8)
    file_type_filter = state.get("file_type_filter")
    date_from        = state.get("date_from")
    date_to          = state.get("date_to")

    intent = detect_intent(query)
    tag_filters = extract_tag_filters(query)

    if not file_type_filter and intent.get("file_type_hint"):
        file_type_filter = intent["file_type_hint"]

    raw_results: list[dict] = []

    # Person cluster
    person_chunk_ids: set[str] | None = None
    if intent["person_name"]:
        person_chunk_ids = await _resolve_person_chunks(intent["person_name"])

    # Text / document
    if not intent["visual"] or intent["code"]:
        from embeddings.text import embed as embed_text
        text_vec = embed_text(query).tolist()
        if not intent["code"]:
            raw_results.extend(leann_search(IndexName.TEXT, text_vec, top_k=top_k))

    # Code
    if intent["code"]:
        from embeddings.code import embed as embed_code
        code_vec = embed_code(query).tolist()
        raw_results.extend(leann_search(IndexName.CODE, code_vec, top_k=top_k))

    # Visual
    if intent["visual"]:
        from embeddings.image import embed_text as clip_text
        clip_vec = clip_text(query)
        raw_results.extend(leann_search(IndexName.IMAGE, clip_vec, top_k=top_k))
        raw_results.extend(leann_search(IndexName.VIDEO, clip_vec, top_k=top_k))

    # Audio
    audio_kw = {"audio", "podcast", "recording", "transcript", "said", "heard",
                "spoke", "call", "meeting", "interview", "voice"}
    if not raw_results or bool(set(query.lower().split()) & audio_kw):
        if "text_vec" not in dir():
            from embeddings.text import embed as embed_text
            text_vec = embed_text(query).tolist()
        raw_results.extend(leann_search(IndexName.AUDIO, text_vec, top_k=top_k))

    # Visual fallback
    if not raw_results:
        from embeddings.image import embed_text as clip_text
        clip_vec = clip_text(query)
        raw_results.extend(leann_search(IndexName.IMAGE, clip_vec, top_k=top_k))

    # Filter + fetch docs
    results: list[QueryResult] = []
    seen: set[str] = set()

    for hit in sorted(raw_results, key=lambda x: x["score"], reverse=True):
        if hit.get("score", 0) < 0.05:  # very low bar — score_route decides routing
            break
        chunk_id = hit.get("chunk_id")
        if not chunk_id or chunk_id in seen:
            continue
        if person_chunk_ids is not None and chunk_id not in person_chunk_ids:
            continue
        seen.add(chunk_id)

        doc = get_chunk_by_id(chunk_id)
        if not doc:
            continue
        if file_type_filter and doc.get("file_type") != file_type_filter:
            continue
        if tag_filters:
            chunk_tags = set(doc.get("tags", []))
            if not all(t in chunk_tags for t in tag_filters):
                continue

        meta = doc.get("metadata", {})
        if date_from or date_to:
            doc_date = meta.get("created_at") or meta.get("modified_at")
            if doc_date:
                if date_from and doc_date < date_from:
                    continue
                if date_to and doc_date > date_to:
                    continue

        if intent.get("device_hint"):
            if intent["device_hint"] not in (meta.get("device") or "").lower():
                continue

        cid = str(doc["_id"])
        results.append(QueryResult(
            chunk_id=cid,
            score=hit["score"],
            file_type=doc.get("file_type", "unknown"),
            source_path=doc.get("source_path", ""),
            text=doc.get("text_content"),
            metadata=meta,
            thumbnail_url=f"/chunk/{cid}/thumbnail" if doc.get("file_type") == "image" else None,
        ))
        if len(results) >= top_k:
            break

    # MongoDB fallback
    if not results:
        results = await _mongo_fallback(
            query=query, intent=intent,
            file_type_filter=file_type_filter,
            date_from=date_from, date_to=date_to,
            top_k=top_k, device_hint=intent.get("device_hint"),
        )

    return {**state, "intent": intent, "results": results}


async def _node_expand_doc(state: OmnexState) -> OmnexState:
    """If all results come from a single source, fetch all its chunks for full context."""
    results = state.get("results", [])
    if not results:
        return {**state, "expansion_ids": set()}

    source_paths = {r.source_path for r in results}
    if len(source_paths) != 1:
        return {**state, "expansion_ids": set()}

    from storage.mongo import get_db
    db = get_db()
    source_path = next(iter(source_paths))
    all_docs = list(db["chunks"].find(
        {"source_path": source_path},
        {"_id": 1, "file_type": 1, "source_path": 1, "text_content": 1, "metadata": 1}
    ).sort("chunk_index", 1))

    seen = {r.chunk_id for r in results}
    expansion_ids: set[str] = set()
    for doc in all_docs:
        cid = str(doc["_id"])
        if cid not in seen:
            results.append(QueryResult(
                chunk_id=cid,
                score=0.0,
                file_type=doc.get("file_type", "unknown"),
                source_path=doc.get("source_path", ""),
                text=doc.get("text_content"),
                metadata=doc.get("metadata", {}),
            ))
            seen.add(cid)
            expansion_ids.add(cid)

    return {**state, "results": results, "expansion_ids": expansion_ids}


_EXPRESSION_GUIDE = (
    "You can use paralinguistic expression tags to make your voice responses more natural and expressive. "
    "Use them sparingly and only where they genuinely fit the tone. "
    "Available tags: [laugh] [chuckle] [sigh] [gasp] [cough] [clear throat] [sniff] [groan] [shush] "
    "Example: 'Let me check that for you. [clear throat] Here is what I found.' "
    "Example: 'I could not find anything matching that. [sigh] Try rephrasing your search.' "
    "Never overuse tags — one per response at most unless the content genuinely calls for more."
)

_ANSWER_SYSTEM = """You are Omnex, a personal AI memory system.
Search results from the user's indexed personal files are provided below.
NEVER invent, fabricate, or guess any names, dates, facts, or details.
ONLY state information that appears in the provided search results.
If a detail is not in the results, say it was not found.
NEVER say you lack access to the user's data — you have already searched it.
NEVER ask the user to share files — you already have the search results.

Respond in this EXACT format (no deviation):

RELEVANT_IDS: ["id1", "id2"]
RESPONSE: Your answer here in plain sentences. No markdown, bullets, or bold.
FILTERS: [{"label": "Only documents", "query": "<original query> documents only", "file_type": "document"}, ...]

Rules for FILTERS:
- Only include a filter if the retrieved results ACTUALLY CONTAIN that file type or date range.
- If results are all documents, offer a "Only documents" filter with file_type="document".
- If results contain multiple file types, offer one filter per type present.
- If results span multiple years/months, offer a date filter for the most relevant period.
- If no useful filter applies, output FILTERS: []
- Never invent filters for data that isn't in the results.

""" + _EXPRESSION_GUIDE

_CHAT_SYSTEM = (
    "You are Omnex, a personal AI memory system and assistant. "
    "The user is having a casual conversation — no data search is needed. "
    "Respond naturally and helpfully. Keep replies short. "
    "If asked what you can do: you index and search the user's personal files "
    "(documents, photos, videos, audio, code) so they can find anything in plain language.\n\n"
    + _EXPRESSION_GUIDE
)


async def _node_answer(state: OmnexState) -> OmnexState:
    """LLM answers using retrieved search results."""
    import logging, json
    log = logging.getLogger("omnex.graph")

    results       = state.get("results", [])
    expansion_ids = state.get("expansion_ids", set())
    query         = state["query"]
    history       = state.get("history", [])

    context  = _build_context(results)
    messages = _dedup_history(history)

    user_content = f'User request: "{query}"'
    if context:
        user_content += (
            f"\n\nRetrieved from the user's personal data index:\n\n{context}\n\n"
            f"Select only the items relevant to the request. "
            f"Start with RELEVANT_IDS: [...] then RESPONSE: then FILTERS: [...]"
        )
    else:
        user_content += (
            "\n\nNo items were retrieved. Tell the user you searched but found nothing. "
            "Suggest they may not have indexed that content yet, or try different wording. "
            "Output: RELEVANT_IDS: []\nRESPONSE: <message>\nFILTERS: []"
        )
    messages.append({"role": "user", "content": user_content})

    try:
        raw = await _llm_call(messages, system=_ANSWER_SYSTEM, max_tokens=1024)
        log.info(f"[answer] raw: {raw[:400]}")
        kept_ids, response_text, applicable_filters = _parse_llm_response(raw, results)

        if expansion_ids:
            final_results = results
        else:
            id_set = set(kept_ids)
            final_results = [r for r in results if r.chunk_id in id_set] if kept_ids else results

        return {**state, "llm_response": response_text, "final_results": final_results,
                "applicable_filters": applicable_filters}

    except Exception as e:
        log.warning(f"[answer] LLM failed: {e}")
        return {**state, "llm_response": None, "final_results": results,
                "applicable_filters": []}


async def _node_chat(state: OmnexState) -> OmnexState:
    """Pure conversational reply — no index access."""
    import logging
    log = logging.getLogger("omnex.graph")

    query   = state["query"]
    history = state.get("history", [])

    messages = _dedup_history(history)
    messages.append({"role": "user", "content": query})

    try:
        response = await _llm_call(messages, system=_CHAT_SYSTEM, max_tokens=256)
        log.info(f"[chat] response: {response[:200]}")
    except Exception as e:
        log.warning(f"[chat] LLM failed: {e}")
        response = "Hey! Ask me anything about your files and data."

    return {**state, "llm_response": response, "final_results": [], "applicable_filters": []}


# ── Score-based routing edge ───────────────────────────────────────────────────

def _route_after_search(state: OmnexState) -> Literal["expand_doc", "chat"]:
    import logging
    log = logging.getLogger("omnex.graph")
    results = state.get("results", [])
    if not results:
        log.info("[score_route] no results → chat")
        return "chat"
    best_score = max((r.score for r in results), default=0.0)
    route = "expand_doc" if best_score >= _SCORE_THRESHOLD else "chat"
    log.info(f"[score_route] best_score={best_score:.3f} → {route}")
    return route


# ── Build the graph ───────────────────────────────────────────────────────────

from langgraph.graph import StateGraph, END, START

def _build_graph():
    g = StateGraph(OmnexState)

    g.add_node("search_index", _node_search_index)
    g.add_node("expand_doc",   _node_expand_doc)
    g.add_node("answer",       _node_answer)
    g.add_node("chat",         _node_chat)

    g.add_edge(START, "search_index")
    g.add_conditional_edges("search_index", _route_after_search, {
        "expand_doc": "expand_doc",
        "chat":       "chat",
    })
    g.add_edge("expand_doc", "answer")
    g.add_edge("answer",     END)
    g.add_edge("chat",       END)

    return g.compile()


_graph = _build_graph()


# ── Public API ─────────────────────────────────────────────────────────────────

async def search(
    query: str,
    top_k: int = 8,
    file_type_filter: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    session_id: str | None = None,
    history: list[dict] | None = None,
) -> QueryResponse:

    initial: OmnexState = {
        "query":            query,
        "top_k":            top_k,
        "file_type_filter": file_type_filter,
        "date_from":        date_from,
        "date_to":          date_to,
        "session_id":       session_id,
        "history":          history or [],
        "results":          [],
        "expansion_ids":    set(),
        "final_results":    [],
        "applicable_filters": [],
    }

    result = await _graph.ainvoke(initial)

    final_results      = result.get("final_results") or []
    applicable_filters = result.get("applicable_filters") or []

    return QueryResponse(
        query=query,
        results=final_results,
        total=len(final_results),
        llm_response=result.get("llm_response"),
        applicable_filters=applicable_filters,
        session_id=session_id,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_context(results: list[QueryResult]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        meta = r.metadata
        date   = meta.get("exif_datetime") or meta.get("created_at") or meta.get("modified_at") or ""
        device = meta.get("device", "")
        gps    = meta.get("gps")
        lang   = meta.get("language", "")
        parts.append(
            f"[{i}] ID:{r.chunk_id} | {r.file_type.upper()} | {r.source_path}"
            + (f" | {date}" if date else "")
            + (f" | device:{device}" if device else "")
            + (f" | gps:{gps['lat']:.4f},{gps['lng']:.4f}" if gps else "")
            + (f" | lang:{lang}" if lang else "")
            + f"\n{r.text or ''}"
        )
    return "\n\n".join(parts)


def _parse_llm_response(
    raw: str, candidates: list[QueryResult]
) -> tuple[list[str], str, list[ApplicableFilter]]:
    import json
    all_ids = [r.chunk_id for r in candidates]

    # Extract RELEVANT_IDS
    kept_ids = all_ids
    id_match = re.search(r'RELEVANT_IDS:\s*(\[.*?\])', raw, re.DOTALL)
    if id_match:
        try:
            parsed = json.loads(id_match.group(1))
            valid = [i for i in parsed if i in set(all_ids)]
            if valid:
                kept_ids = valid
        except Exception:
            pass

    # Extract RESPONSE
    response_text = raw
    resp_match = re.search(r'RESPONSE:\s*(.*?)(?=\nFILTERS:|$)', raw, re.DOTALL)
    if resp_match:
        response_text = resp_match.group(1).strip()

    # Extract FILTERS
    applicable_filters: list[ApplicableFilter] = []
    filters_match = re.search(r'FILTERS:\s*(\[.*?\])', raw, re.DOTALL)
    if filters_match:
        try:
            raw_filters = json.loads(filters_match.group(1))
            for f in raw_filters:
                if isinstance(f, dict) and f.get("label") and f.get("query"):
                    applicable_filters.append(ApplicableFilter(
                        label=f["label"],
                        query=f["query"],
                        file_type=f.get("file_type"),
                        date_from=f.get("date_from"),
                        date_to=f.get("date_to"),
                    ))
        except Exception:
            pass

    return kept_ids, response_text or raw, applicable_filters


# ── MongoDB broad fallback ─────────────────────────────────────────────────────

async def _mongo_fallback(
    query: str,
    intent: dict,
    file_type_filter: str | None,
    date_from: datetime | None,
    date_to: datetime | None,
    top_k: int,
    device_hint: str | None = None,
) -> list[QueryResult]:
    from storage.mongo import get_db

    db = get_db()
    q: dict = {"chunk_index": 0}

    if file_type_filter:
        q["file_type"] = file_type_filter
    if device_hint:
        q["metadata.device"] = {"$regex": device_hint, "$options": "i"}

    now = datetime.now(timezone.utc)
    ql  = query.lower()

    if not date_from and not date_to:
        if re.search(r"\b(today)\b", ql):
            from datetime import timedelta
            date_from = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif re.search(r"\b(yesterday)\b", ql):
            from datetime import timedelta
            date_from = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            date_to   = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif re.search(r"\b(this week|last week|recently|lately)\b", ql):
            from datetime import timedelta
            date_from = now - timedelta(days=7)
        elif re.search(r"\b(this month|last month)\b", ql):
            from datetime import timedelta
            date_from = now - timedelta(days=30)
        elif re.search(r"\b(this year|last year)\b", ql):
            from datetime import timedelta
            date_from = now - timedelta(days=365)
        elif re.search(r"\b(\d+)\s+days?\s+ago\b", ql):
            m = re.search(r"\b(\d+)\s+days?\s+ago\b", ql)
            from datetime import timedelta
            date_from = now - timedelta(days=int(m.group(1)))
        elif re.search(r"\b(\d+)\s+weeks?\s+ago\b", ql):
            m = re.search(r"\b(\d+)\s+weeks?\s+ago\b", ql)
            from datetime import timedelta
            date_from = now - timedelta(weeks=int(m.group(1)))

    if date_from:
        q.setdefault("created_at", {})["$gte"] = date_from
    if date_to:
        q.setdefault("created_at", {})["$lte"] = date_to

    from embeddings.tagger import extract_tag_filters
    specific_tags = [t for t in extract_tag_filters(query) if not t.startswith("type-")]
    if specific_tags:
        q["tags"] = {"$all": specific_tags}

    stop = {"what", "did", "i", "do", "work", "on", "the", "a", "an", "my", "me",
            "show", "find", "list", "get", "have", "has", "is", "are", "was",
            "recently", "lately", "today", "yesterday", "last", "this", "week",
            "month", "year", "any", "all", "some", "about", "tell", "give",
            "photos", "pictures", "images", "files", "documents", "videos",
            "audio", "from", "with", "that", "those", "these", "for", "and",
            "january", "february", "march", "april", "may", "june", "july",
            "august", "september", "october", "november", "december",
            "iphone", "samsung", "pixel", "android", "canon", "nikon"}
    keywords = [w for w in re.findall(r"[a-z]+", ql) if len(w) > 2 and w not in stop]

    if keywords:
        try:
            q["$text"] = {"$search": " ".join(keywords)}
            docs = list(db["chunks"].find(q, {"score": {"$meta": "textScore"}})
                        .sort([("score", {"$meta": "textScore"})]).limit(top_k))
        except Exception:
            q.pop("$text", None)
            q["text_content"] = {"$regex": keywords[0], "$options": "i"}
            docs = list(db["chunks"].find(q).sort("created_at", -1).limit(top_k))
    else:
        docs = list(db["chunks"].find(q).sort("created_at", -1).limit(top_k))

    results = []
    for doc in docs:
        cid  = str(doc["_id"])
        meta = doc.get("metadata", {})
        results.append(QueryResult(
            chunk_id=cid,
            score=0.0,
            file_type=doc.get("file_type", "unknown"),
            source_path=doc.get("source_path", ""),
            text=doc.get("text_content"),
            metadata=meta,
            thumbnail_url=f"/chunk/{cid}/thumbnail" if doc.get("file_type") == "image" else None,
        ))
    return results


# ── Person resolution ──────────────────────────────────────────────────────────

async def _resolve_person_chunks(name: str) -> set[str]:
    from storage.mongo import get_db
    db = get_db()
    pattern  = re.compile(re.escape(name), re.IGNORECASE)
    identity = db["identities"].find_one({"label": {"$regex": pattern}})
    if not identity:
        return set()
    cluster_id = identity["cluster_id"]
    face_docs  = db["face_embeddings"].find({"cluster_id": cluster_id}, {"chunk_id": 1})
    return {doc["chunk_id"] for doc in face_docs}
