"""
Omnex — Query Engine (LangGraph)

Graph:
  START → classify → search_index → expand_doc → answer → END
                   ↘ chat ────────────────────────────────↗

Node responsibilities:
  classify     — LLM decides: SEARCH (+ keywords) or CHAT
  search_index — vector search + mongo fallback + filters
  expand_doc   — if single source, fetch all chunks for full context
  answer       — LLM answers using retrieved context
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
class QueryResponse:
    query:       str
    results:     list[QueryResult]
    total:       int
    llm_response: str | None = None
    suggested_refinements: list[str] = field(default_factory=list)
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
    route:            Literal["search", "chat"]
    search_keywords:  str

    # Search
    intent:           dict
    raw_results:      list[dict]
    results:          list[QueryResult]
    expansion_ids:    set[str]

    # Output
    llm_response:     str | None
    final_results:    list[QueryResult]


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

_CLASSIFY_SYSTEM = """You are a routing agent for Omnex, a personal AI memory system.
Decide if the user's message requires searching their indexed personal files or is casual conversation.

Reply with EXACTLY one of:
SEARCH: <concise search keywords>
CHAT

Use SEARCH for: anything about the user's files, documents, photos, videos, audio, code, work, people, events, memories.
Use CHAT for: greetings, thanks, casual chat, questions about what Omnex can do.

Examples:
  hi → CHAT
  thanks → CHAT
  what can you do? → CHAT
  where do I work? → SEARCH: current employer job title
  who are my references? → SEARCH: references contact details
  show me photos from 2023 → SEARCH: photos 2023
  find my Adobe invoice → SEARCH: invoice Adobe
  what did I work on last year? → SEARCH: projects work 2024"""


async def _node_classify(state: OmnexState) -> OmnexState:
    import logging
    log = logging.getLogger("omnex.graph")
    query = state["query"]

    try:
        raw = await _llm_call(
            messages=[{"role": "user", "content": query}],
            system=_CLASSIFY_SYSTEM,
            max_tokens=64,
        )
        log.info(f"[classify] '{query}' → {raw!r}")

        if raw.upper().startswith("CHAT"):
            return {**state, "route": "chat", "search_keywords": ""}
        if raw.upper().startswith("SEARCH:"):
            keywords = raw[7:].strip() or query
            return {**state, "route": "search", "search_keywords": keywords}
        # Unrecognised — default to search
        return {**state, "route": "search", "search_keywords": query}

    except Exception as e:
        log.warning(f"[classify] failed ({e}), defaulting to search")
        return {**state, "route": "search", "search_keywords": query}


async def _node_search_index(state: OmnexState) -> OmnexState:
    from storage.leann_store import IndexName, search as leann_search
    from storage.mongo import get_chunk_by_id
    from embeddings.tagger import extract_tag_filters

    query            = state["query"]
    embed_query      = state.get("search_keywords") or query
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
        text_vec = embed_text(embed_query).tolist()
        if not intent["code"]:
            raw_results.extend(leann_search(IndexName.TEXT, text_vec, top_k=top_k))

    # Code
    if intent["code"]:
        from embeddings.code import embed as embed_code
        code_vec = embed_code(embed_query).tolist()
        raw_results.extend(leann_search(IndexName.CODE, code_vec, top_k=top_k))

    # Visual
    if intent["visual"]:
        from embeddings.image import embed_text as clip_text
        clip_vec = clip_text(embed_query)
        raw_results.extend(leann_search(IndexName.IMAGE, clip_vec, top_k=top_k))
        raw_results.extend(leann_search(IndexName.VIDEO, clip_vec, top_k=top_k))

    # Audio
    audio_kw = {"audio", "podcast", "recording", "transcript", "said", "heard",
                "spoke", "call", "meeting", "interview", "voice"}
    if not raw_results or bool(set(query.lower().split()) & audio_kw):
        if "text_vec" not in dir():
            from embeddings.text import embed as embed_text
            text_vec = embed_text(embed_query).tolist()
        raw_results.extend(leann_search(IndexName.AUDIO, text_vec, top_k=top_k))

    # Visual fallback
    if not raw_results:
        from embeddings.image import embed_text as clip_text
        clip_vec = clip_text(embed_query)
        raw_results.extend(leann_search(IndexName.IMAGE, clip_vec, top_k=top_k))

    # Filter + fetch docs
    results: list[QueryResult] = []
    seen: set[str] = set()
    SCORE_THRESHOLD = 0.15

    for hit in sorted(raw_results, key=lambda x: x["score"], reverse=True):
        if hit.get("score", 0) < SCORE_THRESHOLD:
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


_ANSWER_SYSTEM = (
    "You are Omnex, a personal AI memory system. "
    "Search results from the user's indexed personal files are provided below. "
    "NEVER invent, fabricate, or guess any names, dates, facts, or details. "
    "ONLY state information that appears in the provided search results. "
    "If a detail is not in the results, say it was not found. "
    "NEVER say you lack access to the user's data — you have already searched it. "
    "NEVER ask the user to share files — you already have the search results. "
    "ALWAYS begin your reply with exactly:\n"
    "RELEVANT_IDS: [\"id1\", \"id2\"]\n"
    "List only the IDs (from the ID: field) of results that are relevant to the query. "
    "Then write a concise response using only facts from those results. "
    "If none match: RELEVANT_IDS: [] and say you found nothing. "
    "Do not use markdown, bullet points, or bold text. Write in plain sentences."
)

_CHAT_SYSTEM = (
    "You are Omnex, a personal AI memory system and assistant. "
    "The user is having a casual conversation — no data search is needed. "
    "Respond naturally and helpfully. Keep replies short. "
    "If asked what you can do: you index and search the user's personal files "
    "(documents, photos, videos, audio, code) so they can find anything in plain language."
)


async def _node_answer(state: OmnexState) -> OmnexState:
    """LLM answers using retrieved search results."""
    import logging
    log = logging.getLogger("omnex.graph")

    results      = state.get("results", [])
    expansion_ids = state.get("expansion_ids", set())
    query        = state["query"]
    history      = state.get("history", [])

    context  = _build_context(results)
    messages = _dedup_history(history)

    user_content = f'User request: "{query}"'
    if context:
        user_content += (
            f"\n\nRetrieved from the user's personal data index:\n\n{context}\n\n"
            f"Select only the items relevant to the request. "
            f"Start with RELEVANT_IDS: [...] using the ID values shown, then respond."
        )
    else:
        user_content += (
            "\n\nNo items were retrieved. Tell the user you searched but found nothing. "
            "Suggest they may not have indexed that content yet, or try different wording."
        )
    messages.append({"role": "user", "content": user_content})

    try:
        raw = await _llm_call(messages, system=_ANSWER_SYSTEM, max_tokens=1024)
        log.info(f"[answer] raw: {raw[:300]}")
        kept_ids, response_text = _parse_llm_response(raw, results)

        # When full-doc expansion happened, don't filter by LLM IDs — trust full doc
        if expansion_ids:
            final_results = results
        else:
            id_set = set(kept_ids)
            final_results = [r for r in results if r.chunk_id in id_set] if kept_ids else results

        return {**state, "llm_response": response_text, "final_results": final_results}

    except Exception as e:
        log.warning(f"[answer] LLM failed: {e}")
        return {**state, "llm_response": None, "final_results": results}


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

    return {**state, "llm_response": response, "final_results": []}


# ── Routing edge ──────────────────────────────────────────────────────────────

def _route_after_classify(state: OmnexState) -> Literal["search_index", "chat"]:
    return "search_index" if state.get("route") == "search" else "chat"


# ── Build the graph ───────────────────────────────────────────────────────────

from langgraph.graph import StateGraph, END, START

def _build_graph():
    g = StateGraph(OmnexState)

    g.add_node("classify",     _node_classify)
    g.add_node("search_index", _node_search_index)
    g.add_node("expand_doc",   _node_expand_doc)
    g.add_node("answer",       _node_answer)
    g.add_node("chat",         _node_chat)

    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", _route_after_classify, {
        "search_index": "search_index",
        "chat":         "chat",
    })
    g.add_edge("search_index", "expand_doc")
    g.add_edge("expand_doc",   "answer")
    g.add_edge("answer",       END)
    g.add_edge("chat",         END)

    return g.compile()


_graph = _build_graph()


# ── Public API — same signature as before ─────────────────────────────────────

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
    }

    result = await _graph.ainvoke(initial)

    final_results = result.get("final_results") or []
    intent        = result.get("intent") or {}

    return QueryResponse(
        query=query,
        results=final_results,
        total=len(final_results),
        llm_response=result.get("llm_response"),
        suggested_refinements=_suggest_refinements(query, intent, final_results),
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


def _parse_llm_response(raw: str, candidates: list[QueryResult]) -> tuple[list[str], str]:
    import json
    all_ids = [r.chunk_id for r in candidates]
    match = re.search(r'RELEVANT_IDS:\s*(\[.*?\])', raw, re.DOTALL)
    if match:
        try:
            kept  = json.loads(match.group(1))
            prose = raw[match.end():].strip()
            valid = [i for i in kept if i in set(all_ids)]
            return valid if valid else all_ids, prose or raw
        except Exception:
            pass
    return all_ids, raw


def _suggest_refinements(query: str, intent: dict, results: list[QueryResult]) -> list[str]:
    suggestions = []
    if results:
        types = {r.file_type for r in results}
        if "image" in types:
            suggestions.append("Show only photos")
        if "document" in types:
            suggestions.append("Show only documents")
        if len(results) > 3:
            suggestions.append("Show more like the first result")
    if not intent.get("temporal"):
        suggestions.append("Filter by date range")
    return suggestions[:3]


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
