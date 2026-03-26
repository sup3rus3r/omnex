"""
Omnex — Query Engine
Translates natural language queries into multi-index LEANN searches.

Flow:
  1. Parse intent — detect if query is text, visual, person, temporal, location, code
  2. Embed query — MiniLM for text/code, CLIP text encoder for visual
  3. Multi-index search — run against relevant LEANN indexes in parallel
  4. Apply metadata filters — date range, file type, tags, GPS
  5. Merge + rank results — deduplicate, score by relevance
  6. Assemble context — top-N chunks passed to LLM
  7. Return structured results
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


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


# ── Intent detection ──────────────────────────────────────────────────────────

PERSON_KEYWORDS  = {"with", "photo of", "photos of", "pictures of", "show me"}
VISUAL_KEYWORDS = {
    "photo", "photos", "picture", "pictures", "image", "images",
    "screenshot", "selfie", "video", "videos", "clip", "footage",
    "show me", "find photos", "find images",
}

TEMPORAL_PATTERNS = [
    r"\b(\d{4})\b",                          # year: 2022
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
    r"\b(last|this|next)\s+(week|month|year)\b",
    r"\b(\d+)\s+(days?|weeks?|months?|years?)\s+ago\b",
    r"\brecently\b|\blast\s+\w+\b",
]

LOCATION_KEYWORDS = {"in", "at", "from", "near", "around"}
CODE_KEYWORDS = {"code", "function", "class", "script", "module", "implementation", "def", "import"}


_CONVERSATIONAL = {
    "hi", "hello", "hey", "sup", "yo", "howdy",
    "thanks", "thank you", "cheers", "ok", "okay",
    "bye", "goodbye", "cya", "later",
    "how are you", "what's up", "whats up",
    "good morning", "good afternoon", "good evening",
    "who are you", "what are you", "what can you do",
}

def is_conversational(query: str) -> bool:
    """Returns True for short greetings/chitchat that shouldn't trigger a search."""
    q = query.strip().lower().rstrip("!?.")
    if q in _CONVERSATIONAL:
        return True
    # Short (≤4 words) with no content words
    words = q.split()
    if len(words) <= 2 and not any(w in words for w in ("show", "find", "search", "what", "where", "when", "who", "how", "list", "get")):
        return True
    return False


def detect_intent(query: str) -> dict:
    q = query.lower()
    words = set(q.split())

    is_visual   = bool(words & VISUAL_KEYWORDS)
    is_temporal = any(re.search(p, q) for p in TEMPORAL_PATTERNS)
    is_code     = bool(words & CODE_KEYWORDS)
    is_location = bool(words & LOCATION_KEYWORDS)

    # Person detection — "photos with Sarah", "show me pictures of Dad"
    person_name = _extract_person_name(q)

    return {
        "visual":      is_visual,
        "temporal":    is_temporal,
        "code":        is_code,
        "location":    is_location,
        "person_name": person_name,
    }


def _extract_person_name(query: str) -> str | None:
    """
    Extract a person name from queries like:
    - "photos with Sarah"
    - "find pictures of my sister Sarah"
    - "show me Dad"
    Returns the likely name token or None.
    """
    patterns = [
        r'\bwith\s+([A-Z][a-z]+)\b',
        r'\bof\s+(?:my\s+)?([A-Z][a-z]+)\b',
        r'\bshow\s+(?:me\s+)?([A-Z][a-z]+)\b',
        r'\bfind\s+([A-Z][a-z]+)\b',
    ]
    for p in patterns:
        m = re.search(p, query, re.IGNORECASE)
        if m:
            name = m.group(1)
            # Skip common non-name words
            if name.lower() not in {"photos", "pictures", "images", "video", "files", "documents", "code", "all", "my", "the"}:
                return name
    return None


# ── Main search ───────────────────────────────────────────────────────────────

async def search(
    query: str,
    top_k: int = 8,
    file_type_filter: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    session_id: str | None = None,
    history: list[dict] | None = None,
) -> QueryResponse:
    """
    Execute a natural language query against the Omnex index.
    """
    from storage.leann_store import IndexName, search as leann_search
    from storage.mongo import get_chunk_by_id

    from embeddings.tagger import extract_tag_filters

    # Skip search entirely for greetings / chitchat
    if is_conversational(query):
        llm_response = await _ask_llm(query, [], history=history or [])
        return QueryResponse(
            query=query, results=[], total=0,
            llm_response=llm_response,
            suggested_refinements=[],
            session_id=session_id,
        )

    intent = detect_intent(query)
    tag_filters = extract_tag_filters(query)  # e.g. ["year-2023", "topic-travel"]
    raw_results: list[dict] = []

    # Person-based search — resolve name to cluster, filter image chunks
    person_chunk_ids: set[str] | None = None
    if intent["person_name"]:
        person_chunk_ids = await _resolve_person_chunks(intent["person_name"])

    # Text / document search (MiniLM)
    if not intent["visual"] or intent["code"]:
        from embeddings.text import embed as embed_text
        text_vec = embed_text(query).tolist()

        if not intent["code"]:
            hits = leann_search(IndexName.TEXT, text_vec, top_k=top_k)
            raw_results.extend(hits)

    # Code search — only when query explicitly mentions code concepts
    if intent["code"]:
        from embeddings.code import embed as embed_code
        code_vec = embed_code(query).tolist()
        code_hits = leann_search(IndexName.CODE, code_vec, top_k=top_k)
        raw_results.extend(code_hits)

    # Visual search — CLIP text → image + video frame space
    if intent["visual"]:
        from embeddings.image import embed_text as clip_text
        clip_vec = clip_text(query)
        hits = leann_search(IndexName.IMAGE, clip_vec, top_k=top_k)
        raw_results.extend(hits)
        video_hits = leann_search(IndexName.VIDEO, clip_vec, top_k=top_k)
        raw_results.extend(video_hits)

    # Audio — only search if query seems audio-related or no results yet
    audio_keywords = {"audio", "podcast", "recording", "transcript", "said", "heard", "spoke", "call", "meeting", "interview", "voice"}
    if not raw_results or bool(set(query.lower().split()) & audio_keywords):
        if "text_vec" not in dir():
            from embeddings.text import embed as embed_text
            text_vec = embed_text(query).tolist()
        audio_hits = leann_search(IndexName.AUDIO, text_vec, top_k=top_k)
        raw_results.extend(audio_hits)

    # Fallback — if still nothing, run broad visual search
    if not raw_results:
        from embeddings.image import embed_text as clip_text
        clip_vec = clip_text(query)
        hits = leann_search(IndexName.IMAGE, clip_vec, top_k=top_k)
        raw_results.extend(hits)

    # Fetch chunk documents and apply filters
    results: list[QueryResult] = []
    seen_chunk_ids: set[str] = set()

    # Minimum relevance threshold — drop low-confidence hits
    SCORE_THRESHOLD = 0.25

    for hit in sorted(raw_results, key=lambda x: x["score"], reverse=True):
        if hit.get("score", 0) < SCORE_THRESHOLD:
            break  # sorted descending — everything after is worse
        chunk_id = hit.get("chunk_id")
        if not chunk_id or chunk_id in seen_chunk_ids:
            continue
        # If person filter active, only include chunks featuring that person
        if person_chunk_ids is not None and chunk_id not in person_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)

        doc = get_chunk_by_id(chunk_id)
        if not doc:
            continue

        # Apply filters
        if file_type_filter and doc.get("file_type") != file_type_filter:
            continue

        # Tag-based AND-filter — all extracted tags must be present on the chunk
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

        chunk_id_str = str(doc["_id"])
        thumbnail_url = (
            f"/chunk/{chunk_id_str}/thumbnail"
            if doc.get("file_type") == "image"
            else None
        )

        results.append(QueryResult(
            chunk_id=chunk_id_str,
            score=hit["score"],
            file_type=doc.get("file_type", "unknown"),
            source_path=doc.get("source_path", ""),
            text=doc.get("text_content"),
            metadata=meta,
            thumbnail_url=thumbnail_url,
        ))

        if len(results) >= top_k:
            break

    # LLM filter + response — LLM sees all candidates, returns only relevant ones + answer
    llm_response = None
    final_results = results
    if results:
        llm_response, kept_ids = await _ask_llm(query, results, history=history or [])
        if kept_ids is not None:
            id_set = set(kept_ids)
            final_results = [r for r in results if r.chunk_id in id_set]

    refinements = _suggest_refinements(query, intent, final_results)

    return QueryResponse(
        query=query,
        results=final_results,
        total=len(final_results),
        llm_response=llm_response,
        suggested_refinements=refinements,
        session_id=session_id,
    )


# ── Person resolution ─────────────────────────────────────────────────────────

async def _resolve_person_chunks(name: str) -> set[str]:
    """
    Resolve a person name to the set of chunk IDs featuring that person.
    Matches against labelled identity clusters — case-insensitive.
    Returns empty set if name not found (no filter applied for unknown names).
    """
    from storage.mongo import get_db
    import re as _re

    db = get_db()
    # Fuzzy name match — "Sarah", "sarah", "SARAH" all match
    pattern = _re.compile(_re.escape(name), _re.IGNORECASE)
    identity = db["identities"].find_one({"label": {"$regex": pattern}})
    if not identity:
        return set()

    cluster_id = identity["cluster_id"]
    # Get all chunk IDs that have face embeddings belonging to this cluster
    face_docs = db["face_embeddings"].find({"cluster_id": cluster_id}, {"chunk_id": 1})
    return {doc["chunk_id"] for doc in face_docs}


# ── LLM integration ───────────────────────────────────────────────────────────

async def _ask_llm(
    query: str,
    results: list[QueryResult],
    history: list[dict] | None = None,
) -> tuple[str | None, list[str] | None]:
    """
    Pass candidate results to LLM. LLM acts as intelligent filter:
    1. Decides which candidates actually match the user's request
    2. Returns a natural language response referencing only those items

    Returns (llm_response, list_of_kept_chunk_ids).
    kept_ids is None if LLM unavailable (fall back to returning all results).
    """
    import os, json, logging
    log = logging.getLogger(__name__)
    provider = os.getenv("LLM_PROVIDER", "local").lower()

    context  = _build_context(results)
    log.info(f"Sending {len(results)} candidates to LLM:\n{context[:600]}")
    messages = _build_messages(query, context, history or [])

    try:
        if provider == "anthropic":
            raw = await _llm_anthropic(messages)
        elif provider == "openai":
            raw = await _llm_openai(messages)
        else:
            raw = await _llm_local(messages)

        log.info(f"LLM raw response:\n{raw[:800]}")

        # Parse structured response — LLM returns JSON block + prose
        kept_ids, response_text = _parse_llm_filter_response(raw, results)
        log.info(f"Kept IDs: {kept_ids}")
        return response_text, kept_ids

    except ModuleNotFoundError as e:
        log.warning(f"LLM call skipped (missing package: {e})")
        return None, None
    except Exception as e:
        log.warning(f"LLM call failed: {e}")
        return None, None


def _parse_llm_filter_response(raw: str, candidates: list[QueryResult]) -> tuple[list[str], str]:
    """
    Extract the JSON filter block and prose response from LLM output.
    Expected format:
        RELEVANT_IDS: ["id1", "id2"]
        <prose response>
    Falls back to returning all IDs if parsing fails.
    """
    import re, json
    all_ids = [r.chunk_id for r in candidates]

    match = re.search(r'RELEVANT_IDS:\s*(\[.*?\])', raw, re.DOTALL)
    if match:
        try:
            kept = json.loads(match.group(1))
            prose = raw[match.end():].strip()
            # Validate — only keep IDs that actually exist in candidates
            valid = [i for i in kept if i in set(all_ids)]
            return valid if valid else all_ids, prose or raw
        except Exception:
            pass

    # LLM didn't follow format — return all and use full response as prose
    return all_ids, raw


def _build_context(results: list[QueryResult]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        snippet = (r.text or "")[:400]
        date = r.metadata.get("created_at") or r.metadata.get("modified_at") or ""
        date_str = f" | {date}" if date else ""
        parts.append(
            f"[{i}] ID:{r.chunk_id} | {r.file_type.upper()} | {r.source_path}{date_str}\n{snippet}"
        )
    return "\n\n".join(parts)


_SYSTEM_PROMPT = (
    "You are Omnex, a personal AI memory assistant. "
    "You have access to the user's indexed personal data. "
    "When candidates are provided, they ARE real items from the user's data — treat them as such. "
    "Your job: read the candidates, pick only the ones that match the request, and answer naturally. "
    "ALWAYS begin your reply with exactly this line (no exceptions):\n"
    "RELEVANT_IDS: [\"id1\", \"id2\"]\n"
    "Use the actual ID values from the candidates (the ID: field). "
    "Then write a concise natural response about what you found. "
    "If none match: RELEVANT_IDS: [] then explain what you found instead. "
    "Do not use markdown formatting. Do not use bullet points or bold text. Write in plain sentences."
)


def _build_messages(query: str, context: str, history: list[dict]) -> list[dict]:
    """Build a messages array with system prompt, prior history, and current turn."""
    messages: list[dict] = []

    # Prior conversation turns — skip empty/None content, coerce to string
    for turn in history:
        role    = turn.get("role", "user")
        content = turn.get("content") or ""
        if not isinstance(content, str):
            content = str(content)
        content = content.strip()
        if not content:
            continue  # Anthropic rejects empty-content messages
        if role not in ("user", "assistant"):
            continue
        messages.append({"role": role, "content": content})

    # Anthropic requires alternating user/assistant turns — deduplicate consecutive same roles
    deduped: list[dict] = []
    for msg in messages:
        if deduped and deduped[-1]["role"] == msg["role"]:
            deduped[-1]["content"] += "\n" + msg["content"]
        else:
            deduped.append(msg)
    messages = deduped

    # Current user turn — include retrieved context
    user_content = f'User request: "{query}"'
    if context:
        user_content += (
            f"\n\nThe following items were retrieved from the user's personal data index. "
            f"These are REAL files that exist in their data:\n\n{context}\n\n"
            f"Based on the request, select only the items that match. "
            f"Start your reply with RELEVANT_IDS: [...] using the ID values shown, then respond naturally."
        )
    else:
        user_content += "\n\nNo items were found in the index. Respond conversationally."
    messages.append({"role": "user", "content": user_content})

    return messages


async def _llm_local(messages: list[dict]) -> str:
    import httpx, os
    host  = os.getenv("LOCAL_LLM_HOST", "http://localhost:11434")
    model = os.getenv("LOCAL_LLM_MODEL", "phi3:mini")

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{host}/api/chat",
            json={"model": model, "messages": messages, "stream": False},
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")


async def _llm_anthropic(messages: list[dict]) -> str:
    import anthropic, os
    client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    model  = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    message = await client.messages.create(
        model=model,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=messages,
    )
    return message.content[0].text


async def _llm_openai(messages: list[dict]) -> str:
    from openai import AsyncOpenAI
    import os
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    all_msgs = [{"role": "system", "content": _SYSTEM_PROMPT}] + messages
    resp = await client.chat.completions.create(
        model=model,
        messages=all_msgs,
        max_tokens=1024,
    )
    return resp.choices[0].message.content


# ── Refinement suggestions ────────────────────────────────────────────────────

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
    if not intent["temporal"]:
        suggestions.append("Filter by date range")
    return suggestions[:3]
