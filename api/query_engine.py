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

# Device name patterns — extracted from queries like "photos from my iPhone"
_DEVICE_PATTERNS = re.compile(
    r'\b(iphone|ipad|samsung|galaxy|pixel|huawei|sony|canon|nikon|fuji(?:film)?|gopro|oneplus|xiaomi|android)\b',
    re.IGNORECASE,
)

# File-type hint patterns — "my PDFs", "spreadsheets", "MP3s"
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

    # Device name — "photos from my iPhone", "Samsung pictures"
    device_match = _DEVICE_PATTERNS.search(query)
    device_hint = device_match.group(1).lower() if device_match else None

    # File type override from natural language — "my PDFs", "MP3s"
    file_type_hint = None
    for pattern, ft in _EXT_PATTERNS:
        if pattern.search(query):
            file_type_hint = ft
            break

    return {
        "visual":          is_visual,
        "temporal":        is_temporal,
        "code":            is_code,
        "location":        is_location,
        "person_name":     person_name,
        "device_hint":     device_hint,
        "file_type_hint":  file_type_hint,
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

    # Intent-derived overrides — only if not already set by caller
    if not file_type_filter and intent.get("file_type_hint"):
        file_type_filter = intent["file_type_hint"]

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
    SCORE_THRESHOLD = 0.15

    for hit in sorted(raw_results, key=lambda x: x["score"], reverse=True):
        if hit.get("score", 0) < SCORE_THRESHOLD:
            break  # sorted descending — everything after is worse
        chunk_id = hit.get("chunk_id")
        if not chunk_id or chunk_id in seen_chunk_ids:
            continue
        if person_chunk_ids is not None and chunk_id not in person_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)

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

        # Device filter — "photos from iPhone" → metadata.device contains "iphone"
        if intent.get("device_hint"):
            device = (meta.get("device") or "").lower()
            if intent["device_hint"] not in device:
                continue

        chunk_id_str = str(doc["_id"])
        results.append(QueryResult(
            chunk_id=chunk_id_str,
            score=hit["score"],
            file_type=doc.get("file_type", "unknown"),
            source_path=doc.get("source_path", ""),
            text=doc.get("text_content"),
            metadata=meta,
            thumbnail_url=f"/chunk/{chunk_id_str}/thumbnail" if doc.get("file_type") == "image" else None,
        ))

        if len(results) >= top_k:
            break

    # ── Broad MongoDB fallback ─────────────────────────────────────────────────
    # When vector search returns nothing (sparse index, broad queries like
    # "what did I work on recently?"), fall back to a MongoDB query so the
    # LLM always has something real to reason about.
    if not results:
        results = await _mongo_fallback(
            query=query,
            intent=intent,
            file_type_filter=file_type_filter,
            date_from=date_from,
            date_to=date_to,
            top_k=top_k,
            device_hint=intent.get("device_hint"),
        )

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


# ── MongoDB broad fallback ────────────────────────────────────────────────────

async def _mongo_fallback(
    query: str,
    intent: dict,
    file_type_filter: str | None,
    date_from: datetime | None,
    date_to: datetime | None,
    top_k: int,
    device_hint: str | None = None,
) -> list[QueryResult]:
    """
    Broad MongoDB fallback when vector search finds nothing.
    Handles temporal queries ("recently", "last week"), broad browsing,
    device queries ("photos from iPhone"), and keyword text search.
    """
    from storage.mongo import get_db
    import re

    db = get_db()
    q: dict = {"chunk_index": 0}  # one representative chunk per source file

    if file_type_filter:
        q["file_type"] = file_type_filter

    # Device filter — match metadata.device case-insensitively
    if device_hint:
        q["metadata.device"] = {"$regex": device_hint, "$options": "i"}

    # Resolve relative temporal expressions → date range
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

    # Tag-based filters from query — year, month, topic, scene, location, season
    from embeddings.tagger import extract_tag_filters
    tag_filters = extract_tag_filters(query)
    # Only apply tags that are specific (year, month, topic, scene) — skip type tags
    # since file_type_filter already handles those
    specific_tags = [t for t in tag_filters if not t.startswith("type-")]
    if specific_tags:
        q["tags"] = {"$all": specific_tags}

    # Keyword text search — strip common question words, use content words
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
        # MongoDB text search if index exists, otherwise regex
        try:
            q["$text"] = {"$search": " ".join(keywords)}
            docs = list(db["chunks"].find(q, {"score": {"$meta": "textScore"}})
                        .sort([("score", {"$meta": "textScore"})])
                        .limit(top_k))
        except Exception:
            # Fallback: regex on text_content for the strongest keyword
            q.pop("$text", None)
            q["text_content"] = {"$regex": keywords[0], "$options": "i"}
            docs = list(db["chunks"].find(q).sort("created_at", -1).limit(top_k))
    else:
        # No useful keywords — just return most recent items
        docs = list(db["chunks"].find(q).sort("created_at", -1).limit(top_k))

    results = []
    for doc in docs:
        chunk_id_str = str(doc["_id"])
        meta = doc.get("metadata", {})
        results.append(QueryResult(
            chunk_id=chunk_id_str,
            score=0.0,  # no vector score — fallback result
            file_type=doc.get("file_type", "unknown"),
            source_path=doc.get("source_path", ""),
            text=doc.get("text_content"),
            metadata=meta,
            thumbnail_url=f"/chunk/{chunk_id_str}/thumbnail" if doc.get("file_type") == "image" else None,
        ))
    return results


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
        meta = r.metadata
        date = meta.get("exif_datetime") or meta.get("created_at") or meta.get("modified_at") or ""
        date_str = f" | {date}" if date else ""
        device = meta.get("device", "")
        device_str = f" | device:{device}" if device else ""
        gps = meta.get("gps")
        gps_str = f" | gps:{gps['lat']:.4f},{gps['lng']:.4f}" if gps else ""
        lang = meta.get("language", "")
        lang_str = f" | lang:{lang}" if lang else ""
        parts.append(
            f"[{i}] ID:{r.chunk_id} | {r.file_type.upper()} | {r.source_path}{date_str}{device_str}{gps_str}{lang_str}\n{snippet}"
        )
    return "\n\n".join(parts)


_SYSTEM_PROMPT = (
    "You are Omnex, a personal AI memory system. "
    "You have already searched the user's indexed personal files and data. "
    "The search results (if any) are provided in each message. "
    "NEVER say you lack access to the user's data — you have already searched it. "
    "NEVER ask the user to share files — you already have the search results. "
    "When candidates are provided, they ARE real items found in the user's data. "
    "Your job: pick only the ones that match the request, and answer naturally. "
    "ALWAYS begin your reply with exactly this line (no exceptions):\n"
    "RELEVANT_IDS: [\"id1\", \"id2\"]\n"
    "Use the actual ID values from the candidates (the ID: field). "
    "Then write a concise natural response about what you found. "
    "If none match: RELEVANT_IDS: [] then explain what was searched and found. "
    "When no results were found: say you searched and found nothing, suggest reindexing or different keywords. "
    "Do not use markdown. Do not use bullet points or bold. Write in plain sentences."
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
        user_content += (
            "\n\nNo items were retrieved from the index for this query. "
            "Tell the user you searched their data but found nothing matching their request. "
            "Do not say you lack access to their data — you have already searched and found no matches. "
            "Suggest they may not have indexed that type of content yet, or try different wording."
        )
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
