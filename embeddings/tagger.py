"""
Omnex — Neural Auto-Tagger (Phase 7)

Generates semantic tags for every indexed chunk using a combination of:
  1. Rule-based extraction  — file type, extension, date, path hints
  2. CLIP zero-shot scoring — scene/content classification for images/video
  3. Text keyword matching  — topic extraction for documents and audio
  4. Metadata-derived tags  — GPS→location bucket, duration, language, camera model

Tags are stored as a flat list of lowercase strings on every chunk document.
They power:
  - Tag-based query filtering  ("work documents from 2023")
  - Refinement suggestions in the UI
  - Future fine-tune classifier (user label → tag feedback loop)

Tag taxonomy (first-level categories):
  date:      year-YYYY, month-YYYY-MM, season-spring/summer/autumn/winter, recent
  type:      photo, video, audio, document, code, screenshot, recording
  format:    pdf, mp4, py, docx, jpg …  (file extension)
  scene:     outdoor, indoor, nature, urban, food, people, animal, text, diagram
  topic:     work, personal, travel, music, finance, health, education
  location:  gps-present, [continent/region bucket if GPS present]
  language:  lang-en, lang-fr …  (detected transcript language)
  temporal:  this-week, this-month, this-year, older
  size:      tiny (<10KB), small (<100KB), medium (<10MB), large (>10MB)
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── Scene categories scored via CLIP ──────────────────────────────────────────

CLIP_SCENE_LABELS = [
    "outdoor scene",
    "indoor scene",
    "nature landscape",
    "urban cityscape",
    "food and drink",
    "people and faces",
    "animals and pets",
    "text document screenshot",
    "diagram chart graph",
    "night scene",
    "beach ocean sea",
    "mountains hiking",
    "vehicle car transport",
    "sports activity",
    "work office professional",
]

CLIP_SCENE_TAG_MAP = {
    "outdoor scene":            "scene-outdoor",
    "indoor scene":             "scene-indoor",
    "nature landscape":         "scene-nature",
    "urban cityscape":          "scene-urban",
    "food and drink":           "scene-food",
    "people and faces":         "scene-people",
    "animals and pets":         "scene-animals",
    "text document screenshot": "scene-text",
    "diagram chart graph":      "scene-diagram",
    "night scene":              "scene-night",
    "beach ocean sea":          "topic-beach",
    "mountains hiking":         "topic-outdoors",
    "vehicle car transport":    "topic-transport",
    "sports activity":          "topic-sports",
    "work office professional": "topic-work",
}

# Score threshold — only add CLIP tags above this cosine similarity
CLIP_TAG_THRESHOLD = 0.22


# ── Topic keyword → tag map for text/document content ─────────────────────────

TEXT_TOPIC_PATTERNS: list[tuple[str, list[str]]] = [
    ("topic-finance",     ["invoice", "budget", "expense", "payment", "salary", "tax", "receipt", "bank", "financial", "revenue", "profit", "cost"]),
    ("topic-work",        ["meeting", "project", "deadline", "client", "report", "proposal", "agenda", "sprint", "roadmap", "quarterly"]),
    ("topic-travel",      ["flight", "hotel", "booking", "itinerary", "passport", "visa", "airport", "trip", "vacation", "holiday", "destination"]),
    ("topic-health",      ["doctor", "hospital", "prescription", "medical", "diagnosis", "appointment", "therapy", "fitness", "workout", "symptom"]),
    ("topic-education",   ["lecture", "assignment", "exam", "course", "university", "school", "thesis", "research", "study", "academic"]),
    ("topic-personal",    ["diary", "journal", "note", "reminder", "todo", "personal", "family", "birthday", "friend"]),
    ("topic-legal",       ["contract", "agreement", "terms", "legal", "lawyer", "court", "clause", "liability", "confidential"]),
    ("topic-tech",        ["api", "database", "server", "deployment", "software", "hardware", "network", "algorithm", "debug", "repository"]),
    ("topic-music",       ["song", "album", "playlist", "lyrics", "chord", "tempo", "beat", "track", "recording", "musician"]),
]

# Pre-compile for speed
_COMPILED_TEXT_TOPICS: list[tuple[str, re.Pattern]] = [
    (tag, re.compile(r'\b(?:' + '|'.join(kws) + r')\b', re.IGNORECASE))
    for tag, kws in TEXT_TOPIC_PATTERNS
]


# ── Main tagger ────────────────────────────────────────────────────────────────

def tag_chunk(
    path: Path,
    file_type: str,
    metadata: dict[str, Any],
    text_content: str | None = None,
    image_embedding=None,       # numpy array — CLIP image embedding (512-dim)
    chunk_id: str | None = None,  # if provided, triggers async deep enrichment
) -> list[str]:
    """
    Generate all tags for a single chunk.

    Args:
        path:            Source file path
        file_type:       'image' | 'video' | 'audio' | 'document' | 'code'
        metadata:        Chunk metadata dict (EXIF, timestamps, language, etc.)
        text_content:    Extracted text (transcript, document body, code snippet)
        image_embedding: Pre-computed CLIP image embedding (saves re-encoding)
        chunk_id:        MongoDB chunk ID — enables async GLiNER/moondream enrichment

    Returns:
        Sorted, deduplicated list of lowercase tag strings.
    """
    tags: set[str] = set()

    # 1. Type + format tags
    tags.update(_type_tags(path, file_type))

    # 2. Date / temporal tags
    tags.update(_date_tags(metadata, path))

    # 3. Size tags
    tags.update(_size_tags(metadata))

    # 4. GPS / location tags
    tags.update(_location_tags(metadata))

    # 5. Language tags (audio/video transcripts)
    tags.update(_language_tags(metadata))

    # 6. Text topic tags (documents, audio transcripts, code)
    if text_content:
        tags.update(_text_topic_tags(text_content, file_type))

    # 7. CLIP scene tags (images and video keyframes)
    if image_embedding is not None and file_type in ("image", "video"):
        tags.update(_clip_scene_tags(image_embedding))

    # 8. Path-hint tags (Desktop, Downloads, Documents, etc.)
    tags.update(_path_hint_tags(path))

    # 9. Semantic tagger — spaCy NER + fastText LID + KeyBERT + doc-type (sync)
    try:
        from ingestion.semantic_tagger import tag_sync, tag_async_text, tag_async_image
        sem = tag_sync(
            text=text_content,
            file_type=file_type,
            metadata=metadata,
            source_path=path,
        )
        tags.update(sem.get("semantic_tags", []))

        # Merge enriched metadata fields back into the metadata dict
        for key in ("detected_language", "entities", "people", "organizations",
                    "places", "keywords", "doc_type"):
            if key in sem:
                metadata[key] = sem[key]

        # Fire async deep enrichment if we have a chunk_id
        if chunk_id:
            if text_content and file_type in ("document", "audio", "video", "code"):
                tag_async_text(chunk_id, text_content, file_type)
            elif file_type == "image" and path.exists():
                tag_async_image(chunk_id, path)
    except Exception as e:
        import logging
        logging.getLogger("omnex.tagger").debug(f"Semantic tagger error: {e}")

    return sorted(tags)


# ── Tag generators ─────────────────────────────────────────────────────────────

def _type_tags(path: Path, file_type: str) -> list[str]:
    tags = [f"type-{file_type}"]

    ext = path.suffix.lower().lstrip(".")
    if ext:
        tags.append(f"format-{ext}")

    # Semantic sub-type
    if file_type == "image":
        if ext in ("jpg", "jpeg", "heic", "heif"):
            tags.append("type-photo")
        elif ext == "png":
            # PNG screenshots are common — flag as potential screenshot
            tags.append("type-screenshot-candidate")
        elif ext in ("gif", "webp"):
            tags.append("type-animated")
    elif file_type == "audio":
        tags.append("type-recording")
    elif file_type == "video":
        tags.append("type-recording")
    elif file_type == "document":
        if ext == "pdf":
            tags.append("type-pdf")
        elif ext in ("doc", "docx"):
            tags.append("type-word")
        elif ext in ("xls", "xlsx", "csv"):
            tags.append("type-spreadsheet")
        elif ext in ("ppt", "pptx"):
            tags.append("type-presentation")
        elif ext == "md":
            tags.append("type-markdown")

    return tags


def _date_tags(metadata: dict, path: Path) -> list[str]:
    tags = []

    # Try EXIF first, then filesystem dates
    raw_date = (
        metadata.get("exif_datetime")
        or metadata.get("created_at")
        or metadata.get("modified_at")
    )

    dt: datetime | None = None
    if isinstance(raw_date, datetime):
        dt = raw_date
    elif isinstance(raw_date, str):
        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(raw_date[:19], fmt)
                break
            except ValueError:
                pass

    if dt is None:
        # Fallback to path stat
        try:
            stat = path.stat()
            dt = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        except Exception:
            pass

    if dt:
        year  = dt.year
        month = dt.month
        tags.append(f"year-{year}")
        tags.append(f"month-{year}-{month:02d}")

        # Season (Northern Hemisphere)
        if month in (12, 1, 2):
            tags.append("season-winter")
        elif month in (3, 4, 5):
            tags.append("season-spring")
        elif month in (6, 7, 8):
            tags.append("season-summer")
        else:
            tags.append("season-autumn")

        # Recency
        now = datetime.now(tz=timezone.utc) if dt.tzinfo else datetime.now()
        age_days = (now - dt.replace(tzinfo=None)).days if not dt.tzinfo else (datetime.now(tz=timezone.utc) - dt).days
        if age_days <= 7:
            tags.append("temporal-this-week")
        elif age_days <= 31:
            tags.append("temporal-this-month")
        elif age_days <= 365:
            tags.append("temporal-this-year")
        else:
            tags.append(f"temporal-older")

    return tags


def _size_tags(metadata: dict) -> list[str]:
    size = metadata.get("size_bytes")
    if size is None:
        return []
    if size < 10_000:
        return ["size-tiny"]
    elif size < 100_000:
        return ["size-small"]
    elif size < 10_000_000:
        return ["size-medium"]
    else:
        return ["size-large"]


def _location_tags(metadata: dict) -> list[str]:
    gps = metadata.get("gps")
    if not gps:
        return []

    tags = ["location-gps"]
    lat = gps.get("lat")
    lng = gps.get("lng")

    if lat is not None and lng is not None:
        # Coarse hemisphere/region buckets (good enough for filtering)
        if lat > 0:
            tags.append("location-northern-hemisphere")
        else:
            tags.append("location-southern-hemisphere")

        # Very coarse continent approximation
        if -30 <= lat <= 75 and -25 <= lng <= 65:
            tags.append("location-europe-africa")
        elif -60 <= lat <= 80 and -170 <= lng <= -20:
            tags.append("location-americas")
        elif -60 <= lat <= 80 and 60 <= lng <= 180:
            tags.append("location-asia-pacific")

    return tags


def _language_tags(metadata: dict) -> list[str]:
    lang = metadata.get("language")
    if not lang:
        return []
    # Whisper returns ISO 639-1 codes: 'en', 'fr', 'de' etc.
    return [f"lang-{lang.lower()[:2]}"]


def _text_topic_tags(text: str, file_type: str) -> list[str]:
    tags = []
    # Only run keyword matching on first 2000 chars for speed
    sample = text[:2000]
    for tag, pattern in _COMPILED_TEXT_TOPICS:
        if pattern.search(sample):
            tags.append(tag)
    # Code-specific: language is already captured in metadata, add broad tag
    if file_type == "code":
        tags.append("topic-tech")
    return tags


def _clip_scene_tags(image_embedding) -> list[str]:
    """
    Score the image embedding against CLIP_SCENE_LABELS using the CLIP text encoder.
    Returns tags for labels above CLIP_TAG_THRESHOLD.
    """
    try:
        import numpy as np
        from embeddings.image import embed_text as clip_embed_text

        tags = []
        for label in CLIP_SCENE_LABELS:
            text_vec = clip_embed_text(label)           # (512,) numpy array
            # Cosine similarity — both embeddings should already be normalised
            score = float(np.dot(image_embedding, text_vec))
            if score >= CLIP_TAG_THRESHOLD:
                tag = CLIP_SCENE_TAG_MAP.get(label)
                if tag:
                    tags.append(tag)
        return tags
    except Exception:
        return []


def _path_hint_tags(path: Path) -> list[str]:
    """
    Extract location-in-filesystem hints from the path.
    Desktop, Downloads, Documents, Pictures etc. are strong semantic signals.
    """
    tags = []
    path_lower = path.as_posix().lower()
    hint_map = {
        "/desktop/":    "location-desktop",
        "/downloads/":  "location-downloads",
        "/documents/":  "location-documents",
        "/pictures/":   "location-pictures",
        "/photos/":     "location-pictures",
        "/videos/":     "location-videos",
        "/music/":      "location-music",
        "/code/":       "location-code",
        "/projects/":   "location-code",
        "/work/":       "topic-work",
        "/personal/":   "topic-personal",
        "/backup/":     "location-backup",
        "screenshot":   "type-screenshot",
        "screen shot":  "type-screenshot",
        "screen_shot":  "type-screenshot",
        "wallpaper":    "type-wallpaper",
        "avatar":       "type-avatar",
        "profile":      "type-profile",
        "invoice":      "topic-finance",
        "receipt":      "topic-finance",
        "contract":     "topic-legal",
        "resume":       "topic-work",
        "cv":           "topic-work",
    }
    for hint, tag in hint_map.items():
        if hint in path_lower:
            tags.append(tag)
    return tags


# ── Tag query extraction ───────────────────────────────────────────────────────

# Maps NL query fragments to tag prefixes/values for filter extraction
_QUERY_TAG_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\b(photo|photos|picture|pictures|image|images)\b', re.I), "type-image"),
    (re.compile(r'\b(video|videos|clip|footage|recording)\b', re.I),        "type-video"),
    (re.compile(r'\b(audio|recording|podcast|voice memo)\b', re.I),         "type-audio"),
    (re.compile(r'\b(document|documents|pdf|word|doc)\b', re.I),            "type-document"),
    (re.compile(r'\b(code|script|function|class|repo)\b', re.I),            "type-code"),
    (re.compile(r'\b(screenshot|screen shot)\b', re.I),                     "type-screenshot"),
    (re.compile(r'\b(work|office|professional|meeting|project)\b', re.I),   "topic-work"),
    (re.compile(r'\b(travel|trip|holiday|vacation|beach|abroad)\b', re.I),  "topic-travel"),
    (re.compile(r'\b(finance|invoice|receipt|expense|budget)\b', re.I),     "topic-finance"),
    (re.compile(r'\b(nature|outdoor|landscape|mountain|forest)\b', re.I),   "scene-nature"),
    (re.compile(r'\b(food|restaurant|meal|eating|cooking)\b', re.I),        "scene-food"),
    (re.compile(r'\b(gps|location|map|coordinates)\b', re.I),               "location-gps"),
]

_YEAR_RE  = re.compile(r'\b(20\d{2}|19\d{2})\b')
_MONTH_RE = re.compile(
    r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
    re.I
)
_MONTH_MAP = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}


def extract_tag_filters(query: str) -> list[str]:
    """
    Parse a natural language query and return a list of tag values that
    should be used as AND-filters when searching.

    e.g. "work documents from 2022" → ["type-document", "topic-work", "year-2022"]
    """
    filters: list[str] = []

    for pattern, tag in _QUERY_TAG_PATTERNS:
        if pattern.search(query):
            filters.append(tag)

    # Year extraction
    year_m = _YEAR_RE.search(query)
    if year_m:
        filters.append(f"year-{year_m.group(1)}")

    # Month + year combo
    month_m = _MONTH_RE.search(query)
    year_m2 = _YEAR_RE.search(query)
    if month_m and year_m2:
        m_num = _MONTH_MAP[month_m.group(1).lower()]
        filters.append(f"month-{year_m2.group(1)}-{m_num}")

    # Recency
    if re.search(r'\b(recent|recently|latest|today|yesterday|this week)\b', query, re.I):
        filters.append("temporal-this-week")
    elif re.search(r'\bthis month\b', query, re.I):
        filters.append("temporal-this-month")
    elif re.search(r'\bthis year\b', query, re.I):
        filters.append("temporal-this-year")

    # Season
    if re.search(r'\b(summer)\b', query, re.I):
        filters.append("season-summer")
    elif re.search(r'\b(winter)\b', query, re.I):
        filters.append("season-winter")
    elif re.search(r'\b(spring)\b', query, re.I):
        filters.append("season-spring")
    elif re.search(r'\b(autumn|fall)\b', query, re.I):
        filters.append("season-autumn")

    return list(dict.fromkeys(filters))  # deduplicate, preserve order
