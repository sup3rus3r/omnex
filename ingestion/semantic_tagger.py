"""
Omnex — Semantic Tagger

Enriches every indexed chunk with deep metadata using lightweight local models.
Loaded once at process startup; subsequent calls are fast inference only.

Models:
  spaCy en_core_web_sm  (12.8 MB)   — NER: people, orgs, places, dates, money
  fastText lid.176.ftz  (917 KB)    — Language detection (all modalities)
  KeyBERT on MiniLM     (0 MB extra)— Top keyphrases (reuses existing embedder)
  MiniLM zero-shot      (0 MB extra)— Document type classification
  GLiNER2 ONNX          (188 MB)    — Deep structured entity extraction (async)
  moondream2 4-bit       (~1 GB)    — Image captions (GPU preferred, async)

Architecture:
  tag_sync()  — hot path, ~150-250ms, runs synchronously on every chunk
  tag_async() — deep path, runs in background thread pool, upserts chunk when done
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

log = logging.getLogger("omnex.semantic_tagger")

# ── Model singletons ──────────────────────────────────────────────────────────

_lock         = threading.Lock()
_spacy_nlp    = None
_ft_lid       = None
_kw_model     = None
_type_embs    = None   # pre-computed MiniLM embeddings for doc-type labels
_gliner       = None
_moondream    = None
_moondream_proc = None

# Background executor for async enrichment (GLiNER, moondream)
_async_pool   = ThreadPoolExecutor(max_workers=2, thread_name_prefix="omnex-tagger-async")

# ── Document type labels for zero-shot classification ─────────────────────────

_DOC_TYPE_LABELS = [
    "curriculum vitae resume",
    "invoice receipt payment",
    "contract agreement legal",
    "academic paper research",
    "email message correspondence",
    "report presentation",
    "news article blog post",
    "personal journal diary notes",
    "technical documentation code",
    "spreadsheet financial data",
    "medical health record",
    "travel itinerary booking",
]

_DOC_TYPE_TAG_MAP = {
    "curriculum vitae resume":         "doc-type-cv",
    "invoice receipt payment":         "doc-type-invoice",
    "contract agreement legal":        "doc-type-contract",
    "academic paper research":         "doc-type-academic",
    "email message correspondence":    "doc-type-email",
    "report presentation":             "doc-type-report",
    "news article blog post":          "doc-type-article",
    "personal journal diary notes":    "doc-type-personal",
    "technical documentation code":    "doc-type-technical",
    "spreadsheet financial data":      "doc-type-spreadsheet",
    "medical health record":           "doc-type-medical",
    "travel itinerary booking":        "doc-type-travel",
}

# GLiNER entity types — zero-shot, no retraining needed
_GLINER_ENTITY_TYPES = [
    "person",
    "organization",
    "location",
    "date",
    "money",
    "product",
    "job title",
    "skill",
    "programming language",
    "software framework",
    "email address",
    "phone number",
    "invoice number",
    "contract clause",
    "medical condition",
    "medication",
]


# ── Model loaders (lazy, thread-safe) ─────────────────────────────────────────

def _get_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        with _lock:
            if _spacy_nlp is None:
                try:
                    import spacy
                    _spacy_nlp = spacy.load("en_core_web_sm")
                    log.info("[tagger] spaCy en_core_web_sm loaded")
                except Exception as e:
                    log.warning(f"[tagger] spaCy unavailable: {e}")
                    _spacy_nlp = False
    return _spacy_nlp or None


def _get_fasttext():
    global _ft_lid
    if _ft_lid is None:
        with _lock:
            if _ft_lid is None:
                try:
                    import numpy as np
                    import fasttext

                    model_path = os.getenv(
                        "FASTTEXT_LID_PATH",
                        "/data/models/fasttext/lid.176.ftz"
                    )
                    if not Path(model_path).exists():
                        log.warning(f"[tagger] fastText LID model not found at {model_path}")
                        _ft_lid = False
                    else:
                        model = fasttext.load_model(model_path)

                        # Wrap predict to handle numpy 2.x copy=False incompatibility
                        _orig_predict = model.predict

                        def _safe_predict(text, k=1, threshold=0.0):
                            try:
                                labels, probs = _orig_predict(text, k=k, threshold=threshold)
                                return labels, np.asarray(probs)
                            except ValueError:
                                # numpy 2.x: call the underlying C extension directly
                                labels, probs = model.f.predict(text, k, threshold, "strict")
                                return labels, np.asarray(probs)

                        model.predict = _safe_predict
                        _ft_lid = model
                        log.info("[tagger] fastText LID loaded")
                except Exception as e:
                    log.warning(f"[tagger] fastText unavailable: {e}")
                    _ft_lid = False
    return _ft_lid or None


def _get_kw_model():
    global _kw_model, _type_embs
    if _kw_model is None:
        with _lock:
            if _kw_model is None:
                try:
                    # sentence_transformers must be fully loaded before keybert imports
                    # any of its backends (they re-import sentence_transformers and hit
                    # a torchvision circular import if ST isn't already in sys.modules)
                    from embeddings.text import _get_model as _get_minilm
                    embedder = _get_minilm()

                    # At this point sentence_transformers is fully in sys.modules.
                    # Now import keybert — it will find ST already cached, no re-import.
                    from keybert import KeyBERT
                    import numpy as np

                    # Use the already-loaded embedder object directly via a minimal wrapper
                    # so KeyBERT never needs to re-instantiate SentenceTransformer itself
                    class _EmbedderShim:
                        """Minimal duck-type for KeyBERT's expected backend interface."""
                        def encode(self, sentences, **kwargs):
                            return embedder.encode(
                                sentences, show_progress_bar=False,
                                normalize_embeddings=True
                            )

                    _kw_model = KeyBERT(model=_EmbedderShim())

                    # Pre-compute doc-type label embeddings
                    vecs = embedder.encode(_DOC_TYPE_LABELS, normalize_embeddings=True)
                    _type_embs = vecs
                    log.info("[tagger] KeyBERT + doc-type classifier loaded")
                except Exception as e:
                    log.warning(f"[tagger] KeyBERT unavailable: {e}")
                    _kw_model = False
    return _kw_model or None


def _get_gliner():
    global _gliner
    if _gliner is None:
        with _lock:
            if _gliner is None:
                try:
                    from gliner import GLiNER
                    model_name = os.getenv("GLINER_MODEL", "urchade/gliner_medium-v2.1")
                    _gliner = GLiNER.from_pretrained(model_name)
                    log.info(f"[tagger] GLiNER loaded: {model_name}")
                except Exception as e:
                    log.warning(f"[tagger] GLiNER unavailable: {e}")
                    _gliner = False
    return _gliner or None


def _get_moondream():
    global _moondream, _moondream_proc
    if _moondream is None:
        with _lock:
            if _moondream is None:
                try:
                    import moondream as md
                    model_path = os.getenv(
                        "MOONDREAM_MODEL_PATH",
                        "/data/models/moondream/moondream-2b-int8.mf"
                    )
                    if Path(model_path).exists():
                        _moondream = md.vl(model=model_path)
                        log.info("[tagger] moondream2 loaded from local path")
                    else:
                        # Auto-download (cached to HF_HOME)
                        _moondream = md.vl(model="moondream-2b-int8")
                        log.info("[tagger] moondream2 loaded (auto-download)")
                except Exception as e:
                    log.warning(f"[tagger] moondream unavailable: {e}")
                    _moondream = False
    return _moondream or None


# ── Synchronous hot path ───────────────────────────────────────────────────────

def tag_sync(
    text: str | None,
    file_type: str,
    metadata: dict[str, Any],
    source_path: Path | None = None,
) -> dict[str, Any]:
    """
    Fast synchronous enrichment — runs on every chunk during ingestion.
    Returns a dict of new metadata fields and a 'semantic_tags' list.

    Target latency: <250ms per chunk on CPU.
    """
    result: dict[str, Any] = {"semantic_tags": []}
    tags: list[str] = []

    sample = (text or "")[:3000]  # cap to keep inference fast

    # ── Language detection ────────────────────────────────────────────────────
    if sample.strip():
        ft = _get_fasttext()
        if ft:
            try:
                # fasttext-wheel has numpy 2.x compat issue with predict() returning probs
                # Use get_language() or parse raw output instead
                clean = sample[:500].replace("\n", " ")
                result_ft = ft.predict(clean, k=1)
                labels = result_ft[0] if result_ft else []
                if labels:
                    lang = labels[0].replace("__label__", "")
                    result["detected_language"] = lang
                    tags.append(f"lang-{lang[:2].lower()}")
            except Exception:
                # Fallback: try the line-by-line API
                try:
                    clean = sample[:200].replace("\n", " ")
                    labels = ft.predict(clean, k=1)[0]
                    lang = str(labels[0]).replace("__label__", "")
                    result["detected_language"] = lang
                    tags.append(f"lang-{lang[:2].lower()}")
                except Exception:
                    pass

    # ── NER via spaCy ─────────────────────────────────────────────────────────
    if sample.strip() and file_type in ("document", "audio", "video", "code"):
        nlp = _get_spacy()
        if nlp:
            try:
                doc = nlp(sample[:100_000])
                entities: list[dict] = []
                people, orgs, places = [], [], []
                for ent in doc.ents:
                    entities.append({"text": ent.text, "label": ent.label_})
                    if ent.label_ == "PERSON":
                        people.append(ent.text)
                        tags.append("has-people")
                    elif ent.label_ in ("ORG", "COMPANY"):
                        orgs.append(ent.text)
                        tags.append("has-organizations")
                    elif ent.label_ in ("GPE", "LOC"):
                        places.append(ent.text)
                        tags.append("has-locations")
                    elif ent.label_ == "DATE":
                        tags.append("has-dates")
                    elif ent.label_ == "MONEY":
                        tags.append("has-money")

                if entities:
                    result["entities"] = entities
                if people:
                    result["people"] = list(dict.fromkeys(people))[:10]
                if orgs:
                    result["organizations"] = list(dict.fromkeys(orgs))[:10]
                if places:
                    result["places"] = list(dict.fromkeys(places))[:10]
            except Exception as e:
                log.debug(f"[tagger] spaCy error: {e}")

    # ── Keywords + doc-type classification via MiniLM ─────────────────────────
    if sample.strip() and file_type in ("document", "audio", "video"):
        kw = _get_kw_model()
        if kw and _type_embs is not None:
            try:
                # Keywords
                keywords = kw.extract_keywords(
                    sample,
                    keyphrase_ngram_range=(1, 2),
                    top_n=8,
                    stop_words="english",
                )
                kw_list = [k for k, _ in keywords if k]
                if kw_list:
                    result["keywords"] = kw_list
                    tags += [f"kw-{k.replace(' ', '-').lower()}" for k in kw_list[:5]]

                # Zero-shot doc type
                import numpy as np
                from embeddings.text import _get_model as _get_minilm
                embedder = _get_minilm()
                doc_vec = embedder.encode(sample[:2000], normalize_embeddings=True)
                scores = doc_vec @ _type_embs.T
                best_idx = int(scores.argmax())
                best_score = float(scores[best_idx])
                if best_score >= 0.25:
                    label = _DOC_TYPE_LABELS[best_idx]
                    doc_type_tag = _DOC_TYPE_TAG_MAP.get(label)
                    if doc_type_tag:
                        result["doc_type"] = doc_type_tag.replace("doc-type-", "")
                        tags.append(doc_type_tag)
            except Exception as e:
                log.debug(f"[tagger] KeyBERT/doc-type error: {e}")

    result["semantic_tags"] = sorted(set(tags))
    return result


# ── Async deep enrichment ─────────────────────────────────────────────────────

def tag_async_text(chunk_id: str, text: str, file_type: str) -> None:
    """
    Submit a chunk for deep GLiNER enrichment in the background.
    Updates the chunk's MongoDB document when done — non-blocking.
    """
    _async_pool.submit(_enrich_text_gliner, chunk_id, text, file_type)


def tag_async_image(chunk_id: str, image_path: Path | str) -> None:
    """
    Submit an image chunk for moondream captioning in the background.
    Updates the chunk's MongoDB document when done — non-blocking.
    """
    _async_pool.submit(_enrich_image_moondream, chunk_id, str(image_path))


def _enrich_text_gliner(chunk_id: str, text: str, file_type: str) -> None:
    """Background worker: GLiNER deep entity extraction → upsert chunk metadata."""
    try:
        gliner = _get_gliner()
        if not gliner:
            return

        sample = text[:4000]
        entities = gliner.predict_entities(sample, _GLINER_ENTITY_TYPES, threshold=0.5)

        # Group by type
        grouped: dict[str, list[str]] = {}
        for ent in entities:
            label = ent["label"]
            value = ent["text"].strip()
            if value:
                grouped.setdefault(label, []).append(value)

        # Deduplicate
        structured: dict[str, list[str]] = {
            k: list(dict.fromkeys(v))[:10]
            for k, v in grouped.items()
            if v
        }

        if not structured:
            return

        # Upsert into MongoDB
        from storage.mongo import get_db
        from bson import ObjectId
        db = get_db()
        db["chunks"].update_one(
            {"_id": ObjectId(chunk_id)},
            {"$set": {"metadata.gliner_entities": structured}},
        )
        log.debug(f"[tagger] GLiNER enriched chunk {chunk_id}: {list(structured.keys())}")

    except Exception as e:
        log.debug(f"[tagger] GLiNER async failed for {chunk_id}: {e}")


def _enrich_image_moondream(chunk_id: str, image_path: str) -> None:
    """Background worker: moondream2 image caption → upsert chunk metadata."""
    try:
        model = _get_moondream()
        if not model:
            return

        from PIL import Image
        img = Image.open(image_path)

        # Caption + short object/scene description
        caption = model.caption(img)["caption"]
        objects = model.query(img, "List the main objects and people visible.")["answer"]

        from storage.mongo import get_db
        from bson import ObjectId
        db = get_db()
        db["chunks"].update_one(
            {"_id": ObjectId(chunk_id)},
            {"$set": {
                "metadata.image_caption":  caption,
                "metadata.image_objects":  objects,
                "text_content": caption,   # makes image searchable by text
            }},
        )
        log.debug(f"[tagger] moondream captioned chunk {chunk_id}")

    except Exception as e:
        log.debug(f"[tagger] moondream async failed for {chunk_id}: {e}")


# ── Warm-up ───────────────────────────────────────────────────────────────────

def warm_up() -> None:
    """
    Pre-load all sync models at process startup.
    Call this once from the ingestion entry point before the thread pool starts.
    Prevents multiple threads racing to load the same model.
    """
    log.info("[tagger] warming up semantic models...")
    _get_fasttext()
    _get_spacy()
    _get_kw_model()
    log.info("[tagger] semantic models ready (GLiNER + moondream load lazily on first use)")
