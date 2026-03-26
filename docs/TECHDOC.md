# Omnex — Technical Reference

> v0.1 | Implementation guide for contributors and AI coding assistants
> Stack: Python · FastAPI · Next.js · Go · MongoDB · LEANN · Ollama
> LEANN replaces Qdrant as the vector index -- 97% storage savings, no server process, file-based indexes.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Ingestion Pipeline — Implementation](#2-ingestion-pipeline--implementation)
3. [ML Models — Integration Guide](#3-ml-models--integration-guide)
4. [Storage — Schema & Queries](#4-storage--schema--queries)
5. [FastAPI Backend — Full Spec](#5-fastapi-backend--full-spec)
6. [Next.js Interface — Component Spec](#6-nextjs-interface--component-spec)
7. [FUSE Layer — Go Implementation](#7-fuse-layer--go-implementation)
8. [Configuration Reference](#8-configuration-reference)
9. [Testing Strategy](#9-testing-strategy)
10. [Performance Tuning](#10-performance-tuning)
11. [Error Handling Patterns](#11-error-handling-patterns)
12. [Data Flow Diagrams](#12-data-flow-diagrams)

---

## 1. Environment Setup

### 1.1 Prerequisites

```bash
# Python 3.11+
python --version

# Node.js 20+
node --version

# Go 1.22+
go version

# Docker + Docker Compose
docker --version
docker compose version
```

### 1.2 First-Run Install (Linux)

```bash
git clone https://github.com/sup3rus3r/omnex
cd omnex

# Start backing services
docker compose up -d

# Python environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download ML models (~4GB, runs once)
python models/download.py

# Start API
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

# Start interface (separate terminal)
cd interface && npm install && npm run dev
```

### 1.3 First-Run Install (Windows)

```powershell
# Requires WinFsp for FUSE layer
# https://winfsp.dev/rel/

git clone https://github.com/sup3rus3r/omnex
cd omnex

docker compose up -d

python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python models\download.py

uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### 1.4 docker-compose.yml (Full)

```yaml
version: '3.9'

services:
  mongodb:
    image: mongo:7
    container_name: omnex-mongo
    restart: unless-stopped
    volumes:
      - ./data/mongo:/data/db
    ports:
      - '127.0.0.1:27017:27017'
    environment:
      MONGO_INITDB_DATABASE: omnex

  ollama:
    image: ollama/ollama:latest
    container_name: omnex-ollama
    restart: unless-stopped
    volumes:
      - ./models/ollama:/root/.ollama
    ports:
      - '127.0.0.1:11434:11434'
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

### 1.5 Environment Variables

```bash
# .env
OMNEX_SOURCE_PATH=/mnt/source_drive     # Drive to index
OMNEX_DATA_PATH=/mnt/omnex         # Where omnex stores everything

MONGO_URI=mongodb://localhost:27017
MONGO_DB=omnex

OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=phi3:mini

API_HOST=127.0.0.1
API_PORT=8000

INGEST_WORKERS=4                             # Parallel ingestion workers
GPU_ENABLED=false                            # Set true if CUDA/ROCm available
LOG_LEVEL=INFO
```

---

## 2. Ingestion Pipeline — Implementation

### 2.1 File Detector (`ingestion/detector.py`)

```python
import magic
from pathlib import Path
from enum import Enum

class FileType(str, Enum):
    DOCUMENT = "document"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    CODE = "code"
    ARCHIVE = "archive"
    UNKNOWN = "unknown"

MIME_MAP = {
    "application/pdf": FileType.DOCUMENT,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": FileType.DOCUMENT,
    "text/plain": FileType.DOCUMENT,
    "text/markdown": FileType.DOCUMENT,
    "text/html": FileType.DOCUMENT,
    "image/jpeg": FileType.IMAGE,
    "image/png": FileType.IMAGE,
    "image/heic": FileType.IMAGE,
    "image/webp": FileType.IMAGE,
    "video/mp4": FileType.VIDEO,
    "video/x-matroska": FileType.VIDEO,
    "video/quicktime": FileType.VIDEO,
    "audio/mpeg": FileType.AUDIO,
    "audio/flac": FileType.AUDIO,
    "audio/x-wav": FileType.AUDIO,
    "application/zip": FileType.ARCHIVE,
    "application/x-tar": FileType.ARCHIVE,
}

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".go", ".rs", ".java", ".cpp",
    ".c", ".cs", ".rb", ".php", ".swift", ".kt", ".sh"
}

def detect(path: Path) -> tuple[FileType, str]:
    mime = magic.from_file(str(path), mime=True)
    if path.suffix.lower() in CODE_EXTENSIONS:
        return FileType.CODE, mime
    return MIME_MAP.get(mime, FileType.UNKNOWN), mime
```

### 2.2 Router (`ingestion/router.py`)

```python
from pathlib import Path
from ingestion.detector import detect, FileType
from ingestion.processors import document, image, video, audio, code, archive, unknown

PROCESSOR_MAP = {
    FileType.DOCUMENT: document.process,
    FileType.IMAGE: image.process,
    FileType.VIDEO: video.process,
    FileType.AUDIO: audio.process,
    FileType.CODE: code.process,
    FileType.ARCHIVE: archive.process,
    FileType.UNKNOWN: unknown.process,
}

async def route(path: Path) -> list[dict]:
    """Returns list of chunk dicts ready for storage."""
    file_type, mime_type = detect(path)
    processor = PROCESSOR_MAP[file_type]
    return await processor(path, mime_type)
```

### 2.3 Document Processor (`ingestion/processors/document.py`)

```python
from pathlib import Path
from ingestion.chunker import semantic_chunk
from embeddings.text import embed_texts
import pypdf
import docx

async def process(path: Path, mime_type: str) -> list[dict]:
    text = extract_text(path, mime_type)
    if not text.strip():
        return []

    chunks = semantic_chunk(text, chunk_size=512, overlap=64)
    embeddings = await embed_texts(chunks)

    return [
        {
            "text_content": chunk,
            "embedding": emb,
            "chunk_index": i,
            "chunk_total": len(chunks),
            "file_type": "document",
            "mime_type": mime_type,
        }
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]

def extract_text(path: Path, mime: str) -> str:
    if mime == "application/pdf":
        reader = pypdf.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if "wordprocessingml" in mime:
        doc = docx.Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    return path.read_text(errors="replace")
```

### 2.4 Image Processor (`ingestion/processors/image.py`)

```python
from pathlib import Path
from PIL import Image
import piexif
from embeddings.image import embed_image
from embeddings.faces import detect_faces

async def process(path: Path, mime_type: str) -> list[dict]:
    img = Image.open(path).convert("RGB")
    embedding = await embed_image(img)
    exif = extract_exif(path)
    faces = await detect_faces(img)

    thumbnail_ref = generate_thumbnail(img, path)

    return [{
        "text_content": None,
        "embedding": embedding,
        "chunk_index": 0,
        "chunk_total": 1,
        "file_type": "image",
        "mime_type": mime_type,
        "thumbnail_ref": thumbnail_ref,
        "metadata": {
            "dimensions": {"w": img.width, "h": img.height},
            "gps": exif.get("gps"),
            "created_at": exif.get("datetime"),
            "device": exif.get("device"),
            "faces": [f["cluster_candidate"] for f in faces],
        }
    }]

def extract_exif(path: Path) -> dict:
    try:
        exif_data = piexif.load(str(path))
        gps = parse_gps(exif_data.get("GPS", {}))
        dt = exif_data.get("0th", {}).get(piexif.ImageIFD.DateTime)
        device = exif_data.get("0th", {}).get(piexif.ImageIFD.Make)
        return {"gps": gps, "datetime": dt, "device": device}
    except Exception:
        return {}
```

### 2.5 Chunker (`ingestion/chunker.py`)

```python
from transformers import AutoTokenizer

_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def semantic_chunk(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Token-aware chunker that respects sentence boundaries."""
    sentences = split_sentences(text)
    chunks = []
    current_tokens = []
    current_chunk = []

    for sentence in sentences:
        tokens = _tokenizer.encode(sentence, add_special_tokens=False)
        if len(current_tokens) + len(tokens) > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Overlap: keep last N tokens worth of sentences
            overlap_chunk = []
            overlap_count = 0
            for s in reversed(current_chunk):
                s_tokens = _tokenizer.encode(s, add_special_tokens=False)
                if overlap_count + len(s_tokens) <= overlap:
                    overlap_chunk.insert(0, s)
                    overlap_count += len(s_tokens)
                else:
                    break
            current_chunk = overlap_chunk
            current_tokens = _tokenizer.encode(" ".join(current_chunk), add_special_tokens=False)

        current_chunk.append(sentence)
        current_tokens.extend(tokens)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def split_sentences(text: str) -> list[str]:
    import re
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
```

### 2.6 File Watcher (`ingestion/watcher.py`)

```python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path
import asyncio
from ingestion.router import route
from storage.mongo import save_chunks
from storage.leann import add_to_index

class OmnexHandler(FileSystemEventHandler):
    def __init__(self, loop):
        self.loop = loop

    def on_created(self, event):
        if not event.is_directory:
            asyncio.run_coroutine_threadsafe(
                ingest_file(Path(event.src_path)), self.loop
            )

    def on_modified(self, event):
        if not event.is_directory:
            asyncio.run_coroutine_threadsafe(
                ingest_file(Path(event.src_path)), self.loop
            )

    def on_deleted(self, event):
        if not event.is_directory:
            asyncio.run_coroutine_threadsafe(
                delete_file(Path(event.src_path)), self.loop
            )

async def ingest_file(path: Path):
    from ingestion.hasher import file_hash
    from storage.mongo import get_by_hash

    h = file_hash(path)
    if await get_by_hash(h):
        return  # Already indexed, unchanged

    chunks = await route(path)
    chunk_ids = await save_chunks(chunks, source_path=str(path), source_hash=h)
    await add_to_index(chunks, chunk_ids)

def start_watcher(source_path: str, loop):
    handler = OmnexHandler(loop)
    observer = Observer()
    observer.schedule(handler, source_path, recursive=True)
    observer.start()
    return observer
```

---

## 3. ML Models — Integration Guide

### 3.1 Text Embeddings (`embeddings/text.py`)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

async def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=64)
    return embeddings.tolist()

async def embed_query(query: str) -> list[float]:
    results = await embed_texts([query])
    return results[0]
```

### 3.2 Image Embeddings (`embeddings/image.py`)

```python
import torch
import clip
from PIL import Image

_model = None
_preprocess = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

def get_model():
    global _model, _preprocess
    if _model is None:
        _model, _preprocess = clip.load("ViT-B/32", device=_device)
    return _model, _preprocess

async def embed_image(img: Image.Image) -> list[float]:
    model, preprocess = get_model()
    tensor = preprocess(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        features = model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()[0].tolist()

async def embed_text_for_image_search(text: str) -> list[float]:
    """CLIP text encoder — for cross-modal text→image search."""
    model, _ = get_model()
    tokens = clip.tokenize([text]).to(_device)
    with torch.no_grad():
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()[0].tolist()
```

### 3.3 Face Clustering (`embeddings/faces.py`)

```python
from deepface import DeepFace
from sklearn.cluster import DBSCAN
import numpy as np
from storage.mongo import get_all_face_embeddings, save_identity_cluster

async def detect_faces(img_path: str) -> list[dict]:
    """Returns face crops with embeddings for an image."""
    try:
        results = DeepFace.represent(
            img_path=img_path,
            model_name="Facenet",
            detector_backend="retinaface",
            enforce_detection=False
        )
        return [
            {
                "embedding": r["embedding"],
                "region": r["facial_area"],
                "confidence": r.get("face_confidence", 0)
            }
            for r in results if r.get("face_confidence", 0) > 0.85
        ]
    except Exception:
        return []

async def cluster_all_faces():
    """Run after full ingestion — clusters all detected face embeddings."""
    face_records = await get_all_face_embeddings()
    if not face_records:
        return

    embeddings = np.array([r["embedding"] for r in face_records])
    ids = [r["_id"] for r in face_records]

    clustering = DBSCAN(eps=0.4, min_samples=3, metric="cosine").fit(embeddings)
    labels = clustering.labels_

    clusters = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue  # Noise / outlier face
        clusters.setdefault(str(label), []).append(ids[idx])

    for cluster_id, face_ids in clusters.items():
        await save_identity_cluster(cluster_id, face_ids)
```

### 3.4 Whisper Transcription (`embeddings/audio.py`)

```python
import whisper
import tempfile
import subprocess
from pathlib import Path

_model = None

def get_model():
    global _model
    if _model is None:
        _model = whisper.load_model("small")  # or "medium" for better accuracy
    return _model

async def transcribe(audio_path: str) -> str:
    model = get_model()
    result = model.transcribe(audio_path, language=None)  # Auto language detection
    return result["text"]

async def extract_audio_from_video(video_path: str) -> str:
    """Extract audio track from video for transcription."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        out_path = f.name
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        out_path, "-y"
    ], capture_output=True)
    return out_path
```

---

## 4. Storage — Schema & Queries

### 4.1 MongoDB Client (`storage/mongo.py`)

```python
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import IndexModel, ASCENDING, DESCENDING
from bson import ObjectId
import os

client = AsyncIOMotorClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
db = client[os.getenv("MONGO_DB", "omnex")]

chunks = db.chunks
identities = db.identities
ingest_state = db.ingest_state

async def init_indexes():
    await chunks.create_indexes([
        IndexModel([("source_hash", ASCENDING)]),
        IndexModel([("source_path", ASCENDING)]),
        IndexModel([("file_type", ASCENDING)]),
        IndexModel([("metadata.created_at", DESCENDING)]),
        IndexModel([("tags", ASCENDING)]),
    ])
    await identities.create_indexes([
        IndexModel([("cluster_id", ASCENDING)]),
        IndexModel([("label", ASCENDING)]),
    ])

async def save_chunks(chunk_list: list[dict], source_path: str, source_hash: str) -> list[str]:
    docs = [
        {**chunk, "source_path": source_path, "source_hash": source_hash}
        for chunk in chunk_list
    ]
    result = await chunks.insert_many(docs)
    return [str(id_) for id_ in result.inserted_ids]

async def get_by_hash(hash_: str) -> dict | None:
    return await chunks.find_one({"source_hash": hash_})

async def get_chunks_by_ids(ids: list[str]) -> list[dict]:
    object_ids = [ObjectId(id_) for id_ in ids]
    cursor = chunks.find({"_id": {"$in": object_ids}})
    return await cursor.to_list(length=None)
```

### 4.2 LEANN Client (`storage/leann.py`)

LEANN is file-based -- no server process. Each collection is a `.leann` index file on the destination drive. Embeddings are NOT stored; only the graph structure (~12 bytes/node). Embeddings recompute on-demand during search traversal.

```python
from leann import LeannBuilder, LeannSearcher
import os

DATA_PATH = os.getenv("OMNEX_DATA_PATH", "/mnt/omnex")

INDEXES = {
    "text_chunks":  {"model": "sentence-transformers/all-MiniLM-L6-v2"},
    "image_chunks": {"model": "openai/clip-vit-base-patch32"},
    "video_frames": {"model": "openai/clip-vit-base-patch32"},
    "audio_chunks": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
    "code_chunks":  {"model": "microsoft/codebert-base"},
}

_builders: dict[str, LeannBuilder] = {}
_searchers: dict[str, LeannSearcher] = {}

def index_path(name: str) -> str:
    return os.path.join(DATA_PATH, "leann", f"{name}.leann")

def get_builder(name: str) -> LeannBuilder:
    if name not in _builders:
        _builders[name] = LeannBuilder(
            backend_name="hnsw",
            embedding_model=INDEXES[name]["model"],
        )
    return _builders[name]

def get_searcher(name: str) -> LeannSearcher | None:
    path = index_path(name)
    if name not in _searchers and os.path.exists(path):
        _searchers[name] = LeannSearcher(path)
    return _searchers.get(name)

async def add_to_index(chunks: list[dict], chunk_ids: list[str], collection: str):
    builder = get_builder(collection)
    for i, chunk in enumerate(chunks):
        metadata = {
            "chunk_id": chunk_ids[i],
            "file_type": chunk.get("file_type", ""),
            "tags": ",".join(chunk.get("tags", [])),
            "date": str(chunk.get("metadata", {}).get("created_at", "")),
            "source_path": chunk.get("source_path", ""),
            "faces": ",".join(chunk.get("metadata", {}).get("faces", [])),
            "gps_lat": str(chunk.get("metadata", {}).get("gps", {}).get("lat", "")),
            "gps_lng": str(chunk.get("metadata", {}).get("gps", {}).get("lng", "")),
        }
        if chunk.get("text_content"):
            builder.add_text(chunk["text_content"], metadata=metadata)
        elif chunk.get("embedding"):
            builder.add_vector(chunk["embedding"], metadata=metadata)

async def build_index(collection: str):
    """Finalise and write the LEANN index file. Call after each batch ingestion."""
    builder = get_builder(collection)
    builder.build_index(index_path(collection))
    _searchers.pop(collection, None)

async def search(
    collection: str,
    vector: list[float] | None = None,
    query_text: str | None = None,
    limit: int = 20,
    filters: dict | None = None
) -> list[dict]:
    searcher = get_searcher(collection)
    if not searcher:
        return []
    metadata_filters = build_leann_filters(filters) if filters else None
    if query_text:
        results = searcher.search(query_text, top_k=limit, metadata_filters=metadata_filters)
    elif vector:
        results = searcher.search_by_vector(vector, top_k=limit, metadata_filters=metadata_filters)
    else:
        return []
    return [
        {
            "score": r.score,
            "chunk_id": r.metadata.get("chunk_id"),
            "file_type": r.metadata.get("file_type"),
            "source_path": r.metadata.get("source_path"),
            "tags": r.metadata.get("tags", "").split(","),
        }
        for r in results
    ]

def build_leann_filters(filters: dict) -> dict:
    leann_filters = {}
    if "file_type" in filters:
        leann_filters["file_type"] = {"==": filters["file_type"]}
    if "date_from" in filters:
        leann_filters["date"] = {">=": filters["date_from"]}
    if "date_to" in filters:
        leann_filters.setdefault("date", {})["<="] = filters["date_to"]
    if "faces" in filters:
        leann_filters["faces"] = {"contains": filters["faces"][0]}
    return leann_filters
```


---

## 5. FastAPI Backend — Full Spec

### 5.1 Main App (`api/main.py`)

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from storage.mongo import init_indexes
from storage.leann import build_index
from api.routes import query, ingest, identity, chunks

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_indexes()
    # LEANN indexes build lazily on first ingestion batch
    yield

app = FastAPI(title="Omnex API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router, prefix="/query", tags=["query"])
app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
app.include_router(identity.router, prefix="/identity", tags=["identity"])
app.include_router(chunks.router, prefix="/chunk", tags=["chunks"])
```

### 5.2 Query Route (`api/routes/query.py`)

```python
from fastapi import APIRouter
from pydantic import BaseModel
from api.query_engine import run_query, refine_query

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None
    filters: dict | None = None
    limit: int = 20

class RefineRequest(BaseModel):
    session_id: str
    refinement: str
    limit: int = 20

@router.post("/")
async def search(req: QueryRequest):
    return await run_query(req.query, req.filters, req.limit, req.session_id)

@router.post("/refine")
async def refine(req: RefineRequest):
    return await refine_query(req.session_id, req.refinement, req.limit)
```

### 5.3 Query Engine (`api/query_engine.py`)

```python
from embeddings.text import embed_query
from embeddings.image import embed_text_for_image_search
from storage.leann import search as leann_search
from storage.mongo import get_chunks_by_ids
from ollama import AsyncClient as OllamaClient
import re

ollama = OllamaClient()
_sessions: dict[str, dict] = {}

async def run_query(
    query: str,
    filters: dict | None = None,
    limit: int = 20,
    session_id: str | None = None
) -> dict:
    parsed = parse_query(query)

    # Multi-collection search
    results = []

    if parsed["intent"] in ("visual", "mixed", "general"):
        clip_vec = await embed_text_for_image_search(query)
        img_results = await leann_search("image_chunks", vector=clip_vec, limit=limit // 2, filters=parsed["filters"])
        results.extend(img_results)

    if parsed["intent"] in ("text", "mixed", "general"):
        text_vec = await embed_query(query)
        text_results = await leann_search("text_chunks", query_text=query, limit=limit // 2, filters=parsed["filters"])
        results.extend(text_results)

    # Deduplicate + sort by score
    seen = set()
    deduped = []
    for r in sorted(results, key=lambda x: x["score"], reverse=True):
        if r["chunk_id"] not in seen:
            seen.add(r["chunk_id"])
            deduped.append(r)

    chunk_ids = [r["chunk_id"] for r in deduped[:limit]]
    chunks = await get_chunks_by_ids(chunk_ids)

    # LLM response generation
    context = build_context(query, chunks)
    llm_response = await generate_response(context)

    if session_id:
        _sessions[session_id] = {"query": query, "results": deduped, "chunks": chunks}

    return {
        "response": llm_response,
        "results": deduped[:limit],
        "chunks": chunks,
        "session_id": session_id
    }

def parse_query(query: str) -> dict:
    """Extract intent and filters from natural language query."""
    filters = {}
    intent = "general"

    q_lower = query.lower()

    # Detect visual intent
    visual_keywords = ["photo", "image", "picture", "screenshot", "pic", "video", "clip"]
    if any(k in q_lower for k in visual_keywords):
        intent = "visual"

    # Detect text intent
    text_keywords = ["document", "file", "note", "email", "code", "report"]
    if any(k in q_lower for k in text_keywords):
        intent = "text"

    # Date extraction (basic — enhance with dateparser)
    date_patterns = {
        r"last year": ("2024-01-01", "2024-12-31"),
        r"(\d{4})": None,  # Year extraction
        r"last month": None,
    }

    return {"intent": intent, "filters": filters}

def build_context(query: str, chunks: list[dict]) -> str:
    snippets = []
    for c in chunks[:5]:
        if c.get("text_content"):
            snippets.append(f"- {c['text_content'][:200]}")
    return f"User query: {query}\n\nRelevant data found:\n" + "\n".join(snippets)

async def generate_response(context: str) -> str:
    import os
    response = await ollama.chat(
        model=os.getenv("OLLAMA_MODEL", "phi3:mini"),
        messages=[
            {"role": "system", "content": "You are Omnex, an AI memory assistant. Help the user find and understand their personal data. Be concise."},
            {"role": "user", "content": context}
        ]
    )
    return response["message"]["content"]
```

---

## 6. Next.js Interface — Component Spec

### 6.1 Project Structure

```
interface/
├── app/
│   ├── layout.tsx
│   ├── page.tsx              # Main query interface
│   └── globals.css
├── components/
│   ├── QueryBar/
│   │   ├── index.tsx         # NL input + voice button
│   │   └── VoiceInput.tsx    # Web Speech API
│   ├── ResultGrid/
│   │   ├── index.tsx         # Masonry grid / list toggle
│   │   ├── ImageCard.tsx
│   │   ├── DocumentCard.tsx
│   │   └── VideoCard.tsx
│   ├── PreviewPane/
│   │   └── index.tsx         # Full preview modal
│   ├── RefinementRail/
│   │   └── index.tsx         # Contextual narrowing chips
│   ├── IdentityManager/
│   │   ├── index.tsx         # Face cluster review
│   │   └── ClusterCard.tsx
│   └── IngestionDashboard/
│       └── index.tsx         # Progress + stats
├── lib/
│   ├── api.ts                # FastAPI client
│   └── types.ts              # Shared TypeScript types
└── package.json
```

### 6.2 API Client (`lib/api.ts`)

```typescript
const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

export async function query(
  text: string,
  sessionId?: string,
  filters?: Record<string, unknown>
) {
  const res = await fetch(`${API_BASE}/query/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: text, session_id: sessionId, filters }),
  })
  if (!res.ok) throw new Error(`Query failed: ${res.status}`)
  return res.json()
}

export async function refine(sessionId: string, refinement: string) {
  const res = await fetch(`${API_BASE}/query/refine`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, refinement }),
  })
  return res.json()
}

export async function getIngestStatus() {
  return fetch(`${API_BASE}/ingest/status`).then(r => r.json())
}

export async function labelIdentity(clusterId: string, label: string) {
  return fetch(`${API_BASE}/identity/label`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cluster_id: clusterId, label }),
  }).then(r => r.json())
}
```

### 6.3 TypeScript Types (`lib/types.ts`)

```typescript
export type FileType = "document" | "image" | "video" | "audio" | "code" | "unknown"

export interface Chunk {
  _id: string
  source_path: string
  file_type: FileType
  mime_type: string
  text_content: string | null
  thumbnail_ref: string | null
  tags: string[]
  metadata: {
    created_at: string | null
    gps: { lat: number; lng: number } | null
    dimensions: { w: number; h: number } | null
    duration_seconds: number | null
    faces: string[]
    device: string | null
  }
}

export interface QueryResult {
  response: string
  results: Array<{ chunk_id: string; score: number; file_type: FileType }>
  chunks: Chunk[]
  session_id: string | null
}

export interface Identity {
  _id: string
  label: string | null
  cluster_id: string
  thumbnail_ref: string
  confirmed_count: number
  candidate_count: number
}
```

---

## 7. FUSE Layer — Go Implementation

### 7.1 Module Setup

```bash
mkdir -p omnex/fuse && cd omnex/fuse
go mod init github.com/sup3rus3r/omnex/fuse
go get github.com/winfsp/cgofuse/fuse
```

### 7.2 Filesystem (`fuse/fs.go`)

```go
package main

import (
    "encoding/json"
    "fmt"
    "net/http"
    "os"
    "strings"
    "time"

    "github.com/winfsp/cgofuse/fuse"
)

const apiBase = "http://127.0.0.1:8000"

type GoldenFS struct {
    fuse.FileSystemBase
    client *http.Client
}

func NewGoldenFS() *GoldenFS {
    return &GoldenFS{
        client: &http.Client{Timeout: 10 * time.Second},
    }
}

// Virtual root directories
var rootDirs = []string{"People", "Places", "By Year", "Documents", "Videos", "Code", "Search"}

func (fs *GoldenFS) Getattr(path string, stat *fuse.Stat_t, fh uint64) int {
    now := fuse.Now()
    if path == "/" || isVirtualDir(path) {
        stat.Mode = fuse.S_IFDIR | 0555
        stat.Nlink = 2
        stat.Atim, stat.Mtim, stat.Ctim = now, now, now
        return 0
    }
    // Check if path resolves to a chunk
    chunk := fs.resolveChunk(path)
    if chunk == nil {
        return -fuse.ENOENT
    }
    stat.Mode = fuse.S_IFREG | 0444
    stat.Size = int64(chunk.Size)
    stat.Nlink = 1
    return 0
}

func (fs *GoldenFS) Readdir(
    path string,
    fill func(name string, stat *fuse.Stat_t, ofst int64) bool,
    ofst int64,
    fh uint64,
) int {
    fill(".", nil, 0)
    fill("..", nil, 0)

    if path == "/" {
        for _, d := range rootDirs {
            fill(d, nil, 0)
        }
        return 0
    }

    // Fetch entries from API
    entries := fs.fetchEntries(path)
    for _, e := range entries {
        fill(e.Name, nil, 0)
    }
    return 0
}

func (fs *GoldenFS) Read(
    path string,
    buff []byte,
    ofst int64,
    fh uint64,
) int {
    chunk := fs.resolveChunk(path)
    if chunk == nil {
        return -fuse.ENOENT
    }
    data, err := fs.fetchRaw(chunk.ID)
    if err != nil {
        return -fuse.EIO
    }
    end := int(ofst) + len(buff)
    if end > len(data) {
        end = len(data)
    }
    n := copy(buff, data[ofst:end])
    return n
}

func isVirtualDir(path string) bool {
    parts := strings.Split(strings.Trim(path, "/"), "/")
    if len(parts) == 0 {
        return false
    }
    for _, d := range rootDirs {
        if parts[0] == d {
            return true
        }
    }
    return false
}

type ChunkInfo struct {
    ID   string
    Size int
    Name string
}

func (fs *GoldenFS) resolveChunk(path string) *ChunkInfo {
    resp, err := fs.client.Get(fmt.Sprintf("%s/chunk/resolve?path=%s", apiBase, path))
    if err != nil || resp.StatusCode != 200 {
        return nil
    }
    defer resp.Body.Close()
    var info ChunkInfo
    json.NewDecoder(resp.Body).Decode(&info)
    return &info
}

func (fs *GoldenFS) fetchRaw(chunkID string) ([]byte, error) {
    resp, err := fs.client.Get(fmt.Sprintf("%s/chunk/%s/raw", apiBase, chunkID))
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    buf := make([]byte, 0)
    tmp := make([]byte, 4096)
    for {
        n, err := resp.Body.Read(tmp)
        buf = append(buf, tmp[:n]...)
        if err != nil {
            break
        }
    }
    return buf, nil
}

type DirEntry struct {
    Name string `json:"name"`
}

func (fs *GoldenFS) fetchEntries(path string) []DirEntry {
    resp, err := fs.client.Get(fmt.Sprintf("%s/fuse/ls?path=%s", apiBase, path))
    if err != nil || resp.StatusCode != 200 {
        return nil
    }
    defer resp.Body.Close()
    var entries []DirEntry
    json.NewDecoder(resp.Body).Decode(&entries)
    return entries
}
```

### 7.3 Entry Point (`fuse/main.go`)

```go
package main

import (
    "fmt"
    "os"
    "github.com/winfsp/cgofuse/fuse"
)

func main() {
    mountPoint := os.Getenv("OMNEX_MOUNT")
    if mountPoint == "" {
        mountPoint = "/mnt/omnex"
    }

    fs := NewGoldenFS()
    host := fuse.NewFileSystemHost(fs)

    args := []string{mountPoint}
    if runtime.GOOS == "linux" {
        args = append(args, "-o", "allow_other")
    }

    fmt.Printf("Mounting Omnex at %s\n", mountPoint)
    if !host.Mount("", args) {
        fmt.Fprintln(os.Stderr, "Mount failed")
        os.Exit(1)
    }
}
```

---

## 8. Configuration Reference

### 8.1 `config.py`

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    source_path: str = "/mnt/source"
    data_path: str = "/mnt/omnex"

    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "omnex"

    leann_backend: str = "hnsw"          # hnsw or diskann
    leann_graph_degree: int = 32
    leann_build_complexity: int = 64

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "phi3:mini"

    api_host: str = "127.0.0.1"
    api_port: int = 8000

    ingest_workers: int = 4
    gpu_enabled: bool = False
    log_level: str = "INFO"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64
    video_segment_seconds: int = 30
    audio_segment_seconds: int = 30

    # Face clustering
    face_confidence_threshold: float = 0.85
    dbscan_eps: float = 0.4
    dbscan_min_samples: int = 3

    # Binary store
    max_chunk_size_mb: int = 4

    class Config:
        env_file = ".env"
        env_prefix = "OMNEX_"

settings = Settings()
```

---

## 9. Testing Strategy

### 9.1 Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific layer
pytest tests/ingestion/ -v
pytest tests/embeddings/ -v
pytest tests/storage/ -v
pytest tests/api/ -v
```

### 9.2 Key Test Cases

```python
# tests/ingestion/test_detector.py
def test_detects_pdf_by_content():
    # Rename a PDF to .txt — should still detect as document
    ...

def test_detects_image_heic():
    ...

# tests/ingestion/test_chunker.py
def test_chunk_respects_sentence_boundaries():
    text = "First sentence. Second sentence. Third sentence."
    chunks = semantic_chunk(text, chunk_size=10, overlap=2)
    # No chunk should split mid-sentence
    ...

def test_chunk_overlap_continuity():
    ...

# tests/storage/test_leann.py
async def test_add_and_search():
    # Add text chunks, build index, search, assert results
    ...

async def test_filter_by_date_range():
    # Verify LEANN metadata filters work correctly
    ...

# tests/api/test_query.py
async def test_query_returns_results():
    ...

async def test_compound_query_with_filters():
    ...
```

### 9.3 Integration Test — Full Ingest Cycle

```python
# tests/integration/test_full_cycle.py
async def test_ingest_and_retrieve_image():
    # 1. Ingest a known test image with known EXIF
    # 2. Wait for processing
    # 3. Query with a description matching the image
    # 4. Assert it appears in results
    ...

async def test_face_cluster_and_label():
    # 1. Ingest 5 images of same person
    # 2. Run clustering
    # 3. Assert they appear in same cluster
    # 4. Label the cluster
    # 5. Query by label — assert all 5 images returned
    ...
```

---

## 10. Performance Tuning

### 10.1 Ingestion Throughput

```python
# Use multiprocessing for CPU-bound embedding
from concurrent.futures import ProcessPoolExecutor
import asyncio

async def ingest_batch(paths: list[Path], workers: int = 4):
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks = [loop.run_in_executor(pool, ingest_sync, p) for p in paths]
        return await asyncio.gather(*tasks)
```

### 10.2 GPU Acceleration

```python
# embeddings/text.py — GPU check
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# Batch size tuning per hardware
BATCH_SIZES = {
    "cpu": 32,
    "cuda": 256,
}
batch_size = BATCH_SIZES.get(device, 32)
```

### 10.3 LEANN Index Tuning

```python
# For large collections (>1M chunks) -- tune graph degree and build complexity
builder = LeannBuilder(
    backend_name="hnsw",       # or "diskann" for very large sets
    graph_degree=32,            # higher = better recall, more graph storage
    build_complexity=64,        # higher = better quality, slower build
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
)

# DiskANN backend for collections >5M chunks
builder = LeannBuilder(
    backend_name="diskann",
    graph_degree=64,
    build_complexity=128,
)
```

---

## 11. Error Handling Patterns

### 11.1 Ingestion Errors

```python
# All processors should return empty list on failure, never raise
async def process(path: Path, mime: str) -> list[dict]:
    try:
        ...
    except Exception as e:
        logger.warning(f"Failed to process {path}: {e}")
        # Store a minimal error record so we know this file was seen
        return [{
            "file_type": "unknown",
            "source_path": str(path),
            "error": str(e),
            "chunk_index": 0,
            "chunk_total": 0,
        }]
```

### 11.2 API Error Responses

```python
from fastapi import HTTPException

@router.post("/")
async def search(req: QueryRequest):
    try:
        return await run_query(req.query, req.filters, req.limit, req.session_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail="Query processing failed")
```

---

## 12. Data Flow Diagrams

### 12.1 Ingestion Flow

```
File on source drive
        │
        ▼
[detector.py] ── mime type ──► [router.py]
                                    │
              ┌─────────────────────┼──────────────────────┐
              ▼                     ▼                       ▼
      [document.py]           [image.py]             [video.py]
      extract text            CLIP embed             frame sample
      semantic chunk          EXIF parse             Whisper AST
      MiniLM embed            face detect            chunk embed
              │                     │                       │
              └─────────────────────┼───────────────────────┘
                                    ▼
                          [storage/mongo.py]
                          save chunk metadata
                                    │
                                    ▼
                          [storage/qdrant.py]
                          upsert vectors
                                    │
                                    ▼
                          [storage/binary_store.py]
                          write raw binary chunks
```

### 12.2 Query Flow

```
User types query in Next.js
        │
        ▼
POST /query  (FastAPI)
        │
        ▼
[query_engine.py]
 parse_query() ── extract intent, entities, date hints
        │
        ├── visual intent ──► CLIP text encoder ──► search image_chunks
        │
        ├── text intent ───► MiniLM encoder ─────► search text_chunks
        │
        └── mixed ──────────► both ──────────────► merge + deduplicate
                                    │
                                    ▼
                          re-rank by score
                                    │
                                    ▼
                          fetch full chunks from MongoDB
                                    │
                                    ▼
                          build LLM context
                                    │
                                    ▼
                          Ollama (local LLM) ──► natural language response
                                    │
                                    ▼
                          return { response, results, chunks }
                                    │
                                    ▼
                          Next.js renders result grid + response
```

---

> **Tech doc version mirrors architecture doc version.**
> Update both when making significant changes.
> 
> *Omnex — github.com/sup3rus3r/omnex*
