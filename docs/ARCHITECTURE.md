# Omnex — Architecture & Technical Blueprint
> v0.1 — Living Document | Open Source | Local-First | AI-Native
> 
> *Your data. Your memory. No file system.*

---

## Table of Contents

1. [Vision & Philosophy](#1-vision--philosophy)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Ingestion Pipeline](#3-ingestion-pipeline)
4. [ML Processing Layer](#4-ml-processing-layer)
5. [Storage Layer](#5-storage-layer)
6. [Query Engine](#6-query-engine)
7. [AI Interface](#7-ai-interface-nextjs--ollama)
8. [FUSE Virtual Filesystem Layer](#8-fuse-virtual-filesystem-layer)
9. [Repository Structure](#9-repository-structure)
10. [Build Roadmap](#10-build-roadmap)
11. [Dependencies & System Requirements](#11-dependencies--system-requirements)
12. [Open Source & Contribution](#12-open-source--contribution)

---

## 1. Vision & Philosophy

Omnex is a paradigm shift in how humans relate to their personal data. The traditional file system — hierarchical folders, filenames, manual search — is a relic of an era before AI. Omnex replaces this model with an AI-native memory layer: your data is indexed by meaning, not location.

**The core thesis:** You should never search for a file again. You recall your data the way you recall a memory — by context, time, people, place, feeling — and Omnex retrieves it.

### 1.1 The Problem

- Finding a photo from 3 years ago requires manual scrolling through thousands of images
- Documents are buried in folder structures that made sense when created but are opaque later
- Video and audio content is completely unsearchable by conventional means
- Data spread across drives, devices, and formats has no unified memory layer
- The backup drive paradigm exists only for failure protection — not intelligence

### 1.2 The Solution

> Omnex ingests every file on a source drive — documents, images, video, audio, code — and builds a semantic memory on a second drive. Every piece of data is chunked, embedded, tagged, and made queryable through a natural language interface. The file system is abstracted away entirely.
>
> Long-term: Omnex becomes the AI OS memory layer. Traditional file systems cease to exist as a user-facing concept. A FUSE virtual filesystem ensures existing applications continue to function.

### 1.3 Design Principles

- **Local-first** — All processing, models, and storage run on-device. No cloud dependency.
- **Open source** — AGPL-3.0 licensed, community driven from day one.
- **Privacy by design** — Your data never leaves your hardware.
- **Progressive enhancement** — Starts as a search tool, evolves into an OS memory layer.
- **Platform agnostic** — Windows and Linux first, macOS to follow.
- **Hardware inclusive** — Designed to run on modest hardware. GPU optional.

---

## 2. System Architecture Overview

Omnex is composed of six distinct layers, each with a clear responsibility boundary. They communicate via internal APIs and shared storage interfaces.

```
SOURCE DRIVE  →  [Ingestion Pipeline]  →  [ML Processing Layer]
                        ↓                           ↓
                [Storage Layer: MongoDB + LEANN index + GridFS-style binary chunks]
                        ↓
                [Query Engine]  →  [AI Interface (Next.js + LLM)]
                        ↓
                [FUSE Virtual Filesystem]  ←→  OS / Applications
```

| Layer | Technology | Responsibility |
|---|---|---|
| Ingestion Pipeline | Python | File detection, routing, chunking, preprocessing |
| ML Processing | Python (PyTorch/ONNX) | Embeddings, face clustering, transcription, tagging |
| Storage Layer | MongoDB + LEANN | Chunk storage, vector index, metadata, relationships |
| Query Engine | FastAPI | NL query parsing, vector search, result ranking |
| AI Interface | Next.js + Ollama | Conversational UI, result display, refinement |
| FUSE Layer | Go (cgofuse) | Virtual filesystem, OS integration, transparent R/W |

---

## 3. Ingestion Pipeline

The ingestion pipeline is the entry point for all data. It runs on first-use as a full drive scan and thereafter as a file watcher for incremental updates. It detects file types, routes them to the appropriate processor, chunks them GridFS-style, and hands off to the ML layer.

### 3.1 File Type Detection & Routing

Detection is done via `python-magic` (libmagic bindings) — not file extension — for accuracy on renamed or extensionless files.

| Category | Types | Processor |
|---|---|---|
| Documents | PDF, DOCX, TXT, MD, HTML, XLSX, PPTX | Text extractor → chunker → embedder |
| Images | JPG, PNG, HEIC, RAW, WEBP, GIF | CLIP embedder + EXIF parser + face detector |
| Video | MP4, MKV, MOV, AVI, WEBM | Frame sampler + Whisper transcription + CLIP frames |
| Audio | MP3, WAV, FLAC, M4A, OGG | Whisper transcription → text chunker → embedder |
| Code | PY, JS, TS, GO, RS, JAVA, etc. | AST-aware chunker → code embedder (CodeBERT) |
| Archives | ZIP, TAR, RAR, 7Z | Recursive extraction → route each extracted file |
| Unknown | Any unrecognised | Raw binary chunk + filename/path metadata only |

### 3.2 Chunking Strategy

All data is chunked before storage. Chunking follows different strategies per type, inspired by MongoDB GridFS's approach to binary data segmentation.

- **Text/Documents** — Semantic chunking (512 token window, 64 token overlap). Respects paragraph and sentence boundaries.
- **Images** — Stored as single chunks. Thumbnails generated at 256×256 for UI preview.
- **Video** — Split into 30-second segments. Each segment: frame samples (1fps) + audio chunk. Segments stored independently.
- **Audio** — Split into 30-second segments with 5-second overlap for continuity.
- **Code** — Chunked at function/class boundaries using AST parsing. Falls back to line-based if AST fails.

### 3.3 Incremental Indexing

After initial full-drive scan, a file watcher (`watchdog` on Python) monitors the source drive for changes. New, modified, and deleted files are processed in real-time. Each chunk carries a content hash — unchanged files are never re-processed.

### 3.4 Cold Start Performance Estimates

| Drive Size | File Count (est.) | Est. Time (CPU) | Est. Time (GPU) |
|---|---|---|---|
| 500 GB | ~200K files | 8–12 hours | 2–3 hours |
| 1 TB | ~400K files | 16–24 hours | 4–6 hours |
| 2 TB | ~800K files | 32–48 hours | 8–12 hours |
| 4 TB | ~1.5M files | 60–96 hours | 15–24 hours |

> **Note:** Cold start runs as a background process. The system is queryable from the first indexed files. Users see progressive results as indexing continues. Estimated times assume mixed media workloads. Text-heavy drives will be significantly faster. Video-heavy drives significantly slower.

---

## 4. ML Processing Layer

The ML layer is responsible for generating semantic meaning from raw data. It runs locally on-device using quantized models selected for the best accuracy-to-resource tradeoff.

### 4.1 Model Stack

| Function | Model | Size | Backend |
|---|---|---|---|
| Text embeddings | all-MiniLM-L6-v2 | ~80 MB | sentence-transformers |
| Image/video understanding | CLIP ViT-B/32 | ~350 MB | openai/clip via transformers |
| Face detection | RetinaFace | ~100 MB | deepface |
| Face clustering & identity | FaceNet (via DeepFace) | ~250 MB | deepface |
| Audio/video transcription | Whisper small / medium | 250–500 MB | openai-whisper |
| Code embeddings | CodeBERT base | ~500 MB | transformers |
| Chat / query LLM | Phi-3 Mini or Gemma 3 2B | 2–4 GB | Ollama |
| Neural auto-tagger | Custom fine-tuned classifier | ~200 MB | PyTorch (trained on user data) |

> **Total model footprint:** ~3.5–5 GB | **Minimum RAM:** 8 GB | **GPU:** Optional (CUDA/ROCm supported)

### 4.2 Embedding Architecture

LEANN uses graph-based selective recomputation — it stores a pruned adjacency graph (~12 bytes/node) and recomputes embeddings on-demand during search instead of persisting all float32 vectors. This achieves 97% storage reduction with no accuracy loss.

- **Text index** — 384-dim MiniLM, cosine similarity, ~12 bytes/chunk stored
- **Image/video index** — 512-dim CLIP, cosine similarity, ~12 bytes/chunk stored
- **Cross-modal search** — text query → CLIP text encoder → search image index (allows *"find my photos of Table Mountain"*)
- **Destination drive savings** — a 2TB source drive needs ~same-size destination (not 1.5–2x)

### 4.3 Face Clustering & Identity

Face clustering is a core subsystem for the *"find photos with my sister"* use case. It operates in four phases:

- **Phase 1 — Detection:** RetinaFace detects faces in every image and video frame. Face crops are extracted and stored.
- **Phase 2 — Clustering:** FaceNet generates 128-dim face embeddings. DBSCAN clustering groups faces into identity clusters without requiring labeled training data.
- **Phase 3 — Labeling:** User is presented with cluster samples and asked to name them once. Labels propagate to all matching faces across the entire dataset.
- **Phase 4 — Online learning:** New photos auto-classify against known identity clusters. Confidence threshold triggers user review for ambiguous cases.

### 4.4 Neural Auto-Tagger

The auto-tagger assigns semantic tags to every chunk at ingest time. It starts as a rule-based system and evolves into a fine-tuned classifier trained on the user's own labeling behavior over time.

- **Initial tags:** date, location (GPS or inferred), scene type, people present, file type, source application
- **Learned tags:** user-defined categories emerge from query patterns and explicit labels
- Tags are stored as metadata in MongoDB and as filterable payload fields in LEANN

---

## 5. Storage Layer

The storage layer lives entirely on the secondary (destination) drive. It uses three complementary systems: MongoDB for metadata and relationships, LEANN for the vector index (97% storage savings over Qdrant), and a custom chunked binary store for raw data.

### 5.1 MongoDB Schema

#### `chunks` collection

```json
{
  "_id": "ObjectId",
  "source_path": "String",
  "source_hash": "String",
  "chunk_index": "Int",
  "chunk_total": "Int",
  "file_type": "image | video | audio | document | code",
  "mime_type": "String",
  "data_ref": "String",
  "text_content": "String | null",
  "leann_id": "String",          // LEANN internal node ID
  "tags": ["String"],
  "metadata": {
    "created_at": "Date",
    "ingested_at": "Date",
    "gps": { "lat": "Float", "lng": "Float" },
    "duration_seconds": "Float",
    "dimensions": { "w": "Int", "h": "Int" },
    "faces": ["face_id: String"],
    "device": "String",
    "app_source": "String"
  },
  "embedding_model": "String",
  "created_at": "Date",
  "updated_at": "Date"
}
```

#### `identities` collection

```json
{
  "_id": "ObjectId",
  "label": "String",
  "cluster_id": "String",
  "face_embeddings": ["Array"],
  "confirmed_chunks": ["ObjectId"],
  "candidate_chunks": ["ObjectId"],
  "thumbnail_ref": "String",
  "created_at": "Date"
}
```

### 5.2 LEANN Index Files

LEANN stores each collection as a `.leann` file on the destination drive. No server process required — it's a file-based index.

| Index file | Vector Dim | Metric | Metadata filters |
|---|---|---|---|
| text_chunks.leann | 384 (MiniLM) | Cosine | file_type, tags, date, source_path |
| image_chunks.leann | 512 (CLIP) | Cosine | tags, date, gps, faces, dimensions |
| video_frames.leann | 512 (CLIP) | Cosine | video_id, timestamp, tags, faces |
| audio_chunks.leann | 384 (MiniLM) | Cosine | tags, date, duration, transcript_snippet |
| code_chunks.leann | 768 (CodeBERT) | Cosine | language, tags, source_path |

**Storage comparison (1TB mixed drive):** Qdrant ≈ 180 GB vector storage → LEANN ≈ 8 GB. Binary chunks, MongoDB, and thumbnails are identical for both.

### 5.3 Binary Chunk Store

Raw binary data is stored using a GridFS-inspired chunked approach. Files over 16MB are split into 4MB binary chunks stored as flat files in a content-addressed directory structure.

```
/omnex/
├── chunks/
│   ├── ab/
│   │   ├── ab3f92...chunk0.bin
│   │   └── ab3f92...chunk1.bin
│   └── c7/
│       └── c7e1a4...chunk0.bin
├── thumbnails/
│   └── {chunk_id}.jpg
├── models/
├── db/
└── leann/
```

---

## 6. Query Engine

The query engine exposes a FastAPI backend that translates natural language queries into multi-stage vector searches, applies metadata filters, ranks results, and returns structured responses to the interface layer.

### 6.1 Query Flow

```
1. User submits natural language query via interface
2. Query parser extracts: intent, entities (people, places, dates, file types), time ranges
3. Query is embedded using appropriate model (MiniLM for text, CLIP text encoder for visual)
4. Multi-index LEANN search with metadata pre-filters (date range, tags, file type) — embeddings recomputed on-demand during traversal
5. Results re-ranked using cross-encoder for precision
6. Context window assembled: top-N chunks + metadata passed to local LLM
7. LLM generates response with result references
8. Interface renders results with thumbnails, previews, and refinement suggestions
```

### 6.2 Query Types

| Query Type | Example | Strategy |
|---|---|---|
| Semantic text | "Find my notes about the Python project" | MiniLM embed → text_chunks search |
| Visual scene | "Photos of the beach at sunset" | CLIP text → image_chunks search |
| Person-based | "Photos with my sister Sarah" | Identity filter → image/video search |
| Temporal | "Documents I worked on last March" | Date range filter + semantic search |
| Location | "Photos from Cape Town" | GPS bbox filter + CLIP search |
| Cross-modal | "Video where I talked about the meeting" | Whisper transcript search → video chunks |
| Compound | "My sister at Clifton 3 years ago" | Identity + GPS + date range + CLIP |
| Conversational | "Show me more like that last one" | Similarity search from result embedding |

### 6.3 FastAPI Endpoints

```
POST /query              — Natural language search
POST /query/refine       — Narrow previous results (session-aware)
GET  /chunk/{id}         — Retrieve specific chunk with full metadata
GET  /chunk/{id}/raw     — Stream raw binary (image/video/audio)
POST /identity/label     — Assign name to face cluster
GET  /identity/pending   — Fetch clusters awaiting user labeling
GET  /stats              — Ingestion status, index size, model versions
POST /ingest/trigger     — Force re-scan of source path
GET  /ingest/status      — Current ingestion progress
POST /tag                — Manually tag chunk(s)
DELETE /chunk/{id}       — Remove chunk from index (does not delete source)
```

---

## 7. AI Interface (Next.js + Ollama)

The interface is a locally-served Next.js application that provides a conversational experience for querying and exploring the Omnex index.

### 7.1 Interface Components

- **Query bar** — Full-width natural language input with voice input support (browser Web Speech API)
- **Result grid** — Masonry layout for images/video thumbnails, list view for documents/text
- **Preview pane** — In-app viewer for images, video player, document reader, code viewer
- **Refinement rail** — "More like this", "From same day", "Also with Sarah" — contextual narrowing
- **Timeline view** — Results plotted on a time axis for temporal exploration
- **Identity manager** — Face cluster review, naming, and correction UI
- **Ingestion dashboard** — Live progress, file type breakdown, model status

### 7.2 Local LLM Integration

The chat layer uses Ollama running locally. The LLM receives: the user query, retrieved chunks as context, and metadata summaries. It generates natural language responses, suggests refinements, and can explain why results were returned.

| Model | RAM Required | Use Case |
|---|---|---|
| Phi-3 Mini 4K (3.8B) | ~4 GB | Minimum spec — fast responses, good reasoning |
| Gemma 3 2B | ~3 GB | Lightest option — lower accuracy |
| Llama 3.2 3B | ~4 GB | Balanced alternative |
| Phi-3 Medium (14B) | ~10 GB | High-spec — best comprehension |

### 7.3 Multimodal Query Input

- **Text** — Standard natural language query
- **Image upload** — "Find more images similar to this one" — CLIP embed the uploaded image → search image collection
- **Voice** — Browser speech recognition → text → standard query flow
- **Combined** — "Find photos like this but from last summer" — image embed + date filter

---

## 8. FUSE Virtual Filesystem Layer

The FUSE layer is the long-term OS integration component. It exposes Omnex as a virtual drive to the operating system, making the semantic index transparent to all applications while eliminating the traditional folder hierarchy as a user-facing concept.

### 8.1 Technology

- **Language** — Go (compiles to a single native binary, fast, cross-platform)
- **Library** — cgofuse (single codebase: Linux/libfuse, Windows/WinFsp, macOS/macFUSE)
- **Communication** — cgofuse layer calls Omnex FastAPI backend via local HTTP (127.0.0.1)
- **Mount point** — User-configurable (e.g. `G:\` on Windows, `/mnt/omnex` on Linux)

### 8.2 Virtual Filesystem Design

The virtual drive presents a dynamic directory structure generated from the semantic index. Directories are not real — they are query materializations.

```
G:/                           # Omnex virtual mount
├── People/
│   ├── Sarah/                # All chunks featuring Sarah
│   └── Mom/
├── Places/
│   ├── Cape Town/
│   └── Clifton/
├── By Year/
│   ├── 2022/
│   └── 2023/
├── Documents/
├── Code/
├── Videos/
└── Search/                   # Dynamic — type a folder name to search
    └── [query here]/         # Creates a query result directory on-demand
```

### 8.3 Read/Write Behaviour

- **Read** — File access → FUSE → FastAPI semantic lookup → stream chunk from binary store → return to app
- **Write** — New file written to virtual drive → FUSE intercepts → triggers ingestion pipeline → file stored and indexed
- **Delete** — Marks chunk as deleted in index. Source file optionally deleted based on user config.
- **Rename/Move** — Updates metadata only — no physical movement since there is no physical hierarchy

### 8.4 Build Sequence

- **Phase 1** — Read-only FUSE mount exposing virtual directory structure
- **Phase 2** — Write support triggering ingestion pipeline
- **Phase 3** — Dynamic `Search/` directory supporting on-demand query folders
- **Phase 4** — Full bidirectional sync with source drive reconciliation

---

## 9. Repository Structure

```
omnex/
├── ingestion/              # Python — file detection, routing, chunking
│   ├── detector.py         # python-magic based type detection
│   ├── router.py           # Routes files to correct processor
│   ├── processors/
│   │   ├── document.py     # PDF, DOCX, TXT, MD, HTML
│   │   ├── image.py        # CLIP + EXIF + face detection
│   │   ├── video.py        # Frame sampling + Whisper
│   │   ├── audio.py        # Whisper transcription
│   │   └── code.py         # AST-aware chunker + CodeBERT
│   ├── chunker.py          # Chunking strategies per type
│   ├── watcher.py          # watchdog file system monitor
│   └── hasher.py           # SHA256 content hashing
├── embeddings/             # Python — all ML model wrappers
│   ├── text.py             # MiniLM sentence-transformers
│   ├── image.py            # CLIP ViT-B/32
│   ├── audio.py            # Whisper
│   ├── code.py             # CodeBERT
│   ├── faces.py            # RetinaFace + FaceNet + DBSCAN
│   └── tagger.py           # Auto-tagging pipeline
├── storage/                # Python — data layer
│   ├── mongo.py            # MongoDB client + schema
│   ├── qdrant.py           # LEANN index builder + searcher
│   └── binary_store.py     # Chunked binary file storage
├── api/                    # FastAPI backend
│   ├── main.py
│   ├── routes/
│   │   ├── query.py
│   │   ├── ingest.py
│   │   ├── identity.py
│   │   └── chunks.py
│   └── query_engine.py     # NL parsing + multi-stage search
├── fuse/                   # Go — FUSE virtual filesystem
│   ├── main.go
│   ├── fs.go               # cgofuse filesystem implementation
│   ├── api_client.go       # HTTP client for FastAPI
│   └── go.mod
├── interface/              # Next.js frontend
│   ├── app/
│   ├── components/
│   │   ├── QueryBar/
│   │   ├── ResultGrid/
│   │   ├── PreviewPane/
│   │   ├── IdentityManager/
│   │   └── IngestionDashboard/
│   └── package.json
├── models/                 # Model download scripts + configs
│   └── download.py         # First-run model downloader
├── docker-compose.yml      # MongoDB + LEANN + Ollama services
├── install.sh              # Linux installer
├── install.ps1             # Windows installer
└── docs/
    ├── ARCHITECTURE.md     # This document
    └── CONTRIBUTING.md
```

---

## 10. Build Roadmap

| Phase | Milestone | Components | Status |
|---|---|---|---|
| 0 | Project Setup | Repo, docker-compose, installer scripts | Planned |
| 1 | Ingestion — Text & Docs | detector, router, document processor, chunker, MiniLM, LEANN index build | Planned |
| 2 | Ingestion — Images | image processor, CLIP, EXIF parser, binary store | Planned |
| 3 | Basic Query Interface | FastAPI /query, Next.js query bar + result grid | Planned |
| 4 | Face Clustering | RetinaFace, FaceNet, DBSCAN, identity collection, labeling UI | Planned |
| 5 | Ingestion — Audio & Video | Whisper, frame sampler, video + audio processors | Planned |
| 6 | Ingestion — Code | AST chunker, CodeBERT embedder | Planned |
| 7 | Neural Auto-Tagger | Rule-based tagging → fine-tuned classifier | Planned |
| 8 | File Watcher | watchdog incremental indexing, change detection | Planned |
| 9 | LLM Chat Layer | Ollama integration, context assembly, conversational refinement | Planned |
| 10 | FUSE Layer — Read | Go cgofuse, virtual directory structure, read-only mount | Planned |
| 11 | FUSE Layer — Write | Write interception → ingestion trigger | Planned |
| 12 | Windows Support | WinFsp integration, Windows installer, testing | Planned |

---

## 11. Dependencies & System Requirements

### 11.1 Minimum Hardware

| Component | Minimum | Recommended |
|---|---|---|
| CPU | 4-core x86_64 | 8-core modern CPU |
| RAM | 8 GB | 16–32 GB |
| GPU | None (CPU-only mode) | NVIDIA 8GB+ VRAM (CUDA) or AMD (ROCm) |
| Source Drive | Any size HDD/SSD | SSD preferred for faster ingestion |
| Destination Drive | 1.5x source drive size | 2x source drive size SSD |
| OS | Windows 10+ or Ubuntu 22.04+ | Ubuntu 24.04 LTS |

> **Note:** LEANN requires Python 3.9–3.13. No separate vector DB server process needed — indexes are file-based `.leann` files on the destination drive.

### 11.2 Python Dependencies

```txt
sentence-transformers    # MiniLM text embeddings
transformers             # CLIP, CodeBERT
torch                    # PyTorch inference backend
deepface                 # FaceNet + RetinaFace
openai-whisper           # Audio/video transcription
python-magic             # File type detection
pymongo                  # MongoDB client
leann                    # LEANN vector index — 97% storage savings vs Qdrant
fastapi                  # API backend
uvicorn                  # ASGI server
watchdog                 # File system monitoring
pillow                   # Image processing
opencv-python            # Video frame extraction
pypdf2                   # PDF text extraction
python-docx              # DOCX extraction
scikit-learn             # DBSCAN clustering
numpy
pydantic
```

### 11.3 Services (Docker Compose)

```yaml
services:
  mongodb:
    image: mongo:7
    volumes: ['./data/mongo:/data/db']
    ports: ['27017:27017']

  qdrant:
    image: leann/qdrant:latest
    volumes: ['./data/qdrant:/leann/storage']
    ports: ['6333:6333']

  ollama:
    image: ollama/ollama:latest
    volumes: ['./models:/root/.ollama']
    ports: ['11434:11434']
    deploy:
      resources:
        reservations:
          devices: [{driver: nvidia, count: all, capabilities: [gpu]}]
```

---

## 12. Open Source & Contribution

### 12.1 License

Omnex is released under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. This ensures the project remains open source even when deployed as a service.

### 12.2 Contribution Areas

- New file type processors (e-readers, CAD files, databases)
- Model adapters (support for additional embedding models)
- Platform support (macOS FUSE, ARM/Raspberry Pi)
- Performance optimization (GPU acceleration, parallel ingestion)
- UI components (timeline view, map view for GPS-tagged media)
- Language support (multilingual embeddings, Whisper language detection)

### 12.3 Versioning

Document version is tracked alongside code. Each significant architectural decision should be reflected in an update to this document. Version format: `MAJOR.MINOR` — major for paradigm shifts, minor for component additions or changes.

---

> **This is a living document.** Update it as the architecture evolves. Every major build decision should be captured here before code is written. Use this document as the source of truth when working with AI coding assistants.

---

*Omnex — github.com/sup3rus3r/omnex*
