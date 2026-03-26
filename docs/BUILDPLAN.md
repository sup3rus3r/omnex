# Omnex — Living Build Plan
> v0.1 — Working Document | Updated continuously as decisions are made
> *The AI OS memory layer. Local-first. Open source. Agentic-ready.*

---

## The North Star

Omnex is not a search tool. It is the **memory substrate for the agentic era** — the data layer that sits beneath AI agents, AI operating systems, and human-AI interaction. What exists today (file systems, folder hierarchies, manual search) is legacy. Omnex is the foundation of what comes next.

**What we are building:**
- A local-first, privacy-absolute memory layer for all personal data
- Multi-modal: documents, images, video, audio, code — all queryable by meaning
- Voice-first communication interface — text is secondary
- Dual-consumer architecture: humans via UI today, agents via API tomorrow
- Open source — the community builds what big tech won't

**What we are not building:**
- A cloud sync tool
- A file manager
- A fancy search UI
- Anything that requires your data to leave your hardware

---

## Current Status

| Document | Purpose | Status |
|---|---|---|
| Omnex_Architecture.md | System architecture & technical blueprint | v0.1 — active |
| Omnex_TechDoc.md | Implementation reference for contributors | v0.1 — active |
| Omnex_BuildPlan.md | This document — living task tracker & decisions log | v0.1 — active |

---

## Architecture Decisions Log

> Record every significant decision here with reasoning. This prevents revisiting settled questions.

| # | Decision | Rationale | Date |
|---|---|---|---|
| 001 | Use LEANN over Qdrant for vector storage | 97% storage reduction — allows destination drive to match or be smaller than source drive. Critical for product viability. Will grow with LEANN as it matures. | 2026-03-26 |
| 002 | Local-first, no cloud dependency | Privacy by design. Structural advantage over big tech — they cannot copy this without destroying their own model. | 2026-03-26 |
| 003 | Variative ingestion: file → folder → drive | Allows users to test with small scope (single folder, minutes to index) before committing full drive (hours). Solves cold start anxiety and delivers wow moment fast. | 2026-03-26 |
| 004 | Voice as first-class input, not a feature | If Omnex is the AI OS memory layer, voice is the primary human input modality. Web Speech API for v1, local Whisper-based for full offline in future. | 2026-03-26 |
| 005 | UI is a communication layer, not a file browser | The interface is a conversation with your data. Result display supports the conversation — it is not the destination. | 2026-03-26 |
| 006 | Dual-consumer API design from day one | Humans use the UI. Agents use the API. Same backend serves both. Exposing the API early lets the developer/agent ecosystem build on Omnex before v1. | 2026-03-26 |
| 007 | Human-in-the-loop for cold start tagging | Auto-tagger needs user data to learn. Surface labeling prompts progressively during ingestion. User builds taxonomy naturally. | 2026-03-26 |
| 008 | Cold start time is acceptable | 8–96 hours for full drive indexing is expected for a product of this scope. The value proposition justifies it. Frame it as "building your memory", not "indexing". | 2026-03-26 |
| 009 | FUSE layer is adaptive | WinFsp/cgofuse is the plan. Approach is flexible if platform edge cases require alternatives. Core functionality before OS integration. | 2026-03-26 |

---

## Build Phases

### Phase 0 — Foundation
**Goal:** Repo scaffolded, services running, environment reproducible.

- [ ] Create GitHub repository (omnex)
- [ ] Write `docker-compose.yml` — MongoDB + Ollama services
- [ ] Write `requirements.txt` — full Python dependency list
- [ ] Write `models/download.py` — first-run model downloader
- [ ] Write `install.sh` — Linux one-command installer
- [ ] Write `install.ps1` — Windows one-command installer
- [ ] Confirm LEANN installs and initialises correctly on Windows + Linux
- [ ] Confirm Ollama runs locally with Phi-3 Mini
- [ ] README.md with vision statement + install instructions

**Exit criteria:** `docker compose up` + install script → all services healthy, models downloaded.

---

### Phase 1 — Variative Ingestion Core
**Goal:** User can point Omnex at a file, folder, or drive and it ingests everything.

- [ ] `ingestion/detector.py` — python-magic file type detection (not extension-based)
- [ ] `ingestion/router.py` — routes detected files to correct processor
- [ ] `ingestion/hasher.py` — SHA256 content hashing (skip unchanged files)
- [ ] `ingestion/chunker.py` — chunking strategies: semantic (text), fixed (binary), AST (code)
- [ ] `ingestion/processors/document.py` — PDF, DOCX, TXT, MD, HTML, XLSX, PPTX
- [ ] `storage/mongo.py` — MongoDB client, chunk schema, identities schema
- [ ] `storage/binary_store.py` — GridFS-inspired chunked binary file storage
- [ ] `storage/leann_store.py` — LEANN index builder for text chunks
- [ ] `embeddings/text.py` — MiniLM sentence-transformers wrapper
- [ ] Ingestion scope selector: accept `--path` as file, folder, or drive root
- [ ] Progress reporting: files processed / total, current file, estimated remaining

**Exit criteria:** Point at a folder of documents → all text extracted, chunked, embedded, stored in MongoDB + LEANN. Query returns results.

---

### Phase 2 — Image Ingestion
**Goal:** Images are indexed semantically — by scene, content, GPS, date.

- [ ] `ingestion/processors/image.py` — CLIP embedding + EXIF parser
- [ ] `embeddings/image.py` — CLIP ViT-B/32 wrapper
- [ ] `storage/leann_store.py` — extend to image index (512-dim CLIP)
- [ ] Thumbnail generation — 256×256 JPG stored in `/thumbnails/`
- [ ] GPS metadata extraction → stored in chunk metadata
- [ ] Date/time metadata extraction (EXIF + file timestamps)
- [ ] Binary chunk storage for image files

**Exit criteria:** Point at a folder of photos → images indexed with CLIP embeddings, EXIF metadata, thumbnails generated.

---

### Phase 3 — Query Engine + Basic Interface
**Goal:** First wow moment. User types or speaks a query and gets results.

- [ ] `api/main.py` — FastAPI app initialisation
- [ ] `api/query_engine.py` — NL query parsing, multi-index LEANN search, result ranking
- [ ] `api/routes/query.py` — POST /query, POST /query/refine
- [ ] `api/routes/chunks.py` — GET /chunk/{id}, GET /chunk/{id}/raw
- [ ] `api/routes/ingest.py` — POST /ingest/trigger, GET /ingest/status
- [ ] Next.js app scaffold — `interface/`
- [ ] `QueryBar` component — text input + voice input (Web Speech API)
- [ ] `ResultGrid` component — masonry layout, thumbnails, document previews
- [ ] `PreviewPane` component — image viewer, document reader
- [ ] `IngestionDashboard` component — live progress, scope selector (file/folder/drive)
- [ ] Scope selector UI — user picks file, folder, or drive to ingest
- [ ] Voice input — Web Speech API → text → query flow
- [ ] Cross-modal query — text query searches both text and image indexes

**Exit criteria:** User opens UI → selects a folder → watches ingestion progress → asks "photos from my holiday" by voice or text → sees relevant images returned.

---

### Phase 4 — Face Clustering & Identity
**Goal:** "Find photos with my sister" works.

- [ ] `ingestion/processors/image.py` — extend with face detection
- [ ] `embeddings/faces.py` — RetinaFace detection + FaceNet embeddings + DBSCAN clustering
- [ ] `storage/mongo.py` — extend identities collection
- [ ] `api/routes/identity.py` — POST /identity/label, GET /identity/pending
- [ ] `IdentityManager` component — face cluster review, naming UI, correction
- [ ] Query engine — person-based queries via identity filter
- [ ] Online learning — new photos auto-classify against known clusters

**Exit criteria:** After image ingestion, user is shown face clusters and names them once. "Photos with Sarah" returns correctly filtered results.

---

### Phase 5 — Audio & Video Ingestion
**Goal:** Spoken content and video are searchable by what was said and what was seen.

- [ ] `ingestion/processors/audio.py` — 30-second segmentation + Whisper transcription
- [ ] `ingestion/processors/video.py` — frame sampling (1fps) + Whisper transcription + CLIP frames
- [ ] `embeddings/audio.py` — Whisper wrapper
- [ ] LEANN indexes for audio chunks and video frames
- [ ] Video player component in PreviewPane — seeks to relevant timestamp
- [ ] Cross-modal query — "video where I talked about the project" → transcript search

**Exit criteria:** Point at a folder of videos/recordings → transcripts extracted, frames indexed → spoken content findable by natural language query.

---

### Phase 6 — Code Ingestion
**Goal:** Code across all languages is semantically indexed at function/class level.

- [ ] `ingestion/processors/code.py` — AST-aware chunker (function/class boundaries)
- [ ] `embeddings/code.py` — CodeBERT embeddings wrapper
- [ ] LEANN code index (768-dim CodeBERT)
- [ ] Code viewer in PreviewPane — syntax highlighted
- [ ] Language detection and tagging

**Exit criteria:** Point at a code repository → functions and classes indexed → "authentication middleware code" returns relevant code chunks.

---

### Phase 7 — Neural Auto-Tagger
**Goal:** Every chunk is automatically tagged. Tags improve as user interacts.

- [ ] `embeddings/tagger.py` — rule-based tagging pipeline (initial)
- [ ] Initial tags: date, location, scene type, file type, people present, source app
- [ ] Human-in-the-loop tagging prompts during ingestion (cold start)
- [ ] Tag storage as metadata fields in MongoDB + LEANN payload filters
- [ ] Tag-based filtering in query engine
- [ ] Progressive fine-tuning path — user label behaviour feeds classifier over time

**Exit criteria:** After ingestion, every chunk has meaningful tags. Queries like "work documents from last year" use tag filters to narrow results.

---

### Phase 8 — File Watcher (Incremental Indexing)
**Goal:** Omnex stays current automatically. New files indexed as they appear.

- [ ] `ingestion/watcher.py` — watchdog file system monitor
- [ ] Change detection — new, modified, deleted file events
- [ ] Content hash comparison — skip unchanged files
- [ ] Background processing queue — new files processed without interrupting queries
- [ ] Deleted file handling — mark chunks as removed in index

**Exit criteria:** Add a file to a watched folder → it appears in search results within seconds without manual re-trigger.

---

### Phase 9 — LLM Chat Layer
**Goal:** Conversational interaction with data. Omnex explains, summarises, suggests.

- [ ] Ollama integration — context assembly pipeline
- [ ] `api/routes/query.py` — extend to pass context window to LLM
- [ ] Conversational refinement — "show me more like that" / "from the same trip"
- [ ] LLM-generated result explanations — why was this returned?
- [ ] Refinement rail in UI — contextual narrowing suggestions
- [ ] Session awareness — multi-turn conversation maintains context
- [ ] Voice output — text-to-speech for LLM responses (browser TTS API)

**Exit criteria:** User has a back-and-forth conversation with their data. Asks follow-up questions. Gets coherent, contextually aware responses. Full voice in/voice out supported.

---

### Phase 10 — Agentic API Layer
**Goal:** External agents can query and interact with Omnex programmatically.

- [ ] Document all API endpoints formally (OpenAPI spec)
- [ ] API key / local auth for agent access
- [ ] Agent-oriented endpoints — structured JSON responses optimised for agent consumption
- [ ] Webhook support — agents subscribe to new data events
- [ ] MCP (Model Context Protocol) server — Omnex as an MCP tool for Claude, GPT, etc.
- [ ] Rate limiting and queue management for concurrent agent queries
- [ ] Developer documentation — how to build agents on Omnex

**Exit criteria:** An external agent (Claude, custom script) can query Omnex via API, receive structured results, and act on them without human involvement.

---

### Phase 11 — FUSE Virtual Filesystem (Read)
**Goal:** Omnex mounts as a virtual drive. Apps can read from it transparently.

- [ ] `fuse/main.go` — Go entry point
- [ ] `fuse/fs.go` — cgofuse filesystem implementation (read-only)
- [ ] `fuse/api_client.go` — HTTP client calling FastAPI backend
- [ ] Virtual directory structure: People/, Places/, By Year/, Documents/, Code/, Videos/
- [ ] Dynamic `Search/` directory — type a folder name to execute a query
- [ ] Windows: WinFsp integration + testing
- [ ] Linux: libfuse integration + testing
- [ ] Mount point configuration

**Exit criteria:** Omnex mounts as `G:\` (Windows) or `/mnt/omnex` (Linux). Virtual directories browse correctly. Files open in native applications.

---

### Phase 12 — FUSE Virtual Filesystem (Write + Sync)
**Goal:** Writing to the virtual drive triggers ingestion. Full bidirectional behaviour.

- [ ] Write interception — new file written to virtual drive → ingestion pipeline triggered
- [ ] Delete behaviour — marks chunk deleted, optional source file deletion (user config)
- [ ] Rename/move — metadata update only, no physical movement
- [ ] Source drive reconciliation — changes on source reflected in index
- [ ] Full bidirectional sync

**Exit criteria:** Drag a file onto the Omnex virtual drive → it is ingested and immediately queryable. Delete from virtual drive → removed from index.

---

## Interface Design Principles

These are not negotiable. Every UI decision must pass these checks:

1. **Communication first** — The interface is a conversation with data, not a file browser. Every design decision should make the conversation clearer.
2. **Voice equal to text** — Voice input and text input are equally prominent. Neither is an afterthought.
3. **Progressive disclosure** — Show what's needed now. Surface complexity only when the user needs it.
4. **Trust signals** — User must always feel in control of their data. Show what was indexed, what models are running, what is local.
5. **Future-proof** — The UI is the human interface today. In the AI OS era it becomes a nice-to-have. Design so the API can carry the full product without the UI.

---

## Ingestion Scope UX Flow

```
User opens Omnex
        ↓
"What would you like to remember?"
        ↓
[ Add a file ]  [ Add a folder ]  [ Add a drive ]
        ↓
Scope selected → ingestion begins
        ↓
Live progress: "Remembering 1,240 of 4,820 items..."
        ↓
First results available → prompt user to try a query
        ↓
"Your memory is ready. Ask me anything."
        ↓
Progressive expansion prompt:
"You've remembered 1 folder. Add your full drive to unlock your complete memory."
```

---

## Voice Interface Flow

```
User activates voice (button or wake word — future)
        ↓
Browser Web Speech API captures audio → text
        ↓
Text submitted to query engine (same path as typed query)
        ↓
Results returned + LLM response generated
        ↓
Response read aloud via browser TTS API
        ↓
User continues conversation or refines
```

**Future path:** Replace Web Speech API with local Whisper for fully offline voice operation.

---

## Dual-Consumer Architecture

```
                    ┌─────────────────┐
                    │  Omnex Core │
                    │  (FastAPI + LEANN│
                    │   + MongoDB)     │
                    └────────┬────────┘
                             │
               ┌─────────────┴─────────────┐
               │                           │
    ┌──────────▼──────────┐    ┌───────────▼───────────┐
    │   Human Interface   │    │    Agent Interface     │
    │   (Next.js UI)      │    │    (API / MCP Server)  │
    │   Voice + Text      │    │    Structured JSON     │
    │   Visual results    │    │    Webhooks            │
    └─────────────────────┘    └───────────────────────┘
```

Same backend. Same memory. Two consumers. Humans now. Agents always.

---

## Open Questions

> Track unresolved decisions here. Resolve and move to Architecture Decisions Log when settled.

| # | Question | Context |
|---|---|---|
| Q001 | Wake word support for voice? | Always-on listening vs push-to-talk. Privacy implications of always-on on local hardware. |
| Q002 | Multi-user / family support? | Separate identity spaces on same machine? Shared identity clusters? |
| Q003 | Mobile ingestion path? | Photos from phone — direct WiFi sync to Omnex instance? |
| Q004 | Encryption at rest on destination drive? | User expectation for sensitive data. Performance cost? |
| Q005 | MCP server priority? | Claude/GPT agent integration — does this come before or after FUSE? |

---

## Key Metrics to Track

Once live, these tell us if we're succeeding:

- **Time to wow moment** — how long from install to first meaningful query result
- **Ingestion throughput** — files/minute on reference hardware (CPU and GPU)
- **Query latency** — time from query submission to first result displayed
- **Voice round-trip** — time from end of speech to first result displayed
- **Storage ratio** — destination drive size / source drive size (target: ≤ 1.0x)

---

*This document is the source of truth for what we are building and why. Update it before writing code. Every significant decision goes in the Architecture Decisions Log. Every completed task gets checked off.*

*Omnex — github.com/sup3rus3r/omnex*
