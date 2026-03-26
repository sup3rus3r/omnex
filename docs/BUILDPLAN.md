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
| docs/ARCHITECTURE.md | System architecture & technical blueprint | v0.1 — active |
| docs/TECHDOC.md | Implementation reference for contributors | v0.1 — active |
| docs/BUILDPLAN.md | This document — living task tracker & decisions log | v0.1 — active |

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
| 010 | LLM provider abstraction — local, OpenAI, Anthropic | Query/chat layer routes to whichever provider is configured via LLM_PROVIDER env var. Local (Ollama/LM Studio) is default. OpenAI (gpt-5.4) and Anthropic (claude-sonnet-4-6) supported. All embedding/ML models are always local — provider setting only affects the chat/query LLM. | 2026-03-26 |
| 011 | Omnex as hivemind substrate for multi-agent ecosystems and A2A protocols | The personal memory layer is the foundation. The architecture is designed from day one to serve as the shared semantic memory substrate for multi-agent ecosystems. Agents read/write to the same index via the API. A2A protocol compatibility (MCP server) is a Phase 10 deliverable. The API humans use today is the same API agents will use tomorrow — no separate agent interface. | 2026-03-26 |

---

## Build Phases

### Phase 0 — Foundation ✓ COMPLETE
**Goal:** Repo scaffolded, services running, environment reproducible.

- [x] Create GitHub repository (omnex)
- [x] Write `docker-compose.yml` — MongoDB + Ollama services
- [x] Write `requirements.txt` — full Python dependency list
- [x] Write `models/download.py` — first-run model downloader
- [x] Write `install.sh` — Linux one-command installer
- [x] Write `install.ps1` — Windows one-command installer
- [x] Write `.env.example` — all environment variables documented
- [x] Write `.gitignore`
- [x] README.md with vision statement + install instructions
- [ ] Confirm LEANN installs and initialises correctly on Windows + Linux
- [ ] Confirm Ollama runs locally with Phi-3 Mini

**Exit criteria:** `docker compose up` + install script → all services healthy, models downloaded.

---

### Phase 1 — Variative Ingestion Core ✓ COMPLETE
**Goal:** User can point Omnex at a file, folder, or drive and it ingests everything.

- [x] `ingestion/detector.py` — libmagic content-based type detection (not extension-based)
- [x] `ingestion/router.py` — routes detected files to correct processor, handles archives recursively
- [x] `ingestion/hasher.py` — xxHash3-128 content hashing (skip unchanged files)
- [x] `ingestion/chunker.py` — semantic chunking (text), AST-aware (code), line fallback
- [x] `ingestion/processors/document.py` — PDF, DOCX, TXT, MD, HTML, XLSX, PPTX
- [x] `storage/mongo.py` — MongoDB client, chunk schema, identities schema, indexes
- [x] `storage/binary_store.py` — GridFS-inspired 4MB chunked binary file storage
- [x] `storage/leann_store.py` — LEANN index wrapper (text, image, video, audio, code)
- [x] `embeddings/text.py` — MiniLM 384-dim, GPU-aware, normalised embeddings
- [x] `ingestion/__main__.py` — CLI: `python -m ingestion --path <file|folder|drive>`
- [x] Progress reporting with parallel workers (ThreadPoolExecutor)

**Exit criteria:** Point at a folder of documents → all text extracted, chunked, embedded, stored in MongoDB + LEANN. ✓

---

### Phase 2 — Image Ingestion ✓ COMPLETE
**Goal:** Images are indexed semantically — by scene, content, GPS, date.

- [x] `ingestion/processors/image.py` — CLIP embedding + EXIF parser + thumbnail generation, HEIC support
- [x] `embeddings/image.py` — CLIP ViT-B/32, image→embedding + text→embedding (cross-modal search)
- [x] `storage/leann_store.py` — image index at 512-dim CLIP (already included in Phase 1)
- [x] Thumbnail generation — 256×256 JPEG stored via binary_store
- [x] GPS metadata extraction → lat/lng stored in chunk metadata
- [x] EXIF date/time, device, orientation extraction
- [x] Full pipeline wired in `ingestion/__main__.py` — CLIP embed + store + LEANN index

**Exit criteria:** Point at a folder of photos → images indexed, thumbnails generated, cross-modal text queries return images. ✓

---

### Phase 3 — Query Engine + Basic Interface ✓ COMPLETE
**Goal:** First wow moment. User types or speaks a query and gets results.

- [x] `api/main.py` — FastAPI app initialisation
- [x] `api/query_engine.py` — NL query parsing, multi-index LEANN search, result ranking, LLM provider abstraction (local/OpenAI/Anthropic)
- [x] `api/routes/query.py` — POST /query, POST /query/refine
- [x] `api/routes/chunks.py` — GET /chunk/{id}, GET /chunk/{id}/raw, GET /chunk/{id}/thumbnail
- [x] `api/routes/ingest.py` — POST /ingest/trigger, GET /ingest/status
- [x] `api/routes/identity.py` — POST /identity/label, GET /identity/pending (Phase 4 ready)
- [x] Next.js app scaffold — `interface/`
- [x] `QueryBar` component — text input + voice input (Web Speech API), auto-submit on voice end
- [x] `ResultGrid` component — image grid + file list, thumbnails, score badges
- [x] `PreviewPane` component — image/video/audio/code/document viewer + metadata
- [x] `IngestionDashboard` component — live progress, scope selector (file/folder/drive), expansion prompt
- [x] Voice input — Web Speech API → text → query flow
- [x] Cross-modal query — text query searches both text and image LEANN indexes

**Exit criteria:** User opens UI → selects a folder → watches ingestion progress → asks "photos from my holiday" by voice or text → sees relevant images returned. ✓

---

### Phase 4 — Face Clustering & Identity ✓ COMPLETE
**Goal:** "Find photos with my sister" works.

- [x] `ingestion/processors/image.py` — extended with RetinaFace detection + crop extraction
- [x] `embeddings/faces.py` — RetinaFace + FaceNet 128-dim embeddings + DBSCAN clustering + online classification
- [x] `ingestion/__main__.py` — stores face crops + embeddings, runs clustering after ingestion completes
- [x] `storage/mongo.py` — face_embeddings collection, identities collection with centroids
- [x] `api/routes/identity.py` — POST /identity/label, GET /identity/pending, GET /identity/all
- [x] `IdentityManager` component — face cluster review, circular crop display, name + save + skip
- [x] Query engine — person name extraction from NL query + identity-to-chunk resolution
- [x] Online classification — `classify_face()` matches new faces against known clusters with confidence threshold

**Exit criteria:** After image ingestion, user is shown face clusters and names them once. "Photos with Sarah" returns correctly filtered results. ✓

---

### Phase 5 — Audio & Video Ingestion ✓ COMPLETE
**Goal:** Spoken content and video are searchable by what was said and what was seen.

- [x] `embeddings/audio.py` — Whisper wrapper: lazy model load, GPU-aware, returns TranscriptSegment list with timestamps + language
- [x] `ingestion/processors/audio.py` — Whisper transcription → 30-second segment grouping, returns AudioResult
- [x] `ingestion/processors/video.py` — ffmpeg audio extraction → Whisper transcript + OpenCV 1fps frame sampling (cap MAX_FRAMES=120) + CLIP keyframe embeds + thumbnail at 10%
- [x] `ingestion/__main__.py` — full audio/video pipeline replacing stub: audio→MiniLM→LEANN AUDIO; video transcript→MiniLM→LEANN AUDIO + keyframes→CLIP→LEANN VIDEO
- [x] `api/query_engine.py` — LEANN AUDIO searched on all text queries; LEANN VIDEO searched on visual queries
- [x] `api/routes/chunks.py` — thumbnail route extended: falls back to `{source_hash}_thumb` for video thumbnails
- [x] `ResultGrid` component — VideoCard grid with thumbnail, play overlay, timecode badge
- [x] `PreviewPane` component — video/audio players seek to matched segment timestamp on open; duration + language in metadata footer

**Exit criteria:** Point at a folder of videos/recordings → transcripts extracted, frames indexed → spoken content findable by natural language query. ✓

---

### Phase 6 — Code Ingestion ✓ COMPLETE
**Goal:** Code across all languages is semantically indexed at function/class level.

- [x] `embeddings/code.py` — CodeBERT (microsoft/codebert-base) 768-dim CLS embeddings, L2-normalised, GPU-aware, lazy load
- [x] `ingestion/processors/code.py` — language detection (20+ languages via extension), regex symbol extraction (Python/JS/TS/Go/Rust/Java), per-chunk metadata (symbol_name, symbol_type, start_line, end_line, language)
- [x] `ingestion/__main__.py` — CODE path now uses CodeBERT instead of MiniLM; attaches language+symbol metadata to every code chunk doc
- [x] `api/query_engine.py` — NL queries embed via CodeBERT into code vector space (CodeBERT trained on NL+code pairs — cross-lingual NL→code search works natively); CODE index always searched alongside TEXT for non-visual queries
- [x] `ResultGrid` — code file cards show language badge + symbol name in monospace
- [x] `PreviewPane` — code viewer shows language badge, symbol type + name, line range header; `<code>` element tagged with `language-*` class for future Prism/highlight.js integration

**Exit criteria:** Point at a code repository → functions and classes indexed → "authentication middleware code" returns relevant code chunks. ✓

---

### Phase 7 — Neural Auto-Tagger ✓ COMPLETE
**Goal:** Every chunk is automatically tagged. Tags improve as user interacts.

- [x] `embeddings/tagger.py` — rule-based + CLIP zero-shot tagging pipeline
- [x] Tag taxonomy: date (year/month/season), type, format, scene (CLIP), topic (keyword), location (GPS), language, temporal, size, path-hints
- [x] `extract_tag_filters()` — parses NL queries → AND-filter list for query engine
- [x] Tag-based AND-filtering wired into `api/query_engine.py`
- [x] `_auto_tags()` wired into all ingestion paths in `ingestion/__main__.py` (image, audio, video transcript, video keyframe, text/code)

**Exit criteria:** After ingestion, every chunk has meaningful tags. Queries like "work documents from last year" use tag filters to narrow results. ✓

---

### Phase 8 — File Watcher (Incremental Indexing) ✓ COMPLETE
**Goal:** Omnex stays current automatically. New files indexed as they appear.

- [x] `ingestion/watcher.py` — watchdog-based file system monitor with debounce (3s)
- [x] Change detection — created/modified/moved/deleted file events
- [x] Content hash comparison — skip unchanged files (via existing hasher)
- [x] Background processing queue — ThreadPoolExecutor, non-blocking
- [x] Deleted file handling — removes chunks + LEANN vectors for deleted source paths
- [x] `WatcherHandle` API — `start_watcher()` / `stop_watcher()` for embedding in other processes
- [x] CLI: `python -m ingestion.watcher --path <dir>`

**Exit criteria:** Add a file to a watched folder → it appears in search results within seconds without manual re-trigger. ✓

---

### Phase 9 — LLM Chat Layer + High-Quality TTS ✓ COMPLETE
**Goal:** Conversational interaction with data. Omnex explains, summarises, suggests. Voice output is human-quality, not robotic.

- [x] Multi-provider LLM abstraction — Anthropic Claude, OpenAI GPT, local Ollama
- [x] `api/query_engine.py` — LLM as intelligent filter: receives all candidates, outputs `RELEVANT_IDS: [...]` to return only matching results, then prose response
- [x] Conversation history — last 10 turns passed to every LLM call, strict alternating-role enforcement for Anthropic API
- [x] Session persistence — MongoDB-backed sessions, restored from localStorage
- [x] Refinement suggestions — contextual narrowing pills after each response
- [x] Score threshold (0.25) + top_k=8 — only confident hits reach the LLM
- [x] `api/tts.py` — Qwen3-TTS GPU engine + Kokoro ONNX CPU fallback, auto-downloads models
- [x] `api/routes/tts.py` — `POST /voice/speak` → WAV audio, `GET /voice/info`
- [x] Frontend — WAV blob playback via Web Audio API, TTS brain orb button
- [x] Markdown rendering — `react-markdown` + `.md-prose` CSS in chat messages
- [x] Expandable results — collapsed "N sources retrieved" chevron in chat

**Exit criteria:** Conversational multi-turn dialogue with data. LLM filters results intelligently. Voice output is high-quality local audio. ✓

---

### Phase 10 — Agentic API Layer + Remote Access ✓ COMPLETE
**Goal:** External agents can query and interact with Omnex programmatically from anywhere.

- [x] `api/auth.py` — API key middleware, `X-API-Key` header, `OMNEX_API_KEY` env var
- [x] `api/routes/mcp.py` — JSON-RPC 2.0 MCP server at `/mcp` with tools: `recall`, `ingest`, `remember`, `stats`
- [x] HMAC-SHA256 signed media URLs — time-limited, chunk_id bound
- [x] `api/tunnel.py` — ngrok CLI subprocess + polls local API for tunnel URL, falls back to pyngrok
- [x] ngrok installed via official apt repo in Dockerfile
- [x] Remote Access UI — tunnel status, URL display, copy-to-clipboard, MCP config snippet, curl example
- [x] FUSE mount status in Remote Access panel

**Exit criteria:** Claude Desktop / any MCP client connects. Remote agent queries over ngrok with signed media URLs. ✓

---

### Phase 11 — FUSE Virtual Filesystem (Read) ✓ COMPLETE
**Goal:** Omnex mounts as a virtual drive. Apps can read from it transparently.

- [x] `fuse/omnex_fs.py` — fusepy FUSE implementation, Python, Linux/WSL
- [x] `fuse/Dockerfile` — libfuse + fusepy + omnex deps
- [x] `docker-compose.yml` — `omnex-fuse` service with `--device /dev/fuse` + `SYS_ADMIN` capability
- [x] Virtual directory structure: `documents/`, `images/`, `audio/`, `video/`, `code/`, `by_date/YYYY/MM/`
- [x] Dynamic `search/<query>` directory — read a file named after your query to execute it
- [x] `GET /setup/fuse` — mount status + path for UI
- [x] Remote Access UI — FUSE mount status card

**Exit criteria:** `/mnt/omnex` mounted. Virtual directories browse correctly. Files readable via native apps. ✓

---

### Phase 12 — FUSE Virtual Filesystem (Write + Sync) ✓ COMPLETE
**Goal:** Writing to the virtual drive triggers ingestion. Full bidirectional behaviour.

- [x] `fuse/omnex_fs.py` — `create()`, `write()`, `release()` buffer pattern — file written to staging dir on close
- [x] `_trigger_ingest()` — HTTP POST to `/ingest/trigger` after file release
- [x] `unlink()` — calls `DELETE /ingest/source` to remove from index
- [x] `drop/` magic directory — any file written here is deleted from index by source path
- [x] Mount changed from `ro=True` to `ro=False`
- [x] fusepy import collision fixed — loaded via `importlib.util.spec_from_file_location` from site-packages

**Exit criteria:** Write file to `/mnt/omnex/documents/` → ingested and queryable. Delete → removed from index. ✓

---

### Phase 13 — Local Whisper Voice + Always-Listen Mode ✓ COMPLETE
**Goal:** Fully offline voice input. Always-listening Jarvis-style mode with automatic speech detection.

- [x] `api/routes/tts.py` — `POST /voice/transcribe` — accepts multipart audio (webm/wav/ogg), transcribes via local Whisper, returns `{text, language, duration}`
- [x] Frontend — replaced Web Speech API with `MediaRecorder` + Whisper backend
- [x] Live waveform mic button — 5-bar visualiser animates with real microphone amplitude via Web Audio `AnalyserNode`
- [x] Pulsing brain orb TTS button — expanding pulse rings + glow breath animation when speaking
- [x] Push-to-talk — click to start recording, click again to stop
- [x] Always-listen VAD mode — hold mic button 0.6s to toggle; auto-detects speech onset (~0.025 amplitude threshold), records until 0.75s silence, transcribes, restarts loop
- [x] `_startMicStream()` / `_stopMicStream()` — shared mic stream + analyser lifecycle
- [x] Voice detection uses `navigator.mediaDevices` (not Web Speech API) — works in all browsers

**Exit criteria:** Fully offline voice round-trip. Hold mic button → Omnex listens continuously, hears query, transcribes locally, queries, responds. ✓

---

### Phase 14 — People View + Timeline + Delete UI + Settings ✓ COMPLETE
**Goal:** Full data management UI. People, Timeline, and Settings views functional.

- [x] `api/routes/identity.py` — `GET /identity/clusters` (all clusters with counts + labels), `GET /identity/photos/{cluster_id}` (photo chunks for a person)
- [x] `PeoplePanel` — identity list (named + unnamed), photo grid per person, inline name editor
- [x] `TimelinePanel` — year/month/type filter pills, paginated `ResultGrid`, page navigation
- [x] `SettingsPanel` — live config from API, index stats, copy-to-clipboard `.env` snippets for all settings, Kokoro voice picker
- [x] `IndexedSources` — collapsible source manager in Ingest panel, shows all ingested paths with chunk counts, bulk delete per source via `DELETE /ingest/source`
- [x] Single chunk delete — "Remove from index" button in PreviewPane calls `DELETE /ingest/chunk/{id}`
- [x] `DELETE /ingest/source` + `DELETE /ingest/chunk/{id}` — remove chunks from MongoDB, vector indexes, and binary store

---

### Phase 15 — Progressive UX — Cold Start + Drive Expansion ✓ COMPLETE
**Goal:** First-run experience is frictionless. Empty index is welcoming, not blank.

- [x] `EmptyIndexState` component — shown when `stats.total === 0`; animated database icon, type pills (docs/photos/video/audio/code), "Open Ingest →" CTA
- [x] `ReadyState` component — shown when index has data but no queries yet; breathing brain orb, per-type counts, 6 suggestion chips that fire queries on click
- [x] Drive expansion nudge — bottom-right toast after first folder indexed (< 5000 chunks); "Add folders" CTA + dismiss; session-scoped dismissal
- [x] Ingestion progress toast — bottom-right overlay while any ingestion is running; animated progress bar + folder name + % complete; auto-hides when done
- [x] `fetchStats` polls `/ingest/status` every 10s to drive toast + nudge state
- [x] Estimated time display — `eta_seconds` + `files_per_minute` computed from `started_at` in `/ingest/status`; toast shows "~N min remaining" + files/min

**Exit criteria:** New user installs → clear guided path from zero to first query result with no confusion. ✓

---

### Phase 16 — Multi-Agent Write API ✓ COMPLETE
**Goal:** AI agents can store observations and memories directly into the Omnex index.

- [x] `POST /agents/observe` — agents push text memories: `{text, source, agent_id, metadata}`
- [x] Agent identity tag — every agent-written chunk carries `agent_id` + `agent_name` in metadata, `file_type: "observation"`
- [x] Agent registration — `POST /agents` creates named agent identities with auto-generated API keys
- [x] `GET /agents` — list registered agents; `DELETE /agents/{id}` — remove agent
- [x] MCP `omnex_remember` tool — agents call this via `X-Agent-ID` header to persist observations
- [x] UI — observation results surface in ResultGrid under "Agent Memory" section with purple brain badge + agent name tag

**Exit criteria:** Claude agent calls `omnex_remember("User prefers dark mode")` via MCP → stored in index → recalled on next session. ✓

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
Push-to-talk:
  User clicks mic button → MediaRecorder starts
        ↓
  User speaks → clicks again to stop
        ↓
  Audio blob → POST /voice/transcribe → local Whisper → text
        ↓
  Text → query engine → LLM filter → results
        ↓
  LLM response → POST /voice/speak → Qwen/Kokoro TTS → WAV playback
        ↓
  Brain orb pulses while speaking

Always-listen (Jarvis mode):
  User holds mic button 0.6s → VAD loop starts
        ↓
  Microphone monitored continuously via Web Audio AnalyserNode
        ↓
  Speech onset detected (amplitude > 0.025) → recording begins
        ↓
  0.75s silence → recording stops → transcribe → query
        ↓
  Loop restarts 1.2s after response
        ↓
  Hold mic again to turn off
```

**Fully offline** — no browser Speech API, no cloud voice services. Whisper runs locally in the API container.

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
| ~~Q001~~ | ~~Wake word support for voice?~~ | **Resolved (Phase 13):** Always-listen VAD mode implemented — hold mic button 0.6s to toggle. Monitors amplitude continuously, auto-records on speech onset, auto-stops on silence. No wake word needed. |
| Q002 | Multi-user / family support? | Separate identity spaces on same machine? Shared identity clusters? |
| Q003 | Mobile ingestion path? | Photos from phone — direct WiFi sync to Omnex instance? |
| Q004 | Encryption at rest on destination drive? | User expectation for sensitive data. Performance cost? |
| ~~Q005~~ | ~~MCP server priority?~~ | **Resolved (Phase 10):** MCP server is Phase 10, before FUSE. Remote access via ngrok auto-tunnel. Signed media URLs for binary content delivery to remote agents. |

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
