# Omnex

### *Everything, indexed. Nothing lost. No file system.*

> **The AI OS memory layer — local-first, open source, agentic-ready.**
>
> Built for the era when AI has been fully adopted and the file system as we know it becomes legacy.

---

## What is Omnex?

The way humans store and retrieve data has not fundamentally changed since the 1970s. Files live in folders. Folders live in drives. You remember where you put things — or you don't. Search is keyword-based. Video and audio are completely unsearchable. Your data has no intelligence.

**Omnex changes this at the foundation.**

Omnex ingests your data — every document, image, video, audio file, and line of code — and builds a semantic memory layer on a second drive. Everything is chunked, embedded, and indexed by meaning. You interact with your data the way you access a memory: by context, time, people, place, emotion. Not by filename or folder path.

You never search for a file again. You recall it.

> *"Find the contract I signed around the time we moved."*
> *"Show me photos with my sister from the Cape Town trip."*
> *"Pull up the code I wrote for authentication last year."*
> *"What was I working on the week before the product launch?"*

These are not search queries. They are memories. Omnex retrieves them.

---

## Why This Matters

We are at an inflection point. AI agents are becoming capable of acting on our behalf — scheduling, researching, deciding, creating. For agents to work effectively, they need a memory layer that isn't a hierarchical folder structure built for a pre-AI world.

Omnex is that layer.

- **For humans today** — a voice-first, conversational interface to all your personal data
- **For agents tomorrow** — a structured, queryable API that AI can call to recall, reason over, and act on your data
- **For the AI OS era** — a FUSE virtual filesystem that replaces the traditional file system at the OS level, transparently

This is not RAG bolted onto a file browser. This is the memory substrate the agentic era requires. It doesn't exist yet. We are building it first.

---

## Core Principles

| Principle | What it means |
|---|---|
| **Local-first** | All models, processing, and storage run on your hardware. No cloud. No telemetry. |
| **Privacy by design** | Your data never leaves your machine. Not optional. |
| **Open source** | AGPL-3.0. Community-built from day one. No commercial lock-in. |
| **Voice-first** | Voice and text are equal input modalities. Neither is an afterthought. |
| **Agentic-ready** | The API is a first-class citizen. Humans use the UI. Agents use the API. Same backend. |
| **Hardware inclusive** | Runs on modest hardware. GPU optional. Designed for real people, not just ML labs. |
| **Progressive enhancement** | Start with a single folder. Expand to a full drive. Grow into an OS memory layer. |

---

## How It Works

```
SOURCE DRIVE  →  [Ingestion Pipeline]  →  [ML Processing Layer]
                        ↓                           ↓
              [Storage: MongoDB + LEANN + Binary Chunk Store]
                        ↓
                [Query Engine (FastAPI)]
                        ↓
         ┌──────────────┴──────────────┐
         │                             │
  [Voice + Text UI            [Agent API / MCP Server]
   (Next.js)]                  (Claude, GPT, custom agents)
         │
  [FUSE Virtual Filesystem]  ←→  OS / Applications
```

**Ingestion Pipeline** — Detects file types by content (not extension), routes to the right processor, chunks everything, hashes for deduplication. Supports file, folder, or full drive scope — start small, expand as you build confidence.

**ML Processing Layer** — Runs entirely on-device. Text embeddings (MiniLM), image/video understanding (CLIP), face clustering (RetinaFace + FaceNet), transcription (Whisper), code understanding (CodeBERT). No API calls. No data leaving your machine.

**Storage Layer** — MongoDB for metadata and relationships. LEANN for vector indexing — achieving 97% storage reduction over conventional vector databases, meaning your destination drive can match or be smaller than your source. Content-addressed binary chunk store for raw data.

**Query Engine** — Natural language parsed into multi-stage vector searches with metadata filtering. Results re-ranked for precision. Context assembled and passed to a local LLM (Ollama) for conversational responses.

**Voice + Text Interface** — A conversation with your data, not a file browser. Ask anything. Refine by voice. Results surface with full context — thumbnails, previews, document snippets, video timestamps.

**FUSE Virtual Filesystem** — Long-term: Omnex mounts as a virtual drive. Folders like `People/Sarah/`, `Places/Cape Town/`, `By Year/2023/` are generated from the semantic index, not from physical directories. Applications interact with it transparently.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Ingestion Pipeline | Python — python-magic, watchdog, PyMuPDF, python-docx |
| ML Processing | Python — sentence-transformers, CLIP, Whisper, DeepFace, CodeBERT, PyTorch |
| Vector Index | LEANN — file-based, 97% storage savings vs Qdrant, no server process |
| Metadata Store | MongoDB |
| Query & API Backend | FastAPI + Uvicorn |
| Voice + Text Interface | Next.js + Web Speech API + Browser TTS |
| Local LLM | Ollama — Phi-3 Mini, Gemma 3 2B, or Llama 3.2 |
| FUSE Layer | Go — cgofuse (Linux/libfuse, Windows/WinFsp) |
| Services | Docker Compose |

---

## Minimum Hardware

| Component | Minimum | Recommended |
|---|---|---|
| CPU | 4-core x86_64 | 8-core modern CPU |
| RAM | 8 GB | 16–32 GB |
| GPU | None (CPU-only mode) | NVIDIA 8GB+ VRAM (CUDA) or AMD (ROCm) |
| Source Drive | Any size | SSD preferred |
| Destination Drive | Same size as source | 1.2x source, SSD |
| OS | Windows 10+ or Ubuntu 22.04+ | Ubuntu 24.04 LTS |

---

## Current Status

Omnex is in active early development. The architecture is defined. The build plan is public. We are assembling the team now.

| Phase | Milestone | Status |
|---|---|---|
| 0 | Foundation — repo, docker, installers | In progress |
| 1 | Ingestion — text & documents | Planned |
| 2 | Ingestion — images | Planned |
| 3 | Query engine + basic UI + voice input | Planned |
| 4 | Face clustering & identity | Planned |
| 5 | Audio & video ingestion | Planned |
| 6 | Code ingestion | Planned |
| 7 | Neural auto-tagger | Planned |
| 8 | File watcher — incremental indexing | Planned |
| 9 | LLM chat layer + voice output | Planned |
| 10 | Agentic API + MCP server | Planned |
| 11 | FUSE filesystem — read | Planned |
| 12 | FUSE filesystem — write + sync | Planned |

Full architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
Full build plan: [docs/BUILDPLAN.md](docs/BUILDPLAN.md)
Implementation reference: [docs/TECHDOC.md](docs/TECHDOC.md)

---

## We Are Looking for Collaborators

Omnex is an ambitious, open source project building something that does not exist yet. We are looking for people who want to work on something that matters — not another CRUD app, not another wrapper around someone else's API.

**We need people who can contribute to:**

### Core Infrastructure
- **Python engineers** — ingestion pipeline, ML model integration, FastAPI backend
- **ML engineers** — embedding pipelines, face clustering, model optimization, quantization
- **Go engineers** — FUSE virtual filesystem, OS-level integration, cgofuse

### Intelligence Layer
- **NLP engineers** — query parsing, result ranking, cross-encoder re-ranking
- **Vector search engineers** — LEANN optimization, multi-index search, hybrid retrieval
- **LLM integration** — Ollama integration, context assembly, conversational refinement

### Interface
- **Frontend engineers** — Next.js, voice UI, real-time ingestion dashboard, masonry result grid
- **UX designers** — designing a conversation interface for data, not a file browser

### Platform & DevOps
- **Windows engineers** — WinFsp integration, Windows-specific edge cases, testing
- **Linux engineers** — libfuse, system service configuration, packaging (deb/rpm/flatpak)
- **macOS engineers** — macFUSE path (future milestone)

### Community
- **Technical writers** — architecture documentation, contributor guides, API docs
- **Community builders** — Discord, GitHub discussions, contributor onboarding

---

## What We Value in Contributors

- You care about **privacy and local-first software**
- You want to build **infrastructure, not features** — this is a platform
- You think in **systems** — how layers connect, how data flows, where things fail
- You're comfortable with **ambiguity and early-stage work** — we are defining the category
- You believe the **file system era is ending** and want to build what comes next

You do not need to be an expert in every layer. Deep knowledge in one area with curiosity about the others is enough.

---

## Getting Started as a Contributor

1. **Read the architecture doc** — [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Understand the system before writing a line of code.
2. **Read the build plan** — [docs/BUILDPLAN.md](docs/BUILDPLAN.md). Find the phase that matches your skills.
3. **Open an issue** — introduce yourself, tell us what you want to build or fix.
4. **Join the discussion** — GitHub Discussions for architectural questions and decisions.
5. **Pick a task** — issues are labelled by phase, skill, and difficulty.

**First contribution?** Look for issues tagged `good-first-issue`. These are well-defined, self-contained tasks that don't require understanding the full system.

---

## Repository Structure

```
omnex/
├── ingestion/          # Python — file detection, routing, chunking, processors
├── embeddings/         # Python — all ML model wrappers
├── storage/            # Python — MongoDB, LEANN, binary chunk store
├── api/                # FastAPI backend — query engine, all routes
├── fuse/               # Go — FUSE virtual filesystem
├── interface/          # Next.js — voice + text UI
├── models/             # Model download scripts + configs
├── docs/               # Architecture, build plan, implementation reference
├── docker-compose.yml
├── requirements.txt
├── install.sh
└── install.ps1
```

---

## License

Omnex is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

This means:
- You can use, modify, and distribute Omnex freely
- If you deploy a modified version as a service, you must open source your modifications
- The project stays open — forever

We chose AGPL deliberately. Omnex is infrastructure. Infrastructure should be open.

---

## The Bigger Picture

The cloud era gave us convenience at the cost of privacy and control. The AI era will give us intelligence — but only if the data layer is built correctly from the start.

Big tech will build their version of this. It will be cloud-first, walled-garden, and trained on your data without your meaningful consent. That version already exists in pieces — Google Photos, iCloud, Microsoft Recall. They are siloed, proprietary, and not agentic.

Omnex is the open alternative. Local. Private. Agentic-ready. Built by the community, for everyone.

**The file system had a good run. Its time is ending. Help us build what comes next.**

---

*Omnex — github.com/sup3rus3r/omnex*
*AGPL-3.0 | Local-first | Open source | Agentic-ready*
