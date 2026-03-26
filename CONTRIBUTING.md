# Contributing to Omnex

Thank you for your interest in contributing to Omnex — the AI OS memory layer. Whether you're fixing a bug, building a new ingestion processor, improving the vector search layer, or working on the FUSE filesystem — every contribution moves us closer to the future of data.

Please read these guidelines before getting started.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Ways to Contribute](#ways-to-contribute)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)
- [Development Setup](#development-setup)
- [Branch & Commit Conventions](#branch--commit-conventions)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Code Style & Standards](#code-style--standards)
- [Security Vulnerabilities](#security-vulnerabilities)
- [License](#license)

---

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold a welcoming and respectful environment for everyone.

---

## Ways to Contribute

- **Bug reports** — Found something broken? Open an issue.
- **Feature requests** — Have an idea that fits the vision? Share it.
- **Code contributions** — Fix bugs, implement phases from the [build plan](docs/BUILDPLAN.md), improve performance.
- **ML/model work** — Embedding pipelines, quantization, model optimization.
- **Documentation** — Architecture docs, contributor guides, API docs, inline comments.
- **Testing** — Write tests, reproduce bugs, validate fixes across OS platforms.
- **Platform work** — Windows (WinFsp), Linux (libfuse), macOS (macFUSE) integration and testing.
- **Triage** — Help label and prioritize open issues.

Not sure where to start? Look for issues tagged `good-first-issue`.

---

## Reporting Bugs

Before opening a new issue, search [existing issues](https://github.com/sup3rus3r/omnex/issues) to avoid duplicates.

When filing a bug report, include:

- **A clear and descriptive title**
- **Steps to reproduce** the problem
- **Expected behavior** vs. **actual behavior**
- **Environment details**: OS, Python version, Node.js version, Go version, GPU info
- **Which layer is affected**: ingestion, embeddings, storage, API, interface, FUSE
- **Relevant logs or error messages** (redact any secrets or personal data)
- **Screenshots or recordings** if applicable

> **Do not include API keys, passwords, or personal data in issues.**

---

## Suggesting Features

Feature requests are welcome. To suggest a new feature:

1. Search [existing issues](https://github.com/sup3rus3r/omnex/issues) to see if it has already been proposed.
2. Open a new issue with the label `enhancement`.
3. Describe the problem your feature solves and your proposed solution.
4. Reference the relevant phase in [docs/BUILDPLAN.md](docs/BUILDPLAN.md) if applicable.
5. Provide any relevant examples, mockups, or references.

For large or architectural changes, open an issue to discuss before submitting a PR. Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) first — understand the system before proposing changes to it.

---

## Development Setup

### Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.11+ |
| [uv](https://docs.astral.sh/uv/) | latest (replaces pip/venv) |
| Node.js | 20+ |
| Go | 1.22+ |
| Docker + Docker Compose | latest |
| MongoDB | 7 (local or via Docker) |

### 1. Fork and Clone

```bash
git clone https://github.com/your-username/omnex.git
cd omnex
```

### 2. Environment

```bash
cp .env.example .env
# Edit .env — set drive paths, LLM provider, API keys as needed
```

### 3. Start Services

```bash
docker compose up -d
# Starts MongoDB + Ollama
```

### 4. Python Backend

```bash
uv venv .venv
uv pip install -r requirements.txt
```

### 5. Download Models (first run)

```bash
python models/download.py
# Downloads MiniLM, CLIP, Whisper small, CodeBERT, Ollama phi3:mini
# GPU flag: python models/download.py --gpu
```

### 6. Start API

```bash
uv run uvicorn api.main:app --reload --port 8000
```

### 7. Frontend

```bash
cd interface
npm install
npm run dev
# Runs at http://localhost:3000
```

### 8. Go FUSE Layer (optional — Phase 11+)

```bash
cd fuse
go build ./...
```

### One-command install (Linux/macOS)

```bash
bash install.sh
```

### One-command install (Windows)

```powershell
./install.ps1
```

---

## Branch & Commit Conventions

### Branch Naming

| Prefix | Purpose |
|--------|---------|
| `feature/` | New features or enhancements |
| `fix/` | Bug fixes |
| `docs/` | Documentation-only changes |
| `refactor/` | Code refactoring without behavior change |
| `test/` | Adding or updating tests |
| `chore/` | Maintenance, dependency updates, tooling |
| `phase/` | Work scoped to a specific build phase |

Examples:
```
feature/audio-ingestion
fix/face-clustering-cold-start
phase/5-video-transcription
docs/leann-index-guide
```

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <short description>

[optional body]

[optional footer]
```

**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

**Scopes:** `ingestion`, `embeddings`, `storage`, `api`, `interface`, `fuse`, `models`, `docker`

Examples:
```
feat(ingestion): add audio chunking via Whisper transcription
fix(storage): handle LEANN index rebuild on corrupt vector file
docs(architecture): update FUSE layer diagram for phase 11
chore(docker): pin Ollama image to v0.3.0
```

Keep the subject line under 72 characters. Use the body to explain *why*, not just *what*.

---

## Submitting a Pull Request

1. **Ensure your branch is up to date** with `main`:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Keep PRs focused** — one logical change per PR. Avoid bundling unrelated changes.

3. **Test your changes** locally before submitting.

4. **Open the PR against `main`** with:
   - A clear title following the commit convention above
   - A description explaining **what** changed and **why**
   - References to related issues (e.g., `Closes #42`)
   - The phase this PR belongs to, if applicable
   - Screenshots or recordings for UI changes

5. **Be responsive** to review feedback.

6. All checks must pass before a PR can be merged.

### PR Checklist

- [ ] Code follows the project's style guidelines
- [ ] No secrets, credentials, or personal data are included
- [ ] Backend changes include appropriate error handling
- [ ] Frontend changes have been tested in the browser
- [ ] ML/embedding changes have been validated against the relevant index
- [ ] [docs/BUILDPLAN.md](docs/BUILDPLAN.md) updated if a phase task is completed
- [ ] Documentation updated where necessary
- [ ] Commit history is clean and follows conventions

---

## Code Style & Standards

### Python (ingestion / embeddings / storage / api)

- **Formatter:** Black or Ruff — keep consistent with existing style
- **Type hints:** Use type annotations for all function signatures
- **Pydantic models:** Use for all FastAPI request/response schemas
- **Async:** Prefer `async`/`await` consistent with FastAPI conventions
- **Models:** Lazy-load all ML models — do not load at import time
- Do not commit unused imports or dead code

### TypeScript / JavaScript (interface)

- **Language:** TypeScript — avoid `any` types
- **Formatting:** Prettier
- **Components:** Functional React components with hooks
- **Styling:** Tailwind CSS utility classes — avoid inline styles
- **Data fetching:** SWR for polling/caching, `fetch` for one-off calls

### Go (fuse)

- **Formatting:** `gofmt` — always
- **Error handling:** Explicit, no silent discards
- **Imports:** Grouped (stdlib / external / internal)

### General

- Write self-documenting code; add comments only where logic is not immediately obvious
- Keep functions small and focused
- Do not introduce new dependencies without prior discussion in an issue
- Never hardcode paths, credentials, or model names — use environment variables

---

## Security Vulnerabilities

**Do not report security vulnerabilities through public GitHub issues.**

Use [GitHub's private vulnerability reporting](https://github.com/sup3rus3r/omnex/security/advisories/new) to disclose responsibly. See [SECURITY.md](SECURITY.md) for full details.

---

## License

By contributing to Omnex, you agree that your contributions will be licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** — the same license that governs the project.

See the [LICENSE](LICENSE) file for full terms.

---

*Omnex is infrastructure. Infrastructure should be open. Thank you for helping build it.*
