# Security Policy

## Supported Versions

Only the latest release on the `main` branch receives security updates.

| Version | Supported |
| ------- | --------- |
| `main` (latest) | ✅ |
| Older commits | ❌ |

## Scope

Omnex is **local-first by design** — all processing, models, and storage run on your hardware. No data is sent to external servers unless you explicitly configure a cloud LLM provider (`LLM_PROVIDER=openai` or `LLM_PROVIDER=anthropic`). Privacy is not a feature — it is the architecture.

Security concerns relevant to Omnex include:

- **API security** — the FastAPI backend binds to localhost by default; misconfiguration that exposes it to a network
- **LLM provider credential handling** — API keys stored in `.env` files
- **Ingestion pipeline** — processing of untrusted files (malformed PDFs, malicious archives, EXIF injection)
- **FUSE layer** — OS-level filesystem exposure in Phases 11–12
- **MCP server / A2A protocol** — agent-facing API surface in Phase 10
- **Dependency vulnerabilities** — Python packages, npm packages, Go modules

## Reporting a Vulnerability

**Do not report security vulnerabilities through public GitHub issues.**

To report a vulnerability, use [GitHub's private security advisory](https://github.com/sup3rus3r/omnex/security/advisories/new). This ensures your report remains confidential until a fix is available.

Include as much of the following as possible:

- A description of the vulnerability and its potential impact
- The component or layer affected (ingestion, embeddings, storage, API, interface, FUSE, MCP)
- Step-by-step instructions to reproduce the issue
- Any proof-of-concept code or screenshots
- Suggested mitigations, if any

You will receive a response as quickly as possible. Please allow reasonable time to investigate and patch the issue before any public disclosure.

## Security Best Practices for Users

- Never expose the Omnex API (`localhost:8000`) to an untrusted network without authentication
- Store API keys only in `.env` — never commit `.env` to version control
- When using cloud LLM providers, be aware that query context (top-5 result snippets) is sent to the provider's API
- The FUSE virtual filesystem (Phase 11+) mounts with read-only permissions by default
