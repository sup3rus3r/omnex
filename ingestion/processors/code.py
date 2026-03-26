"""
Omnex — Code Processor
Extracts structured metadata from source code files and prepares them for CodeBERT embedding.

Responsibilities:
  1. Detect language from extension + content
  2. Extract top-level symbols (functions, classes, methods) with line numbers
  3. Tag each chunk with language, symbol type, symbol name
  4. Return CodeResult with per-chunk metadata that enriches the MongoDB doc

The actual chunking is handled by ingestion/chunker.py (_chunk_code).
This processor adds the semantic layer on top — *what* each chunk is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CodeChunkMeta:
    symbol_name: str | None     # function name, class name, etc.
    symbol_type: str | None     # "function", "class", "method", "module"
    start_line:  int
    end_line:    int
    language:    str


@dataclass
class CodeResult:
    language:   str
    chunk_metas: list[CodeChunkMeta]
    line_count: int
    has_symbols: bool


# ── Language detection ─────────────────────────────────────────────────────────

EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py":         "python",
    ".js":         "javascript",
    ".ts":         "typescript",
    ".jsx":        "javascript",
    ".tsx":        "typescript",
    ".go":         "go",
    ".rs":         "rust",
    ".java":       "java",
    ".kt":         "kotlin",
    ".swift":      "swift",
    ".cpp":        "cpp",
    ".cc":         "cpp",
    ".c":          "c",
    ".h":          "c",
    ".hpp":        "cpp",
    ".cs":         "csharp",
    ".rb":         "ruby",
    ".php":        "php",
    ".scala":      "scala",
    ".r":          "r",
    ".sh":         "shell",
    ".bash":       "shell",
    ".zsh":        "shell",
    ".fish":       "shell",
    ".sql":        "sql",
    ".lua":        "lua",
    ".dart":       "dart",
    ".zig":        "zig",
    ".yaml":       "yaml",
    ".yml":        "yaml",
    ".toml":       "toml",
    ".json":       "json",
    ".tf":         "terraform",
    ".hcl":        "hcl",
    ".dockerfile": "dockerfile",
}


def detect_language(path: Path) -> str:
    return EXTENSION_TO_LANGUAGE.get(path.suffix.lower(), "unknown")


# ── Symbol extraction ──────────────────────────────────────────────────────────

def extract_symbols(text: str, language: str) -> list[tuple[str, str, int]]:
    """
    Extract top-level symbol names from source code.
    Returns list of (symbol_name, symbol_type, start_line_number).
    Line numbers are 1-based.
    """
    lines = text.splitlines()
    symbols: list[tuple[str, str, int]] = []

    if language == "python":
        pattern = re.compile(r'^(?:async\s+)?def\s+(\w+)|^class\s+(\w+)')
        for i, line in enumerate(lines, 1):
            m = pattern.match(line)
            if m:
                name = m.group(1) or m.group(2)
                stype = "function" if m.group(1) else "class"
                symbols.append((name, stype, i))

    elif language in ("javascript", "typescript"):
        patterns = [
            (re.compile(r'^(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+(\w+)'), "function"),
            (re.compile(r'^(?:export\s+)?(?:default\s+)?class\s+(\w+)'),                 "class"),
            (re.compile(r'^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\('), "function"),
            (re.compile(r'^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?function'), "function"),
        ]
        for i, line in enumerate(lines, 1):
            for pat, stype in patterns:
                m = pat.match(line)
                if m:
                    symbols.append((m.group(1), stype, i))
                    break

    elif language == "go":
        pattern = re.compile(r'^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(')
        for i, line in enumerate(lines, 1):
            m = pattern.match(line)
            if m:
                symbols.append((m.group(1), "function", i))

    elif language == "rust":
        pattern = re.compile(r'^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)|^(?:pub\s+)?(?:struct|enum|trait|impl)\s+(\w+)')
        for i, line in enumerate(lines, 1):
            m = pattern.match(line)
            if m:
                name  = m.group(1) or m.group(2)
                stype = "function" if m.group(1) else "class"
                symbols.append((name, stype, i))

    elif language == "java":
        pattern = re.compile(r'^\s*(?:public|private|protected|static|\s)+[\w<>\[\]]+\s+(\w+)\s*\(|^\s*(?:public|private|protected)?\s*(?:class|interface|enum)\s+(\w+)')
        for i, line in enumerate(lines, 1):
            m = pattern.match(line)
            if m:
                name  = m.group(1) or m.group(2)
                stype = "function" if m.group(1) else "class"
                symbols.append((name, stype, i))

    # For other languages, return empty (chunk-level metadata will use "module")
    return symbols


# ── Main processor ─────────────────────────────────────────────────────────────

def process(path: Path, chunks: list[str]) -> CodeResult:
    """
    Given a path and its pre-chunked text blocks, return CodeResult with per-chunk metadata.
    The i-th CodeChunkMeta corresponds to the i-th chunk from chunker.chunk_text().
    """
    language  = detect_language(path)
    full_text = path.read_text(encoding="utf-8", errors="replace")
    lines     = full_text.splitlines()
    line_count = len(lines)

    symbols   = extract_symbols(full_text, language)
    has_symbols = bool(symbols)

    chunk_metas: list[CodeChunkMeta] = []
    for chunk_text in chunks:
        meta = _match_chunk_to_symbol(chunk_text, full_text, symbols, language, lines)
        chunk_metas.append(meta)

    return CodeResult(
        language=language,
        chunk_metas=chunk_metas,
        line_count=line_count,
        has_symbols=has_symbols,
    )


def _match_chunk_to_symbol(
    chunk: str,
    full_text: str,
    symbols: list[tuple[str, str, int]],
    language: str,
    lines: list[str],
) -> CodeChunkMeta:
    """
    Find which symbol a chunk belongs to by locating it in the full source.
    Falls back to "module" if no match.
    """
    # Find the chunk's position in the full text
    pos = full_text.find(chunk[:min(80, len(chunk))])
    if pos == -1:
        return CodeChunkMeta(
            symbol_name=None, symbol_type="module",
            start_line=1, end_line=len(lines), language=language,
        )

    chunk_start_line = full_text[:pos].count("\n") + 1
    chunk_end_line   = chunk_start_line + chunk.count("\n")

    # Find the closest symbol that starts at or before this chunk
    matched_symbol: tuple[str, str, int] | None = None
    for sym_name, sym_type, sym_line in reversed(symbols):
        if sym_line <= chunk_start_line:
            matched_symbol = (sym_name, sym_type, sym_line)
            break

    if matched_symbol:
        return CodeChunkMeta(
            symbol_name=matched_symbol[0],
            symbol_type=matched_symbol[1],
            start_line=chunk_start_line,
            end_line=chunk_end_line,
            language=language,
        )

    return CodeChunkMeta(
        symbol_name=None,
        symbol_type="module",
        start_line=chunk_start_line,
        end_line=chunk_end_line,
        language=language,
    )
