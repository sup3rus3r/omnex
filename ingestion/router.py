"""
Omnex — Ingestion Router
Routes detected files to the correct processor and returns extracted text.
Handles archives recursively.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from ingestion.detector import FileType, detect, is_indexable

logger = logging.getLogger(__name__)


def route(path: Path) -> tuple[str, FileType, str]:
    """
    Route a file to the correct processor.

    Returns:
        (extracted_text, file_type, mime_type)
        extracted_text is empty string for binary types (image/video/audio)
        — those are handled separately by their respective processors.
    """
    if not is_indexable(path):
        return "", FileType.UNKNOWN, "application/octet-stream"

    file_type, mime = detect(path)

    try:
        if file_type == FileType.DOCUMENT:
            from ingestion.processors.document import extract
            text = extract(path)
            return text, file_type, mime

        elif file_type == FileType.CODE:
            text = path.read_text(encoding="utf-8", errors="replace")
            return text, file_type, mime

        elif file_type == FileType.IMAGE:
            # Text extraction not applicable — return empty, signal type
            return "", file_type, mime

        elif file_type == FileType.AUDIO:
            return "", file_type, mime

        elif file_type == FileType.VIDEO:
            return "", file_type, mime

        elif file_type == FileType.ARCHIVE:
            # Recursively extract and route — returns combined text from all contents
            text = _route_archive(path)
            return text, file_type, mime

        else:
            return "", FileType.UNKNOWN, mime

    except Exception as e:
        logger.warning(f"Routing failed for {path}: {e}")
        return "", FileType.UNKNOWN, mime


def _route_archive(path: Path) -> str:
    """Extract archive and route each contained file. Returns combined text."""
    import zipfile
    import tarfile

    extracted_texts: list[str] = []

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            if zipfile.is_zipfile(path):
                with zipfile.ZipFile(path, "r") as zf:
                    zf.extractall(tmp_path)

            elif tarfile.is_tarfile(path):
                with tarfile.open(path, "r:*") as tf:
                    tf.extractall(tmp_path)

            else:
                logger.warning(f"Unsupported archive format: {path}")
                return ""

            for extracted in tmp_path.rglob("*"):
                if extracted.is_file():
                    text, _, _ = route(extracted)
                    if text:
                        extracted_texts.append(f"[{extracted.name}]\n{text}")

    except Exception as e:
        logger.warning(f"Archive extraction failed for {path}: {e}")

    return "\n\n".join(extracted_texts)
