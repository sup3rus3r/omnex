"""
Omnex — Audio Embeddings (Whisper)
Transcribes audio segments using OpenAI Whisper running fully on-device.

Usage:
    from embeddings.audio import transcribe, transcribe_segments
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

_model = None
_model_name: str | None = None


def _get_model():
    global _model, _model_name
    name = os.getenv("WHISPER_MODEL", "small")
    if _model is None or _model_name != name:
        import whisper
        logger.info(f"Loading Whisper model: {name}")
        _model = whisper.load_model(name)
        _model_name = name
    return _model


class TranscriptSegment(NamedTuple):
    text: str
    start: float   # seconds
    end: float     # seconds
    language: str


def transcribe(audio_path: Path | str) -> list[TranscriptSegment]:
    """
    Transcribe an audio file. Returns one TranscriptSegment per Whisper segment.
    Whisper handles language detection automatically.
    """
    model = _get_model()
    result = model.transcribe(str(audio_path), verbose=False)

    language = result.get("language", "unknown")
    segments = []
    for seg in result.get("segments", []):
        text = seg.get("text", "").strip()
        if text:
            segments.append(TranscriptSegment(
                text=text,
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", 0.0)),
                language=language,
            ))
    return segments


def transcribe_to_text(audio_path: Path | str) -> str:
    """Transcribe and return the full text (no timestamps)."""
    segments = transcribe(audio_path)
    return " ".join(s.text for s in segments)
