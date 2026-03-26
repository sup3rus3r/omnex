"""
Omnex — Audio Processor
Transcribes audio files via Whisper, returning chunked transcript segments.

Supports: MP3, WAV, FLAC, OGG, M4A, AAC, OPUS, WMA, and any format
ffmpeg can decode (Whisper uses ffmpeg under the hood).

Flow:
  1. Run Whisper transcription on the full file
  2. Group segments into ~30-second windows (for chunking alignment)
  3. Return AudioResult with segments + duration metadata
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

SEGMENT_WINDOW_SECONDS = 30.0  # group Whisper segments into ~30s chunks


@dataclass
class AudioResult:
    segments: list[dict]          # [{text, start, end, language}]
    full_text: str
    duration_seconds: float
    language: str
    metadata: dict = field(default_factory=dict)


def process(path: Path) -> AudioResult | None:
    """
    Transcribe an audio file and return grouped transcript segments.
    Returns None if transcription fails.
    """
    try:
        from embeddings.audio import transcribe
        raw_segments = transcribe(path)
    except Exception as e:
        logger.error(f"Whisper transcription failed for {path}: {e}")
        return None

    if not raw_segments:
        logger.warning(f"No transcript segments for {path}")
        return None

    language = raw_segments[0].language if raw_segments else "unknown"
    duration = raw_segments[-1].end if raw_segments else 0.0

    # Group raw segments into ~SEGMENT_WINDOW_SECONDS windows
    grouped = _group_segments(raw_segments, SEGMENT_WINDOW_SECONDS)

    full_text = " ".join(s.text for s in raw_segments)

    metadata = {
        "duration_seconds": duration,
        "language": language,
        "segment_count": len(grouped),
    }

    return AudioResult(
        segments=grouped,
        full_text=full_text,
        duration_seconds=duration,
        language=language,
        metadata=metadata,
    )


def _group_segments(segments, window_seconds: float) -> list[dict]:
    """
    Merge Whisper segments into fixed-size time windows.
    Each resulting chunk has: text, start, end, language.
    """
    if not segments:
        return []

    groups: list[dict] = []
    current_texts: list[str] = []
    current_start = segments[0].start
    current_end = segments[0].end
    language = segments[0].language

    for seg in segments:
        if seg.end - current_start > window_seconds and current_texts:
            groups.append({
                "text": " ".join(current_texts),
                "start": current_start,
                "end": current_end,
                "language": language,
            })
            current_texts = []
            current_start = seg.start

        current_texts.append(seg.text)
        current_end = seg.end

    if current_texts:
        groups.append({
            "text": " ".join(current_texts),
            "start": current_start,
            "end": current_end,
            "language": language,
        })

    return groups
