"""
Omnex — Video Processor
Extracts both semantic content dimensions from video:
  1. Visual — CLIP embeddings of sampled keyframes (1 fps, capped at MAX_FRAMES)
  2. Transcript — Whisper transcription of the audio track

Flow:
  1. Extract audio track to temp WAV via ffmpeg
  2. Transcribe audio with Whisper
  3. Sample frames at 1 fps (up to MAX_FRAMES)
  4. CLIP-embed each sampled frame
  5. Generate thumbnail from the frame nearest 10% into the video
  6. Return VideoResult

Requires: ffmpeg on PATH (bundled via docker or system install)
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_FRAMES = 120        # cap frame samples to keep memory bounded
FRAME_SAMPLE_FPS = 1    # sample 1 frame per second
THUMBNAIL_OFFSET_PCT = 0.10  # thumbnail at 10% into the video


@dataclass
class VideoFrame:
    timestamp: float          # seconds into the video
    embedding: list[float]    # 512-dim CLIP vector


@dataclass
class VideoResult:
    frames: list[VideoFrame]
    transcript_segments: list[dict]   # [{text, start, end, language}]
    full_transcript: str
    duration_seconds: float
    thumbnail: bytes | None
    language: str
    metadata: dict = field(default_factory=dict)


def process(path: Path) -> VideoResult | None:
    """
    Process a video file: transcribe audio + embed keyframes.
    Returns None if both audio extraction and frame sampling fail.
    """
    duration = _get_duration(path)
    transcript_segments: list[dict] = []
    full_transcript = ""
    language = "unknown"

    # ── Audio track → transcript ──────────────────────────────────────────────
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_audio = Path(tmp.name)

        _extract_audio(path, tmp_audio)
        from ingestion.processors.audio import process as process_audio
        audio_result = process_audio(tmp_audio)
        if audio_result:
            transcript_segments = audio_result.segments
            full_transcript = audio_result.full_text
            language = audio_result.language
            if not duration:
                duration = audio_result.duration_seconds
    except Exception as e:
        logger.warning(f"Audio extraction failed for {path}: {e}")
    finally:
        try:
            tmp_audio.unlink(missing_ok=True)
        except Exception:
            pass

    # ── Frame sampling → CLIP embeddings ─────────────────────────────────────
    frames: list[VideoFrame] = []
    thumbnail: bytes | None = None

    try:
        frames, thumbnail = _process_frames(path, duration)
    except Exception as e:
        logger.warning(f"Frame processing failed for {path}: {e}")

    if not frames and not transcript_segments:
        logger.error(f"Video processing yielded no content for {path}")
        return None

    metadata = {
        "duration_seconds": duration,
        "language": language,
        "frame_count": len(frames),
        "transcript_segments": len(transcript_segments),
    }

    return VideoResult(
        frames=frames,
        transcript_segments=transcript_segments,
        full_transcript=full_transcript,
        duration_seconds=duration or 0.0,
        thumbnail=thumbnail,
        language=language,
        metadata=metadata,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_duration(path: Path) -> float:
    """Get video duration in seconds via ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                str(path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        import json
        info = json.loads(result.stdout)
        return float(info.get("format", {}).get("duration", 0))
    except Exception:
        return 0.0


def _extract_audio(video_path: Path, out_path: Path) -> None:
    """Extract audio track from video to WAV via ffmpeg."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn",                   # no video
            "-acodec", "pcm_s16le",  # Whisper expects WAV
            "-ar", "16000",          # 16kHz
            "-ac", "1",              # mono
            str(out_path),
        ],
        capture_output=True,
        timeout=300,
        check=True,
    )


def _process_frames(path: Path, duration: float) -> tuple[list[VideoFrame], bytes | None]:
    """
    Sample frames at FRAME_SAMPLE_FPS, cap at MAX_FRAMES evenly distributed.
    Returns (frames, thumbnail_bytes).
    """
    from embeddings.image import embed_image
    from PIL import Image
    import cv2
    import io

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 cannot open {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps

    # Determine which frame indices to sample
    sample_every = max(1, int(fps / FRAME_SAMPLE_FPS))
    candidate_indices = list(range(0, total_frames, sample_every))

    # Cap at MAX_FRAMES — evenly spaced
    if len(candidate_indices) > MAX_FRAMES:
        step = len(candidate_indices) // MAX_FRAMES
        candidate_indices = candidate_indices[::step][:MAX_FRAMES]

    # Thumbnail target frame
    thumbnail_frame_idx = int(total_frames * THUMBNAIL_OFFSET_PCT)

    frames: list[VideoFrame] = []
    thumbnail: bytes | None = None

    for frame_idx in candidate_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, bgr = cap.read()
        if not ret:
            continue

        timestamp = frame_idx / fps

        # Convert BGR → RGB PIL
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # CLIP embed
        try:
            embedding = embed_image(pil_img)
            frames.append(VideoFrame(timestamp=timestamp, embedding=embedding))
        except Exception as e:
            logger.debug(f"Frame embed failed at {timestamp:.1f}s: {e}")

        # Thumbnail
        if thumbnail is None and frame_idx >= thumbnail_frame_idx:
            pil_img.thumbnail((256, 256))
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=85)
            thumbnail = buf.getvalue()

    cap.release()
    return frames, thumbnail
