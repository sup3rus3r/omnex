"""
Omnex — TTS Engine

Engines (selectable at runtime via TTS_ENGINE env var or API):
  chatterbox  — ChatterboxTurboTTS on GPU, expressive, ~200ms first chunk
  kokoro      — Kokoro ONNX on CPU, fast, lighter

Streaming: WAV header sent first, then PCM chunks as they generate.
Frontend can start playing on first chunk — no waiting for full audio.
"""

from __future__ import annotations

import io
import logging
import os
import struct
import threading
from functools import lru_cache
from pathlib import Path
from typing import Generator

log = logging.getLogger("omnex.tts")

# ── Config ────────────────────────────────────────────────────────────────────

TTS_ENGINE    = os.getenv("TTS_ENGINE", "chatterbox").lower()   # "chatterbox" | "kokoro"
SAMPLE_RATE   = 24000   # Chatterbox Turbo output sample rate

# Kokoro
_KOKORO_DIR    = Path(os.getenv("OMNEX_DATA_PATH", "/data")) / "models" / "kokoro"
_KOKORO_MODEL  = _KOKORO_DIR / "kokoro-v1.0.onnx"
_KOKORO_VOICES = _KOKORO_DIR / "voices-v1.0.bin"
KOKORO_VOICE   = os.getenv("TTS_KOKORO_VOICE", "af_heart")
KOKORO_VOICES  = ["af_heart", "af_bella", "af_sarah", "af_nicole",
                  "am_adam", "am_michael", "bm_george", "bf_emma"]

# Chatterbox
CHATTERBOX_VOICE = os.getenv("TTS_CHATTERBOX_VOICE", "default")


# ── WAV header ────────────────────────────────────────────────────────────────

def _wav_header(sample_rate: int, num_channels: int = 1, bits: int = 16) -> bytes:
    byte_rate   = sample_rate * num_channels * bits // 8
    block_align = num_channels * bits // 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 0xFFFFFFFF,
        b"WAVE",
        b"fmt ", 16,
        1, num_channels, sample_rate,
        byte_rate, block_align, bits,
        b"data", 0xFFFFFFFF,
    )


# ── Chatterbox Turbo ──────────────────────────────────────────────────────────

_chatterbox_model = None
_chatterbox_lock  = threading.Lock()


def _get_chatterbox():
    global _chatterbox_model
    if _chatterbox_model is None:
        with _chatterbox_lock:
            if _chatterbox_model is None:
                import torch
                from chatterbox.tts_turbo import ChatterboxTurboTTS
                device = "cuda" if torch.cuda.is_available() else "cpu"
                log.info(f"[chatterbox] loading model on {device}")
                _chatterbox_model = ChatterboxTurboTTS.from_pretrained(device=device)
                log.info("[chatterbox] model ready")
    return _chatterbox_model


def _stream_chatterbox(text: str) -> Generator[bytes, None, None]:
    import torch
    import numpy as np

    model = _get_chatterbox()
    yield _wav_header(SAMPLE_RATE)

    # Generate full audio — Turbo is fast enough (~200ms on GPU)
    # We yield in chunks so the frontend can start playing immediately
    wav = model.generate(text)

    # wav is a torch tensor — convert to int16 PCM
    if torch.is_tensor(wav):
        wav = wav.detach().cpu().squeeze().numpy()
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.reshape(-1)
    pcm = (np.clip(wav, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

    # Yield in 8KB chunks so streaming starts immediately
    chunk_size = 8192
    for i in range(0, len(pcm), chunk_size):
        yield pcm[i:i + chunk_size]


# ── Kokoro ────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_kokoro():
    from kokoro_onnx import Kokoro
    if not _KOKORO_MODEL.exists() or not _KOKORO_VOICES.exists():
        _download_kokoro()
    log.info("Loading Kokoro TTS")
    return Kokoro(str(_KOKORO_MODEL), str(_KOKORO_VOICES))


def _download_kokoro():
    import urllib.request
    _KOKORO_DIR.mkdir(parents=True, exist_ok=True)
    base = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
    for fname, dest in [
        ("kokoro-v1.0.onnx", _KOKORO_MODEL),
        ("voices-v1.0.bin",  _KOKORO_VOICES),
    ]:
        if not dest.exists():
            log.info(f"Downloading Kokoro model: {fname}")
            urllib.request.urlretrieve(f"{base}/{fname}", dest)


def _stream_kokoro(text: str, voice: str) -> Generator[bytes, None, None]:
    import soundfile as sf
    import numpy as np

    kokoro = _load_kokoro()
    if voice not in KOKORO_VOICES:
        voice = KOKORO_VOICE

    samples, sr = kokoro.create(text, voice=voice, speed=1.0, lang="en-us")
    yield _wav_header(sr)

    pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    chunk_size = 8192
    for i in range(0, len(pcm), chunk_size):
        yield pcm[i:i + chunk_size]


# ── GPU / engine checks ───────────────────────────────────────────────────────

def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available() and os.getenv("GPU_ENABLED", "false").lower() == "true"
    except ImportError:
        return False


def _chatterbox_available() -> bool:
    try:
        from chatterbox.tts_turbo import ChatterboxTurboTTS  # noqa: F401
        return True
    except ImportError:
        return False


def _active_engine(engine_override: str | None = None) -> str:
    """Resolve which engine to use. Override > env var > auto-detect."""
    engine = (engine_override or TTS_ENGINE).lower()
    if engine == "chatterbox" and not _chatterbox_available():
        log.warning("[tts] chatterbox not available, falling back to kokoro")
        return "kokoro"
    return engine


# ── Public API ────────────────────────────────────────────────────────────────

def synthesize_stream(
    text: str,
    voice: str | None = None,
    engine: str | None = None,
) -> Generator[bytes, None, None]:
    active = _active_engine(engine)
    if active == "chatterbox":
        yield from _stream_chatterbox(text)
    else:
        yield from _stream_kokoro(text, voice or KOKORO_VOICE)


def synthesize(text: str, voice: str | None = None, engine: str | None = None) -> bytes:
    return b"".join(synthesize_stream(text, voice, engine))


def engine_info() -> dict:
    active = _active_engine()
    return {
        "engine":               active,
        "available_engines":    ["chatterbox", "kokoro"],
        "chatterbox_available": _chatterbox_available(),
        "chatterbox_voice":     CHATTERBOX_VOICE,
        "kokoro_voice":         KOKORO_VOICE,
        "kokoro_voices":        KOKORO_VOICES,
        "kokoro_ready":         _KOKORO_MODEL.exists() and _KOKORO_VOICES.exists(),
        "gpu_enabled":          _gpu_available(),
    }
