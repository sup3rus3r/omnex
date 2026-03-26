"""
Omnex — TTS Engine
Qwen3-TTS on GPU, Kokoro ONNX on CPU fallback.

/voice/speak  POST  { text: str, voice?: str }  → audio/wav stream
"""

from __future__ import annotations

import io
import logging
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

log = logging.getLogger("omnex.tts")

# Model file paths for Kokoro
_KOKORO_DIR   = Path(os.getenv("OMNEX_DATA_PATH", "/data")) / "models" / "kokoro"
_KOKORO_MODEL = _KOKORO_DIR / "kokoro-v1.0.onnx"
_KOKORO_VOICES = _KOKORO_DIR / "voices-v1.0.bin"

# Default voices
QWEN_VOICE    = os.getenv("TTS_QWEN_VOICE",   "Ryan")
KOKORO_VOICE  = os.getenv("TTS_KOKORO_VOICE", "af_heart")


def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available() and os.getenv("GPU_ENABLED", "false").lower() == "true"
    except ImportError:
        return False


@lru_cache(maxsize=1)
def _load_qwen():
    from qwen_tts import Qwen3TTSModel
    import torch
    model_id = os.getenv("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    log.info(f"Loading Qwen TTS: {model_id}")
    return Qwen3TTSModel.from_pretrained(
        model_id,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )


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
        ("kokoro-v1.0.onnx",  _KOKORO_MODEL),
        ("voices-v1.0.bin",   _KOKORO_VOICES),
    ]:
        if not dest.exists():
            log.info(f"Downloading Kokoro model: {fname}")
            urllib.request.urlretrieve(f"{base}/{fname}", dest)


def _to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV")
    return buf.getvalue()


def synthesize(text: str, voice: str | None = None) -> bytes:
    """
    Synthesize text to WAV bytes.
    Uses Qwen on GPU if available, Kokoro ONNX as fallback.
    Returns raw WAV bytes.
    """
    if _gpu_available():
        try:
            return _synthesize_qwen(text, voice or QWEN_VOICE)
        except Exception as e:
            log.warning(f"Qwen TTS failed ({e}), falling back to Kokoro")

    return _synthesize_kokoro(text, voice or KOKORO_VOICE)


def _synthesize_qwen(text: str, voice: str) -> bytes:
    model = _load_qwen()
    wavs, sr = model.generate_custom_voice(
        text=text,
        speaker=voice,
        language="English",
    )
    return _to_wav_bytes(wavs[0], sr)


def _synthesize_kokoro(text: str, voice: str) -> bytes:
    kokoro = _load_kokoro()
    samples, sr = kokoro.create(text, voice=voice, speed=1.0, lang="en-us")
    return _to_wav_bytes(samples, sr)


def engine_info() -> dict:
    gpu = _gpu_available()
    return {
        "engine":       "qwen"   if gpu else "kokoro",
        "gpu_enabled":  gpu,
        "qwen_voice":   QWEN_VOICE,
        "kokoro_voice": KOKORO_VOICE,
        "kokoro_ready": _KOKORO_MODEL.exists() and _KOKORO_VOICES.exists(),
    }
