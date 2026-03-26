"""
Omnex — TTS Engine
VibeVoice-Realtime-0.5B on GPU  → realtime streaming PCM (~300ms first token)
                                   runs in an isolated subprocess to avoid
                                   heap conflicts with onnxruntime-gpu/insightface
Kokoro ONNX on CPU              → full WAV fallback

/voice/speak  POST  { text, voice? }  → audio/wav (streaming or full)
"""

from __future__ import annotations

import io
import logging
import os
import struct
from functools import lru_cache
from pathlib import Path
from typing import Generator

log = logging.getLogger("omnex.tts")

# ── Kokoro paths ──────────────────────────────────────────────────────────────
_KOKORO_DIR    = Path(os.getenv("OMNEX_DATA_PATH", "/data")) / "models" / "kokoro"
_KOKORO_MODEL  = _KOKORO_DIR / "kokoro-v1.0.onnx"
_KOKORO_VOICES = _KOKORO_DIR / "voices-v1.0.bin"

# ── VibeVoice voices ──────────────────────────────────────────────────────────
VIBEVOICE_VOICES      = ["Carter", "Davis", "Emma", "Frank", "Grace", "Mike"]
VIBEVOICE_SAMPLE_RATE = 24000

# ── Defaults ──────────────────────────────────────────────────────────────────
VIBEVOICE_VOICE = os.getenv("VIBEVOICE_VOICE", "Emma")
KOKORO_VOICE    = os.getenv("TTS_KOKORO_VOICE", "af_heart")


def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available() and os.getenv("GPU_ENABLED", "false").lower() == "true"
    except ImportError:
        return False


def _vibevoice_installed() -> bool:
    """Check if the VibeVoice package is installed without importing it."""
    import importlib.util
    return importlib.util.find_spec("streamingtts") is not None


def _wav_header(sample_rate: int, num_channels: int = 1, bits: int = 16) -> bytes:
    """Return a WAV header for a streaming (unknown length) file."""
    data_size   = 0xFFFFFFFF
    header_size = 44
    fmt_size    = 16
    byte_rate   = sample_rate * num_channels * bits // 8
    block_align = num_channels * bits // 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", data_size + header_size - 8,
        b"WAVE",
        b"fmt ", fmt_size,
        1, num_channels, sample_rate,
        byte_rate, block_align, bits,
        b"data", data_size,
    )


# ── VibeVoice subprocess worker ───────────────────────────────────────────────

def _vibevoice_worker_main():
    """
    Runs in an isolated subprocess. Reads (text, voice) from stdin as a JSON
    line, writes raw PCM int16 bytes to stdout, then exits.
    Heap corruption stays contained in this process.
    """
    import sys, json
    import numpy as np

    line = sys.stdin.readline()
    req  = json.loads(line)
    text  = req["text"]
    voice = req["voice"]

    model_path = os.getenv("VIBEVOICE_MODEL_PATH", "/opt/vibevoice")
    from streamingtts import StreamingTTS
    model = StreamingTTS(model_path)

    for chunk in model.stream(text, voice=voice):
        pcm = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
        sys.stdout.buffer.write(pcm.tobytes())
        sys.stdout.buffer.flush()


def _stream_vibevoice_subprocess(text: str, voice: str) -> Generator[bytes, None, None]:
    """
    Spawn an isolated subprocess for VibeVoice to avoid heap corruption
    contaminating the main API process.
    """
    import subprocess, json, sys

    if voice not in VIBEVOICE_VOICES:
        voice = VIBEVOICE_VOICE

    yield _wav_header(VIBEVOICE_SAMPLE_RATE)

    cmd = [sys.executable, "-c",
           "from api.tts import _vibevoice_worker_main; _vibevoice_worker_main()"]

    env = {**os.environ, "PYTHONPATH": "/app"}

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=env,
    )

    try:
        payload = json.dumps({"text": text, "voice": voice}) + "\n"
        proc.stdin.write(payload.encode())
        proc.stdin.close()

        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            yield chunk

        proc.wait(timeout=60)
    except Exception as e:
        log.error(f"VibeVoice subprocess failed: {e}")
        proc.kill()
        raise


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
    kokoro = _load_kokoro()
    samples, sr = kokoro.create(text, voice=voice, speed=1.0, lang="en-us")
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV")
    yield buf.getvalue()


# ── Public API ────────────────────────────────────────────────────────────────

def synthesize_stream(text: str, voice: str | None = None) -> Generator[bytes, None, None]:
    """
    Yield WAV header + PCM chunks.
    Uses VibeVoice (subprocess) on GPU, Kokoro on CPU.
    """
    if _gpu_available() and _vibevoice_installed():
        yield from _stream_vibevoice_subprocess(text, voice or VIBEVOICE_VOICE)
    else:
        yield from _stream_kokoro(text, voice or KOKORO_VOICE)


def synthesize(text: str, voice: str | None = None) -> bytes:
    return b"".join(synthesize_stream(text, voice))


def engine_info() -> dict:
    gpu  = _gpu_available()
    vibe = _vibevoice_installed()
    return {
        "engine":           "vibevoice" if (gpu and vibe) else "kokoro",
        "gpu_enabled":      gpu,
        "vibevoice_ready":  vibe,
        "vibevoice_voice":  VIBEVOICE_VOICE,
        "vibevoice_voices": VIBEVOICE_VOICES,
        "kokoro_voice":     KOKORO_VOICE,
        "kokoro_ready":     _KOKORO_MODEL.exists() and _KOKORO_VOICES.exists(),
    }
