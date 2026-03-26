"""
Omnex — TTS Engine
VibeVoice-Realtime-0.5B on GPU  → realtime streaming PCM (~300ms first token)
                                   runs in a persistent isolated subprocess
                                   (avoids heap conflicts with onnxruntime-gpu)
Kokoro ONNX on CPU              → full WAV fallback

/voice/speak  POST  { text, voice? }  → audio/wav (streaming or full)
"""

from __future__ import annotations

import io
import json
import logging
import os
import struct
import subprocess
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Generator

log = logging.getLogger("omnex.tts")

# ── Kokoro paths ──────────────────────────────────────────────────────────────
_KOKORO_DIR    = Path(os.getenv("OMNEX_DATA_PATH", "/data")) / "models" / "kokoro"
_KOKORO_MODEL  = _KOKORO_DIR / "kokoro-v1.0.onnx"
_KOKORO_VOICES = _KOKORO_DIR / "voices-v1.0.bin"

# ── VibeVoice config ──────────────────────────────────────────────────────────
# Voice names map to .pt files in /opt/vibevoice/demo/voices/streaming_model/
# e.g. "Emma" → "en-Emma_woman.pt", "Carter" → "en-Carter_man.pt"
VIBEVOICE_VOICES      = ["Carter", "Davis", "Emma", "Frank", "Grace", "Mike"]
VIBEVOICE_SAMPLE_RATE = 24000
VIBEVOICE_VOICE       = os.getenv("VIBEVOICE_VOICE", "Emma")
KOKORO_VOICE          = os.getenv("TTS_KOKORO_VOICE", "af_heart")

# Path to the cloned VibeVoice GitHub repo (set in docker-compose / Dockerfile)
_VIBEVOICE_REPO = os.getenv("VIBEVOICE_REPO_PATH", "/opt/vibevoice")
_VIBEVOICE_HF   = os.getenv("VIBEVOICE_MODEL_ID", "microsoft/VibeVoice-Realtime-0.5B")


# ── Persistent VibeVoice worker process ───────────────────────────────────────

# Protocol (over stdin/stdout of the worker):
#   Request  (main → worker): JSON line  {"text": "...", "voice": "Emma"}\n
#   Response (worker → main): binary frames, each prefixed with a 4-byte LE
#                              uint32 length. Length 0 = end of stream.

_WORKER_SCRIPT = r"""
import sys, os, json, struct, copy, traceback
import torch
import numpy as np

repo_path  = os.getenv("VIBEVOICE_REPO_PATH", "/opt/vibevoice")
model_id   = os.getenv("VIBEVOICE_MODEL_ID",  "microsoft/VibeVoice-Realtime-0.5B")
voices_dir = os.path.join(repo_path, "demo", "voices", "streaming_model")

# Add the repo to sys.path so vibevoice package is importable
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
from vibevoice.modular.streamer import AudioStreamer

# ── Voice preset loader ───────────────────────────────────────────────────────

def _load_voice_presets():
    # Scan voices dir and build name->path map (case-insensitive partial match)
    import glob
    presets = {}
    if os.path.isdir(voices_dir):
        for pt in glob.glob(os.path.join(voices_dir, "*.pt")):
            stem = os.path.splitext(os.path.basename(pt))[0].lower()
            presets[stem] = pt
    return presets

_voice_presets = _load_voice_presets()

def _resolve_voice(name: str) -> str:
    # Return the .pt file path for the given voice name
    key = name.lower()
    # exact match
    if key in _voice_presets:
        return _voice_presets[key]
    # partial match (e.g. "emma" matches "en-emma_woman")
    matches = [p for k, p in _voice_presets.items() if key in k]
    if matches:
        return matches[0]
    # fallback to first available
    if _voice_presets:
        return next(iter(_voice_presets.values()))
    raise RuntimeError(f"No voice presets found in {voices_dir}")

# ── Model loading ─────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    processor = VibeVoiceStreamingProcessor.from_pretrained(model_id)
    if device == "cuda":
        try:
            import flash_attn  # noqa: F401
            _attn = "flash_attention_2"
        except ImportError:
            _attn = "sdpa"
        model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation=_attn,
        )
    else:
        model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
            attn_implementation="sdpa",
        )
    model.eval()
    model.set_ddpm_inference_steps(num_steps=5)
    _model_ok = True
except Exception as e:
    sys.stderr.write(f"[vibevoice-worker] model load failed: {e}\n")
    sys.stderr.flush()
    _model_ok = False

# Signal ready (or failure)
if _model_ok:
    sys.stdout.buffer.write(b"READY\n")
else:
    sys.stdout.buffer.write(b"FAILED\n")
sys.stdout.buffer.flush()

if not _model_ok:
    sys.exit(1)

# ── GPU warmup — run one silent inference so first real request is fast ────────
try:
    _warmup_voice = next(iter(_voice_presets.values()), None)
    if _warmup_voice:
        _wp = torch.load(_warmup_voice, map_location=device, weights_only=False)
        _wi = processor.process_input_with_cached_prompt(
            text="hi",
            cached_prompt=_wp,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        for k, v in _wi.items():
            if torch.is_tensor(v):
                _wi[k] = v.to(device)
        import threading as _wt
        _ws = AudioStreamer(batch_size=1, stop_signal=None, timeout=10.0)
        def _wrun():
            try:
                model.generate(**_wi, max_new_tokens=None, cfg_scale=1.5,
                               tokenizer=processor.tokenizer,
                               generation_config={"do_sample": False},
                               verbose=False, all_prefilled_outputs=copy.deepcopy(_wp),
                               audio_streamer=_ws)
            finally:
                _ws.end()
        _wthread = _wt.Thread(target=_wrun, daemon=True)
        _wthread.start()
        for _ in _ws.get_stream(0):
            pass
        _wthread.join()
except Exception as _we:
    sys.stderr.write(f"[vibevoice-worker] warmup error (non-fatal): {_we}\n")

# ── Request loop ──────────────────────────────────────────────────────────────

while True:
    line = sys.stdin.readline()
    if not line:
        break
    try:
        req   = json.loads(line)
        text  = req["text"]
        voice = req.get("voice", "Emma")

        voice_path = _resolve_voice(voice)
        all_prefilled = torch.load(voice_path, map_location=device, weights_only=False)

        inputs = processor.process_input_with_cached_prompt(
            text=text,
            cached_prompt=all_prefilled,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(device)

        import threading as _threading
        stop_event = _threading.Event()
        audio_streamer = AudioStreamer(batch_size=1, stop_signal=stop_event, timeout=60.0)

        def _generate():
            try:
                model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=1.5,
                    tokenizer=processor.tokenizer,
                    generation_config={"do_sample": False},
                    verbose=False,
                    all_prefilled_outputs=copy.deepcopy(all_prefilled),
                    audio_streamer=audio_streamer,
                    stop_check_fn=stop_event.is_set,
                )
            except Exception as exc:
                sys.stderr.write(f"[vibevoice-worker] generate error: {exc}\n")
                sys.stderr.flush()
            finally:
                audio_streamer.end()

        t = _threading.Thread(target=_generate, daemon=True)
        t.start()

        import numpy as np
        for chunk in audio_streamer.get_stream(0):
            if chunk is None:
                break
            if torch.is_tensor(chunk):
                chunk = chunk.detach().cpu().to(torch.float32).numpy()
            else:
                chunk = np.asarray(chunk, dtype=np.float32)
            if chunk.ndim > 1:
                chunk = chunk.reshape(-1)
            pcm = np.clip(chunk, -1.0, 1.0)
            data = (pcm * 32767.0).astype(np.int16).tobytes()
            sys.stdout.buffer.write(struct.pack("<I", len(data)))
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()

        stop_event.set()
        t.join()

    except Exception as e:
        sys.stderr.write(f"[vibevoice-worker] request error: {e}\n{traceback.format_exc()}\n")
        sys.stderr.flush()

    # End-of-stream sentinel
    sys.stdout.buffer.write(struct.pack("<I", 0))
    sys.stdout.buffer.flush()
"""


class _VibeVoiceWorker:
    """
    A single long-lived subprocess running the VibeVoice model.
    The model is loaded once. Requests are serialised — one at a time.
    """

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._ready = False

    def _spawn(self) -> bool:
        env = {
            **os.environ,
            "PYTHONPATH": f"/app:{_VIBEVOICE_REPO}",
            "VIBEVOICE_REPO_PATH": _VIBEVOICE_REPO,
            "VIBEVOICE_MODEL_ID":  _VIBEVOICE_HF,
        }
        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-c", _WORKER_SCRIPT],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
            # Wait for READY signal (model loaded — can take 30-60s on first run)
            line = self._proc.stdout.readline()
            if line.strip() == b"READY":
                log.info("[vibevoice] worker ready")
                self._ready = True
                return True
            else:
                stderr = self._proc.stderr.read(2048) if self._proc.stderr else b""
                log.error(f"[vibevoice] worker startup failed: {line!r}\n{stderr.decode(errors='replace')}")
                self._proc.kill()
                self._proc = None
                return False
        except Exception as e:
            log.error(f"[vibevoice] failed to spawn worker: {e}")
            self._proc = None
            return False

    def _is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stream(self, text: str, voice: str) -> Generator[bytes, None, None]:
        with self._lock:
            # Spawn or respawn if dead
            if not self._is_alive():
                self._ready = False
                log.info("[vibevoice] spawning worker process")
                if not self._spawn():
                    raise RuntimeError("VibeVoice worker failed to start")

            # Send request
            req = json.dumps({"text": text, "voice": voice}) + "\n"
            self._proc.stdin.write(req.encode())
            self._proc.stdin.flush()

            # Read framed PCM chunks until sentinel
            while True:
                header = self._proc.stdout.read(4)
                if len(header) < 4:
                    break
                length = struct.unpack("<I", header)[0]
                if length == 0:
                    break
                data = self._proc.stdout.read(length)
                if not data:
                    break
                yield data

    def shutdown(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._proc.wait(timeout=5)


# Module-level singleton — created lazily on first TTS request
_worker: _VibeVoiceWorker | None = None
_worker_init_lock = threading.Lock()


def _get_worker() -> _VibeVoiceWorker:
    global _worker
    if _worker is None:
        with _worker_init_lock:
            if _worker is None:
                _worker = _VibeVoiceWorker()
    return _worker


# ── Capability checks ─────────────────────────────────────────────────────────

def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available() and os.getenv("GPU_ENABLED", "false").lower() == "true"
    except ImportError:
        return False


def _vibevoice_installed() -> bool:
    """Check that the VibeVoice repo is present and importable."""
    repo = Path(_VIBEVOICE_REPO)
    return (repo / "vibevoice" / "modular" / "modeling_vibevoice_streaming_inference.py").exists()


# ── WAV header ────────────────────────────────────────────────────────────────

def _wav_header(sample_rate: int, num_channels: int = 1, bits: int = 16) -> bytes:
    data_size   = 0xFFFFFFFF
    header_size = 44
    fmt_size    = 16
    byte_rate   = sample_rate * num_channels * bits // 8
    block_align = num_channels * bits // 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 0xFFFFFFFF,
        b"WAVE",
        b"fmt ", fmt_size,
        1, num_channels, sample_rate,
        byte_rate, block_align, bits,
        b"data", data_size,
    )


# ── VibeVoice stream ──────────────────────────────────────────────────────────

def _stream_vibevoice(text: str, voice: str) -> Generator[bytes, None, None]:
    if voice not in VIBEVOICE_VOICES:
        voice = VIBEVOICE_VOICE

    worker = _get_worker()
    yield _wav_header(VIBEVOICE_SAMPLE_RATE)
    yield from worker.stream(text, voice)


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
    if _gpu_available() and _vibevoice_installed():
        yield from _stream_vibevoice(text, voice or VIBEVOICE_VOICE)
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
