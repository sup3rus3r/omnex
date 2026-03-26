"""
Omnex — Model Manager
Checks which models are present and downloads missing ones with progress callbacks.

Models required:
  - all-MiniLM-L6-v2        text embeddings       ~90MB
  - clip-vit-base-patch32    image/video           ~350MB
  - microsoft/codebert-base  code embeddings       ~500MB
  - whisper-base             audio transcription   ~140MB
  - deepface (retinaface + facenet)                ~200MB
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# HuggingFace cache root
HF_CACHE = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
WHISPER_CACHE = Path.home() / ".cache" / "whisper"


@dataclass
class ModelInfo:
    id:          str         # display identifier
    name:        str         # human label
    size_mb:     int         # approximate download size
    optional:    bool = False      # if True, errors don't block setup completion
    status:      str = "pending"   # pending | downloading | ready | error
    progress:    float = 0.0       # 0–100
    error:       str = ""


# Registry of all required models
MODELS: list[ModelInfo] = [
    ModelInfo("minilm",    "MiniLM-L6-v2 (Text)",      90),
    ModelInfo("clip",      "CLIP ViT-B/32 (Images)",   350),
    ModelInfo("codebert",  "CodeBERT (Code)",           500),
    ModelInfo("whisper",   "Whisper Base (Audio)",      140),
    ModelInfo("deepface",  "InsightFace buffalo_l",      320),
]

# Global state — updated by download threads, read by SSE stream
_lock = threading.Lock()


def get_model(mid: str) -> ModelInfo | None:
    return next((m for m in MODELS if m.id == mid), None)


def all_ready() -> bool:
    return all(_is_present(m) or m.optional for m in MODELS)


def status_snapshot() -> list[dict]:
    result = []
    for m in MODELS:
        present = _is_present(m)
        with _lock:
            s = m.status if m.status in ("downloading", "error") else ("ready" if present else "pending")
            p = m.progress if m.status == "downloading" else (100.0 if present else 0.0)
        result.append({
            "id":       m.id,
            "name":     m.name,
            "size_mb":  m.size_mb,
            "status":   s,
            "progress": p,
            "error":    m.error,
        })
    return result


# ── Presence checks ────────────────────────────────────────────────────────────

def _is_present(m: ModelInfo) -> bool:
    try:
        if m.id == "minilm":
            return _hf_model_present("sentence-transformers/all-MiniLM-L6-v2")
        if m.id == "clip":
            return _hf_model_present("openai/clip-vit-base-patch32")
        if m.id == "codebert":
            return _hf_model_present("microsoft/codebert-base")
        if m.id == "whisper":
            return (WHISPER_CACHE / "base.pt").exists()
        if m.id == "deepface":
            return _deepface_present()
    except Exception:
        pass
    return False


def _hf_model_present(repo_id: str) -> bool:
    """Check if a HuggingFace model has been downloaded to local cache."""
    # HF cache layout: ~/.cache/huggingface/hub/models--org--name/snapshots/
    slug = "models--" + repo_id.replace("/", "--")
    model_dir = HF_CACHE / "hub" / slug
    if not model_dir.exists():
        return False
    snapshots = list((model_dir / "snapshots").glob("*")) if (model_dir / "snapshots").exists() else []
    return len(snapshots) > 0


def _deepface_present() -> bool:
    # Check by file presence — avoids importing onnxruntime which crashes in Anaconda shells
    venv_site = Path(__file__).parent.parent / ".venv" / "Lib" / "site-packages"
    if not venv_site.exists():
        # Try relative to sys.prefix
        import sys
        venv_site = Path(sys.prefix) / "Lib" / "site-packages"
    return (venv_site / "insightface").exists() and (
        (venv_site / "onnxruntime").exists() or
        (venv_site / "onnxruntime_gpu").exists()
    )


# ── Downloads ─────────────────────────────────────────────────────────────────

def download_all(on_progress: Callable[[str, float, str], None] | None = None) -> None:
    """
    Download all missing models sequentially.
    on_progress(model_id, percent, status) called on each update.
    """
    for m in MODELS:
        if _is_present(m):
            with _lock:
                m.status = "ready"
                m.progress = 100.0
            if on_progress:
                on_progress(m.id, 100.0, "ready")
            continue

        with _lock:
            m.status = "downloading"
            m.progress = 0.0

        if on_progress:
            on_progress(m.id, 0.0, "downloading")

        try:
            _download_model(m, on_progress)
            with _lock:
                m.status = "ready"
                m.progress = 100.0
            if on_progress:
                on_progress(m.id, 100.0, "ready")
        except Exception as e:
            with _lock:
                m.status = "error"
                m.error = str(e)
            if on_progress:
                on_progress(m.id, 0.0, "error")


def _download_model(m: ModelInfo, on_progress: Callable | None) -> None:
    def tick(p: float):
        with _lock:
            m.progress = p
        if on_progress:
            on_progress(m.id, p, "downloading")

    if m.id == "minilm":
        _download_hf_sentence_transformer("sentence-transformers/all-MiniLM-L6-v2", tick)
    elif m.id == "clip":
        _download_hf_clip("openai/clip-vit-base-patch32", tick)
    elif m.id == "codebert":
        _download_hf_transformers("microsoft/codebert-base", tick)
    elif m.id == "whisper":
        _download_whisper("base", tick)
    elif m.id == "deepface":
        _download_deepface(tick)


def _download_hf_sentence_transformer(repo_id: str, tick: Callable) -> None:
    _hf_snapshot(repo_id, tick)


def _download_hf_clip(repo_id: str, tick: Callable) -> None:
    _hf_snapshot(repo_id, tick)


def _download_hf_transformers(repo_id: str, tick: Callable) -> None:
    _hf_snapshot(repo_id, tick)


def _download_whisper(size: str, tick: Callable) -> None:
    """Download whisper model weights directly via requests — no torch needed."""
    import urllib.request

    WHISPER_URLS = {
        "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    }
    url = WHISPER_URLS.get(size)
    if not url:
        raise RuntimeError(f"Unknown whisper model size: {size}")

    WHISPER_CACHE.mkdir(parents=True, exist_ok=True)
    dest = WHISPER_CACHE / f"{size}.pt"

    tick(5.0)
    _download_url_with_progress(url, dest, tick, start=5.0, end=98.0)
    tick(100.0)


def _download_deepface(tick: Callable) -> None:
    """Verify InsightFace is installed by checking package files on disk.

    Avoids importing onnxruntime directly (crashes in Anaconda shells).
    buffalo_l model weights download automatically on first image ingestion.
    """
    tick(10.0)
    if not _deepface_present():
        raise RuntimeError(
            "InsightFace not installed. "
            "Run: .venv\\Scripts\\pip install insightface onnxruntime-gpu"
        )
    tick(100.0)


# ── Direct download helpers ────────────────────────────────────────────────────

_UNUSED = r"""
import sys, os

kind  = sys.argv[1]
param = sys.argv[2] if len(sys.argv) > 2 else ""

if kind == "sentence_transformer":
    from sentence_transformers import SentenceTransformer
    SentenceTransformer(param, device="cpu")

elif kind == "clip":
    from transformers import CLIPProcessor, CLIPModel
    CLIPProcessor.from_pretrained(param)
    CLIPModel.from_pretrained(param)

elif kind == "transformers":
    from transformers import AutoTokenizer, AutoModel
    AutoTokenizer.from_pretrained(param)
    AutoModel.from_pretrained(param)

elif kind == "whisper":
    import whisper
    whisper.load_model(param)

elif kind == "deepface":
    import numpy as np
    from deepface import DeepFace
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    try:
        DeepFace.represent(dummy, model_name="Facenet512",
                           detector_backend="retinaface", enforce_detection=False)
    except Exception:
        pass

print("ok")
"""


def _clean_env() -> dict:
    """Return os.environ with Anaconda/conda directories stripped from PATH.

    Anaconda injects its own DLLs (MKL, libiomp5, etc.) before the venv's,
    which causes torch c10.dll to fail on Windows with CUDA 12.4.
    We keep everything except conda/Anaconda path entries.
    """
    env = dict(os.environ)
    path_parts = env.get("PATH", "").split(os.pathsep)
    cleaned = [
        p for p in path_parts
        if not any(kw in p.lower() for kw in ("anaconda", "conda", "miniconda"))
    ]
    env["PATH"] = os.pathsep.join(cleaned)
    return env


def _run_in_subprocess(kind: str, param: str, tick: Callable, midpoint: float) -> None:
    """Run a download in a clean subprocess, reporting progress at midpoint."""
    proc = subprocess.Popen(
        [sys.executable, "-c", _DOWNLOADER_SCRIPT, kind, param],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_clean_env(),
    )
    # Emit midpoint progress while waiting
    tick(midpoint)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        # Write full stderr to a log file for debugging
        log_path = Path(os.getenv("TEMP", "/tmp")) / f"omnex_{kind}_error.log"
        log_path.write_text(stderr, encoding="utf-8")
        err_lines = [l for l in stderr.strip().splitlines() if l.strip() and not l.startswith(" ")]
        msg = err_lines[-1] if err_lines else f"{kind} download failed"
        raise RuntimeError(f"{msg} (full log: {log_path})")


def _hf_snapshot(repo_id: str, tick: Callable) -> None:
    """Download a HuggingFace repo via huggingface_hub — no torch import needed.

    snapshot_download doesn't expose per-byte progress, so we approximate by
    polling the cache directory size against the expected total while the
    download runs in a thread.
    """
    import threading as _threading
    from huggingface_hub import snapshot_download, repo_info
    from huggingface_hub.utils import HfHubHTTPError

    tick(3.0)

    # Get total expected size from repo metadata (best-effort)
    total_bytes = 0
    try:
        info = repo_info(repo_id, files_metadata=True)
        if hasattr(info, "siblings"):
            total_bytes = sum(
                getattr(s, "size", 0) or 0
                for s in info.siblings
                if s.rfilename and not any(
                    s.rfilename.endswith(ext)
                    for ext in (".msgpack", "flax_model.safetensors", "tf_model.h5", "rust_model.ot")
                )
            )
    except Exception:
        pass

    # HF cache path for this repo
    slug = "models--" + repo_id.replace("/", "--")
    blobs_dir = HF_CACHE / "hub" / slug / "blobs"

    result: dict = {"done": False, "error": None}

    def _do_download():
        try:
            snapshot_download(
                repo_id=repo_id,
                ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
            )
        except Exception as e:
            result["error"] = e
        finally:
            result["done"] = True

    t = _threading.Thread(target=_do_download, daemon=True)
    t.start()

    import time
    while not result["done"]:
        if total_bytes > 0 and blobs_dir.exists():
            downloaded = sum(f.stat().st_size for f in blobs_dir.iterdir() if f.is_file())
            pct = 5.0 + min(downloaded / total_bytes, 0.97) * 93.0
            tick(pct)
        time.sleep(1.5)

    if result["error"]:
        raise result["error"]
    tick(100.0)


def _download_url_with_progress(url: str, dest: Path, tick: Callable, start: float, end: float) -> None:
    """Stream a URL to dest, calling tick() with progress in [start, end]."""
    import urllib.request

    tmp = dest.with_suffix(".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "omnex/0.1"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk = 1024 * 256  # 256 KB
            with tmp.open("wb") as f:
                while True:
                    block = resp.read(chunk)
                    if not block:
                        break
                    f.write(block)
                    downloaded += len(block)
                    if total > 0:
                        pct = start + (downloaded / total) * (end - start)
                        tick(min(pct, end))
        tmp.rename(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
