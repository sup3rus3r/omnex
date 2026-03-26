"""
Omnex — First-Run Model Downloader
Downloads and caches all required ML models to the local models/ directory.
Run once before starting Omnex for the first time.

Usage:
    python models/download.py
    python models/download.py --gpu          # Verify GPU availability
    python models/download.py --list         # List models without downloading
"""

import argparse
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).parent

MODELS = [
    {
        "name": "MiniLM (Text Embeddings)",
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "size": "~80 MB",
        "type": "sentence_transformer",
        "required": True,
    },
    {
        "name": "CLIP ViT-B/32 (Image/Video Understanding)",
        "model_id": "openai/clip-vit-base-patch32",
        "size": "~350 MB",
        "type": "clip",
        "required": True,
    },
    {
        "name": "Whisper Small (Audio/Video Transcription)",
        "model_id": "small",
        "size": "~250 MB",
        "type": "whisper",
        "required": True,
    },
    {
        "name": "CodeBERT Base (Code Embeddings)",
        "model_id": "microsoft/codebert-base",
        "size": "~500 MB",
        "type": "transformers",
        "required": False,
    },
    {
        "name": "Whisper Medium (Higher Accuracy Transcription)",
        "model_id": "medium",
        "size": "~500 MB",
        "type": "whisper",
        "required": False,
    },
]

OLLAMA_MODELS = [
    {
        "name": "Phi-3 Mini (Default LLM — 3.8B)",
        "model_id": "phi3:mini",
        "size": "~2.3 GB",
        "required": True,
    },
]


def print_header():
    print("\n" + "=" * 60)
    print("  Omnex — Model Downloader")
    print("  Everything, indexed. Nothing lost.")
    print("=" * 60 + "\n")


def list_models():
    print_header()
    print("Required models:")
    for m in MODELS:
        if m["required"]:
            print(f"  ✓  {m['name']} ({m['size']})")
    print("\nOptional models:")
    for m in MODELS:
        if not m["required"]:
            print(f"  ○  {m['name']} ({m['size']})")
    print("\nOllama LLM models (pulled via Ollama):")
    for m in OLLAMA_MODELS:
        print(f"  ✓  {m['name']} ({m['size']})")
    print()


def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU detected: {device} ({vram:.1f} GB VRAM)")
            print("  GPU_ENABLED=true recommended in your .env\n")
        else:
            print("  No CUDA GPU detected — CPU-only mode")
            print("  GPU_ENABLED=false in your .env\n")
    except ImportError:
        print("  torch not installed yet — run pip install -r requirements.txt first\n")


def download_sentence_transformer(model_id: str, name: str):
    from sentence_transformers import SentenceTransformer
    print(f"  Downloading {name}...")
    SentenceTransformer(model_id, cache_folder=str(MODELS_DIR / "cache"))
    print(f"  ✓ {name} ready\n")


def download_clip(model_id: str, name: str):
    from transformers import CLIPProcessor, CLIPModel
    print(f"  Downloading {name}...")
    CLIPModel.from_pretrained(model_id, cache_dir=str(MODELS_DIR / "cache"))
    CLIPProcessor.from_pretrained(model_id, cache_dir=str(MODELS_DIR / "cache"))
    print(f"  ✓ {name} ready\n")


def download_whisper(model_id: str, name: str):
    import whisper
    print(f"  Downloading {name}...")
    whisper.load_model(model_id, download_root=str(MODELS_DIR / "cache" / "whisper"))
    print(f"  ✓ {name} ready\n")


def download_transformers(model_id: str, name: str):
    from transformers import AutoTokenizer, AutoModel
    print(f"  Downloading {name}...")
    AutoTokenizer.from_pretrained(model_id, cache_dir=str(MODELS_DIR / "cache"))
    AutoModel.from_pretrained(model_id, cache_dir=str(MODELS_DIR / "cache"))
    print(f"  ✓ {name} ready\n")


def pull_ollama_model(model_id: str, name: str):
    import subprocess
    print(f"  Pulling {name} via Ollama...")
    result = subprocess.run(
        ["ollama", "pull", model_id],
        capture_output=False,
    )
    if result.returncode == 0:
        print(f"  ✓ {name} ready\n")
    else:
        print(f"  ✗ Failed to pull {name} — is Ollama running? (docker compose up -d)\n")


def download_all(include_optional: bool = False):
    print_header()
    print("Checking GPU...\n")
    check_gpu()

    (MODELS_DIR / "cache").mkdir(parents=True, exist_ok=True)

    print("Downloading ML models...\n")
    for model in MODELS:
        if not model["required"] and not include_optional:
            continue
        try:
            if model["type"] == "sentence_transformer":
                download_sentence_transformer(model["model_id"], model["name"])
            elif model["type"] == "clip":
                download_clip(model["model_id"], model["name"])
            elif model["type"] == "whisper":
                download_whisper(model["model_id"], model["name"])
            elif model["type"] == "transformers":
                download_transformers(model["model_id"], model["name"])
        except Exception as e:
            print(f"  ✗ Failed to download {model['name']}: {e}\n")

    print("Pulling Ollama LLM models...\n")
    for model in OLLAMA_MODELS:
        try:
            pull_ollama_model(model["model_id"], model["name"])
        except FileNotFoundError:
            print("  ✗ Ollama CLI not found — start services with: docker compose up -d\n")

    print("=" * 60)
    print("  All models downloaded. Omnex is ready to run.")
    print("  Start with: uvicorn api.main:app --host 127.0.0.1 --port 8000")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Omnex model downloader")
    parser.add_argument("--list", action="store_true", help="List models without downloading")
    parser.add_argument("--gpu", action="store_true", help="Check GPU availability only")
    parser.add_argument("--optional", action="store_true", help="Also download optional models")
    args = parser.parse_args()

    if args.list:
        list_models()
    elif args.gpu:
        print_header()
        check_gpu()
    else:
        download_all(include_optional=args.optional)
