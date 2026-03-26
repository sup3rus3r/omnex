"""
Omnex — Image Embeddings
CLIP ViT-B/32 wrapper for image and text-to-image search.
512-dimensional embeddings, cosine similarity.

Supports:
  - Image → embedding (for indexing)
  - Text  → embedding (for cross-modal search: "find photos of sunset")
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_model     = None
_processor = None
MODEL_ID   = "openai/clip-vit-base-patch32"
EMBEDDING_DIM = 512


def _get_model():
    global _model, _processor
    if _model is None:
        from transformers import CLIPModel, CLIPProcessor
        cache_dir = str(
            Path(os.getenv("OMNEX_DATA_PATH", "./data")).parent / "models" / "cache"
        )
        # Run on CPU to avoid torch CUDA init in the main uvicorn process.
        # CUDA init alongside jemalloc + onnxruntime-gpu causes SIGABRT.
        _processor = CLIPProcessor.from_pretrained(MODEL_ID, cache_dir=cache_dir)
        _model     = CLIPModel.from_pretrained(MODEL_ID, cache_dir=cache_dir)
        _model.eval()

    return _model, _processor


def embed_image(image) -> list[float]:
    """
    Embed a PIL Image.

    Args:
        image: PIL.Image.Image (RGB)

    Returns:
        List of 512 floats (L2-normalised)
    """
    import torch
    model, processor = _get_model()

    inputs = processor(images=image, return_tensors="pt")
    if False:  # GPU disabled — see _get_model comment
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        features = model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)

    return features[0].cpu().numpy().astype(np.float32).tolist()


def embed_image_batch(images: list) -> np.ndarray:
    """
    Embed a batch of PIL Images.

    Returns:
        np.ndarray of shape (N, 512), dtype float32, L2-normalised
    """
    import torch
    if not images:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    model, processor = _get_model()
    inputs = processor(images=images, return_tensors="pt", padding=True)
    if False:  # GPU disabled — see _get_model comment
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        features = model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy().astype(np.float32)


def embed_text(text: str) -> list[float]:
    """
    Embed a text query into CLIP's image embedding space.
    Enables cross-modal search: text query → image results.

    Returns:
        List of 512 floats (L2-normalised)
    """
    import torch
    model, processor = _get_model()

    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    if False:  # GPU disabled — see _get_model comment
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        features = model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)

    return features[0].cpu().numpy().astype(np.float32).tolist()


def embed_text_batch(texts: list[str]) -> np.ndarray:
    """
    Embed a list of text queries into CLIP space.

    Returns:
        np.ndarray of shape (N, 512), dtype float32
    """
    import torch
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    model, processor = _get_model()
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    if False:  # GPU disabled — see _get_model comment
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        features = model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy().astype(np.float32)


def _gpu_available() -> bool:
    if os.getenv("GPU_ENABLED", "false").lower() != "true":
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
