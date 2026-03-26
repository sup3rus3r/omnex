"""
Omnex — Face Detection, Embedding & Clustering
Phase 4: "Find photos with my sister" works.

Pipeline:
  1. Detection  — InsightFace (buffalo_l) detects faces in every image
  2. Embedding  — ArcFace generates 512-dim identity embeddings per face crop
  3. Clustering — DBSCAN groups faces into identity clusters (no labels needed)
  4. Labelling  — User names each cluster once via IdentityManager UI
  5. Online     — New photos auto-classify against known clusters

All runs locally via InsightFace (ONNX). No cloud, no external API.
"""

from __future__ import annotations

import io
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

FACE_EMBEDDING_DIM = 512   # InsightFace ArcFace embedding dimension
DBSCAN_EPS         = 0.4   # cosine distance threshold for same-identity
DBSCAN_MIN_SAMPLES = 2     # min faces to form a cluster
CONFIDENCE_THRESHOLD = 0.75  # auto-classify confidence — below triggers user review


@dataclass
class DetectedFace:
    crop_bytes:  bytes           # JPEG crop of the face
    embedding:   list[float]     # 512-dim ArcFace embedding
    bbox:        dict            # {x, y, w, h} in original image coords
    confidence:  float           # detection confidence


@dataclass
class ClusterResult:
    cluster_id:  str
    face_count:  int
    embeddings:  list[list[float]]
    sample_crops: list[bytes]    # up to 5 representative crops for UI review
    label:        str | None = None


# ── Face detection & embedding ────────────────────────────────────────────────

def detect_faces(image_path: Path) -> list[DetectedFace]:
    """
    Detect all faces in an image and return ArcFace embeddings + crops.
    Uses InsightFace buffalo_l model pack (ONNX, no TF/torch DLL issues).

    Returns:
        List of DetectedFace (empty if no faces found or detection fails)
    """
    try:
        import cv2
        from insightface.app import FaceAnalysis

        # Initialise once — InsightFace caches the model in ~/.insightface
        app = _get_face_app()

        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return []

        detected = app.get(img_bgr)
        if not detected:
            return []

        from PIL import Image
        img_pil = Image.open(image_path).convert("RGB")
        faces: list[DetectedFace] = []

        for face in detected:
            embedding  = face.embedding.tolist() if face.embedding is not None else []
            confidence = float(face.det_score) if face.det_score is not None else 1.0

            if not embedding or confidence < 0.5:
                continue

            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            facial_area = {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
            crop_bytes  = _crop_face(img_pil, facial_area)
            if crop_bytes is None:
                continue

            faces.append(DetectedFace(
                crop_bytes=crop_bytes,
                embedding=embedding,
                bbox=facial_area,
                confidence=confidence,
            ))

        return faces

    except Exception as e:
        logger.debug(f"Face detection failed for {image_path}: {e}")
        return []


_face_app = None
_face_app_lock = None

def _get_face_app():
    """Return a cached InsightFace FaceAnalysis app (lazy init)."""
    import threading
    global _face_app, _face_app_lock
    if _face_app_lock is None:
        _face_app_lock = threading.Lock()
    with _face_app_lock:
        if _face_app is None:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            app.prepare(ctx_id=0, det_size=(640, 640))
            _face_app = app
    return _face_app


def _crop_face(image, facial_area: dict, padding: float = 0.2) -> bytes | None:
    """Crop face region with padding. Returns JPEG bytes."""
    try:
        x = facial_area.get("x", 0)
        y = facial_area.get("y", 0)
        w = facial_area.get("w", 0)
        h = facial_area.get("h", 0)

        if w == 0 or h == 0:
            return None

        # Add padding
        pad_x = int(w * padding)
        pad_y = int(h * padding)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(image.width,  x + w + pad_x)
        y2 = min(image.height, y + h + pad_y)

        crop = image.crop((x1, y1, x2, y2))
        crop = crop.resize((160, 160))  # Standard FaceNet input size

        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    except Exception:
        return None


# ── Clustering ────────────────────────────────────────────────────────────────

def cluster_embeddings(
    embeddings: list[list[float]],
    chunk_ids: list[str],
) -> list[ClusterResult]:
    """
    Cluster face embeddings using DBSCAN.
    Groups visually similar faces into identity clusters.

    Args:
        embeddings : list of 128-dim FaceNet vectors
        chunk_ids  : corresponding chunk IDs (for MongoDB linkage)

    Returns:
        List of ClusterResult — one per identified cluster.
        Noise points (label=-1) are excluded.
    """
    if len(embeddings) < DBSCAN_MIN_SAMPLES:
        return []

    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import normalize

    X = normalize(np.array(embeddings, dtype=np.float32))

    db = DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
        metric="cosine",
        n_jobs=-1,
    ).fit(X)

    labels = db.labels_
    unique_labels = set(labels) - {-1}  # -1 = noise

    clusters: list[ClusterResult] = []
    for label in unique_labels:
        mask     = labels == label
        idxs     = np.where(mask)[0]
        vecs     = [embeddings[i] for i in idxs]

        clusters.append(ClusterResult(
            cluster_id=str(uuid.uuid4()),
            face_count=len(idxs),
            embeddings=vecs,
            sample_crops=[],  # filled in by caller with actual crop bytes
        ))

    # Sort largest clusters first
    clusters.sort(key=lambda c: c.face_count, reverse=True)
    return clusters


# ── Online classification ─────────────────────────────────────────────────────

def classify_face(
    embedding: list[float],
    known_identities: list[dict],
) -> tuple[str | None, float]:
    """
    Classify a new face embedding against known identity clusters.

    Args:
        embedding        : 128-dim FaceNet vector for the new face
        known_identities : list of identity docs from MongoDB
                           each must have 'cluster_id' and 'face_embeddings'

    Returns:
        (cluster_id, confidence) — cluster_id is None if no confident match
    """
    if not known_identities:
        return None, 0.0

    from sklearn.preprocessing import normalize

    query = normalize(np.array(embedding, dtype=np.float32).reshape(1, -1))[0]

    best_id    = None
    best_score = 0.0

    for identity in known_identities:
        stored = identity.get("face_embeddings", [])
        if not stored:
            continue

        vecs  = normalize(np.array(stored, dtype=np.float32))
        # Cosine similarity = dot product of L2-normalised vectors
        sims  = vecs @ query
        score = float(np.max(sims))

        if score > best_score:
            best_score = score
            best_id    = identity["cluster_id"]

    if best_score >= CONFIDENCE_THRESHOLD:
        return best_id, best_score

    return None, best_score  # below threshold — needs user review


# ── Centroid helper ───────────────────────────────────────────────────────────

def cluster_centroid(embeddings: list[list[float]]) -> list[float]:
    """Compute the mean embedding for a cluster — used for fast similarity."""
    arr = np.array(embeddings, dtype=np.float32)
    mean = arr.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm > 0:
        mean /= norm
    return mean.tolist()
