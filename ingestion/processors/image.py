"""
Omnex — Image Processor
Handles all image file types: JPG, PNG, HEIC, WEBP, GIF, TIFF, BMP, RAW
Extracts:
  - CLIP embedding (semantic scene understanding)
  - EXIF metadata (GPS, date/time, device, dimensions)
  - Thumbnail (256x256 JPG)
  - Face detection crops (stored for Phase 4 face clustering)
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ImageResult:
    embedding:    list[float]        # 512-dim CLIP vector
    thumbnail:    bytes              # 256x256 JPEG bytes
    metadata:     dict = field(default_factory=dict)
    detected_faces: list = field(default_factory=list)  # list[DetectedFace]
    text_content: str = ""           # OCR if applicable (future)


def process(path: Path) -> ImageResult | None:
    """
    Full image processing pipeline.

    Returns:
        ImageResult or None if the file cannot be processed.
    """
    try:
        image = _load_image(path)
        if image is None:
            return None

        embedding      = _embed(image)
        thumbnail      = _make_thumbnail(image)
        metadata       = _extract_exif(path, image)
        detected_faces = _detect_faces(path)

        return ImageResult(
            embedding=embedding,
            thumbnail=thumbnail,
            metadata=metadata,
            detected_faces=detected_faces,
        )

    except Exception as e:
        logger.warning(f"Image processing failed for {path}: {e}")
        return None


# ── Image loading ─────────────────────────────────────────────────────────────

def _load_image(path: Path):
    """Load image as PIL Image. Handles HEIC via pillow-heif if available."""
    from PIL import Image

    suffix = path.suffix.lower()

    if suffix in {".heic", ".heif"}:
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError:
            logger.warning(f"pillow-heif not installed — cannot process {path.name}")
            return None

    try:
        img = Image.open(path)
        img.load()  # Force decode — catches corrupt files early
        # Normalise to RGB (handles RGBA, palette, grayscale)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        return img
    except Exception as e:
        logger.warning(f"Cannot open image {path}: {e}")
        return None


# ── CLIP embedding ────────────────────────────────────────────────────────────

def _embed(image) -> list[float]:
    from embeddings.image import embed_image
    return embed_image(image)


# ── Thumbnail ─────────────────────────────────────────────────────────────────

def _make_thumbnail(image, size: tuple[int, int] = (256, 256)) -> bytes:
    from PIL import Image
    thumb = image.copy()
    thumb.thumbnail(size, Image.LANCZOS)
    buf = io.BytesIO()
    # Convert to RGB before saving as JPEG (handles grayscale)
    if thumb.mode != "RGB":
        thumb = thumb.convert("RGB")
    thumb.save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue()


# ── EXIF metadata ─────────────────────────────────────────────────────────────

def _extract_exif(path: Path, image) -> dict:
    metadata: dict[str, Any] = {
        "dimensions": {"w": image.width, "h": image.height},
    }

    try:
        from PIL import ExifTags
        exif_data = image._getexif()
        if not exif_data:
            return metadata

        tag_map = {v: k for k, v in ExifTags.TAGS.items()}

        # Date/time
        dt_tag = exif_data.get(tag_map.get("DateTimeOriginal"))
        if not dt_tag:
            dt_tag = exif_data.get(tag_map.get("DateTime"))
        if dt_tag:
            metadata["exif_datetime"] = dt_tag

        # Device
        make  = exif_data.get(tag_map.get("Make", 0), "")
        model = exif_data.get(tag_map.get("Model", 0), "")
        if make or model:
            metadata["device"] = f"{make} {model}".strip()

        # GPS
        gps_info = exif_data.get(tag_map.get("GPSInfo", 0))
        if gps_info:
            gps = _parse_gps(gps_info)
            if gps:
                metadata["gps"] = gps

        # Orientation
        orientation = exif_data.get(tag_map.get("Orientation", 0))
        if orientation:
            metadata["orientation"] = orientation

    except (AttributeError, Exception):
        pass  # Not all images have EXIF — that's fine

    return metadata


def _detect_faces(path: Path) -> list:
    """Run face detection on the image. Returns list[DetectedFace]."""
    try:
        from embeddings.faces import detect_faces
        return detect_faces(path)
    except Exception as e:
        logger.debug(f"Face detection skipped for {path}: {e}")
        return []


def _parse_gps(gps_info: dict) -> dict | None:
    """Convert raw EXIF GPS tuples to decimal lat/lng."""
    try:
        from PIL.ExifTags import GPSTAGS
        gps = {GPSTAGS.get(k, k): v for k, v in gps_info.items()}

        def to_decimal(dms, ref: str) -> float:
            d, m, s = dms
            decimal = float(d) + float(m) / 60 + float(s) / 3600
            if ref in ("S", "W"):
                decimal *= -1
            return round(decimal, 6)

        lat = to_decimal(gps["GPSLatitude"],  gps.get("GPSLatitudeRef",  "N"))
        lng = to_decimal(gps["GPSLongitude"], gps.get("GPSLongitudeRef", "E"))
        return {"lat": lat, "lng": lng}

    except (KeyError, TypeError, ZeroDivisionError):
        return None
