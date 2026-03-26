"""
Omnex — File Type Detector
Detects file types by content using libmagic, not file extension.
This ensures accurate detection of renamed or extensionless files.
"""

from enum import Enum
from pathlib import Path

try:
    import magic
    _MAGIC_AVAILABLE = True
except ImportError:
    _MAGIC_AVAILABLE = False

import mimetypes


class FileType(str, Enum):
    DOCUMENT = "document"
    IMAGE    = "image"
    VIDEO    = "video"
    AUDIO    = "audio"
    CODE     = "code"
    ARCHIVE  = "archive"
    UNKNOWN  = "unknown"


MIME_TO_TYPE: dict[str, FileType] = {
    # Documents
    "application/pdf":                                                                      FileType.DOCUMENT,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document":             FileType.DOCUMENT,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":                   FileType.DOCUMENT,
    "application/vnd.openxmlformats-officedocument.presentationml.presentation":           FileType.DOCUMENT,
    "application/msword":                                                                   FileType.DOCUMENT,
    "application/vnd.ms-excel":                                                             FileType.DOCUMENT,
    "application/vnd.ms-powerpoint":                                                        FileType.DOCUMENT,
    "text/plain":                                                                           FileType.DOCUMENT,
    "text/markdown":                                                                        FileType.DOCUMENT,
    "text/html":                                                                            FileType.DOCUMENT,
    "text/xml":                                                                             FileType.DOCUMENT,
    "application/xml":                                                                      FileType.DOCUMENT,
    "application/json":                                                                     FileType.DOCUMENT,
    # Images
    "image/jpeg":                                                                           FileType.IMAGE,
    "image/png":                                                                            FileType.IMAGE,
    "image/gif":                                                                            FileType.IMAGE,
    "image/webp":                                                                           FileType.IMAGE,
    "image/heic":                                                                           FileType.IMAGE,
    "image/heif":                                                                           FileType.IMAGE,
    "image/tiff":                                                                           FileType.IMAGE,
    "image/bmp":                                                                            FileType.IMAGE,
    "image/svg+xml":                                                                        FileType.IMAGE,
    # Video
    "video/mp4":                                                                            FileType.VIDEO,
    "video/x-matroska":                                                                     FileType.VIDEO,
    "video/quicktime":                                                                      FileType.VIDEO,
    "video/x-msvideo":                                                                      FileType.VIDEO,
    "video/webm":                                                                           FileType.VIDEO,
    "video/mpeg":                                                                           FileType.VIDEO,
    "video/x-ms-wmv":                                                                       FileType.VIDEO,
    # Audio
    "audio/mpeg":                                                                           FileType.AUDIO,
    "audio/flac":                                                                           FileType.AUDIO,
    "audio/x-wav":                                                                          FileType.AUDIO,
    "audio/wav":                                                                            FileType.AUDIO,
    "audio/mp4":                                                                            FileType.AUDIO,
    "audio/x-m4a":                                                                          FileType.AUDIO,
    "audio/ogg":                                                                            FileType.AUDIO,
    "audio/aac":                                                                            FileType.AUDIO,
    # Archives
    "application/zip":                                                                      FileType.ARCHIVE,
    "application/x-tar":                                                                    FileType.ARCHIVE,
    "application/x-rar-compressed":                                                         FileType.ARCHIVE,
    "application/x-7z-compressed":                                                          FileType.ARCHIVE,
    "application/gzip":                                                                     FileType.ARCHIVE,
}

CODE_EXTENSIONS: set[str] = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".go", ".rs", ".java", ".kt", ".swift",
    ".cpp", ".c", ".cc", ".h", ".hpp",
    ".cs", ".rb", ".php", ".scala", ".r",
    ".sh", ".bash", ".zsh", ".fish",
    ".sql", ".lua", ".dart", ".zig",
    ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".dockerfile", ".tf", ".hcl",
}

# Text-based mimetypes that should be routed as CODE if extension matches
CODE_MIMES: set[str] = {
    "text/plain",
    "text/x-python",
    "text/x-script.python",
    "application/javascript",
    "text/javascript",
    "application/typescript",
    "text/x-go",
    "text/x-java",
    "text/x-rustsrc",
    "text/x-csrc",
    "text/x-c++src",
    "text/x-shellscript",
}


def detect(path: Path) -> tuple[FileType, str]:
    """
    Detect the file type and MIME type of a file.

    Returns:
        (FileType, mime_type_string)

    Detection order:
    1. Code extension check — overrides MIME for known code extensions
    2. libmagic content detection (if available)
    3. mimetypes fallback (extension-based, less reliable)
    """
    suffix = path.suffix.lower()

    # Code detection by extension — takes priority
    if suffix in CODE_EXTENSIONS:
        mime = _get_mime(path)
        return FileType.CODE, mime

    mime = _get_mime(path)

    # Check MIME map
    file_type = MIME_TO_TYPE.get(mime, FileType.UNKNOWN)

    # Secondary code check — some code files detected as text/plain
    if file_type == FileType.UNKNOWN or (file_type == FileType.DOCUMENT and mime in CODE_MIMES):
        if suffix in CODE_EXTENSIONS:
            return FileType.CODE, mime

    return file_type, mime


def _get_mime(path: Path) -> str:
    """Get MIME type using libmagic if available, else mimetypes fallback."""
    if _MAGIC_AVAILABLE:
        try:
            return magic.from_file(str(path), mime=True)
        except Exception:
            pass
    # Fallback: extension-based
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def is_indexable(path: Path) -> bool:
    """Returns True if the file is a type Omnex can index."""
    if not path.is_file():
        return False
    if path.stat().st_size == 0:
        return False
    file_type, _ = detect(path)
    return file_type != FileType.UNKNOWN
