"""Working out what kind of thing the user pointed `--source` at."""

from __future__ import annotations

import sys
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}

VIDEO = "video"
IMAGES = "images"


def resolve_sources(source: Path) -> tuple[str, list[Path]]:
    """Classify `source` as VIDEO or IMAGES and list the files to process.

    Directories are searched recursively. Exits with a message rather than
    raising: a mistyped path is a user error, not an exceptional condition.
    """
    if source.is_dir():
        images = sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            sys.exit(f"No images found under {source}")
        return IMAGES, images
    if source.suffix.lower() in VIDEO_SUFFIXES:
        return VIDEO, [source]
    if source.suffix.lower() in IMAGE_SUFFIXES:
        return IMAGES, [source]
    sys.exit(f"Unrecognised source type: {source}")
