"""Working out what kind of thing the user pointed `--source` at."""

from __future__ import annotations

from pathlib import Path

from ..errors import UsageError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}

VIDEO = "video"
IMAGES = "images"


def resolve_video(directory: Path, stem: str) -> Path:
    """Find `stem`'s video in `directory`, whatever container it is in.

    `.mp4` was hardcoded while every dataset was a `.mp4` release. Our own
    processed footage is not: a lossless crop has to be FFV1, which MP4 cannot
    carry, so `data/processed/SOFA-O4/videos/` holds `.avi`. Preferring `.mp4`
    keeps ARD-MAV and ARD100 resolving exactly as before.

    Raises `UsageError` with the directory listing: a missing video is a typo
    in `--video-names`, and the listing is usually the whole fix.
    """
    preferred = [".mp4", *sorted(VIDEO_SUFFIXES - {".mp4"})]
    for suffix in preferred:
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    known = sorted(p.name for p in directory.glob(f"{stem}.*")) if directory.is_dir() else []
    raise UsageError(f"No video named {stem!r} in {directory} "
             f"({'found: ' + ', '.join(known) if known else 'directory empty or missing'})")


def resolve_sources(source: Path) -> tuple[str, list[Path]]:
    """Classify `source` as VIDEO or IMAGES and list the files to process.

    Directories are searched recursively. A mistyped path raises `UsageError`,
    which the CLI turns into a one-line message rather than a traceback.
    """
    if source.is_dir():
        images = sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise UsageError(f"No images found under {source}")
        return IMAGES, images
    if source.suffix.lower() in VIDEO_SUFFIXES:
        return VIDEO, [source]
    if source.suffix.lower() in IMAGE_SUFFIXES:
        return IMAGES, [source]
    raise UsageError(f"Unrecognised source type: {source}")
