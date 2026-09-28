"""Writing a labelling session out: YOLO label files and `verified.jsonl`.

Both the batch tracker (`seed_track`) and the interactive annotator
(`annotation.Session`) end by writing these two things, in the same formats, so
the formats live here rather than in either of them.

`verified.jsonl` is the load-bearing one. It records which frames a person
actually adjudicated, keyed exactly as a run's `detections.jsonl` is, so
`src.evaluate --keys-from` accepts it directly. Frames absent from it were never
looked at, and scoring them as negatives would invent precision nobody measured.

Note `write_verified` here **merges**, while `annotation.Session.export` replaces
its own stem's rows wholesale. That is a real difference, not an oversight: the
batch tracker only ever adds frames, whereas the annotator is authoritative for
its stem and must be able to *remove* a frame the user un-verified. Sharing one
of those would silently resurrect deleted work, so they stay separate and only
the formats are shared.
"""

from __future__ import annotations

import json
from pathlib import Path

from .template_track import TrackPoint

def to_yolo(box: tuple[float, float, float, float], width: int,
            height: int) -> tuple[float, float, float, float]:
    """Frame-pixel `(x, y, w, h)` -> normalised YOLO `(cx, cy, w, h)`.

    Clamped to the frame: a tracker that drifted off the edge would otherwise
    write a label outside [0, 1] that every later reader has to second-guess.
    """
    x, y, w, h = box
    cx = min(max((x + w / 2) / width, 0.0), 1.0)
    cy = min(max((y + h / 2) / height, 0.0), 1.0)
    return cx, cy, min(w / width, 1.0), min(h / height, 1.0)



def yolo_line(box: tuple[float, float, float, float],
              width: int, height: int) -> str:
    """One YOLO label line for a single-class box: `0 cx cy w h`, normalised.

    Six decimals, which at 4K is well under a tenth of a pixel -- the precision
    is free and keeps a round-trip through the file lossless at any resolution
    this project uses.
    """
    cx, cy, w, h = to_yolo(box, width, height)
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"


def write_labels(points: list[TrackPoint], out_dir: Path, stem: str,
                 width: int, height: int) -> None:
    """One YOLO label file per proposed box, class 0."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for point in points:
        cx, cy, w, h = to_yolo(point.box, width, height)
        path = out_dir / f"{stem}_{point.frame:04d}.txt"
        path.write_text(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n", encoding="utf-8")


def write_verified(frames: set[int], path: Path, images_dir: Path,
                   stem: str) -> int:
    """Record which frames a human has adjudicated, merging with earlier runs.

    Keyed exactly as the run's `detections.jsonl` is, so `src.evaluate
    --keys-from` accepts this file directly. Frames absent from it were never
    looked at and must not be scored -- counting them as negatives would invent
    precision nobody measured.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                keys[Path(record["image"]).stem] = record
    for frame in frames:
        name = f"{stem}_{frame:04d}"
        keys[name] = {"image": str(images_dir / f"{name}.jpg"), "detections": []}
    with path.open("w", encoding="utf-8") as handle:
        for name in sorted(keys):
            handle.write(json.dumps(keys[name]) + "\n")
    return len(keys)
