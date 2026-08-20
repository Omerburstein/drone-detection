"""Reading ground-truth labels and pairing them with recorded predictions."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..algo.detections import Detections

LABEL_FIELDS = 5  # `<cls> <cx> <cy> <w> <h>` per YOLO label line

# Row keys that are the record's structure rather than something the run chose
# to record about the frame.
RESERVED_KEYS = frozenset({"image", "frame", "detections"})


@dataclass
class EvalFrame:
    """One image's ground truth and predictions, in absolute xyxy pixels.

    Named `EvalFrame` rather than `Frame` to keep it distinct from
    `src.data.frames.Frame`, which is a frame on its way *into* the detector;
    this is one coming back out with labels attached for scoring.
    """

    key: str
    gt_boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    gt_classes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    preds: Detections = field(default_factory=Detections.empty)
    # Whatever else the run recorded on this frame -- GLAD writes `branch`, the
    # state-machine path that produced the box. Carried rather than dropped so
    # `src.eval.records` can put it in the dump; nothing in the metrics reads it.
    extras: dict[str, Any] = field(default_factory=dict)


def yolo_to_xyxy(rows: np.ndarray, width: int, height: int) -> np.ndarray:
    """Normalised YOLO cx,cy,w,h -> absolute xyxy."""
    if len(rows) == 0:
        return np.zeros((0, 4))
    cx, cy = rows[:, 0] * width, rows[:, 1] * height
    half_w, half_h = rows[:, 2] * width / 2, rows[:, 3] * height / 2
    return np.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], axis=1)


def load_label_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a YOLO .txt label file into (cxcywh rows, class ids).

    A missing file means a legitimately empty frame, not an error — negatives
    are part of the measurement.
    """
    if not path.exists():
        return np.zeros((0, 4)), np.zeros(0, dtype=int)

    classes, rows = [], []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < LABEL_FIELDS:
            sys.exit(f"{path}:{line_no}: expected '<cls> <cx> <cy> <w> <h>', got {line!r}")
        classes.append(int(float(parts[0])))
        rows.append([float(v) for v in parts[1:LABEL_FIELDS]])
    return np.array(rows).reshape(-1, 4), np.array(classes, dtype=int)


def _resolve_size(record: dict[str, Any],
                  frame_size: tuple[int, int] | None) -> tuple[str, int, int]:
    """Label-file stem and frame dimensions for one prediction record.

    Records keyed by `image` can resolve their size from the image itself;
    records keyed by `frame` (video runs) cannot, because the JSONL carries no
    frame dimensions — those need --frame-size.
    """
    if "image" not in record:
        if not frame_size:
            sys.exit("Video-keyed predictions need --frame-size W H.")
        return str(record["frame"]), *frame_size

    image_path = Path(record["image"])
    if frame_size:
        return image_path.stem, *frame_size
    if not image_path.exists():
        sys.exit(f"Cannot determine size for {image_path} (missing). "
                 f"Pass --frame-size W H.")
    with Image.open(image_path) as img:
        width, height = img.size
    return image_path.stem, width, height


def load_frames(pred_path: Path, labels_dir: Path,
                frame_size: tuple[int, int] | None,
                key_filter: Callable[[str], bool] | None = None) -> list[EvalFrame]:
    """Pair each prediction record in a run's JSONL with its label file.

    `key_filter` keeps only the frames whose key it accepts, and is applied
    *before* the label file is read — a caller after one video out of fifteen
    (`src.render_video`) would otherwise pay 28,000 file reads to discard 26,000
    of the results.
    """
    frames = []
    for line in pred_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        stem, width, height = _resolve_size(record, frame_size)
        if key_filter is not None and not key_filter(stem):
            continue
        rows, gt_classes = load_label_file(labels_dir / f"{stem}.txt")
        frames.append(EvalFrame(
            key=stem,
            gt_boxes=yolo_to_xyxy(rows, width, height),
            gt_classes=gt_classes,
            preds=Detections.from_records(record.get("detections", [])),
            extras={k: v for k, v in record.items() if k not in RESERVED_KEYS},
        ))
    return frames
