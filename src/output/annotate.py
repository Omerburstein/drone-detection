"""Annotated media output — the eyeball check on what the detector did.

Sinks own the drawing as well as the writing, so a run with annotation disabled
skips the draw work entirely rather than rendering frames nobody will see.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import cv2
import numpy as np

from ..algo.detections import Detections
from ..data.frames import DEFAULT_FPS
from ..data.sources import IMAGES, VIDEO

BOX_COLOUR = (0, 255, 0)  # BGR
BOX_THICKNESS = 2
LABEL_SCALE = 0.5
LABEL_MIN_Y = 12  # keeps the text on-frame for boxes near the top edge
LABEL_Y_OFFSET = 6  # px above the box top edge


def draw(frame: np.ndarray, dets: Detections, names) -> np.ndarray:
    """Copy of `frame` with labelled detection boxes burned in."""
    out = frame.copy()
    for (x1, y1, x2, y2), score, cls in dets:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, BOX_COLOUR, BOX_THICKNESS)
        label = f"{names.get(int(cls), int(cls))} {score:.2f}"
        cv2.putText(out, label, (p1[0], max(LABEL_MIN_Y, p1[1] - LABEL_Y_OFFSET)),
                    cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, BOX_COLOUR, 1, cv2.LINE_AA)
    return out


class AnnotationSink(ABC):
    """Destination for annotated frames."""

    @abstractmethod
    def write(self, frame: np.ndarray, name: str, dets: Detections, names) -> None:
        ...

    def close(self) -> None:
        """Finalise the output. Safe to call when nothing was written."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


class NullSink(AnnotationSink):
    """Drops everything — for --no-save-frames, where only the JSONL matters."""

    def write(self, frame: np.ndarray, name: str, dets: Detections, names) -> None:
        pass


class VideoSink(AnnotationSink):
    """Writes one annotated video.

    The writer is opened on the first frame because its dimensions come from the
    footage, not from the arguments.
    """

    def __init__(self, path: Path, fps: float) -> None:
        self._path = path
        self._fps = fps
        self._writer: cv2.VideoWriter | None = None

    def write(self, frame: np.ndarray, _name: str, dets: Detections, names) -> None:
        if self._writer is None:
            h, w = frame.shape[:2]
            self._writer = cv2.VideoWriter(
                str(self._path), cv2.VideoWriter_fourcc(*"mp4v"), self._fps, (w, h))
        self._writer.write(draw(frame, dets, names))

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


class ImageDirSink(AnnotationSink):
    """Writes one annotated image per input, under a directory."""

    def __init__(self, directory: Path) -> None:
        self._dir = directory
        self._dir.mkdir(exist_ok=True)

    def write(self, frame: np.ndarray, name: str, dets: Detections, names) -> None:
        cv2.imwrite(str(self._dir / name), draw(frame, dets, names))


def open_sink(kind: str, out_dir: Path, save_frames: bool,
              fps: float | None) -> AnnotationSink:
    """Build the sink matching a `resolve_sources` kind, or a null sink."""
    if not save_frames:
        return NullSink()
    if kind == VIDEO:
        return VideoSink(out_dir / "annotated.mp4", fps or DEFAULT_FPS)
    if kind == IMAGES:
        return ImageDirSink(out_dir / "annotated")
    raise ValueError(f"Unknown source kind: {kind}")
