"""Writing an mp4 whose dimensions are only known once a frame arrives.

Both video sinks in this package have the same problem: the writer needs a frame
size, and the frame size comes from the footage rather than from the arguments.
Opening lazily on the first frame is the whole trick, and it lives here so the
annotated-run sink and the ground-truth overlay share one implementation of it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Self

import cv2
import numpy as np

FOURCC = "mp4v"  # available in every opencv-python wheel, unlike H.264


def open_video(path: Path) -> tuple[cv2.VideoCapture, int, int, float]:
    """Open a video and read back the geometry the labels must be scaled by.

    Here rather than in a CLI because every consumer of a recorded run needs
    the same three numbers to make sense of it: `src.render_video` to scale
    ground truth onto the frame, `src.crops` to size the boxes it crops.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        sys.exit(f"Could not open {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    return capture, width, height, fps


class LazyVideoWriter:
    """An mp4 writer that opens on the first frame written to it.

    Closing is idempotent and safe when nothing was ever written, so a run that
    produced no frames leaves no half-written file behind.
    """

    def __init__(self, path: Path, fps: float) -> None:
        self._path = path
        self._fps = fps
        self._writer: cv2.VideoWriter | None = None
        self.frames = 0

    def write(self, frame: np.ndarray) -> None:
        """Append one frame, opening the file if this is the first."""
        if self._writer is None:
            height, width = frame.shape[:2]
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = cv2.VideoWriter(
                str(self._path), cv2.VideoWriter_fourcc(*FOURCC), self._fps,
                (width, height))
            if not self._writer.isOpened():
                raise RuntimeError(f"Could not open {self._path} for writing")
        self._writer.write(frame)
        self.frames += 1

    def close(self) -> None:
        """Finalise the file. Safe to call twice, or when nothing was written."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
