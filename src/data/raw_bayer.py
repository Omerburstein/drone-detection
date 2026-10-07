"""Reading our camera's headerless `.raw` capture: 8-bit Bayer frames, back to back.

The FIELD recorder writes two files per recording: an H.264 `.mp4` and a `.raw` holding
the sensor's own pixels. Neither carries a header, so the layout below was established
by measurement on `captured_raw_20260616_040253_004` (2026-10-07):

* **4128x3008, 8 bits, one byte per pixel, BG Bayer** -- the byte autocorrelation repeats
  every 8256 bytes (two sensor rows), and the file is exactly 3600 x 4128 x 3008 bytes;
* **mounted upside down** -- demosaiced and rotated 180 degrees, it matches the `.mp4`;
* **the `.mp4` is this, shrunk 4x** -- after `INTER_AREA` to 1032x752 every frame checked
  (0, 1350, 3599) correlates 0.996-0.9995 with the same-index `.mp4` frame and 0.85-0.95
  with its neighbours, so the two files are frame-aligned with no offset.

`RawBayerCapture` is a drop-in for the slice of `cv2.VideoCapture` the experiment
scripts use (`set(CAP_PROP_POS_FRAMES)`, `read`, `get`, `release`), so a run at any
resolution reads the raw directly instead of a re-encoded copy -- full-resolution BGR
would be 134 GB for this one file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

SENSOR_W, SENSOR_H = 4128, 3008
BAYER = cv2.COLOR_BayerBG2BGR


@dataclass(frozen=True)
class RawLayout:
    """Frame geometry of a headerless 8-bit Bayer file."""

    width: int = SENSOR_W
    height: int = SENSOR_H
    bayer: int = BAYER
    rotate_180: bool = True

    @property
    def frame_bytes(self) -> int:
        return self.width * self.height

    def n_frames(self, path: str) -> int:
        """Whole frames in `path`; a trailing partial frame is not counted."""
        return os.path.getsize(path) // self.frame_bytes


def decode(mosaic: np.ndarray, layout: RawLayout = RawLayout(),
           size: tuple[int, int] | None = None) -> np.ndarray:
    """One Bayer mosaic to an upright BGR image, shrunk to `size` (w, h) when given.

    Shrinking uses `INTER_AREA`, the filter that reproduces the recorder's own `.mp4`."""
    img = cv2.cvtColor(mosaic, layout.bayer)
    if layout.rotate_180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    if size is not None and size != (img.shape[1], img.shape[0]):
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


class RawBayerCapture:
    """The `cv2.VideoCapture` subset the experiment scripts call, over a `.raw` file.

    `size` is the (w, h) frames come out at; default is the sensor's own resolution."""

    def __init__(self, path: str, size: tuple[int, int] | None = None,
                 layout: RawLayout = RawLayout(), fps: float = 30.0) -> None:
        self.layout = layout
        self.size = size or (layout.width, layout.height)
        self.fps = fps
        self.count = layout.n_frames(path)
        self.pos = 0
        self._fh = open(path, "rb")

    def isOpened(self) -> bool:  # noqa: N802 -- cv2's spelling
        return not self._fh.closed

    def set(self, prop: int, value: float) -> bool:
        if prop != cv2.CAP_PROP_POS_FRAMES:
            return False
        self.pos = max(0, int(value))
        return True

    def get(self, prop: int) -> float:
        return {cv2.CAP_PROP_POS_FRAMES: self.pos, cv2.CAP_PROP_FRAME_COUNT: self.count,
                cv2.CAP_PROP_FPS: self.fps, cv2.CAP_PROP_FRAME_WIDTH: self.size[0],
                cv2.CAP_PROP_FRAME_HEIGHT: self.size[1]}.get(prop, 0.0)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._fh.closed or self.pos >= self.count:
            return False, None
        n = self.layout.frame_bytes
        self._fh.seek(self.pos * n)
        buf = self._fh.read(n)
        if len(buf) < n:
            return False, None
        self.pos += 1
        mosaic = np.frombuffer(buf, np.uint8).reshape(self.layout.height, self.layout.width)
        return True, decode(mosaic, self.layout, self.size)

    def release(self) -> None:
        self._fh.close()


def open_capture(path: str, size: tuple[int, int] | None = None):
    """A `RawBayerCapture` for a `.raw` path, `cv2.VideoCapture` for anything else."""
    if str(path).lower().endswith(".raw"):
        return RawBayerCapture(str(path), size)
    return cv2.VideoCapture(path)
