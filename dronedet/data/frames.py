"""Frame sources: a uniform iteration over video frames or image files.

Video and stills differ in more than decoding — they stride differently, they
count progress differently, and they are recorded under different JSONL keys.
Each source owns those decisions and yields a fully-described `Frame`, which is
what lets one run loop serve both.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from .sources import IMAGES, VIDEO

PROGRESS_EVERY = 25  # frames between in-flight progress lines
DEFAULT_FPS = 30.0  # fallback when a container reports no frame rate


@dataclass(frozen=True)
class Frame:
    """One frame to process, carrying everything the run loop needs about it."""

    image: np.ndarray
    key: dict[str, Any]  # JSONL identity: {"frame": n} for video, {"image": path} for stills
    name: str  # filename to use if this frame is written out annotated
    progress_label: str | None  # set only on frames where a progress line is due


class FrameSource(ABC):
    """Iterable of frames to run the detector over."""

    #: Frame rate for annotated video output; None for sources that aren't video.
    output_fps: float | None = None

    @abstractmethod
    def __iter__(self) -> Iterator[Frame]:
        ...

    def close(self) -> None:
        """Release any decoder held by the source."""


class VideoFrameSource(FrameSource):
    """Frames decoded from a video file, sampled every `stride` frames.

    `max_frames` caps *processed* frames, not source frames, so it composes with
    `stride` to sample a long clip cheaply.
    """

    def __init__(self, path: Path, stride: int, max_frames: int | None) -> None:
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            sys.exit(f"Could not open video {path}")
        self._stride = stride
        self._max_frames = max_frames
        source_fps = self._cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
        # Annotated output plays at real-world speed despite the striding.
        self.output_fps = source_fps / max(1, stride)

    def __iter__(self) -> Iterator[Frame]:
        idx = -1
        processed = 0
        try:
            while True:
                ok, image = self._cap.read()
                if not ok:
                    break
                idx += 1
                if idx % self._stride:
                    continue
                processed += 1
                due = processed % PROGRESS_EVERY == 0
                yield Frame(
                    image=image,
                    key={"frame": idx},
                    name=f"{idx:08d}.png",
                    progress_label=f"frame {idx}" if due else None,
                )
                if self._max_frames and processed >= self._max_frames:
                    break
        finally:
            self._cap.release()

    def close(self) -> None:
        self._cap.release()


class ImageFrameSource(FrameSource):
    """Frames read from a list of image files.

    Note the deliberate asymmetry with video: `stride` is applied only when
    `max_frames` is also set, because image directories are usually already
    sparse samples rather than dense video. Documented in
    docs/baseline_detect.md.
    """

    def __init__(self, paths: list[Path], stride: int, max_frames: int | None) -> None:
        if max_frames:
            paths = paths[::stride][:max_frames]
        self._paths = paths

    def __iter__(self) -> Iterator[Frame]:
        total = len(self._paths)
        for i, path in enumerate(self._paths):
            image = cv2.imread(str(path))
            if image is None:
                print(f"  ! unreadable: {path}", file=sys.stderr)
                continue
            due = (i + 1) % PROGRESS_EVERY == 0
            yield Frame(
                image=image,
                key={"image": str(path)},
                name=path.name,
                progress_label=f"{i + 1}/{total}" if due else None,
            )


def open_source(kind: str, paths: list[Path], stride: int,
                max_frames: int | None) -> FrameSource:
    """Build the frame source matching a `resolve_sources` kind."""
    if kind == VIDEO:
        return VideoFrameSource(paths[0], stride, max_frames)
    if kind == IMAGES:
        return ImageFrameSource(paths, stride, max_frames)
    raise ValueError(f"Unknown source kind: {kind}")
