"""The detector's per-frame output as a single value."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Detections:
    """Boxes, scores and class ids for one frame.

    These three arrays are meaningless apart and are always the same length, so
    they travel as one value rather than as a tuple threaded through every
    signature. Coordinates are absolute pixels in the *original* frame with a
    top-left origin — tiled runs are mapped back before results get here, which
    is what makes tiled and whole-frame output directly comparable.
    """

    boxes: np.ndarray  # (N, 4) in xyxy order
    scores: np.ndarray  # (N,) confidence, 0-1
    classes: np.ndarray  # (N,) integer class ids

    @classmethod
    def empty(cls) -> Detections:
        """A found-nothing result with correctly shaped empty arrays."""
        return cls(np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int))

    def __len__(self) -> int:
        return len(self.boxes)

    def __iter__(self) -> Iterator[tuple[np.ndarray, float, int]]:
        """Iterate as (box, score, class) triples."""
        return zip(self.boxes, self.scores, self.classes)
