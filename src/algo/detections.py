"""The detector's per-frame output as a single value."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

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

    # The two methods below define the `detections.jsonl` row schema. The
    # recorder writes it and the evaluator reads it back, so it lives in one
    # place here rather than being spelled out at both ends where the two
    # copies could drift.

    @classmethod
    def from_records(cls, rows: list[dict[str, Any]]) -> Detections:
        """Rebuild detections from `detections.jsonl` rows."""
        return cls(
            boxes=np.array([r["bbox"] for r in rows], dtype=float).reshape(-1, 4),
            scores=np.array([r["conf"] for r in rows], dtype=float),
            classes=np.array([r["cls"] for r in rows], dtype=int),
        )

    def to_records(self, bbox_decimals: int,
                   conf_decimals: int) -> list[dict[str, Any]]:
        """Serialise to `detections.jsonl` rows, rounded to the given precision."""
        return [
            {"bbox": [round(float(v), bbox_decimals) for v in box],
             "conf": round(float(score), conf_decimals),
             "cls": int(cls)}
            for box, score, cls in self
        ]
