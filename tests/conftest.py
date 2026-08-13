"""Shared fixtures for the test suite.

Everything here is synthetic: no dataset reads, no weight downloads, no network.
The suite must pass offline on a CPU-only machine in seconds.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.algo.detections import Detections
from src.eval.labels import EvalFrame


def _make_frame(gt: list[list[float]] | None = None,
                preds: list[tuple[list[float], float, int]] | None = None,
                gt_classes: list[int] | None = None,
                key: str = "f") -> EvalFrame:
    """Build an EvalFrame from plain lists.

    `preds` entries are (xyxy, confidence, class id). Ground-truth boxes default
    to class 0, the single `Drone` class this project uses.
    """
    gt_boxes = np.array(gt, dtype=float).reshape(-1, 4) if gt else np.zeros((0, 4))
    classes = (np.array(gt_classes, dtype=int) if gt_classes is not None
               else np.zeros(len(gt_boxes), dtype=int))

    if preds:
        boxes, scores, pred_classes = zip(*[(b, s, c) for b, s, c in preds])
        detections = Detections(
            boxes=np.array(boxes, dtype=float).reshape(-1, 4),
            scores=np.array(scores, dtype=float),
            classes=np.array(pred_classes, dtype=int),
        )
    else:
        detections = Detections.empty()

    return EvalFrame(key=key, gt_boxes=gt_boxes, gt_classes=classes, preds=detections)


@pytest.fixture
def make_frame():
    """Factory for EvalFrames.

    Exposed as a fixture rather than imported, because an unrelated `tests`
    package on sys.path shadows this one and breaks `from tests.conftest import`.
    """
    return _make_frame


@pytest.fixture
def unit_box() -> np.ndarray:
    """A single 100x100 box at the origin, as (1, 4)."""
    return np.array([[0.0, 0.0, 100.0, 100.0]])
