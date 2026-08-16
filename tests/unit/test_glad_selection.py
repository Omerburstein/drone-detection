"""Tests for the three detector operating points in the GLAD port.

The three upstream detectors share one network and differ only in a confidence
threshold and in which survivor they keep. Those rules are the difference
between GAD's 0.17 recall and the full pipeline's 0.81, so each constant is
pinned here — including the boundary behaviour, which is strict `>` on
confidence and strict `<` on distance throughout.

`_select` takes no backend, so the detectors are constructed with `None` for it:
these are rule tests, deliberately not model tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.algo.glad.yolo import (AcquireDetector, Candidate, GlobalDetector,
                                TrackingDetector)

NO_BACKEND = None


def candidate(x: int, y: int, size: int = 20, conf: float = 0.9) -> Candidate:
    """A square candidate with its top-left at (x, y)."""
    return Candidate(x, y, x + size, y + size, conf)


class TestGlobalDetector:
    """GAD: full frame, conf 0.5, keep the most confident box."""

    def test_threshold_is_the_published_one(self):
        assert GlobalDetector.CONF_THRESH == 0.5

    def test_keeps_the_most_confident(self):
        detector = GlobalDetector(NO_BACKEND)
        box = detector._select([candidate(10, 10, conf=0.6),
                                candidate(90, 90, conf=0.95),
                                candidate(50, 50, conf=0.7)])
        assert box == pytest.approx([90, 90, 20, 20])

    def test_nothing_found_is_none(self):
        assert GlobalDetector(NO_BACKEND)._select([]) is None

    def test_confidence_at_the_threshold_is_rejected(self):
        """Upstream seeds `ref_score` with the threshold itself, so `>` not `>=`."""
        assert GlobalDetector(NO_BACKEND)._select([candidate(10, 10, conf=0.5)]) is None

    def test_returns_top_left_and_size(self):
        box = GlobalDetector(NO_BACKEND)._select([Candidate(30, 40, 55, 70, 0.9)])
        assert box == pytest.approx([30, 40, 25, 30])


class TestTrackingDetector:
    """LAD locked on: conf 0.1, within 50 px of the last box, both refs tighten."""

    def test_constants_are_the_published_ones(self):
        assert TrackingDetector.CONF_THRESH == 0.1
        assert TrackingDetector.MAX_DISTANCE == 50

    def test_low_confidence_is_accepted_inside_the_region(self):
        """The whole point of the local regime: a weak response is still evidence."""
        box = TrackingDetector(NO_BACKEND)._select([candidate(100, 100, conf=0.15)],
                                                   110, 110)
        assert box == pytest.approx([100, 100, 20, 20])

    def test_rejects_a_box_beyond_the_radius(self):
        far = candidate(300, 300, conf=0.99)
        assert TrackingDetector(NO_BACKEND)._select([far], 10, 10) is None

    def test_rejects_a_box_below_the_threshold(self):
        weak = candidate(100, 100, conf=0.05)
        assert TrackingDetector(NO_BACKEND)._select([weak], 110, 110) is None

    def test_both_references_tighten_together(self):
        """A later box must beat the incumbent on confidence *and* distance.

        This is what makes the rule order-dependent, and it is upstream's.
        `nearer` is closer to the anchor but less confident, so it loses.
        """
        confident = candidate(100, 100, conf=0.8)   # centre (110, 110), dist 0
        nearer = candidate(101, 101, conf=0.4)      # centre (111, 111), dist ~1.4
        assert (TrackingDetector(NO_BACKEND)._select([confident, nearer], 110, 110)
                == pytest.approx([100, 100, 20, 20]))

    def test_a_stronger_and_closer_box_wins(self):
        first = candidate(120, 120, conf=0.3)   # centre (130, 130), dist ~28
        better = candidate(101, 101, conf=0.6)  # centre (111, 111), dist ~1.4
        assert (TrackingDetector(NO_BACKEND)._select([first, better], 110, 110)
                == pytest.approx([101, 101, 20, 20]))

    def test_distance_exactly_at_the_radius_is_rejected(self):
        at_radius = candidate(150, 100)  # centre (160, 110), exactly 50 px away
        assert TrackingDetector(NO_BACKEND)._select([at_radius], 110, 110) is None


class TestAcquireDetector:
    """LAD confirming a motion candidate: conf 0.5, nearest within 10 px."""

    def test_constants_are_the_published_ones(self):
        assert AcquireDetector.CONF_THRESH == 0.5
        assert AcquireDetector.MAX_DISTANCE == 10

    def test_takes_the_nearest_ignoring_confidence(self):
        """Motion already proposed the location; this only has to agree with it."""
        near_weak = candidate(103, 103, conf=0.51)   # centre (113, 113), dist ~4.2
        far_strong = candidate(108, 108, conf=0.99)  # centre (118, 118), dist ~11.3
        box = AcquireDetector(NO_BACKEND)._select([far_strong, near_weak], 110, 110)
        assert box == pytest.approx([103, 103, 20, 20])

    def test_rejects_everything_outside_ten_pixels(self):
        """The tight radius is what keeps the motion branch's precision up."""
        outside = candidate(120, 120, conf=0.99)  # centre (130, 130), dist ~28
        assert AcquireDetector(NO_BACKEND)._select([outside], 110, 110) is None

    def test_nothing_found_is_none(self):
        assert AcquireDetector(NO_BACKEND)._select([], 110, 110) is None


class _StubBackend:
    """Returns canned decoded predictions, so `candidates` runs without a model."""

    def __init__(self, rows: np.ndarray, height: int, width: int) -> None:
        self._rows, self._height, self._width = rows, height, width

    def raw_predictions(self, image: np.ndarray) -> tuple[np.ndarray, int, int]:  # noqa: ARG002 -- mirrors Yolov5Backend
        return self._rows, self._height, self._width


class TestCandidates:
    """Threshold, NMS and the coordinate map, wired together."""

    def _detector(self, rows: list[list[float]]) -> GlobalDetector:
        # 320x320 keeps the letterbox scale at exactly 2 with no padding, so the
        # expected coordinates stay readable.
        return GlobalDetector(_StubBackend(np.array(rows), 320, 320))

    def test_drops_boxes_below_the_threshold(self):
        detector = self._detector([[220.0, 220.0, 40.0, 40.0, 0.4, 0.0]])
        assert detector.candidates(None) == []

    def test_maps_a_surviving_box_back_to_image_pixels(self):
        detector = self._detector([[220.0, 220.0, 40.0, 40.0, 0.9, 0.0]])
        found = detector.candidates(None)
        assert len(found) == 1
        assert found[0].as_xywh() == pytest.approx([100, 100, 20, 20])
        assert found[0].conf == pytest.approx(0.9)

    def test_nms_collapses_overlapping_duplicates(self):
        detector = self._detector([[220.0, 220.0, 40.0, 40.0, 0.9, 0.0],
                                   [222.0, 222.0, 40.0, 40.0, 0.8, 0.0]])
        assert len(detector.candidates(None)) == 1

    def test_nms_keeps_boxes_that_do_not_overlap(self):
        detector = self._detector([[100.0, 100.0, 40.0, 40.0, 0.9, 0.0],
                                   [500.0, 500.0, 40.0, 40.0, 0.8, 0.0]])
        assert len(detector.candidates(None)) == 2
