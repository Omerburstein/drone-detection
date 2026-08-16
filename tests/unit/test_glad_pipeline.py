"""Tests for GLAD's global/local state machine.

The state machine is where the paper's recall comes from: 0.17 for appearance
alone, 0.51 once the local regime exists, 0.81 once motion feeds it. Which
branch runs when, and where its box lands, is therefore the whole result.

Every collaborator is stubbed, so nothing here needs weights, the vendored
clone, or a frame that contains anything. What is asserted is the wiring: which
detector is consulted, what anchor it is given, and how a region-relative box
becomes an absolute one.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.algo.glad.pipeline import (FIRST_FRAME, GLOBAL_MISS, GLOBAL_MOD, GLOBAL_YOLO,
                                    LOCAL_MISS, LOCAL_MOD, LOCAL_YOLO,
                                    MAX_LOCAL_MISSES, GladPipeline, StepResult)

FRAME_W, FRAME_H = 1920, 1080
MISS = None


def frame(marker: int = 0) -> np.ndarray:
    """A distinguishable full-size frame."""
    image = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    image[0, 0, 0] = marker
    return image


class StubDetector:
    """Returns scripted results and records the anchor it was called with."""

    def __init__(self, *results) -> None:
        self._results = list(results)
        self.calls: list[tuple[float, ...]] = []

    def detect(self, image: np.ndarray, *anchor: float):  # noqa: ARG002 -- mirrors GladDetector
        self.calls.append(anchor)
        result = self._results.pop(0) if self._results else MISS
        return None if result is None else np.array(result)


class StubMotion:
    """Scripted `MOD2_global` / `MOD2_local`, returning `[]` for a miss as upstream does."""

    def __init__(self, global_results=(), local_results=()) -> None:
        self._global = list(global_results)
        self._local = list(local_results)
        self.global_calls = 0
        self.local_calls = 0

    def MOD2_global(self, previous, current):  # noqa: ARG002, N802 -- mirrors MOD2
        self.global_calls += 1
        return self._global.pop(0) if self._global else []

    def MOD2_local(self, previous, current, x_prev, y_prev):  # noqa: ARG002, N802
        self.local_calls += 1
        return self._local.pop(0) if self._local else []


def build(global_results=(), tracking_results=(), acquire_results=(),
          motion_global=(), motion_local=()):
    """A pipeline over stubs, plus the stubs themselves for assertions."""
    stubs = {
        "global_detector": StubDetector(*global_results),
        "tracking_detector": StubDetector(*tracking_results),
        "acquire_detector": StubDetector(*acquire_results),
        "motion": StubMotion(motion_global, motion_local),
    }
    return GladPipeline(**stubs), stubs


class TestFirstFrame:

    def test_is_always_empty(self):
        """No previous frame means nothing to difference against."""
        pipeline, _ = build(global_results=[[500, 400, 20, 20]])
        result = pipeline.step(frame())
        assert result.box is None
        assert result.branch == FIRST_FRAME

    def test_consults_no_detector(self):
        pipeline, stubs = build(global_results=[[500, 400, 20, 20]])
        pipeline.step(frame())
        assert stubs["global_detector"].calls == []
        assert stubs["motion"].global_calls == 0


class TestGlobalRegime:

    def test_appearance_hit_reports_the_box_and_locks(self):
        pipeline, stubs = build(global_results=[[500, 400, 20, 20]],
                                tracking_results=[[160, 160, 20, 20]])
        pipeline.step(frame(1))
        result = pipeline.step(frame(2))
        assert result.branch == GLOBAL_YOLO
        assert result.box == pytest.approx([500, 400, 20, 20])

        pipeline.step(frame(3))
        # Locked: the next frame goes to the local detector, not the global one.
        assert len(stubs["global_detector"].calls) == 1
        assert len(stubs["tracking_detector"].calls) == 1

    def test_both_branches_missing_stays_unlocked(self):
        pipeline, stubs = build()
        pipeline.step(frame(1))
        assert pipeline.step(frame(2)).branch == GLOBAL_MISS
        assert pipeline.step(frame(3)).branch == GLOBAL_MISS
        assert len(stubs["global_detector"].calls) == 2
        assert stubs["tracking_detector"].calls == []

    def test_motion_alone_does_not_count_as_a_detection(self):
        """A motion candidate LAD refuses to confirm is discarded, not reported.

        This is the gate that keeps GMD's false positives out of the output.
        """
        pipeline, stubs = build(motion_global=[(500, 400, 20, 20)],
                                acquire_results=[MISS])
        pipeline.step(frame(1))
        result = pipeline.step(frame(2))
        assert result.branch == GLOBAL_MISS
        assert result.box is None
        assert stubs["acquire_detector"].calls  # it was consulted

    def test_confirmed_motion_candidate_is_reported_in_frame_coordinates(self):
        """The confirming detector works in region coordinates; output is absolute."""
        pipeline, _ = build(motion_global=[(500, 400, 20, 20)],
                            acquire_results=[[170, 150, 22, 18]])
        pipeline.step(frame(1))
        result = pipeline.step(frame(2))
        assert result.branch == GLOBAL_MOD
        # region_x = 500 - 160 = 340, region_y = 400 - 160 = 240
        assert result.box == pytest.approx([170 + 340, 150 + 240, 22, 18])

    def test_confirming_detector_is_anchored_on_the_motion_candidate(self):
        pipeline, stubs = build(motion_global=[(500, 400, 20, 20)],
                                acquire_results=[MISS])
        pipeline.step(frame(1))
        pipeline.step(frame(2))
        # The candidate's centre, expressed inside the region: (160 + 10, 160 + 10).
        assert stubs["acquire_detector"].calls[0] == pytest.approx((170.0, 170.0))

    def test_appearance_is_tried_before_motion(self):
        """Motion is the expensive path; it only runs when appearance fails."""
        pipeline, stubs = build(global_results=[[500, 400, 20, 20]])
        pipeline.step(frame(1))
        pipeline.step(frame(2))
        assert stubs["motion"].global_calls == 0


class TestLocalRegime:

    @pytest.fixture
    def locked(self):
        """A pipeline locked onto a box at (500, 400), region origin (340, 240)."""
        def _locked(**kwargs):
            pipeline, stubs = build(global_results=[[500, 400, 20, 20]], **kwargs)
            pipeline.step(frame(1))
            pipeline.step(frame(2))
            return pipeline, stubs
        return _locked

    def test_tracking_hit_is_mapped_out_of_the_region(self, locked):
        pipeline, _ = locked(tracking_results=[[170, 150, 22, 18]])
        result = pipeline.step(frame(3))
        assert result.branch == LOCAL_YOLO
        assert result.box == pytest.approx([170 + 340, 150 + 240, 22, 18])

    def test_tracking_is_anchored_on_the_last_box_centre(self, locked):
        pipeline, stubs = locked(tracking_results=[MISS])
        pipeline.step(frame(3))
        # The lock put the box at (160, 160) inside the region; centre is +10.
        assert stubs["tracking_detector"].calls[0] == pytest.approx((170.0, 170.0))

    def test_motion_covers_a_tracking_miss(self, locked):
        pipeline, stubs = locked(tracking_results=[MISS],
                                 motion_local=[(165, 155, 20, 20)])
        result = pipeline.step(frame(3))
        assert result.branch == LOCAL_MOD
        assert result.box == pytest.approx([165 + 340, 155 + 240, 20, 20])
        assert stubs["motion"].local_calls == 1

    def test_motion_is_skipped_when_tracking_succeeds(self, locked):
        pipeline, stubs = locked(tracking_results=[[170, 150, 22, 18]])
        pipeline.step(frame(3))
        assert stubs["motion"].local_calls == 0

    def test_both_missing_reports_a_local_miss(self, locked):
        pipeline, _ = locked()
        assert pipeline.step(frame(3)).branch == LOCAL_MISS

    def test_anchor_lags_one_frame_behind_the_region(self, locked):
        """Pins an upstream quirk: the stored position is not re-based.

        After a local hit the region recentres on the new box, but the relative
        position kept for the next anchor still refers to the *old* region. The
        error equals one frame of target motion, which is small against the
        50 px radius — but it is real, and it is what the released code does.
        """
        pipeline, stubs = locked(tracking_results=[[170, 150, 22, 18], MISS])
        pipeline.step(frame(3))
        pipeline.step(frame(4))
        # Re-based, the box would sit at (160, 160) in the new region and the
        # anchor would be (171, 169). Upstream keeps (170, 150) + half the size.
        assert stubs["tracking_detector"].calls[1] == pytest.approx((181.0, 159.0))


class TestFallbackToGlobal:
    """30 consecutive local misses give up and re-acquire from the full frame."""

    def _locked(self, tracking_results):
        pipeline, stubs = build(global_results=[[500, 400, 20, 20]],
                                tracking_results=tracking_results)
        pipeline.step(frame(1))
        pipeline.step(frame(2))
        return pipeline, stubs

    def test_stays_local_below_the_miss_limit(self):
        pipeline, stubs = self._locked([MISS] * (MAX_LOCAL_MISSES - 1))
        for _ in range(MAX_LOCAL_MISSES - 1):
            pipeline.step(frame())
        assert len(stubs["tracking_detector"].calls) == MAX_LOCAL_MISSES - 1
        assert len(stubs["global_detector"].calls) == 1

    def test_falls_back_after_the_miss_limit(self):
        pipeline, stubs = self._locked([MISS] * MAX_LOCAL_MISSES)
        for _ in range(MAX_LOCAL_MISSES):
            pipeline.step(frame())
        pipeline.step(frame())
        assert len(stubs["global_detector"].calls) == 2

    def test_a_hit_resets_the_miss_count(self):
        """29 misses, a hit, then 29 more must not add up to a fallback."""
        near_miss = [MISS] * (MAX_LOCAL_MISSES - 1)
        pipeline, stubs = self._locked(
            near_miss + [[160, 160, 20, 20]] + near_miss)
        for _ in range(2 * MAX_LOCAL_MISSES - 1):
            pipeline.step(frame())
        assert len(stubs["global_detector"].calls) == 1


class TestReset:

    def test_clears_the_lock(self):
        pipeline, stubs = build(global_results=[[500, 400, 20, 20]])
        pipeline.step(frame(1))
        pipeline.step(frame(2))
        pipeline.reset()
        pipeline.step(frame(3))  # first frame of the "new video": no detection
        assert pipeline.step(frame(4)).branch == GLOBAL_MISS
        assert stubs["tracking_detector"].calls == []

    def test_clears_the_previous_frame(self):
        """Otherwise the first frame of a video is differenced against the last
        frame of the previous one — a scene cut, and pure motion noise."""
        pipeline, _ = build()
        pipeline.step(frame(1))
        pipeline.reset()
        assert pipeline.step(frame(2)).branch == FIRST_FRAME


class TestAsDetections:

    def test_converts_xywh_to_xyxy(self):
        dets = StepResult(np.array([500, 400, 20, 30]), GLOBAL_YOLO).as_detections(1.0)
        assert len(dets) == 1
        assert dets.boxes[0] == pytest.approx([500, 400, 520, 430])

    def test_class_is_the_single_drone_class(self):
        dets = StepResult(np.array([500, 400, 20, 30]), GLOBAL_YOLO).as_detections(1.0)
        assert dets.classes.tolist() == [0]

    def test_confidence_is_the_constant_it_was_given(self):
        """GLAD emits no score, so the run has one operating point, not a ranking."""
        dets = StepResult(np.array([1, 2, 3, 4]), LOCAL_YOLO).as_detections(1.0)
        assert dets.scores.tolist() == [1.0]

    def test_a_miss_is_an_empty_detection_set(self):
        assert len(StepResult(None, GLOBAL_MISS).as_detections(1.0)) == 0
