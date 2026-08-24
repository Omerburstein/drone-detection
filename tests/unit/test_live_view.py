"""The live window's arithmetic and its caveat line.

Nothing here opens a window -- `render` returns a canvas and is pure, which is
why the drawing was separated from the showing. What is asserted is the part
that can mislead: the duty cycle, and the fact that a duty-cycled detection is
always labelled with the policy that produced it. A box drawn without its policy
invites the viewer to read it as a full-rate result.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.algo.deployment import choose_policy
from src.algo.detections import Detections
from src.output import live_view
from src.output.live_view import HUD_HEIGHT, LiveStatus

SOURCE_FPS = 30.0


def a_frame(width: int = 640, height: int = 360) -> np.ndarray:
    """A mid-grey frame, so drawn elements are distinguishable from it."""
    return np.full((height, width, 3), 128, dtype=np.uint8)


def a_detection() -> Detections:
    """One small box, roughly the size this project actually deals with."""
    return Detections(boxes=np.array([[300.0, 180.0, 318.0, 198.0]]),
                      scores=np.array([1.0]), classes=np.zeros(1, dtype=int))


def status(detections: Detections | None = None, processed: int = 10,
           dropped: int = 90, measured: float = 2.79) -> LiveStatus:
    """A LiveStatus with the fields under test set explicitly."""
    return LiveStatus(
        policy=choose_policy(measured, SOURCE_FPS),
        branch="global mod",
        detections=detections if detections is not None else Detections.empty(),
        processed=processed,
        dropped=dropped,
        sustained_fps=measured,
        latency=0.35,
    )


class TestLiveStatus:

    def test_duty_cycle_is_processed_over_everything_delivered(self):
        assert status(processed=10, dropped=90).duty_cycle == pytest.approx(0.10)

    def test_duty_cycle_of_a_feed_that_delivered_nothing_is_zero(self):
        # Guards the first frame, before any counter has moved.
        assert status(processed=0, dropped=0).duty_cycle == 0.0

    def test_locked_follows_the_detections(self):
        assert not status().locked
        assert status(a_detection()).locked


class TestRender:

    def test_canvas_is_the_frame_plus_the_hud_strip(self):
        canvas = live_view.render(a_frame(640, 360), status(), max_width=1280)
        assert canvas.shape[:2] == (360 + HUD_HEIGHT, 640)

    def test_a_wide_frame_is_scaled_down_to_fit(self):
        canvas = live_view.render(a_frame(1920, 1080), status(), max_width=960)
        assert canvas.shape[1] == 960
        assert canvas.shape[0] == 540 + HUD_HEIGHT

    def test_a_narrow_frame_is_not_scaled_up(self):
        canvas = live_view.render(a_frame(320, 240), status(), max_width=1280)
        assert canvas.shape[1] == 320

    def test_the_source_frame_is_not_modified(self):
        """The frame belongs to the capture thread; drawing on it would race."""
        frame = a_frame()
        before = frame.copy()
        live_view.render(frame, status(a_detection()))
        assert np.array_equal(frame, before)

    def test_a_detection_marks_the_canvas(self):
        empty = live_view.render(a_frame(), status())
        drawn = live_view.render(a_frame(), status(a_detection()))
        assert not np.array_equal(empty, drawn)

    def test_render_survives_a_frame_smaller_than_the_inset(self):
        # `fit_zoom` returns 0 rather than raising; this pins that the caller
        # handles it instead of pasting out of bounds.
        canvas = live_view.render(a_frame(80, 60), status(a_detection()))
        assert canvas.shape[:2] == (60 + HUD_HEIGHT, 80)

    def test_a_box_at_the_frame_edge_does_not_raise(self):
        edge = Detections(boxes=np.array([[630.0, 350.0, 645.0, 365.0]]),
                          scores=np.array([1.0]), classes=np.zeros(1, dtype=int))
        assert live_view.render(a_frame(), status(edge)) is not None


class TestDisplayGuard:

    def test_headless_opencv_is_reported_with_the_fix(self):
        """The message has to carry the uninstall command, not just a complaint.

        `opencv-python-headless` shares the `cv2` namespace with `opencv-python`,
        so this failure looks like a code bug and is not one.
        """
        try:
            live_view.require_display()
        except live_view.DisplayUnavailable as exc:
            assert "opencv-python-headless" in str(exc)
            assert "pip uninstall" in str(exc)
        # A build with a GUI raises nothing, and that is equally correct.
