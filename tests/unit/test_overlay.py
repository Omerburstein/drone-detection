"""Unit tests for the ground-truth overlay.

The overlay's job is to be *readable* and to *agree with the scoring*. Colour is
the whole message — green is a target, blue is a hit, red is a false alarm — so
these tests assert on the pixels that come out, not on the calls that went in.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.metrics import CENTER, IOU, MatchCriterion
from src.output import overlay
from src.output.overlay import (Style, View, crop_window, fit_zoom, judge,
                                render_frame)

BLANK = (40, 40, 40)
CENTRED = MatchCriterion(CENTER, 1.0)


@pytest.fixture
def canvas():
    """A 400x600 mid-grey frame, distinguishable from every overlay colour."""
    return np.full((400, 600, 3), BLANK, dtype=np.uint8)


def colours(image: np.ndarray) -> set[tuple[int, int, int]]:
    """Every distinct BGR value present, as plain tuples."""
    return {tuple(int(c) for c in row) for row in np.unique(image.reshape(-1, 3), axis=0)}


def has_colour(image: np.ndarray, colour: tuple[int, int, int]) -> bool:
    """Is this exact colour drawn anywhere? Anti-aliased edges keep the pure
    value at the rectangle's straight runs, so an exact test is safe."""
    return bool((image == np.array(colour, dtype=np.uint8)).all(axis=2).any())


class TestView:

    def test_identity_view_leaves_points_alone(self):
        assert View().point(12.4, 30.6) == (12, 31)

    def test_inset_view_shifts_then_scales_then_pastes(self):
        view = View(origin=(100, 50), scale=4, offset=(200, 10))
        assert view.point(100, 50) == (200, 10)  # crop corner lands at the paste corner
        assert view.point(110, 60) == (240, 50)  # 10 px in the source, 40 on the canvas


class TestJudge:

    def test_marks_the_target_found_and_the_prediction_a_true_positive(self,
                                                                      make_frame):
        frame = make_frame(gt=[[10, 10, 30, 30]],
                           preds=[([11, 11, 31, 31], 0.9, 0)])
        verdict = judge(frame, CENTRED)
        assert verdict.tp.tolist() == [True]
        assert verdict.found.tolist() == [True]
        assert verdict.match_iou[0] == pytest.approx(361 / 439, abs=1e-3)

    def test_distant_prediction_is_a_false_alarm_and_the_target_stays_missed(
            self, make_frame):
        frame = make_frame(gt=[[10, 10, 30, 30]],
                           preds=[([300, 300, 320, 320], 0.9, 0)])
        verdict = judge(frame, CENTRED)
        assert verdict.tp.tolist() == [False]
        assert verdict.found.tolist() == [False]

    def test_summary_counts_every_outcome_in_the_frame(self, make_frame):
        frame = make_frame(gt=[[10, 10, 30, 30], [200, 200, 220, 220]],
                           preds=[([11, 11, 31, 31], 0.9, 0),
                                  ([400, 100, 420, 120], 0.4, 0)])
        assert judge(frame, CENTRED).summary == "1 TP, 1 FP, 1 missed"

    def test_criterion_decides_the_verdict(self, make_frame):
        """A 2 px offset on a 12 px target: found under centre, missed under IoU.

        This is the project's own EXP-004 finding in miniature -- the overlay
        must show whichever rule it was given, not a fixed one.
        """
        frame = make_frame(gt=[[100, 100, 112, 112]],
                           preds=[([104, 104, 116, 116], 0.9, 0)])
        assert judge(frame, CENTRED).tp.tolist() == [True]
        assert judge(frame, MatchCriterion(IOU, 0.5)).tp.tolist() == [False]


class TestCropWindow:

    def test_centres_on_the_target(self, make_frame):
        frame = make_frame(gt=[[290, 190, 310, 210]])
        assert crop_window(frame, (400, 600), 100) == (250, 150, 100)

    def test_clamps_to_the_frame_at_the_edge(self, make_frame):
        frame = make_frame(gt=[[0, 0, 10, 10]])
        assert crop_window(frame, (400, 600), 100) == (0, 0, 100)

    def test_falls_back_to_the_prediction_when_there_is_no_target(self, make_frame):
        """A false alarm on an empty frame is the case worth looking at."""
        frame = make_frame(preds=[([290, 190, 310, 210], 0.5, 0)])
        assert crop_window(frame, (400, 600), 100) == (250, 150, 100)

    def test_uses_the_frame_centre_when_there_is_nothing_at_all(self, make_frame):
        assert crop_window(make_frame(), (400, 600), 100) == (250, 150, 100)

    def test_span_larger_than_the_frame_shrinks_to_fit(self, make_frame):
        frame = make_frame(gt=[[10, 10, 30, 30]])
        assert crop_window(frame, (400, 600), 900) == (0, 0, 400)


class TestFitZoom:
    """The inset has to fit inside the frame it is pasted into."""

    def test_keeps_the_requested_magnification_when_it_fits(self):
        assert fit_zoom((1080, 1920), 110, 5) == 5

    def test_reduces_magnification_rather_than_overflowing(self):
        # 110 px span at 5x is 550, taller than a 400 px frame minus margins.
        assert fit_zoom((400, 600), 110, 5) == 3

    def test_returns_zero_when_even_1x_will_not_fit(self):
        assert fit_zoom((64, 64), 60, 5) == 0


class TestRenderFrame:

    def test_leaves_the_source_frame_untouched(self, canvas, make_frame):
        frame = make_frame(gt=[[100, 100, 120, 120]],
                           preds=[([101, 101, 121, 121], 0.9, 0)])
        render_frame(canvas, frame, CENTRED)
        assert colours(canvas) == {BLANK}

    def test_true_positive_draws_ground_truth_green_and_prediction_blue(
            self, canvas, make_frame):
        frame = make_frame(gt=[[100, 100, 120, 120]],
                           preds=[([101, 101, 121, 121], 0.9, 0)])
        out = render_frame(canvas, frame, CENTRED, Style(zoom=0, caption=False))
        assert has_colour(out, overlay.GT_COLOUR)
        assert has_colour(out, overlay.TP_COLOUR)
        assert not has_colour(out, overlay.FP_COLOUR)
        assert not has_colour(out, overlay.MISS_COLOUR)

    def test_false_alarm_draws_red_and_leaves_the_target_orange(self, canvas,
                                                               make_frame):
        frame = make_frame(gt=[[100, 100, 120, 120]],
                           preds=[([300, 300, 320, 320], 0.9, 0)])
        out = render_frame(canvas, frame, CENTRED, Style(zoom=0, caption=False))
        assert has_colour(out, overlay.FP_COLOUR)
        assert has_colour(out, overlay.MISS_COLOUR)
        assert not has_colour(out, overlay.GT_COLOUR)
        assert not has_colour(out, overlay.TP_COLOUR)

    def test_ground_truth_box_is_drawn_outside_the_prediction(self, canvas,
                                                             make_frame):
        """Both boxes have to stay visible when they coincide, which on a
        correct detection of a 10 px drone is the normal case."""
        box = [100, 100, 110, 110]
        frame = make_frame(gt=[box], preds=[(box, 0.9, 0)])
        out = render_frame(canvas, frame, CENTRED, Style(zoom=0, caption=False))
        gt_cols = np.where((out == np.array(overlay.GT_COLOUR, np.uint8)).all(axis=2))[1]
        pred_cols = np.where((out == np.array(overlay.TP_COLOUR, np.uint8)).all(axis=2))[1]
        assert gt_cols.min() < pred_cols.min()
        assert gt_cols.max() > pred_cols.max()

    def test_inset_magnifies_the_target_region(self, canvas, make_frame):
        """The inset must be a real magnification: a 4 px target that occupies
        4 columns in the frame occupies 4 * zoom in the inset."""
        canvas[100:104, 100:104] = (255, 255, 255)
        frame = make_frame(gt=[[100, 100, 104, 104]])
        style = Style(zoom=5, span=60, caption=False)
        out = render_frame(canvas, frame, CENTRED, style)

        side = style.span * style.zoom
        inset = out[overlay.INSET_MARGIN:overlay.INSET_MARGIN + side,
                    out.shape[1] - side - overlay.INSET_MARGIN:
                    out.shape[1] - overlay.INSET_MARGIN]
        white_cols = np.where((inset == 255).all(axis=2).any(axis=0))[0]
        assert len(white_cols) == 4 * style.zoom

    def test_zoom_zero_draws_no_inset(self, canvas, make_frame):
        frame = make_frame(gt=[[100, 100, 120, 120]])
        out = render_frame(canvas, frame, CENTRED, Style(zoom=0, caption=False))
        corner = out[:200, -200:]
        assert colours(corner) == {BLANK}

    def test_caption_darkens_a_strip_at_the_bottom(self, canvas, make_frame):
        frame = make_frame(gt=[[100, 100, 120, 120]], key="phantom19_0031")
        out = render_frame(canvas, frame, CENTRED, Style(zoom=0, caption=True))
        strip = out[-overlay.CAPTION_HEIGHT:, :]
        # Median, not mean: the strip is mostly darkened background, but the
        # white caption text is bright enough to pull an average back up.
        assert np.median(strip) < np.median(canvas)
        assert colours(out[:-overlay.CAPTION_HEIGHT, :]) != {BLANK}  # boxes survive

    def test_output_keeps_the_frame_size(self, canvas, make_frame):
        out = render_frame(canvas, make_frame(gt=[[10, 10, 30, 30]]), CENTRED)
        assert out.shape == canvas.shape
