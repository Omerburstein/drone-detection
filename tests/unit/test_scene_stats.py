"""Tests for the measured condition axes in `src.data.scene_stats`.

These numbers decide which bucket a frame is scored in, so a sign error or an
off-by-one crop would silently re-label the split and move every per-bucket
metric without failing anything. Everything here is synthetic -- a drawn square
on a flat field, whose true contrast is known exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.scene_stats import (BACKLIT, NO_TARGET, FrameStats, build_axes,
                                  lighting_label, measure_frame, range_label,
                                  reference_size, summarise)

SIDE = 200  # synthetic frame side, large enough for a 3x ring around the box


def field_with_square(background: int, target: int, box=(90, 90, 110, 110)):
    """A flat grey frame with one square of a different grey drawn on it."""
    image = np.full((SIDE, SIDE), background, dtype=np.uint8)
    x0, y0, x1, y1 = box
    image[y0:y1, x0:x1] = target
    return image, np.array([[x0, y0, x1, y1]], dtype=float)


def stats_with(**overrides) -> FrameStats:
    """A FrameStats with plausible defaults, for the labelling rules."""
    base = dict(key="v_0001", n_targets=1, contrast=20.0, size=20.0,
                brightness=128.0, clipped_hi=0.0, clipped_lo=0.0)
    return FrameStats(**{**base, **overrides})


class TestMeasureFrame:

    @pytest.mark.parametrize("background, target, expected", [
        (100, 160, 60.0),
        (160, 100, 60.0),   # sign does not matter -- separation is what counts
        (120, 120, 0.0),    # a target indistinguishable from its background
    ])
    def test_contrast_is_the_luminance_gap(self, background, target, expected):
        image, boxes = field_with_square(background, target)
        assert measure_frame(image, boxes, "v_0001").contrast == pytest.approx(expected)

    def test_background_is_the_ring_not_the_whole_frame(self):
        """A bright corner far from the target must not count as its background."""
        image, boxes = field_with_square(100, 160)
        image[0:40, 0:40] = 255
        assert measure_frame(image, boxes, "v_0001").contrast == pytest.approx(60.0)

    def test_size_is_the_equal_area_square_side(self):
        image, boxes = field_with_square(100, 160, box=(90, 95, 110, 105))
        # 20 x 10 -> sqrt(200)
        assert measure_frame(image, boxes, "v_0001").size == pytest.approx(np.sqrt(200))

    def test_accepts_a_colour_frame(self):
        gray, boxes = field_with_square(100, 160)
        colour = np.stack([gray] * 3, axis=2)
        assert (measure_frame(colour, boxes, "v_0001").contrast
                == pytest.approx(measure_frame(gray, boxes, "v_0001").contrast))

    def test_worst_target_and_nearest_target_govern_the_frame(self):
        """A frame is only as easy as its hardest target, as near as its nearest."""
        image = np.full((SIDE, SIDE), 100, dtype=np.uint8)
        image[20:40, 20:40] = 160    # 20 px, strong contrast
        image[150:156, 150:156] = 105  # 6 px, weak contrast
        boxes = np.array([[20, 20, 40, 40], [150, 150, 156, 156]], dtype=float)

        stats = measure_frame(image, boxes, "v_0001")
        assert stats.n_targets == 2
        assert stats.contrast == pytest.approx(5.0)    # the weak one
        assert stats.size == pytest.approx(20.0)       # the near one

    def test_no_targets_leaves_contrast_undefined(self):
        """NaN, not 0.0: nothing was measured, rather than measured as invisible."""
        image = np.full((SIDE, SIDE), 100, dtype=np.uint8)
        stats = measure_frame(image, np.zeros((0, 4)), "v_0001")
        assert stats.n_targets == 0
        assert stats.contrast != stats.contrast
        assert stats.brightness == pytest.approx(100.0)

    def test_degenerate_box_is_skipped_not_counted(self):
        image = np.full((SIDE, SIDE), 100, dtype=np.uint8)
        stats = measure_frame(image, np.array([[50.0, 50.0, 50.0, 50.0]]), "v_0001")
        assert stats.n_targets == 0

    def test_clipping_fractions(self):
        image = np.full((SIDE, SIDE), 128, dtype=np.uint8)
        image[:20, :] = 255   # 10% of rows blown
        image[20:40, :] = 0   # 10% crushed
        stats = measure_frame(image, np.zeros((0, 4)), "v_0001")
        assert stats.clipped_hi == pytest.approx(0.10)
        assert stats.clipped_lo == pytest.approx(0.10)


class TestLightingLabel:

    @pytest.mark.parametrize("contrast, expected", [
        (0.0, "invisible (<5)"),
        (4.9, "invisible (<5)"),
        (5.0, "low (5-15)"),
        (14.9, "low (5-15)"),
        (15.0, "moderate (15-30)"),
        (30.0, "strong (>=30)"),
        (200.0, "strong (>=30)"),
    ])
    def test_buckets_are_half_open_and_cover_the_range(self, contrast, expected):
        assert lighting_label(stats_with(contrast=contrast)) == expected

    def test_backlighting_overrides_the_contrast_bucket(self):
        """A distinct optical regime, not a point on the contrast scale."""
        assert lighting_label(stats_with(contrast=50.0, clipped_hi=0.05)) == BACKLIT

    def test_below_the_clip_threshold_the_contrast_bucket_stands(self):
        assert lighting_label(stats_with(contrast=50.0, clipped_hi=0.01)) != BACKLIT

    def test_a_frame_with_no_target_is_not_called_invisible(self):
        assert lighting_label(stats_with(n_targets=0,
                                         contrast=float("nan"))) == NO_TARGET


class TestRelativeRange:

    def test_reference_is_a_high_percentile_of_apparent_size(self):
        stats = [stats_with(size=float(s)) for s in range(1, 101)]
        assert reference_size(stats) == pytest.approx(95.0, abs=1.0)

    def test_frames_without_targets_do_not_drag_the_reference(self):
        stats = ([stats_with(size=float(s)) for s in range(1, 101)]
                 + [stats_with(n_targets=0, size=float("nan"))] * 50)
        assert reference_size(stats) == pytest.approx(95.0, abs=1.0)

    @pytest.mark.parametrize("size, expected", [
        (100.0, "near (<2x)"),      # at the reference
        (60.0, "near (<2x)"),       # 1.67x
        (40.0, "mid (2-3x)"),       # 2.5x
        (25.0, "far (3-5x)"),       # 4x
        (10.0, "very far (>5x)"),   # 10x
    ])
    def test_range_is_the_reference_over_apparent_size(self, size, expected):
        assert range_label(stats_with(size=size), reference=100.0) == expected

    def test_a_frame_with_no_target_has_no_range(self):
        assert range_label(stats_with(n_targets=0, size=float("nan")),
                           reference=100.0) == NO_TARGET

    def test_halving_apparent_size_doubles_relative_range(self):
        """The whole basis of the axis: one airframe, so d is proportional to 1/size."""
        near = range_label(stats_with(size=50.0), reference=100.0)
        far = range_label(stats_with(size=25.0), reference=100.0)
        assert (near, far) == ("mid (2-3x)", "far (3-5x)")


class TestBuildAxes:

    @pytest.fixture
    def axes(self):
        stats = [stats_with(key=f"v_{i:04d}", size=float(10 * i + 10),
                            contrast=float(i * 8))
                 for i in range(5)]
        return build_axes(stats)

    def test_declares_both_derived_axes_at_frame_level(self, axes):
        assert set(axes) == {"lighting", "relative_range"}
        assert all(spec["level"] == "frame" for spec in axes.values())

    def test_every_frame_gets_a_label_on_every_axis(self, axes):
        for spec in axes.values():
            assert len(spec["labels"]) == 5

    def test_declares_a_bucket_order(self, axes):
        assert axes["relative_range"]["order"][0] == "near (<2x)"
        assert axes["lighting"]["order"][0] == "invisible (<5)"

    def test_records_the_reference_it_scaled_by(self, axes):
        """Without it, the buckets cannot be related back to pixels later."""
        assert axes["relative_range"]["reference_size_px"] > 0


class TestSummarise:

    def test_groups_by_video_and_counts_the_dark_tail(self):
        stats = ([stats_with(key=f"phantom05_{i:04d}", contrast=1.0) for i in range(4)]
                 + [stats_with(key=f"phantom09_{i:04d}", contrast=40.0) for i in range(6)])
        summary = summarise(stats)
        assert set(summary) == {"phantom05", "phantom09"}
        assert summary["phantom05"]["frames"] == 4
        assert summary["phantom05"]["frac_below_5"] == pytest.approx(1.0)
        assert summary["phantom09"]["frac_below_5"] == pytest.approx(0.0)

    def test_a_video_with_no_measurable_target_reports_none_not_zero(self):
        summary = summarise([stats_with(key="phantom05_0001", n_targets=0,
                                        contrast=float("nan"))])
        assert summary["phantom05"]["contrast_p50"] is None
