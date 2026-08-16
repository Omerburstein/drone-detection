"""Tests for the coordinate handling in the GLAD port.

Every failure mode here is silent. A wrong letterbox offset, a wrong scale, or a
search region anchored on the box centre instead of its top-left all produce
boxes that land *near* the drone rather than on it — plausible output, wrong
numbers, and no exception anywhere. These are the tests that catch it.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.algo.glad.pipeline import REGION_HALF, search_region
from src.algo.glad.yolo import (INPUT_H, INPUT_W, PAD_STYLES, TRAINED_PAD, _letterbox,
                                _to_original)

FRAME_W, FRAME_H = 1920, 1080
CROP = 320  # the search region, which is square and so takes the other pad branch


class TestSearchRegion:
    """Upstream `enlarge_region2`: a 320x320 square anchored on the box top-left."""

    def test_anchors_on_the_top_left_not_the_centre(self):
        """The box's *corner* lands at the region centre. Reads like a bug; isn't."""
        assert search_region(500, 400, REGION_HALF, FRAME_W, FRAME_H) == (340, 240, 320, 320)

    @pytest.mark.parametrize("x, y, expected_origin", [
        (500, 400, (340, 240)),      # interior
        (10, 400, (0, 240)),         # left edge clamps to 0
        (500, 10, (340, 0)),         # top edge clamps to 0
        (1900, 400, (1600, 240)),    # right edge shifts back inside
        (500, 1070, (340, 760)),     # bottom edge shifts back inside
        (5, 5, (0, 0)),              # both low edges at once
    ])
    def test_clamping(self, x, y, expected_origin):
        region = search_region(x, y, REGION_HALF, FRAME_W, FRAME_H)
        assert region[:2] == expected_origin

    @pytest.mark.parametrize("x, y", [(500, 400), (0, 0), (1919, 1079), (10, 1070)])
    def test_region_is_always_exactly_square(self, x, y):
        """Clamping *shifts* the region rather than shrinking it.

        The local detectors are trained on 320x320 crops, so a short region at a
        frame edge would silently change their input scale.
        """
        _, _, width, height = search_region(x, y, REGION_HALF, FRAME_W, FRAME_H)
        assert (width, height) == (2 * REGION_HALF, 2 * REGION_HALF)

    @pytest.mark.parametrize("x, y", [(500, 400), (0, 0), (1919, 1079)])
    def test_region_stays_inside_the_frame(self, x, y):
        region_x, region_y, width, height = search_region(x, y, REGION_HALF,
                                                          FRAME_W, FRAME_H)
        assert 0 <= region_x and region_x + width <= FRAME_W
        assert 0 <= region_y and region_y + height <= FRAME_H

    def test_returns_ints(self):
        """The result indexes a numpy slice; floats would raise there, not here."""
        assert all(isinstance(v, int)
                   for v in search_region(500.7, 400.2, REGION_HALF, FRAME_W, FRAME_H))

    @pytest.mark.slow
    def test_matches_the_vendored_original(self):
        """Pins the port against `third_party/GLAD`, when that clone is present.

        Marked slow because it needs the vendored tree, which is gitignored — so
        the offline suite verifies the behaviour above, and this verifies that
        the behaviour is upstream's.
        """
        from src.algo.glad.vendor import import_enlarge_region  # noqa: PLC0415

        try:
            vendored = import_enlarge_region()
        except FileNotFoundError as missing:
            pytest.skip(str(missing))

        for x in (0, 5, 160, 500, 1600, 1900, 1919):
            for y in (0, 5, 160, 400, 900, 1070, 1079):
                assert (search_region(x, y, REGION_HALF, FRAME_W, FRAME_H)
                        == vendored(x, y, REGION_HALF, FRAME_W, FRAME_H))


class TestLetterbox:
    """tensorrtx's preprocessing, which is not yolov5's."""

    def test_shape_and_range(self):
        out = _letterbox(np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8), TRAINED_PAD)
        assert out.shape == (3, INPUT_H, INPUT_W)
        assert out.dtype == np.float32
        assert 0.0 <= out.min() and out.max() <= 1.0

    @pytest.mark.parametrize("pad_value", sorted(PAD_STYLES.values()))
    def test_fills_the_bars_with_the_requested_value(self, pad_value):
        """The bug this replaces passed `value` positionally, into `dst`.

        OpenCV silently discarded it and filled with 0, so every style produced
        black. Parametrised over all three to keep that from coming back.
        """
        out = _letterbox(np.full((FRAME_H, FRAME_W, 3), 255, dtype=np.uint8), pad_value)
        # 1920x1080 scales to 640x360, leaving 140 pad rows top and bottom.
        assert out[:, 0, 0] == pytest.approx(pad_value / 255.0)
        assert out[:, -1, -1] == pytest.approx(pad_value / 255.0)
        assert out[:, INPUT_H // 2, INPUT_W // 2] == pytest.approx(1.0)

    def test_styles_are_the_three_values_that_matter(self):
        """`released` is upstream's actual output; `trained` is what yolov5 used."""
        assert PAD_STYLES == {"released": 0, "trained": 114, "tensorrtx": 128}

    @pytest.mark.parametrize("pad_value", sorted(PAD_STYLES.values()))
    def test_square_input_is_never_padded(self, pad_value):
        """A search-region crop fills the input exactly, so the fill cannot reach it.

        This is why the bug only ever touched the global detector.
        """
        image = np.full((CROP, CROP, 3), 200, dtype=np.uint8)
        out = _letterbox(image, pad_value)
        assert out.min() == pytest.approx(200 / 255.0)

    def test_channels_are_swapped_to_rgb(self):
        image = np.zeros((CROP, CROP, 3), dtype=np.uint8)
        image[:, :, 0] = 255  # blue in the BGR frame
        out = _letterbox(image, TRAINED_PAD)
        assert out[2].max() == pytest.approx(1.0)  # ...is channel 2 in RGB
        assert out[0].max() == pytest.approx(0.0)


class TestToOriginal:
    """The inverse map. An error here is exactly the near-miss failure mode."""

    @pytest.mark.parametrize("box, frame_size, expected", [
        # 1920x1080: scale 1/3, 140 pad rows. A 30x30 box at (600, 300) has its
        # centre at (615, 315) -> (205, 245) in letterbox space.
        ([205.0, 245.0, 10.0, 10.0], (FRAME_H, FRAME_W), [600, 300, 30, 30]),
        # 320x320: scale 2, no padding at all.
        ([220.0, 220.0, 40.0, 40.0], (CROP, CROP), [100, 100, 20, 20]),
    ])
    def test_known_mappings(self, box, frame_size, expected):
        out = _to_original(np.array([box]), *frame_size)
        assert out[0] == pytest.approx(expected)

    def test_round_trips_a_box_through_the_forward_map(self):
        """Independent of the constants above: forward, then back, is identity."""
        original = np.array([[600.0, 300.0, 30.0, 30.0]])  # xywh
        scale = INPUT_W / FRAME_W
        pad = (INPUT_H - scale * FRAME_H) / 2
        letterboxed = np.array([[
            (original[0, 0] + original[0, 2] / 2) * scale,
            (original[0, 1] + original[0, 3] / 2) * scale + pad,
            original[0, 2] * scale,
            original[0, 3] * scale,
        ]])
        assert _to_original(letterboxed, FRAME_H, FRAME_W)[0] == pytest.approx(original[0])

    def test_handles_several_boxes_at_once(self):
        boxes = np.array([[205.0, 245.0, 10.0, 10.0], [100.0, 200.0, 6.0, 6.0]])
        assert _to_original(boxes, FRAME_H, FRAME_W).shape == (2, 4)
