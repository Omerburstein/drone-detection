"""Unit tests for the seed-and-track labeller.

The parsing and geometry are pinned because they decide what lands in a label
file, and a label file is ground truth for every number taken afterwards — a
silently clamped or transposed box would not fail, it would just make the
metrics wrong.

The tracker itself is exercised on synthetic frames: a structured patch moved by
a known offset, so "did it follow" has an exact answer rather than an eyeball
one.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.data.seed_track import (TemplateTracker, TrackPoint, parse_range,
                                 parse_seed, search_window, to_yolo)


def target_patch(size: int = 12) -> np.ndarray:
    """A fixed, structured patch standing in for the drone.

    Structured on purpose. Normalised cross-correlation is ill-conditioned on a
    flat patch — a uniform block has no variance to normalise by and scores
    near zero against a perfect match. Real target pixels always carry
    structure (body, arms, props), so a flat square would test a case that
    never occurs and fail for the wrong reason.
    """
    rng = np.random.default_rng(20260917)
    patch = rng.integers(40, 250, size=(size, size), dtype=np.int16)
    return np.repeat(patch[:, :, None], 3, axis=2).astype(np.uint8)


def frame_with_square(x: int, y: int, size: int = 12,
                      shape: tuple[int, int] = (200, 320)) -> np.ndarray:
    """A textured frame carrying the target patch at `(x, y)`."""
    rng = np.random.default_rng(7)
    image = rng.integers(20, 60, size=(*shape, 3), dtype=np.int16).astype(np.uint8)
    patch = target_patch(12)
    if size != 12:
        patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_LINEAR)
    image[y:y + size, x:x + size] = patch
    return image


class TestParseSeed:

    def test_reads_frame_and_box(self):
        assert parse_seed(["962", "738", "382", "80", "34"]) == (
            962, (738.0, 382.0, 80.0, 34.0))

    def test_rejects_wrong_arity(self):
        with pytest.raises(ValueError, match="FRAME X Y W H"):
            parse_seed(["962", "738", "382"])

    def test_rejects_non_numbers(self):
        with pytest.raises(ValueError, match="numbers"):
            parse_seed(["962", "738", "382", "eighty", "34"])

    @pytest.mark.parametrize("box", [["1", "0", "0", "0", "10"],
                                     ["1", "0", "0", "10", "-2"]])
    def test_rejects_empty_box(self, box):
        with pytest.raises(ValueError, match="positive size"):
            parse_seed(box)

    def test_rejects_zero_frame_because_keys_are_one_based(self):
        """Frame keys are `<stem>_0001` for the *first* decoded frame; a 0 here
        would write labels one frame out for the whole segment."""
        with pytest.raises(ValueError, match="1-based"):
            parse_seed(["0", "1", "1", "10", "10"])


class TestParseRange:

    def test_inclusive_of_both_ends(self):
        assert list(parse_range("10-13")) == [10, 11, 12, 13]

    def test_single_frame_range(self):
        assert list(parse_range("7-7")) == [7]

    @pytest.mark.parametrize("text", ["10", "10-9", "0-5", "a-b", "1-2-3"])
    def test_rejects_malformed(self, text):
        with pytest.raises(ValueError):
            parse_range(text)


class TestToYolo:

    def test_converts_corner_box_to_normalised_centre(self):
        cx, cy, w, h = to_yolo((100.0, 50.0, 40.0, 20.0), 1440, 1080)
        assert cx == pytest.approx(120 / 1440)
        assert cy == pytest.approx(60 / 1080)
        assert w == pytest.approx(40 / 1440)
        assert h == pytest.approx(20 / 1080)

    def test_clamps_a_box_that_drifted_off_the_edge(self):
        """Out-of-range labels are legal-looking and poison every later reader."""
        cx, cy, _, _ = to_yolo((-40.0, -30.0, 20.0, 10.0), 100, 100)
        assert cx == 0.0
        assert cy == 0.0

    def test_never_exceeds_one(self):
        cx, cy, w, h = to_yolo((90.0, 90.0, 400.0, 400.0), 100, 100)
        assert max(cx, cy, w, h) <= 1.0


class TestSearchWindow:

    def test_expands_by_target_sizes_and_clamps(self):
        x, y, w, h = search_window((100.0, 50.0, 10.0, 10.0), 2.0, (200, 320))
        assert (x, y) == (80, 30)
        assert (w, h) == (50, 50)

    def test_clamped_at_the_origin(self):
        x, y, _, _ = search_window((5.0, 5.0, 10.0, 10.0), 4.0, (200, 320))
        assert (x, y) == (0, 0)

    def test_never_returns_an_empty_window(self):
        _, _, w, h = search_window((0.0, 0.0, 1.0, 1.0), 0.0, (200, 320))
        assert w >= 1 and h >= 1


class TestTemplateTracker:

    def test_follows_the_target(self):
        seed = frame_with_square(100, 80)
        tracker = TemplateTracker(seed, (100.0, 80.0, 12.0, 12.0))
        found = tracker.step(frame_with_square(108, 86))
        assert found is not None
        (x, y, _, _), score = found
        assert x == pytest.approx(108, abs=2)
        assert y == pytest.approx(86, abs=2)
        assert score > 0.9

    def test_gives_up_when_the_target_is_gone(self):
        seed = frame_with_square(100, 80)
        tracker = TemplateTracker(seed, (100.0, 80.0, 12.0, 12.0))
        rng = np.random.default_rng(99)
        empty = rng.integers(20, 60, size=(200, 320, 3), dtype=np.int16).astype(np.uint8)
        assert tracker.step(empty) is None

    def test_does_not_jump_across_the_frame(self):
        """A duplicate of the target far away must not capture the track — that
        is what the local search window is for."""
        seed = frame_with_square(100, 80)
        tracker = TemplateTracker(seed, (100.0, 80.0, 12.0, 12.0), search=1.0)
        moved = frame_with_square(104, 82)
        moved[20:32, 260:272] = target_patch(12)  # a decoy in the far corner
        found = tracker.step(moved)
        assert found is not None
        (x, _, _, _), _ = found
        assert x < 200

    def test_pinned_template_is_the_default(self):
        """Blending is the drift trap documented in the module docstring."""
        tracker = TemplateTracker(frame_with_square(100, 80),
                                  (100.0, 80.0, 12.0, 12.0))
        assert tracker.update == 0.0

    def test_seed_box_outside_the_frame_is_refused(self):
        with pytest.raises(ValueError, match="does not lie inside"):
            TemplateTracker(frame_with_square(10, 10), (400.0, 400.0, 12.0, 12.0))

    def test_tracks_a_growing_target(self):
        """An intercept target grows; a fixed-scale template would lose it."""
        seed = frame_with_square(100, 80, size=12)
        tracker = TemplateTracker(seed, (100.0, 80.0, 12.0, 12.0))
        found = tracker.step(frame_with_square(100, 80, size=13))
        assert found is not None
        (_, _, w, _), _ = found
        assert w >= 12


class TestTrackPoint:

    def test_is_hashable_and_frozen(self):
        point = TrackPoint(12, (1.0, 2.0, 3.0, 4.0), 0.9)
        with pytest.raises(AttributeError):
            point.score = 0.1  # type: ignore[misc]
