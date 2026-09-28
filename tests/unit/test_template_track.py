"""Unit tests for the template tracker, lifted out of test_seed_track.py.

Exercised on synthetic frames: a structured patch moved by a known offset, so
"did it follow" has an exact answer rather than an eyeball one.

The pinned-template default is the one to read first. With blending on, a track
that slips off the drone onto terrain adopts the terrain as its template and
then matches it at 0.99 -- the highest scores on the review sheet were the worst
proposals, which is the opposite of what the sheet is for.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.data.template_track import (TemplateTracker, TrackPoint,
                                     search_window)


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

