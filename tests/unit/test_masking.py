"""Unit tests for the HUD veto.

Two failures matter here and they pull in opposite directions. Letting a glyph
through is what EXP-011 measured: 21 of 24 sampled detections were HUD, and a
lock on a battery digit survived 300 frames. Vetoing too eagerly is worse and
quieter — the reticle sits at the centre of the frame, exactly where a target
being flown at appears, so an over-broad rule would delete real detections and
show up only as unexplained recall loss.

The fraction threshold is what separates them, so it is tested from both sides.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.algo.masking import HUD_VETO_FRACTION, is_masked, overlap_fraction


@pytest.fixture
def mask() -> np.ndarray:
    """A 100x200 frame with a 20x20 block of HUD at (50, 10)."""
    grid = np.zeros((100, 200), dtype=bool)
    grid[10:30, 50:70] = True
    return grid


class TestOverlapFraction:

    def test_box_entirely_on_the_mask(self, mask):
        assert overlap_fraction((50, 10, 20, 20), mask) == 1.0

    def test_box_entirely_off_the_mask(self, mask):
        assert overlap_fraction((120, 60, 20, 20), mask) == 0.0

    def test_half_on(self, mask):
        assert overlap_fraction((50, 10, 40, 20), mask) == pytest.approx(0.5)

    def test_box_outside_the_frame_overlaps_nothing(self, mask):
        """Off the frame is off the HUD; the alternative is an index error."""
        assert overlap_fraction((-50, -50, 10, 10), mask) == 0.0

    def test_degenerate_box_is_not_an_error(self, mask):
        assert overlap_fraction((50, 10, 0, 0), mask) == 0.0

    def test_box_clipped_by_the_frame_edge(self, mask):
        """Only the visible part counts, so a box half off-screen is judged on
        what is actually there rather than on its nominal area."""
        assert overlap_fraction((190, 10, 20, 20), mask) == 0.0


class TestIsMasked:

    def test_vetoes_a_glyph(self, mask):
        assert is_masked((50, 10, 20, 20), mask)

    def test_passes_a_box_in_open_scene(self, mask):
        assert not is_masked((120, 60, 20, 20), mask)

    def test_no_mask_never_vetoes(self):
        """The default. It is what keeps EXP-004/005/010 reproducing exactly."""
        assert not is_masked((50, 10, 20, 20), None)

    def test_a_target_merely_crossing_a_glyph_survives(self, mask):
        """The case that makes this a fraction and not a centre test.

        A 60x60 target overlapping the 20x20 glyph is 11% HUD. A centre-point
        rule would have deleted it, and the reticle sits where targets fly.
        """
        box = (40, 5, 60, 60)
        assert overlap_fraction(box, mask) < HUD_VETO_FRACTION
        assert not is_masked(box, mask)

    def test_threshold_is_inclusive(self, mask):
        assert is_masked((50, 10, 40, 20), mask, threshold=0.5)

    def test_threshold_is_configurable(self, mask):
        box = (50, 10, 40, 20)  # exactly half on
        assert not is_masked(box, mask, threshold=0.75)
