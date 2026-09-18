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

from src.algo.masking import (HUD_VETO_FRACTION, OSD_COLUMNS, TWIN_SCORE, has_twin,
                              is_masked, overlap_fraction, twin_score)


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


# --- the HUD that moves ----------------------------------------------------

WIDTH, HEIGHT = 960, 720
COLUMN = WIDTH // OSD_COLUMNS  # 32 px, as on the analog captures


def sky() -> np.ndarray:
    """A grey frame with mild noise, so nothing is perfectly flat."""
    rng = np.random.default_rng(0)
    return (150 + rng.normal(0, 1.5, (HEIGHT, WIDTH))).clip(0, 255).astype(np.uint8)


def dash(gray: np.ndarray, x: int, y: int) -> None:
    """One horizon dash as the analog OSD draws it: white over a black outline."""
    gray[y:y + 2, x:x + 8] = 215
    gray[y + 3:y + 5, x:x + 8] = 85


def drone(gray: np.ndarray, x: int, y: int) -> None:
    """A small dark airframe with a lit top: a dash's contrast, but alone."""
    gray[y:y + 2, x + 1:x + 9] = 210
    gray[y + 2:y + 6, x:x + 10] = 70


class TestTwins:
    """A dash has identical copies one OSD column away; a drone does not.

    EXP-013's worst lock was on a horizon dash that looks like a quadcopter at
    10 px. Both sides are tested, because the failure that matters more is the
    quiet one: vetoing the drone.
    """

    def test_a_dash_in_a_row_of_dashes_twins(self):
        gray = sky()
        for k in range(-3, 4):
            dash(gray, 470 + k * COLUMN, 300)
        assert has_twin(gray, (470, 300, 8, 5))

    def test_a_lone_drone_does_not_twin(self):
        gray = sky()
        drone(gray, 470, 300)
        assert twin_score(gray, (470, 300, 10, 6)) < TWIN_SCORE

    def test_a_drone_beside_the_horizon_bar_does_not_twin(self):
        """The bar is a row of *dashes*, and a drone is not one of them."""
        gray = sky()
        for k in range(-3, 4):
            dash(gray, 470 + k * COLUMN, 300)
        drone(gray, 486, 330)
        assert not has_twin(gray, (486, 330, 10, 6))

    def test_a_tilted_row_still_twins(self):
        """Roll tilts the horizon; the vertical slack must cover a column's step."""
        gray = sky()
        for k in range(-3, 4):
            dash(gray, 470 + k * COLUMN, 300 + 6 * k)
        assert has_twin(gray, (470, 300, 8, 5))

    def test_the_box_itself_is_not_its_own_twin(self):
        gray = sky()
        dash(gray, 470, 300)
        assert twin_score(gray, (470, 300, 8, 5)) < TWIN_SCORE

    def test_a_flat_box_never_twins(self):
        """Normalised correlation stretches flat sky to full contrast."""
        gray = np.full((HEIGHT, WIDTH), 150, dtype=np.uint8)
        assert twin_score(gray, (470, 300, 10, 10)) == 0.0

    def test_a_box_at_the_frame_edge_is_not_an_error(self):
        gray = sky()
        dash(gray, 0, 0)
        assert 0.0 <= twin_score(gray, (0, 0, 8, 5)) <= 1.0

    def test_a_box_outside_the_frame_scores_nothing(self):
        assert twin_score(sky(), (2000, 2000, 10, 10)) == 0.0
