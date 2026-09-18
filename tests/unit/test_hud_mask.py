"""Unit tests for the OSD text-block half of the HUD mask builder.

The loose threshold is what catches telemetry whose digits change, and it is
also what would paint the middle of the sky if it escaped its rows. Both sides
are pinned: blocks outside the picture rows are filled whole, and nothing
inside them is touched however often it is white.
"""

from __future__ import annotations

import argparse

import numpy as np
import pytest

from src.data.hud_mask import osd_blocks, parse_rows

HEIGHT, WIDTH = 720, 960
PICTURE = (150, 530)


def frequency() -> np.ndarray:
    """White frequency with a sparse telemetry line on top and a smear mid-sky."""
    grid = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    for x in range(700, 880, 30):  # characters whose pixels are white 10% of the time
        grid[60:75, x:x + 12] = 0.10
    grid[250:260, 400:560] = 0.10  # the horizon bar smearing across samples
    return grid


class TestOsdBlocks:

    def test_a_line_of_characters_becomes_one_rectangle(self):
        blocks = osd_blocks(frequency(), 0.08, PICTURE)
        # The gaps between characters are covered, not just the glyph pixels.
        assert blocks[67, 700:862].all()

    def test_the_picture_rows_are_never_touched(self):
        blocks = osd_blocks(frequency(), 0.08, PICTURE)
        assert not blocks[PICTURE[0]:PICTURE[1]].any()

    def test_below_the_fraction_nothing_is_masked(self):
        assert not osd_blocks(frequency(), 0.2, PICTURE).any()

    def test_distant_blocks_stay_separate(self):
        grid = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        grid[60:70, 50:80] = 0.5
        grid[60:70, 700:730] = 0.5
        blocks = osd_blocks(grid, 0.08, PICTURE)
        assert not blocks[65, 200:600].any()


class TestParseRows:

    def test_parses(self):
        assert parse_rows("150:530") == (150, 530)

    @pytest.mark.parametrize("text", ["530:150", "150", "a:b", "-1:10", "5:5"])
    def test_rejects(self, text):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_rows(text)
