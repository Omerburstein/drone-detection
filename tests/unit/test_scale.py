"""Unit tests for pixel constants quoted at a reference width.

The property that matters is exactness at the width a number was tuned at. Every analog run
in the ledger was made with absolute pixels at 960 wide, and quoting them at 1440 must give
the same floats back, not 19.999999. Otherwise a `<=` on a merge radius flips and an old
result stops reproducing.
"""

from __future__ import annotations

import pytest

from src.algo.scale import REF_WIDTH, PixelScale


def test_the_reference_is_the_o4_crop():
    assert REF_WIDTH == 1440


@pytest.mark.parametrize("quoted, at_960", [(13.5, 9.0), (37.5, 25.0), (6.0, 4.0),
                                            (15.0, 10.0), (30.0, 20.0), (45.0, 30.0)])
def test_analog_numbers_come_back_exactly(quoted, at_960):
    assert PixelScale(960).px(quoted) == at_960


def test_a_number_quoted_at_the_clips_own_width_is_unchanged():
    assert PixelScale(4128, ref_width_px=4128).px(9.0) == 9.0


def test_lengths_scale_with_width():
    assert PixelScale(4128).px(30.0) == pytest.approx(30.0 * 4128 / 1440)
    assert PixelScale(2880).factor == 2.0


def test_quoted_inverts_px():
    s = PixelScale(1032, ref_width_px=960)
    assert s.quoted(s.px(17.0)) == pytest.approx(17.0)


@pytest.mark.parametrize("width, ref", [(0, 1440), (960, 0), (-1, 1440)])
def test_a_width_must_be_positive(width, ref):
    with pytest.raises(ValueError):
        PixelScale(width, ref)
