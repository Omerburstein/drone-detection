"""Tests for the contact-sheet primitives.

These decide whether a sheet can be *read*, which is the only thing it is for.
Three properties carry that:

**The crop scales with the box.** A fixed window renders a 6 px blob and a 30 px
drone equally unreadable, one lost and the other clipped. What is asserted is
that the window tracks box size and still floors, so a tiny target keeps context
around it rather than filling the cell with two pixels.

**The box survives the crop.** A cell whose drawn rectangle has drifted is worse
than no cell: it looks authoritative and points at the wrong thing.

**Grouping puts the rare branches first.** `global yolo` fires six times in
3,600 frames, and in frame order those six cells are invisible.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.output.contact import (BACKDROP, GROUP_ORDER, UNGROUPED, Layout, cell,
                                colour_for, grid, group_sort_key, legend, sheet)

SMALL = Layout(cell=40, cols=3, window=4.0, min_window=20)


def image(width: int = 200, height: int = 150) -> np.ndarray:
    """A frame with a horizontal gradient, so a crop's position is visible."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 1] = np.linspace(0, 255, width, dtype=np.uint8)
    return frame


class TestSourceWindow:
    """How much of the frame one cell shows."""

    def test_scales_with_the_box(self) -> None:
        box = np.array([100.0, 100.0, 130.0, 110.0])
        assert SMALL.source_window(box, (500, 500)) == 120  # 4x the 30 px side

    def test_floors_at_min_window(self) -> None:
        """A 3 px box would otherwise get a 12 px window and no context."""
        box = np.array([10.0, 10.0, 13.0, 13.0])
        assert SMALL.source_window(box, (500, 500)) == SMALL.min_window

    def test_never_exceeds_the_frame(self) -> None:
        box = np.array([0.0, 0.0, 400.0, 400.0])
        assert SMALL.source_window(box, (150, 200)) == 150

    def test_uses_the_longest_side(self) -> None:
        wide = np.array([0.0, 0.0, 40.0, 5.0])
        tall = np.array([0.0, 0.0, 5.0, 40.0])
        assert (SMALL.source_window(wide, (500, 500))
                == SMALL.source_window(tall, (500, 500)))


class TestCell:
    """One rendered crop."""

    def test_shape_is_the_layout_plus_border_and_caption(self) -> None:
        tile = cell(image(), np.array([90.0, 70.0, 100.0, 80.0]), "1", "local yolo",
                    SMALL)
        assert tile.shape[1] == SMALL.cell + 4          # 2 px border each side
        assert tile.shape[0] > tile.shape[1]            # caption strip below

    def test_border_carries_the_group_colour(self) -> None:
        tile = cell(image(), np.array([90.0, 70.0, 100.0, 80.0]), "1", "local mod",
                    SMALL)
        assert tuple(int(v) for v in tile[0, 0]) == colour_for("local mod")

    def test_an_unknown_group_still_renders(self) -> None:
        tile = cell(image(), np.array([90.0, 70.0, 100.0, 80.0]), "1", "surprise",
                    SMALL)
        assert tile is not None
        assert tuple(int(v) for v in tile[0, 0]) == colour_for(UNGROUPED)

    def test_a_box_at_the_frame_edge_is_clamped_not_dropped(self) -> None:
        """The window is shifted inward rather than shrunk, so the cell stays square."""
        tile = cell(image(), np.array([0.0, 0.0, 8.0, 8.0]), "1", "local yolo", SMALL)
        assert tile is not None
        assert tile.shape[1] == SMALL.cell + 4

    def test_a_box_outside_the_frame_returns_none(self) -> None:
        assert cell(np.zeros((0, 0, 3), dtype=np.uint8),
                    np.array([0.0, 0.0, 4.0, 4.0]), "1", "local yolo", SMALL) is None

    def test_the_drawn_box_lands_inside_the_cell(self) -> None:
        """The rectangle is drawn in the group colour; it must appear somewhere
        in the image area, not off the edge of it."""
        colour = colour_for("global yolo")
        tile = cell(image(), np.array([95.0, 70.0, 105.0, 80.0]), "1", "global yolo",
                    SMALL)
        inner = tile[2:2 + SMALL.cell, 2:2 + SMALL.cell]
        assert (inner == np.array(colour, dtype=np.uint8)).all(axis=2).any()


class TestGrid:
    """Tiles laid out."""

    def test_rows_wrap_at_cols(self) -> None:
        tiles = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(7)]
        out = grid(tiles, Layout(cols=3))
        assert out.shape == (30, 30, 3)  # 3 rows of 3, last one padded

    def test_the_last_row_is_padded_with_backdrop(self) -> None:
        tiles = [np.full((10, 10, 3), 200, dtype=np.uint8) for _ in range(4)]
        out = grid(tiles, Layout(cols=3))
        assert tuple(int(v) for v in out[15, 25]) == BACKDROP

    def test_no_tiles_is_an_error_not_an_empty_image(self) -> None:
        with pytest.raises(ValueError, match="no tiles"):
            grid([], SMALL)


class TestGrouping:
    """Which cells come first."""

    def test_declared_branches_sort_before_anything_else(self) -> None:
        groups = ["zzz", "local yolo", "global yolo", "aaa"]
        assert sorted(groups, key=group_sort_key)[:2] == ["global yolo", "local yolo"]

    def test_declared_order_is_the_declared_order(self) -> None:
        assert sorted(GROUP_ORDER, key=group_sort_key) == list(GROUP_ORDER)

    def test_undeclared_groups_sort_alphabetically_among_themselves(self) -> None:
        assert sorted(["b", "a"], key=group_sort_key) == ["a", "b"]


class TestLegend:
    """The band that says what the colours mean."""

    def test_spans_the_requested_width(self) -> None:
        assert legend(600, {"local yolo": 3}, "title").shape[1] == 600

    def test_a_zero_count_group_is_omitted_rather_than_shown_as_zero(self) -> None:
        painted = legend(600, {"local yolo": 0, "local mod": 5}, "t")
        assert not (painted == np.array(colour_for("local yolo"),
                                        dtype=np.uint8)).all(axis=2).any()

    def test_empty_counts_do_not_divide_by_zero(self) -> None:
        assert legend(400, {}, "nothing").shape == (64, 400, 3)


class TestSheet:
    """Legend and grid together."""

    def test_is_the_grid_plus_the_band(self) -> None:
        tiles = [np.zeros((20, 20, 3), dtype=np.uint8) for _ in range(3)]
        out = sheet(tiles, {"local yolo": 3}, "t", Layout(cols=3))
        assert out.shape == (64 + 20, 60, 3)
