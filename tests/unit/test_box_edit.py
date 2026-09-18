"""Unit tests for the annotator's box geometry.

Pinned because this is where a mouse gesture becomes a label: a handle that
grabs the wrong edge, a drag that creeps, or a zoom whose image<->window
mapping is off by a fraction of a pixel would all produce boxes that look
right on screen and are wrong in the file.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.box_edit import (MOVE, Viewport, clamp_box, drag, from_corners,
                               hit_test)

BOX = (100.0, 50.0, 40.0, 20.0)  # x0=100 y0=50 x1=140 y1=70


class TestHitTest:

    @pytest.mark.parametrize("point, handle", [
        ((100, 50), "nw"), ((140, 50), "ne"), ((100, 70), "sw"), ((140, 70), "se"),
        ((120, 50), "n"), ((120, 70), "s"), ((100, 60), "w"), ((140, 60), "e"),
        ((120, 60), MOVE),
    ])
    def test_each_region_grabs_its_handle(self, point, handle):
        assert hit_test(BOX, point, tol=3) == handle

    def test_outside_grabs_nothing(self):
        assert hit_test(BOX, (160, 60), tol=3) is None

    def test_centre_of_a_tiny_box_still_moves_it(self):
        """A 6 px box with an 8 px tolerance would otherwise be all handle."""
        assert hit_test((10.0, 10.0, 6.0, 6.0), (13, 13), tol=8) == MOVE


class TestDrag:

    def test_move_keeps_size(self):
        assert drag(BOX, MOVE, 5, -3) == (105.0, 47.0, 40.0, 20.0)

    def test_corner_moves_only_its_two_edges(self):
        assert drag(BOX, "se", 10, 5) == (100.0, 50.0, 50.0, 25.0)
        assert drag(BOX, "nw", 10, 5) == (110.0, 55.0, 30.0, 15.0)

    def test_edge_moves_one_side(self):
        assert drag(BOX, "e", 7, 99) == (100.0, 50.0, 47.0, 20.0)

    def test_dragging_past_the_opposite_edge_normalises(self):
        assert drag(BOX, "e", -60, 0) == (80.0, 50.0, 20.0, 20.0)

    def test_from_corners_accepts_either_order(self):
        assert from_corners(30, 40, 10, 5) == (10, 5, 20, 35)


class TestClamp:

    def test_clips_to_frame(self):
        assert clamp_box((-5.0, -5.0, 20.0, 20.0), 100, 100) == (0.0, 0.0, 15.0, 15.0)

    def test_box_off_the_edge_keeps_minimum_size(self):
        x, y, w, h = clamp_box((120.0, 10.0, 10.0, 10.0), 100, 100)
        assert w >= 2 and x + w <= 100


class TestViewport:

    def test_fit_shows_whole_frame(self):
        view = Viewport((1440, 1080), (720, 540))
        assert view.zoom == pytest.approx(0.5)
        assert view.to_image(0, 0) == pytest.approx((0, 0))
        assert view.to_image(720, 540) == pytest.approx((1440, 1080))

    def test_round_trip(self):
        view = Viewport((1440, 1080), (720, 540))
        view.zoom_at(100, 200, 3.0)
        assert view.to_image(*view.to_display(333.3, 777.7)) == pytest.approx(
            (333.3, 777.7))

    def test_zoom_keeps_the_point_under_the_cursor(self):
        view = Viewport((1440, 1080), (720, 540))
        before = view.to_image(100, 200)
        view.zoom_at(100, 200, 4.0)
        assert view.to_image(100, 200) == pytest.approx(before)

    def test_cannot_zoom_out_past_the_whole_frame(self):
        view = Viewport((1440, 1080), (720, 540))
        view.zoom_at(0, 0, 0.1)
        assert view.zoom == pytest.approx(view.fit_zoom)

    def test_matrix_agrees_with_to_display(self):
        """`render` draws the picture through `matrix` and the boxes through
        `to_display`; if they disagree every box is drawn off its target."""
        view = Viewport((1440, 1080), (720, 540))
        view.zoom_at(300, 100, 2.5)
        view.pan(17, -9)
        point = np.array([640.0, 480.0, 1.0])
        assert view.matrix() @ point == pytest.approx(view.to_display(640, 480))

    def test_keep_in_view_recentres_only_when_zoomed_and_escaping(self):
        view = Viewport((1440, 1080), (720, 540))
        assert not view.keep_in_view((1400.0, 1000.0, 10.0, 10.0))  # not zoomed
        view.focus((700.0, 500.0, 20.0, 20.0))
        assert not view.keep_in_view((705.0, 505.0, 20.0, 20.0))    # still central
        assert view.keep_in_view((900.0, 500.0, 20.0, 20.0))
        assert (view.cx, view.cy) == (910.0, 510.0)
