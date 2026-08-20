"""Cross-cutting the dump by size band and condition at once.

The traps here are all about the false alarm. It has no target, so it has no
target size — which is why `curves.py` refuses to bin precision on `gt_size`. A
cross-cut still has to place it somewhere, and the choice made is: the frame it
occurred in has a size, so the alarm inherits the frame's band. These tests pin
that, pin the two cases where a frame genuinely has no single band, and pin that
the cells still account for every row of the dump exactly once — the property
`records.py` guarantees and every number here relies on.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.crosscut import (MIXED, NO_TARGET, UNKNOWN, Cell, cell_rows,
                               cross_cut, frame_band, group_frames, pooled,
                               select)

AXIS = "scene_category"


def row(key, outcome, gt_size=None, scene="complex", iou=None, offset=None):
    """One dump row, carrying only the columns the cross-cut reads."""
    return {"key": key,
            "outcome": outcome,
            "gt_size": "" if gt_size is None else str(gt_size),
            "iou": "" if iou is None else str(iou),
            "center_dist_rel": "" if offset is None else str(offset),
            AXIS: scene}


def cell_at(cells, band, condition):
    """The one cell with this band and condition, or None."""
    found = [c for c in cells if c.band == band and c.condition == condition]
    assert len(found) <= 1, f"duplicate cell for {band}/{condition}"
    return found[0] if found else None


class TestFrameBand:
    """Which size band a whole frame belongs to."""

    def test_single_target_gives_its_own_band(self):
        assert frame_band([row("f1", "tp", gt_size=6.0)]) == "<8"
        assert frame_band([row("f1", "tp", gt_size=10.0)]) == "8-12"

    def test_a_missed_target_still_sizes_the_frame(self):
        """An fn carries `gt_size`; a frame is not unsized just because it failed."""
        assert frame_band([row("f1", "fn", gt_size=6.0)]) == "<8"

    def test_frame_with_only_a_false_alarm_has_no_band(self):
        """The alarm's own box size is not a target size and must not stand in."""
        assert frame_band([row("f1", "fp")]) == NO_TARGET

    def test_targets_straddling_an_edge_are_not_forced_into_one_band(self):
        rows = [row("f1", "tp", gt_size=6.0), row("f1", "fn", gt_size=30.0)]
        assert frame_band(rows) == MIXED

    def test_targets_in_one_band_stay_in_it(self):
        rows = [row("f1", "tp", gt_size=5.0), row("f1", "fn", gt_size=7.9)]
        assert frame_band(rows) == "<8"

    def test_custom_edges_are_respected(self):
        assert frame_band([row("f1", "tp", gt_size=10.0)], (0.0, 8.0,
                                                            float("inf"))) == ">=8"


class TestCrossCut:
    """Cells, and what lands in them."""

    def test_size_and_condition_both_split(self):
        rows = [row("a", "tp", gt_size=6.0, scene="complex"),
                row("b", "tp", gt_size=20.0, scene="complex"),
                row("c", "fn", gt_size=6.0, scene="ordinary")]
        cells = cross_cut(rows, AXIS)

        assert {(c.band, c.condition) for c in cells} == {
            ("<8", "complex"), ("20-24", "complex"), ("<8", "ordinary")}
        assert cell_at(cells, "<8", "ordinary").pd == pytest.approx(0.0)
        assert cell_at(cells, "<8", "complex").pd == pytest.approx(1.0)

    def test_false_alarm_lands_in_the_band_of_the_frame_it_fired_in(self):
        """The whole point: alarms raised while hunting a 6 px drone count there."""
        rows = [row("a", "fn", gt_size=6.0), row("a", "fp"),
                row("b", "tp", gt_size=20.0)]
        cells = cross_cut(rows, AXIS)

        small = cell_at(cells, "<8", "complex")
        assert small.fp == 1
        assert small.far == pytest.approx(1.0)  # one alarm over one frame
        assert cell_at(cells, "20-24", "complex").fp == 0

    def test_far_is_per_frame_not_per_target(self):
        """Two clean frames and one with two alarms is 0.667 FA/frame, not 2."""
        rows = [row("a", "tp", gt_size=6.0),
                row("b", "tp", gt_size=6.0),
                row("c", "fn", gt_size=6.0), row("c", "fp"), row("c", "fp")]
        cell = cell_at(cross_cut(rows, AXIS), "<8", "complex")
        assert cell.n_frames == 3
        assert cell.far == pytest.approx(2 / 3)

    def test_every_row_is_counted_exactly_once(self):
        """The dump's core guarantee, carried through the cross-cut."""
        rows = [row("a", "tp", gt_size=6.0), row("a", "fp"),
                row("b", "fn", gt_size=30.0),
                row("c", "fp", scene="ordinary"),
                row("d", "tp", gt_size=6.0, scene="ordinary")]
        cells = cross_cut(rows, AXIS)
        assert sum(c.tp + c.fp + c.fn for c in cells) == len(rows)
        assert sum(c.n_frames for c in cells) == len(group_frames(rows))

    def test_unlabelled_frames_are_kept_as_uncategorised(self):
        """Dropping them would change the denominator of every other cell."""
        rows = [{"key": "a", "outcome": "tp", "gt_size": "6.0", "iou": "0.7",
                 "center_dist_rel": "0.1"}]
        cells = cross_cut(rows, AXIS)
        assert cell_at(cells, "<8", UNKNOWN).n_gt == 1

    def test_matched_statistics_use_matched_pairs_only(self):
        """A missed target has no IoU; averaging a blank in as 0 would libel the box."""
        rows = [row("a", "tp", gt_size=6.0, iou=0.6, offset=0.2),
                row("b", "fn", gt_size=6.0)]
        cell = cell_at(cross_cut(rows, AXIS), "<8", "complex")
        assert cell.mean_iou == pytest.approx(0.6)
        assert cell.loc_err == pytest.approx(0.2)

    def test_cell_with_no_match_reports_nan_not_zero(self):
        cell = cell_at(cross_cut([row("a", "fn", gt_size=6.0)], AXIS), "<8", "complex")
        assert np.isnan(cell.mean_iou)
        assert np.isnan(cell.loc_err)
        assert cell.pd == pytest.approx(0.0)  # this one *is* a real zero

    def test_cell_with_no_target_has_no_pd(self):
        """No targets means Pd is undefined, not 0 — the report must not print 0."""
        cell = cell_at(cross_cut([row("a", "fp")], AXIS), NO_TARGET, "complex")
        assert np.isnan(cell.pd)
        assert cell.precision == pytest.approx(0.0)  # the alarm was still wrong

    def test_bands_come_back_in_size_order(self):
        rows = [row("a", "tp", gt_size=30.0), row("b", "tp", gt_size=6.0),
                row("c", "fp")]
        bands = [c.band for c in cross_cut(rows, AXIS)]
        assert bands == ["<8", "24-32", NO_TARGET]

    def test_small_cells_are_flagged_unreliable(self):
        rows = [row(f"f{i}", "tp", gt_size=6.0) for i in range(5)]
        assert not cell_at(cross_cut(rows, AXIS), "<8", "complex").reliable
        rows = [row(f"f{i}", "tp", gt_size=6.0) for i in range(40)]
        assert cell_at(cross_cut(rows, AXIS), "<8", "complex").reliable


class TestSelectAndPool:
    """Filtering to a cell, and summing a selection honestly."""

    @pytest.fixture
    def cells(self):
        rows = [row("a", "tp", gt_size=6.0, scene="complex"),
                row("b", "fn", gt_size=6.0, scene="ordinary"),
                row("c", "tp", gt_size=20.0, scene="complex")]
        return cross_cut(rows, AXIS)

    def test_empty_filters_keep_everything(self, cells):
        assert select(cells) == cells

    def test_both_filters_intersect(self, cells):
        picked = select(cells, bands=("<8",), conditions=("complex",))
        assert [(c.band, c.condition) for c in picked] == [("<8", "complex")]

    def test_one_filter_leaves_the_other_axis_whole(self, cells):
        assert len(select(cells, bands=("<8",))) == 2

    def test_pooled_recomputes_ratios_from_summed_counts(self):
        """Averaging per-cell Pd would weight a 1-target cell like a 99-target one."""
        big = Cell("<8", "a", 99, 99, 0, 0, 99, float("nan"), float("nan"),
                   float("nan"))
        small = Cell("<8", "b", 1, 1, 1, 0, 0, 0.7, 0.1, 0.2)
        total = pooled([big, small])
        assert total.n_gt == 100
        assert total.pd == pytest.approx(0.01)  # not the 0.5 an average would give

    def test_pooled_labels_a_shared_value_and_generalises_the_rest(self):
        cells = [Cell("<8", "complex", 1, 1, 1, 0, 0, 0.7, 0.1, 0.2),
                 Cell("8-12", "complex", 1, 1, 1, 0, 0, 0.7, 0.1, 0.2)]
        total = pooled(cells)
        assert (total.band, total.condition) == ("all", "complex")

    def test_pooled_leaves_unsummable_statistics_absent(self):
        """A median of medians is not a median; NaN says so rather than guessing."""
        cells = [Cell("<8", "a", 1, 1, 1, 0, 0, 0.7, 0.1, 0.2),
                 Cell("<8", "b", 1, 1, 1, 0, 0, 0.9, 0.3, 0.4)]
        assert np.isnan(pooled(cells).loc_err)

    def test_pooling_nothing_is_empty_not_an_error(self):
        assert pooled([]).n_gt == 0


class TestCellRows:
    """The CSV shape the table is written out in."""

    def test_rows_carry_series_counts_and_ratios(self):
        cells = cross_cut([row("a", "tp", gt_size=6.0, iou=0.6, offset=0.2)], AXIS)
        rows = cell_rows({"centre@1x": cells}, AXIS)
        assert len(rows) == 1
        assert rows[0]["series"] == "centre@1x"
        assert rows[0]["axis"] == AXIS
        assert rows[0]["band"] == "<8"
        assert rows[0]["condition"] == "complex"
        assert rows[0]["pd"] == 1.0
        assert rows[0]["far"] == 0.0

    def test_absent_numbers_are_none_not_zero(self):
        rows = cell_rows({"s": cross_cut([row("a", "fp")], AXIS)}, AXIS)
        assert rows[0]["pd"] is None
        assert rows[0]["mean_iou"] is None

    def test_several_series_share_one_table(self):
        cells = cross_cut([row("a", "tp", gt_size=6.0)], AXIS)
        rows = cell_rows({"centre@1x": cells, "IoU@0.50": cells}, AXIS)
        assert [r["series"] for r in rows] == ["centre@1x", "IoU@0.50"]
