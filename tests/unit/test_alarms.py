"""Binning false alarms by distance from the nearest real target.

Three traps here. First, "nearest" is not "matched" — a false alarm has no
matched target by definition, so the distance has to be computed against every
target in the frame regardless of what claimed what. Second, an alarm in a frame
with no targets has no distance at all, and dropping it into the far bin would
manufacture evidence of clutter the run never produced. Third, the module reads
the dump's own `nearest_gt_dist` column when present and re-derives it when not;
those two paths must agree exactly, or the same run answers differently
depending on when its dump was written.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.alarms import (NO_TARGET, PX, REL, REL_EDGES, Alarm, alarms,
                             bin_alarms, by_group, frame_alarms, table_rows)

BOX_COLUMNS = ("x0", "y0", "x1", "y1")


def row(key, outcome, pred=None, gt=None, video="v1", group="complex",
        nearest=None):
    """One dump row.

    `nearest` supplies the recorded columns when a test wants the read path
    rather than the derive path; omitting it produces a pre-column dump.
    """
    record = {"key": key, "outcome": outcome, "video": video,
              "scene_category": group}
    for prefix, box in (("pred", pred), ("gt", gt)):
        for i, name in enumerate(BOX_COLUMNS):
            record[f"{prefix}_{name}"] = "" if box is None else str(box[i])
    if nearest is not None:
        distance, relative, size = nearest
        record["nearest_gt_dist"] = "" if distance is None else str(distance)
        record["nearest_gt_dist_rel"] = "" if relative is None else str(relative)
        record["nearest_gt_size"] = "" if size is None else str(size)
    return record


def square(cx, cy, side):
    """An axis-aligned box of `side` px centred on (cx, cy)."""
    half = side / 2
    return (cx - half, cy - half, cx + half, cy + half)


class TestFrameAlarms:
    """Distance from one frame's alarms to the targets in that frame."""

    def test_distance_is_to_the_nearest_target_not_the_matched_one(self):
        """An fp has no matched target; every target in the frame is a candidate."""
        rows = [row("f1", "fn", gt=square(100, 100, 10)),
                row("f1", "fn", gt=square(300, 100, 10)),
                row("f1", "fp", pred=square(320, 100, 10))]
        alarm = frame_alarms(rows)[0]
        assert alarm.distance == pytest.approx(20.0)      # the far target, not the near
        assert alarm.distance_rel == pytest.approx(2.0)   # 20 px over a 10 px target
        assert alarm.target_size == pytest.approx(10.0)

    def test_a_matched_target_is_still_a_candidate(self):
        """The drone the detector found is exactly what a duplicate box sits on."""
        rows = [row("f1", "tp", pred=square(100, 100, 10), gt=square(100, 100, 10)),
                row("f1", "fp", pred=square(105, 100, 10))]
        alarm = frame_alarms(rows)[0]
        assert alarm.distance == pytest.approx(5.0)
        assert not alarm.orphan

    def test_relative_distance_uses_the_nearest_targets_own_size(self):
        """5 px from a 5 px drone and from a 50 px drone are not the same event."""
        small = frame_alarms([row("f1", "fn", gt=square(100, 100, 5)),
                              row("f1", "fp", pred=square(105, 100, 5))])[0]
        large = frame_alarms([row("f1", "fn", gt=square(100, 100, 50)),
                              row("f1", "fp", pred=square(105, 100, 50))])[0]
        assert small.distance == large.distance == pytest.approx(5.0)
        assert small.distance_rel == pytest.approx(1.0)
        assert large.distance_rel == pytest.approx(0.1)

    def test_alarm_in_an_empty_frame_has_no_distance(self):
        alarm = frame_alarms([row("f1", "fp", pred=square(100, 100, 10))])[0]
        assert alarm.orphan
        assert np.isnan(alarm.distance)
        assert np.isnan(alarm.distance_rel)

    def test_frame_with_no_alarms_contributes_nothing(self):
        assert frame_alarms([row("f1", "tp", pred=square(1, 1, 2),
                                 gt=square(1, 1, 2))]) == []

    def test_context_travels_with_the_alarm(self):
        rows = [row("f1", "fn", gt=square(100, 100, 10), video="phantom63"),
                row("f1", "fp", pred=square(200, 100, 10), video="phantom63")]
        alarm = frame_alarms(rows, group="scene_category")[0]
        assert (alarm.key, alarm.video, alarm.group) == ("f1", "phantom63", "complex")


class TestRecordedColumnsAgree:
    """The read path and the derive path are one quantity, not two."""

    def test_recorded_columns_are_used_when_present(self):
        """A dump that carries the answer is not silently recomputed."""
        rows = [row("f1", "fn", gt=square(100, 100, 10), nearest=(None, None, None)),
                row("f1", "fp", pred=square(160, 100, 10), nearest=(60.0, 6.0, 10.0))]
        alarm = frame_alarms(rows)[0]
        assert alarm.distance == pytest.approx(60.0)
        assert alarm.distance_rel == pytest.approx(6.0)

    def test_both_paths_give_the_same_answer(self):
        """The property the whole fallback rests on."""
        geometry = [row("f1", "fn", gt=square(100, 100, 12)),
                    row("f1", "fp", pred=square(148, 136, 8))]
        derived = frame_alarms(geometry)[0]

        recorded = frame_alarms([
            dict(geometry[0], nearest_gt_dist="", nearest_gt_dist_rel="",
                 nearest_gt_size=""),
            dict(geometry[1], nearest_gt_dist=str(derived.distance),
                 nearest_gt_dist_rel=str(derived.distance_rel),
                 nearest_gt_size=str(derived.target_size)),
        ])[0]
        assert recorded.distance == pytest.approx(derived.distance)
        assert recorded.distance_rel == pytest.approx(derived.distance_rel)

    def test_blank_recorded_cell_is_an_orphan_not_a_missing_column(self):
        """A post-column dump blanks the cell only when the frame had no target."""
        rows = [row("f1", "fp", pred=square(100, 100, 10),
                    nearest=(None, None, None))]
        assert frame_alarms(rows)[0].orphan


class TestBinning:
    """Counts into bins, and what the shares are over."""

    def test_alarms_land_in_the_bin_their_distance_names(self):
        found = [Alarm("f", "v", "", 5.0, 0.5, 10.0),
                 Alarm("f", "v", "", 15.0, 1.5, 10.0),
                 Alarm("f", "v", "", 50.0, 5.0, 10.0)]
        table = bin_alarms(found)
        counts = dict(zip(table.labels, table.counts))
        assert counts["<1"] == 1
        assert counts["1-2"] == 1
        assert counts["4-8"] == 1

    def test_bins_are_half_open_at_the_top(self):
        """A tolerance of 1.0 accepts exactly what lands below the `<1` edge."""
        table = bin_alarms([Alarm("f", "v", "", 10.0, 1.0, 10.0)])
        assert dict(zip(table.labels, table.counts))["<1"] == 0
        assert dict(zip(table.labels, table.counts))["1-2"] == 1

    def test_orphans_are_counted_apart_from_the_ladder(self):
        """Putting them in the top bin would invent clutter the run never made."""
        found = [Alarm("f", "v", "", np.nan, np.nan, np.nan),
                 Alarm("f", "v", "", 50.0, 5.0, 10.0)]
        table = bin_alarms(found)
        assert table.orphans == 1
        assert table.counts.sum() == 1
        assert table.total == 2

    def test_shares_include_orphans_so_the_column_sums_to_one(self):
        found = [Alarm("f", "v", "", np.nan, np.nan, np.nan),
                 Alarm("f", "v", "", 50.0, 5.0, 10.0)]
        table = bin_alarms(found)
        assert table.share.sum() + table.orphans / table.total == pytest.approx(1.0)

    def test_cumulative_reads_from_the_near_end(self):
        found = [Alarm("f", "v", "", 5.0, 0.5, 10.0),
                 Alarm("f", "v", "", 15.0, 1.5, 10.0)]
        table = bin_alarms(found)
        assert table.cumulative[0] == pytest.approx(0.5)
        assert table.cumulative[1] == pytest.approx(1.0)

    def test_pixel_unit_bins_the_other_quantity(self):
        """The same alarm sits in different bins under the two units."""
        found = [Alarm("f", "v", "", 30.0, 0.5, 60.0)]
        assert dict(zip(bin_alarms(found, REL).labels,
                        bin_alarms(found, REL).counts))["<1"] == 1
        table = bin_alarms(found, PX)
        assert dict(zip(table.labels, table.counts))["25-50"] == 1

    def test_custom_edges_are_respected(self):
        table = bin_alarms([Alarm("f", "v", "", 50.0, 5.0, 10.0)],
                           REL, (0.0, 1.0, float("inf")))
        assert table.labels == ["<1", ">=1"]
        assert list(table.counts) == [0, 1]

    def test_empty_table_reports_nan_shares_not_zero(self):
        table = bin_alarms([])
        assert table.total == 0
        assert np.isnan(table.share).all()


class TestAlarmsOverADump:
    """The whole-file entry point and the optional grouping."""

    def test_alarms_are_collected_across_frames(self):
        rows = [row("f1", "fn", gt=square(100, 100, 10)),
                row("f1", "fp", pred=square(200, 100, 10)),
                row("f2", "fp", pred=square(100, 100, 10))]
        found = alarms(rows)
        assert len(found) == 2
        assert [a.key for a in found] == ["f1", "f2"]
        assert found[1].orphan  # f2 held no target

    def test_grouping_splits_and_orders_by_size(self):
        rows = [row("a_1", "fn", gt=square(100, 100, 10), group="complex"),
                row("a_1", "fp", pred=square(200, 100, 10), group="complex"),
                row("b_1", "fn", gt=square(100, 100, 10), group="small_mav"),
                row("b_1", "fp", pred=square(200, 100, 10), group="small_mav"),
                row("b_2", "fn", gt=square(100, 100, 10), group="small_mav"),
                row("b_2", "fp", pred=square(200, 100, 10), group="small_mav")]
        tables = by_group(alarms(rows, "scene_category"))
        assert list(tables) == ["small_mav", "complex"]  # biggest first
        assert tables["small_mav"].total == 2
        assert tables["complex"].total == 1


class TestTableRows:
    """The CSV shape."""

    def test_every_alarm_appears_on_exactly_one_row(self):
        found = [Alarm("f", "v", "", 5.0, 0.5, 10.0),
                 Alarm("f", "v", "", np.nan, np.nan, np.nan)]
        rows = table_rows({"centre@1x": bin_alarms(found)})
        assert sum(r["alarms"] for r in rows) == len(found)

    def test_orphans_get_their_own_labelled_row(self):
        found = [Alarm("f", "v", "", np.nan, np.nan, np.nan)]
        rows = table_rows({"s": bin_alarms(found)})
        orphan = [r for r in rows if r["bin"] == NO_TARGET]
        assert len(orphan) == 1
        assert orphan[0]["alarms"] == 1
        assert orphan[0]["bin_lo"] is None

    def test_rows_carry_series_unit_and_edges(self):
        rows = table_rows({"centre@1x": bin_alarms([Alarm("f", "v", "", 5.0, 0.5,
                                                          10.0)])})
        assert rows[0]["series"] == "centre@1x"
        assert rows[0]["unit"] == REL
        assert (rows[0]["bin_lo"], rows[0]["bin_hi"]) == (REL_EDGES[0], REL_EDGES[1])
