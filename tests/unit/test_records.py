"""The per-object dump: every prediction and every target, exactly once.

The dump's whole claim is that the metric block can be recomputed from it. These
tests hold that claim up: row counts against `tp`/`fp`/`fn`, one row per object
with nothing duplicated or dropped, and the geometry columns that let a
different `--match-tol` be read off the file without re-scoring.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from src.algo.detections import Detections
from src.eval.conditions import Axis
from src.eval.labels import EvalFrame
from src.eval.metrics import CENTER, IOU, MatchCriterion, evaluate
from src.eval.records import BASE_COLUMNS, columns, frame_rows, write_dump


def box(x0, y0, x1, y1):
    """One xyxy box as the array shape the frames carry."""
    return [float(x0), float(y0), float(x1), float(y1)]


def make_frame(key, gt, preds, scores=None, extras=None):
    """An EvalFrame from plain lists, defaulting every prediction to full confidence."""
    preds = np.array(preds, dtype=float).reshape(-1, 4)
    scores = np.ones(len(preds)) if scores is None else np.array(scores, dtype=float)
    return EvalFrame(
        key=key,
        gt_boxes=np.array(gt, dtype=float).reshape(-1, 4),
        gt_classes=np.zeros(len(gt), dtype=int),
        preds=Detections(preds, scores, np.zeros(len(preds), dtype=int)),
        extras=extras or {},
    )


@pytest.fixture
def frames():
    """Three frames covering a hit, a false alarm, a miss and an empty frame."""
    return [
        # One well-placed prediction and one on nothing.
        make_frame("vid_0001", [box(100, 100, 120, 120)],
                   [box(101, 101, 121, 121), box(500, 500, 520, 520)],
                   scores=[0.9, 0.4], extras={"branch": "local yolo"}),
        # A target nothing claimed.
        make_frame("vid_0002", [box(200, 200, 212, 212)], [],
                   extras={"branch": "local miss"}),
        # No ground truth, no predictions.
        make_frame("vid_0003", [], [], extras={"branch": "global miss"}),
    ]


def test_every_object_appears_exactly_once(frames):
    """Row count equals predictions plus unclaimed targets, per frame."""
    rows = [r for f in frames for r in frame_rows(f, MatchCriterion(CENTER, 1.0))]
    # frame 1 contributes a hit and a false alarm, frame 2 an unclaimed target,
    # frame 3 nothing at all.
    assert len(rows) == 3
    assert [r["outcome"] for r in rows] == ["tp", "fp", "fn"]


def test_counts_reproduce_the_metric_block(frames):
    """tp/fp/fn row counts equal what `evaluate` reports. This is the point."""
    criterion = MatchCriterion(CENTER, 1.0)
    metrics = evaluate(frames, criterion)
    rows = [r for f in frames for r in frame_rows(f, criterion)]

    counted = {outcome: sum(r["outcome"] == outcome for r in rows)
               for outcome in ("tp", "fp", "fn")}
    assert counted == {"tp": metrics.tp, "fp": metrics.fp, "fn": metrics.fn}


def test_counts_reproduce_the_metric_block_under_iou(frames):
    """The same, under COCO's rule -- the dump follows the criterion it was given."""
    criterion = MatchCriterion(IOU, 0.5)
    metrics = evaluate(frames, criterion)
    rows = [r for f in frames for r in frame_rows(f, criterion)]

    counted = {outcome: sum(r["outcome"] == outcome for r in rows)
               for outcome in ("tp", "fp", "fn")}
    assert counted == {"tp": metrics.tp, "fp": metrics.fp, "fn": metrics.fn}


def test_a_miss_and_a_false_alarm_leave_the_other_side_blank(frames):
    """An fp row has no target columns; an fn row has no prediction columns."""
    rows = [r for f in frames for r in frame_rows(f, MatchCriterion(CENTER, 1.0))]
    false_alarm = next(r for r in rows if r["outcome"] == "fp")
    miss = next(r for r in rows if r["outcome"] == "fn")

    assert false_alarm["gt_size"] == "" and false_alarm["iou"] == ""
    assert false_alarm["pred_size"] != ""
    assert miss["pred_size"] == "" and miss["score"] == ""
    assert miss["gt_size"] != ""


def test_iou_is_reported_even_when_centre_matching_decided(frames):
    """Localisation quality stays a real measurement, not a restatement of the rule."""
    row = next(r for r in frame_rows(frames[0], MatchCriterion(CENTER, 1.0))
               if r["outcome"] == "tp")
    assert 0.0 < row["iou"] < 1.0


def test_center_dist_rel_is_the_tolerance_the_criterion_thresholds_on():
    """A row's `center_dist_rel` predicts whether a given --match-tol accepts it."""
    # 12 px target, prediction centred 6 px away: 0.5 target sizes.
    frame = make_frame("vid_0001", [box(100, 100, 112, 112)],
                       [box(106, 100, 118, 112)])
    row = next(iter(frame_rows(frame, MatchCriterion(CENTER, 1.0))))
    assert row["center_dist_rel"] == pytest.approx(0.5)

    assert next(iter(frame_rows(frame, MatchCriterion(CENTER, 0.6))))["outcome"] == "tp"
    assert next(iter(frame_rows(frame, MatchCriterion(CENTER, 0.4))))["outcome"] == "fp"


def test_size_columns_are_the_side_of_the_equal_area_square():
    """`gt_size` is sqrt(w*h) -- the same notion the recall buckets bin on."""
    frame = make_frame("vid_0001", [box(0, 0, 40, 10)], [])
    row = next(iter(frame_rows(frame, MatchCriterion(CENTER, 1.0))))
    assert row["gt_size"] == pytest.approx(20.0)
    assert row["gt_area"] == pytest.approx(400.0)


def test_axis_labels_and_recorded_fields_land_on_every_row(frames):
    """Condition axes and per-frame extras are carried through to each row."""
    axes = [Axis("scene_category", "video", {"vid": "ordinary"})]
    rows = list(frame_rows(frames[0], MatchCriterion(CENTER, 1.0), axes))
    assert all(r["scene_category"] == "ordinary" for r in rows)
    assert all(r["branch"] == "local yolo" for r in rows)


def test_video_is_recovered_from_the_frame_key(frames):
    """Grouping by sequence has to work without a separate lookup."""
    rows = list(frame_rows(frames[0], MatchCriterion(CENTER, 1.0)))
    assert all(r["video"] == "vid" for r in rows)


def test_header_collects_extras_from_every_frame_not_just_the_first():
    """A field that first appears late still gets a column."""
    frames = [make_frame("vid_0001", [], []),
              make_frame("vid_0002", [], [], extras={"branch": "local yolo"})]
    header = columns(frames)
    assert header[:len(BASE_COLUMNS)] == list(BASE_COLUMNS)
    assert "branch" in header


def test_written_csv_round_trips(tmp_path, frames):
    """The file on disk carries the same rows the generator produced."""
    axes = [Axis("scene_category", "video", {"vid": "ordinary"})]
    path = tmp_path / "matches.csv"
    written = write_dump(path, frames, MatchCriterion(CENTER, 1.0), axes)

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert written == len(rows) == 3
    assert [r["outcome"] for r in rows] == ["tp", "fp", "fn"]
    assert rows[0]["scene_category"] == "ordinary"
    assert float(rows[0]["gt_size"]) == pytest.approx(20.0)


def test_write_overwrites_rather_than_appending(tmp_path, frames):
    """Two criteria's rows in one file would be double-counted by any GROUP BY."""
    path = tmp_path / "matches.csv"
    write_dump(path, frames, MatchCriterion(CENTER, 1.0))
    write_dump(path, frames, MatchCriterion(CENTER, 1.0))

    with path.open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 3


class TestNearestTargetColumns:
    """`nearest_gt_*`: the distance to the closest target, matched or not.

    Distinct from `center_dist`, which is the offset from the target a prediction
    *claimed*. A false alarm claims nothing and so has no `center_dist` at all —
    these columns are the only place its geometry is recorded, and they are what
    `src.eval.alarms` bins.
    """

    def test_false_alarm_gets_a_distance_where_center_dist_is_blank(self):
        frame = make_frame("f", gt=[box(100, 100, 110, 110)],
                           preds=[box(100, 100, 110, 110), box(200, 100, 210, 110)])
        rows = list(frame_rows(frame, MatchCriterion(CENTER, 1.0)))
        alarm = [r for r in rows if r["outcome"] == "fp"][0]

        assert alarm["center_dist"] == ""          # nothing was claimed
        assert alarm["nearest_gt_dist"] == pytest.approx(100.0)
        assert alarm["nearest_gt_dist_rel"] == pytest.approx(10.0)
        assert alarm["nearest_gt_size"] == pytest.approx(10.0)

    def test_nearest_is_nearest_not_matched(self):
        """Two targets: the alarm is measured against whichever is closer."""
        frame = make_frame("f", gt=[box(0, 0, 10, 10), box(300, 0, 310, 10)],
                           preds=[box(0, 0, 10, 10), box(280, 0, 290, 10)])
        rows = list(frame_rows(frame, MatchCriterion(CENTER, 1.0)))
        alarm = [r for r in rows if r["outcome"] == "fp"][0]
        assert alarm["nearest_gt_dist"] == pytest.approx(20.0)

    def test_true_positive_carries_it_too(self):
        """So a prediction matched to something other than its nearest is visible."""
        frame = make_frame("f", gt=[box(100, 100, 110, 110)],
                           preds=[box(102, 100, 112, 110)])
        row = list(frame_rows(frame, MatchCriterion(CENTER, 1.0)))[0]
        assert row["outcome"] == "tp"
        assert row["nearest_gt_dist"] == pytest.approx(2.0)

    def test_missed_target_has_no_prediction_and_so_no_distance(self):
        frame = make_frame("f", gt=[box(100, 100, 110, 110)], preds=[])
        row = list(frame_rows(frame, MatchCriterion(CENTER, 1.0)))[0]
        assert row["outcome"] == "fn"
        assert row["nearest_gt_dist"] == ""
        assert row["nearest_gt_size"] == ""

    def test_alarm_on_an_empty_frame_is_blank_not_zero(self):
        """A zero would read as 'landed exactly on a drone' — the opposite."""
        frame = make_frame("f", gt=[], preds=[box(200, 100, 210, 110)])
        row = list(frame_rows(frame, MatchCriterion(CENTER, 1.0)))[0]
        assert row["outcome"] == "fp"
        assert row["nearest_gt_dist"] == ""
        assert row["nearest_gt_dist_rel"] == ""

    def test_columns_are_in_the_written_header(self, frames, tmp_path):
        path = tmp_path / "dump.csv"
        write_dump(path, frames, MatchCriterion(CENTER, 1.0))
        header = list(csv.DictReader(path.open(encoding="utf-8")).fieldnames)
        for name in ("nearest_gt_dist", "nearest_gt_dist_rel", "nearest_gt_size"):
            assert name in header
            assert name in BASE_COLUMNS

    def test_a_dumped_alarm_round_trips_into_the_alarms_module(self, tmp_path):
        """The read path in `alarms` and the write path here are one quantity.

        Equal to the dump's own precision, not to the bit: the CSV rounds to
        `DECIMALS`, so the recorded column and a fresh derivation agree to 1e-4.
        That is the tolerance every consumer of the file already lives with.
        """
        from src.eval.alarms import frame_alarms
        from src.eval.curves import load_dump

        frame = make_frame("f", gt=[box(100, 100, 112, 112)],
                           preds=[box(160, 130, 172, 142)])
        path = tmp_path / "dump.csv"
        write_dump(path, [frame], MatchCriterion(CENTER, 1.0))

        rows = load_dump(path)
        recorded = frame_alarms(rows)[0]
        derived = frame_alarms([{k: v for k, v in r.items()
                                 if not k.startswith("nearest_")} for r in rows])[0]
        assert recorded.distance == pytest.approx(derived.distance, abs=1e-4)
        assert recorded.distance_rel == pytest.approx(derived.distance_rel, abs=1e-4)
        assert recorded.distance == pytest.approx(np.hypot(60.0, 30.0), abs=1e-4)
