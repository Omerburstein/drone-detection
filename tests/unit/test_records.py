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
