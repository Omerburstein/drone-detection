"""One row per detection outcome: the scoring spelled out object by object.

Every metric in `metrics.py` is a summary, and a summary answers one question
while discarding what the next one needs. `precision 0.99` cannot be re-cut by
target size; `recall by size` cannot be re-cut by video; neither survives a
change of matching criterion without re-running the scorer.

So this writes the table those summaries are computed *from*: one row per true
positive, false positive and missed target, carrying both boxes, their sizes,
the offset between them, every condition-axis label, and whatever per-frame
fields the run itself recorded (`branch`, for GLAD). Precision, recall, F1,
recall-by-size and every per-axis breakdown are then `GROUP BY` over this file,
as are the questions the metric block does not answer -- precision against
predicted box size, localisation error against range, false alarms per video.

**Every prediction and every ground-truth box appears exactly once**, as a `tp`
row pairing them, an `fp` row with no target, or an `fn` row with no prediction.
That is what makes counting rows a valid way to recompute the headline numbers
rather than an approximation of them.

CSV rather than JSONL, unlike the rest of the pipeline: this file exists to be
loaded by something else -- pandas, a spreadsheet, a notebook -- and its rows are
genuinely flat and uniform, which the nested `detections.jsonl` rows are not.

The matching itself is `metrics.match_frame`, called here rather than
reimplemented, so a row's `outcome` cannot disagree with the metric block it is
supposed to explain.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .conditions import Axis, video_of
from .metrics import MatchCriterion, as_criterion, box_size, iou_matrix, match_frame
from .labels import EvalFrame

TP, FP, FN = "tp", "fp", "fn"

# Columns every row carries, in the order they are written. Fixed first so a
# reader can rely on the left of the table regardless of which axes a run had.
BASE_COLUMNS = (
    "key",            # frame stem, e.g. phantom05_0001
    "video",          # sequence the frame belongs to
    "outcome",        # tp | fp | fn
    "score",          # prediction confidence (blank for fn)
    "pred_x0", "pred_y0", "pred_x1", "pred_y1",
    "pred_size",      # sqrt(w*h) of the prediction, px
    "pred_area",      # w*h of the prediction, px^2
    "gt_x0", "gt_y0", "gt_x1", "gt_y1",
    "gt_size",        # sqrt(w*h) of the target, px -- the size the buckets bin on
    "gt_area",
    "gt_class",
    "pred_class",
    "iou",            # real IoU of the pair, whatever criterion matched them
    "center_dx", "center_dy",
    "center_dist",    # centre-to-centre distance, px
    "center_dist_rel",  # the same, in multiples of gt_size -- the centre criterion
)

DECIMALS = 4


def _round(value: float) -> float | str:
    """A number for the CSV, or blank where the quantity does not exist.

    NaN is written as an empty cell rather than the string `nan`: a missing
    prediction has no confidence, and an empty cell is what every reader already
    understands as absent.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return round(float(value), DECIMALS)


def _box_columns(prefix: str, box: np.ndarray | None) -> dict[str, Any]:
    """The four corners, size and area of one box under a column prefix."""
    if box is None:
        return {f"{prefix}_{name}": "" for name in
                ("x0", "y0", "x1", "y1", "size", "area")}
    x0, y0, x1, y1 = (float(v) for v in box)
    width, height = max(x1 - x0, 0.0), max(y1 - y0, 0.0)
    return {
        f"{prefix}_x0": _round(x0), f"{prefix}_y0": _round(y0),
        f"{prefix}_x1": _round(x1), f"{prefix}_y1": _round(y1),
        f"{prefix}_size": _round(np.sqrt(width * height)),
        f"{prefix}_area": _round(width * height),
    }


def _geometry(pred: np.ndarray | None, gt: np.ndarray | None) -> dict[str, Any]:
    """Offset between a matched pair, blank when there is no pair.

    `center_dist_rel` is the quantity `--match center` thresholds on, written out
    so the effect of a different `--match-tol` can be read off the file without
    re-scoring: a tolerance of *t* accepts exactly the rows below *t*.
    """
    if pred is None or gt is None:
        return {"center_dx": "", "center_dy": "", "center_dist": "",
                "center_dist_rel": ""}

    pred_centre = (pred[:2] + pred[2:]) / 2
    gt_centre = (gt[:2] + gt[2:]) / 2
    delta = pred_centre - gt_centre
    distance = float(np.linalg.norm(delta))
    size = float(box_size(gt.reshape(1, 4))[0])
    return {
        "center_dx": _round(float(delta[0])),
        "center_dy": _round(float(delta[1])),
        "center_dist": _round(distance),
        "center_dist_rel": _round(distance / size) if size > 0 else "",
    }


def _context(frame: EvalFrame, axes: list[Axis]) -> dict[str, Any]:
    """Axis labels and recorded per-frame fields, shared by every row of a frame."""
    context: dict[str, Any] = {axis.name: axis.label_for(frame.key) for axis in axes}
    context.update({key: value for key, value in frame.extras.items()})
    return context


def frame_rows(frame: EvalFrame, criterion: float | MatchCriterion,
               axes: list[Axis] | None = None) -> Iterator[dict[str, Any]]:
    """Every outcome in one frame, as CSV-ready dicts.

    Order is predictions first (true positives and false alarms in the order the
    detector emitted them), then the targets nothing claimed. Within a frame that
    keeps a prediction next to the row that explains it.
    """
    criterion = as_criterion(criterion)
    axes = axes or []
    context = _context(frame, axes)
    tp, matched_gt, _ = match_frame(frame, criterion)

    # Real IoU for every pair, so a centre-matched row still reports how well the
    # box was actually placed -- the check that keeps centre matching honest.
    ious = iou_matrix(frame.preds.boxes, frame.gt_boxes)

    for index, (box, score, cls) in enumerate(frame.preds):
        gt_index = int(matched_gt[index])
        gt_box = frame.gt_boxes[gt_index] if gt_index >= 0 else None
        yield {
            "key": frame.key,
            "video": video_of(frame.key),
            "outcome": TP if tp[index] else FP,
            "score": _round(float(score)),
            "pred_class": int(cls),
            "gt_class": int(frame.gt_classes[gt_index]) if gt_index >= 0 else "",
            "iou": _round(float(ious[index, gt_index])) if gt_index >= 0 else "",
            **_box_columns("pred", np.asarray(box, dtype=float)),
            **_box_columns("gt", gt_box),
            **_geometry(np.asarray(box, dtype=float), gt_box),
            **context,
        }

    claimed = set(int(i) for i in matched_gt[matched_gt >= 0])
    for gt_index, gt_box in enumerate(frame.gt_boxes):
        if gt_index in claimed:
            continue
        yield {
            "key": frame.key,
            "video": video_of(frame.key),
            "outcome": FN,
            "score": "",
            "pred_class": "",
            "gt_class": int(frame.gt_classes[gt_index]),
            "iou": "",
            **_box_columns("pred", None),
            **_box_columns("gt", gt_box),
            **_geometry(None, gt_box),
            **context,
        }


def columns(frames: list[EvalFrame], axes: list[Axis] | None = None) -> list[str]:
    """Header for the dump: fixed columns, then axes, then recorded frame fields.

    Extra fields are collected across every frame and sorted, so the header does
    not depend on which frame happened to be first -- a run whose first rows
    predate a field would otherwise silently drop it from the whole file.
    """
    axes = axes or []
    extras: set[str] = set()
    for frame in frames:
        extras.update(frame.extras)
    axis_names = [axis.name for axis in axes]
    return [*BASE_COLUMNS, *axis_names,
            *sorted(extras - set(axis_names) - set(BASE_COLUMNS))]


def write_dump(path: Path, frames: list[EvalFrame],
               criterion: float | MatchCriterion,
               axes: list[Axis] | None = None) -> int:
    """Write every outcome to `path` as CSV; return the number of rows.

    Overwrites rather than appends. Unlike the results log, this file is not a
    history: it is the full detail of one scoring, and two criteria's worth of
    rows in one file would be silently double-counted by anything that grouped
    it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    header = columns(frames, axes)
    written = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for frame in frames:
            for row in frame_rows(frame, criterion, axes):
                writer.writerow(row)
                written += 1
    return written
