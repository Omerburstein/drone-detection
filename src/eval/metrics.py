"""IoU matching and detection metrics.

Detection is a *matching* problem before it is a regression problem: the model
emits an arbitrary number of boxes and nothing says which one corresponds to
which ground-truth drone. So the pipeline is always

    match predictions to ground truth by IoU  ->  count TP / FP / FN  ->  AP

Everything here follows the COCO protocol: greedy matching in descending
confidence order, 101-point interpolated AP, and mAP averaged over IoU
thresholds 0.50:0.05:0.95.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace

import numpy as np

from .conditions import group_by_condition
from .labels import EvalFrame

# Ground-truth area buckets in pixels^2, as (label, lo, hi).
# Deliberately finer at the small end than COCO's 32^2/96^2 split: on air-to-air
# data almost every target would land in COCO's "small" bucket, which tells us
# nothing about the regime that actually matters here.
AREA_BUCKETS = (
    ("tiny   (<16px)", 0.0, 16.0 ** 2),
    ("small  (16-32)", 16.0 ** 2, 32.0 ** 2),
    ("medium (32-96)", 32.0 ** 2, 96.0 ** 2),
    ("large  (>96px)", 96.0 ** 2, float("inf")),
)

IOU_SWEEP = np.arange(0.5, 0.96, 0.05)  # COCO's 0.50:0.05:0.95
AP_RECALL_POINTS = 101  # COCO's 101-point interpolation grid
EPS = 1e-12  # floor for divisions whose denominator can legitimately be zero
INELIGIBLE = -1.0  # IoU sentinel for a candidate that may not be matched


def f1_score(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall.

    Defined here rather than in the report so the per-condition table and the
    headline block cannot drift apart.
    """
    return 2 * precision * recall / max(precision + recall, EPS)


@dataclass(frozen=True)
class ConditionScore:
    """One scene category's scores, for comparison against published per-category
    figures. GLAD reports precision/recall/F1 per category, so those are carried
    explicitly rather than left to be recomputed by a reader."""

    label: str
    n_frames: int
    n_gt: int
    precision: float
    recall: float
    f1: float
    ap50: float


@dataclass(frozen=True)
class Metrics:
    """The scored result of one run.

    Field order is the JSON schema written to `--json-out` and cited by the
    experiment ledger, so adding a field is fine but reordering is not.
    """

    n_frames: int
    n_gt: int
    n_pred: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    ap50: float
    map: float
    mean_iou: float
    frames_with_miss: int
    by_size: list[tuple[str, int, float]]
    by_condition: list[ConditionScore] = field(default_factory=list)


def _concat(chunks: list[np.ndarray], dtype: type = float) -> np.ndarray:
    """Join per-frame arrays, giving a correctly typed empty array for no frames."""
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=dtype)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes, shaped (len(a), len(b))."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))

    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]

    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), axis=1)
    area_b = np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, EPS), 0.0)


def match_frame(frame: EvalFrame,
                iou_thres: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedily match predictions to ground truth in confidence order.

    Returns (is_true_positive per prediction, matched-GT index per prediction or
    -1, IoU of each match). Each ground-truth box can be claimed once; extra
    predictions on an already-matched target become false positives, which is
    what penalises duplicate boxes.
    """
    n_pred = len(frame.preds)
    tp = np.zeros(n_pred, dtype=bool)
    matched_gt = np.full(n_pred, -1, dtype=int)
    match_iou = np.zeros(n_pred)
    if n_pred == 0 or len(frame.gt_boxes) == 0:
        return tp, matched_gt, match_iou

    ious = iou_matrix(frame.preds.boxes, frame.gt_boxes)
    claimed = np.zeros(len(frame.gt_boxes), dtype=bool)

    for pred_idx in np.argsort(-frame.preds.scores):
        candidates = ious[pred_idx].copy()
        candidates[claimed] = INELIGIBLE
        # Class-aware: a drone box may not be satisfied by a bird prediction.
        candidates[frame.gt_classes != frame.preds.classes[pred_idx]] = INELIGIBLE

        best = int(np.argmax(candidates))
        if candidates[best] >= iou_thres:
            tp[pred_idx] = True
            matched_gt[pred_idx] = best
            match_iou[pred_idx] = candidates[best]
            claimed[best] = True
    return tp, matched_gt, match_iou


def average_precision(tp: np.ndarray, scores: np.ndarray, n_gt: int) -> float:
    """101-point interpolated AP (COCO convention) over all frames' predictions."""
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0

    order = np.argsort(-scores)
    tp_sorted = tp[order]
    cum_tp = np.cumsum(tp_sorted)
    cum_fp = np.cumsum(~tp_sorted)

    recall = cum_tp / n_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, EPS)

    # Make precision monotonically decreasing, then sample at 101 recall points.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    grid = np.linspace(0, 1, AP_RECALL_POINTS)
    return float(np.mean(np.interp(grid, recall, precision,
                                   left=precision[0], right=0.0)))


def _sweep_ap(frames: list[EvalFrame], n_gt: int, iou_sweep: np.ndarray) -> list[float]:
    """AP at each IoU threshold in the sweep; their mean is mAP."""
    aps = []
    for thres in iou_sweep:
        tps, scores = [], []
        for frame in frames:
            tp, _, _ = match_frame(frame, float(thres))
            tps.append(tp)
            scores.append(frame.preds.scores)
        aps.append(average_precision(_concat(tps, bool), _concat(scores), n_gt))
    return aps


@dataclass(frozen=True)
class _PrimaryPass:
    """Per-frame detail gathered at the primary IoU threshold, for the breakdowns."""

    tp: np.ndarray
    matched_ious: np.ndarray
    gt_areas: np.ndarray
    gt_found: np.ndarray
    frames_with_miss: int


def _primary_pass(frames: list[EvalFrame], primary_iou: float) -> _PrimaryPass:
    """Match every frame once at the primary threshold and pool the detail."""
    tp_all, iou_all, gt_areas, gt_found = [], [], [], []
    frames_with_miss = 0
    for frame in frames:
        tp, matched_gt, match_iou = match_frame(frame, primary_iou)
        tp_all.append(tp)
        iou_all.append(match_iou[tp])

        found = np.zeros(len(frame.gt_boxes), dtype=bool)
        found[matched_gt[matched_gt >= 0]] = True
        wh = np.clip(frame.gt_boxes[:, 2:] - frame.gt_boxes[:, :2], 0, None)
        gt_areas.append(wh[:, 0] * wh[:, 1])
        gt_found.append(found)
        if len(frame.gt_boxes) and not found.all():
            frames_with_miss += 1

    return _PrimaryPass(
        tp=_concat(tp_all, bool),
        matched_ious=_concat(iou_all),
        gt_areas=_concat(gt_areas),
        gt_found=_concat(gt_found, bool),
        frames_with_miss=frames_with_miss,
    )


def _recall_by_size(areas: np.ndarray,
                    found: np.ndarray) -> list[tuple[str, int, float]]:
    """Recall within each ground-truth area bucket, as (label, count, recall)."""
    by_size = []
    for label, lo, hi in AREA_BUCKETS:
        sel = (areas >= lo) & (areas < hi)
        by_size.append((label, int(sel.sum()),
                        float(found[sel].mean()) if sel.any() else float("nan")))
    return by_size


def _score(frames: list[EvalFrame], primary_iou: float,
           iou_sweep: np.ndarray = IOU_SWEEP) -> Metrics:
    """Full metric sweep over already-paired frames."""
    n_gt = sum(len(f.gt_boxes) for f in frames)
    if n_gt == 0:
        sys.exit("No ground-truth boxes found -- check --labels points at the right split.")

    aps = _sweep_ap(frames, n_gt, iou_sweep)
    primary = _primary_pass(frames, primary_iou)

    n_tp = int(primary.tp.sum())
    n_fp = int((~primary.tp).sum())
    return Metrics(
        n_frames=len(frames),
        n_gt=n_gt,
        n_pred=int(sum(len(f.preds) for f in frames)),
        tp=n_tp,
        fp=n_fp,
        fn=n_gt - n_tp,
        precision=n_tp / max(n_tp + n_fp, 1),
        recall=n_tp / n_gt,
        ap50=float(aps[0]),
        map=float(np.nanmean(aps)),
        mean_iou=(float(primary.matched_ious.mean())
                  if len(primary.matched_ious) else float("nan")),
        frames_with_miss=primary.frames_with_miss,
        by_size=_recall_by_size(primary.gt_areas, primary.gt_found),
    )


def _score_condition(label: str, frames: list[EvalFrame], primary_iou: float,
                     iou_sweep: np.ndarray) -> ConditionScore:
    """Score one scene category.

    A category with no ground truth is reported with NaN scores rather than
    skipped: an empty category usually means the conditions file and the run
    disagree about which frames exist, and that is worth seeing.
    """
    n_gt = sum(len(f.gt_boxes) for f in frames)
    if n_gt == 0:
        return ConditionScore(label, len(frames), 0, float("nan"), float("nan"),
                              float("nan"), float("nan"))

    scored = _score(frames, primary_iou, iou_sweep)
    return ConditionScore(
        label=label,
        n_frames=scored.n_frames,
        n_gt=scored.n_gt,
        precision=scored.precision,
        recall=scored.recall,
        f1=f1_score(scored.precision, scored.recall),
        ap50=scored.ap50,
    )


def evaluate(frames: list[EvalFrame], primary_iou: float,
             iou_sweep: np.ndarray = IOU_SWEEP,
             conditions: dict[str, str] | None = None) -> Metrics:
    """Score a run, optionally broken down by scene category.

    The per-category pass reuses the same scorer on each subset, so a category's
    numbers are computed exactly as the headline ones are -- there is no second
    implementation to drift.
    """
    metrics = _score(frames, primary_iou, iou_sweep)
    if not conditions:
        return metrics

    grouped = group_by_condition(frames, conditions)
    return replace(metrics, by_condition=[
        _score_condition(label, subset, primary_iou, iou_sweep)
        for label, subset in sorted(grouped.items())
    ])
