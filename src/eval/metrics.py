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

from dataclasses import dataclass, field, replace

import numpy as np

from ..errors import UsageError
from .conditions import LEGACY_KEY, Axis, as_axes, group_by_axis
from .labels import EvalFrame
from .matching import (EPS, IOU, MatchCriterion, _concat, as_criterion,
                       box_centres, box_size, match_frame)

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

def f1_score(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall.

    Defined here rather than in the report so the per-condition table and the
    headline block cannot drift apart.
    """
    return 2 * precision * recall / max(precision + recall, EPS)


@dataclass(frozen=True)
class ConditionScore:
    """One bucket's scores, for comparison against published per-category
    figures. GLAD reports precision/recall/F1 per category, so those are carried
    explicitly rather than left to be recomputed by a reader.

    `axis` names which split the bucket belongs to. It is last and defaulted so
    the JSON key order of every field that predates the multi-axis breakdown is
    unchanged, and so a bare `scene_category` run reads exactly as before. `far`
    is appended after it for the same reason.

    `far` is per bucket what it is overall: this bucket's false alarms over this
    bucket's frames. Precision says what fraction of the alarms were wrong; `far`
    says how often the alarm goes off at all, and a bucket can look precise while
    alarming constantly.
    """

    label: str
    n_frames: int
    n_gt: int
    precision: float
    recall: float
    f1: float
    ap50: float
    axis: str = LEGACY_KEY
    far: float = float("nan")


@dataclass(frozen=True)
class LocError:
    """Size-normalised localisation error within one ground-truth size bucket.

    The distance between a matched pair's centres divided by the target's own
    size (`sqrt(w*h)`), so 0.5 means "half a drone-width off centre" whether the
    drone is 10 px or 100 px across. A raw pixel offset cannot be compared across
    buckets -- 2 px is fatal on a 12 px target and invisible on a 90 px one, and
    that asymmetry is precisely why a fixed IoU threshold behaves so differently
    by size on this data.

    `n` counts **matched pairs, not targets**: an unmatched target has no offset,
    so it is absent here rather than counted as an infinite error. Read this
    beside the bucket's recall, never instead of it -- a detector that finds the
    one easy target in a bucket and misses the rest posts a beautiful error.
    """

    label: str
    n: int          # matched pairs contributing -- not the bucket's target count
    mean: float     # in multiples of the target's own size
    median: float   # the honest centre: the mean is dragged by boundary near-misses
    p90: float      # the tail, which is where a tracker loses lock


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
    # Which rule decided a match. Recorded because P/R/F1 are not comparable
    # across criteria, and a JSON file outlives the command that produced it.
    criterion: str = f"IoU@{0.5:.2f}"
    # Appended after `criterion` so every key that predates them keeps its
    # position in the JSON. NaN rather than 0.0 as the default: an unset
    # false-alarm rate must never read as a perfect one.
    far: float = float("nan")          # false alarms per frame -- fp / n_frames
    loc_err: float = float("nan")      # mean centre offset, in target sizes
    loc_err_p90: float = float("nan")
    loc_by_size: list[LocError] = field(default_factory=list)



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


def _sweep_ap(frames: list[EvalFrame], n_gt: int,
              criteria: list[MatchCriterion]) -> list[float]:
    """AP under each criterion in turn; for the COCO IoU sweep, their mean is mAP."""
    aps = []
    for criterion in criteria:
        tps, scores = [], []
        for frame in frames:
            tp, _, _ = match_frame(frame, criterion)
            tps.append(tp)
            scores.append(frame.preds.scores)
        aps.append(average_precision(_concat(tps, bool), _concat(scores), n_gt))
    return aps


def _ap_criteria(primary: MatchCriterion,
                 iou_sweep: np.ndarray) -> tuple[list[MatchCriterion], bool]:
    """The criteria to compute AP under, and whether their mean is a valid mAP.

    mAP@0.50:0.95 is defined by averaging over *IoU* thresholds. Under centre
    matching there is no such axis -- sweeping the distance tolerance instead
    would produce a number that looks like mAP, is not, and would be compared to
    published mAP by someone eventually. So only the primary point is computed
    and mAP is reported as NaN.
    """
    if primary.kind == IOU:
        return [MatchCriterion(IOU, float(t)) for t in iou_sweep], True
    return [primary], False


@dataclass(frozen=True)
class _PrimaryPass:
    """Per-frame detail gathered at the primary criterion, for the breakdowns."""

    tp: np.ndarray
    matched_ious: np.ndarray
    gt_areas: np.ndarray
    gt_found: np.ndarray
    frames_with_miss: int
    # One entry per matched pair, in the same order: the pair's centre offset in
    # multiples of the target's size, and the area of the target it was matched
    # to. Kept as a pair of parallel arrays so the error can be binned on the
    # target's size with exactly the buckets recall already uses.
    match_offsets: np.ndarray
    match_gt_areas: np.ndarray


def _primary_pass(frames: list[EvalFrame],
                  primary: MatchCriterion) -> _PrimaryPass:
    """Match every frame once at the primary criterion and pool the detail."""
    tp_all, iou_all, gt_areas, gt_found = [], [], [], []
    offsets, offset_areas = [], []
    frames_with_miss = 0
    for frame in frames:
        tp, matched_gt, match_iou = match_frame(frame, primary)
        tp_all.append(tp)
        iou_all.append(match_iou[tp])

        found = np.zeros(len(frame.gt_boxes), dtype=bool)
        found[matched_gt[matched_gt >= 0]] = True
        wh = np.clip(frame.gt_boxes[:, 2:] - frame.gt_boxes[:, :2], 0, None)
        areas = wh[:, 0] * wh[:, 1]
        gt_areas.append(areas)
        gt_found.append(found)
        if len(frame.gt_boxes) and not found.all():
            frames_with_miss += 1

        pairs = matched_gt[tp]
        offsets.append(_relative_offset(frame.preds.boxes[tp], frame.gt_boxes[pairs]))
        offset_areas.append(areas[pairs])

    return _PrimaryPass(
        tp=_concat(tp_all, bool),
        matched_ious=_concat(iou_all),
        gt_areas=_concat(gt_areas),
        gt_found=_concat(gt_found, bool),
        frames_with_miss=frames_with_miss,
        match_offsets=_concat(offsets),
        match_gt_areas=_concat(offset_areas),
    )


def _relative_offset(preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
    """Centre offset of each already-matched pair, in multiples of target size.

    Paired element-wise, not pairwise: these boxes are matched already, so the
    (n, n) grid `center_distance` builds would be thrown away but for its
    diagonal.

    A degenerate target has no scale to normalise by and yields NaN rather than
    an infinity, so it drops out of a mean instead of destroying it. Under centre
    matching such a target cannot be matched at all; under IoU it can, which is
    the case this guard exists for.
    """
    if len(preds) == 0:
        return np.zeros(0)
    distance = np.linalg.norm(box_centres(preds) - box_centres(gts), axis=1)
    sizes = box_size(gts)
    return np.where(sizes > 0, distance / np.maximum(sizes, EPS), np.nan)


def _loc_error_by_size(areas: np.ndarray, offsets: np.ndarray) -> list[LocError]:
    """Size-normalised centre offset within each ground-truth area bucket.

    Bins on the *target's* area, the same quantity recall bins on, so a bucket's
    error and its recall are two statements about one set of drones. Binning on
    the predicted box instead would put a badly oversized box in the wrong
    bucket -- exactly the boxes whose offset most needs reading.
    """
    by_size = []
    for label, lo, hi in AREA_BUCKETS:
        sel = (areas >= lo) & (areas < hi) & ~np.isnan(offsets)
        bucket = offsets[sel]
        by_size.append(LocError(
            label, int(len(bucket)),
            float(bucket.mean()) if len(bucket) else float("nan"),
            float(np.median(bucket)) if len(bucket) else float("nan"),
            float(np.percentile(bucket, 90)) if len(bucket) else float("nan"),
        ))
    return by_size


def _recall_by_size(areas: np.ndarray,
                    found: np.ndarray) -> list[tuple[str, int, float]]:
    """Recall within each ground-truth area bucket, as (label, count, recall)."""
    by_size = []
    for label, lo, hi in AREA_BUCKETS:
        sel = (areas >= lo) & (areas < hi)
        by_size.append((label, int(sel.sum()),
                        float(found[sel].mean()) if sel.any() else float("nan")))
    return by_size


def _score(frames: list[EvalFrame], primary: float | MatchCriterion,
           iou_sweep: np.ndarray = IOU_SWEEP) -> Metrics:
    """Full metric sweep over already-paired frames."""
    primary = as_criterion(primary)
    n_gt = sum(len(f.gt_boxes) for f in frames)
    if n_gt == 0:
        raise UsageError("No ground-truth boxes found -- check --labels points at the right split.")

    criteria, sweep_is_map = _ap_criteria(primary, iou_sweep)
    aps = _sweep_ap(frames, n_gt, criteria)
    pass_ = _primary_pass(frames, primary)

    n_tp = int(pass_.tp.sum())
    n_fp = int((~pass_.tp).sum())
    offsets = pass_.match_offsets[~np.isnan(pass_.match_offsets)]
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
        map=float(np.nanmean(aps)) if sweep_is_map else float("nan"),
        mean_iou=(float(pass_.matched_ious.mean())
                  if len(pass_.matched_ious) else float("nan")),
        frames_with_miss=pass_.frames_with_miss,
        by_size=_recall_by_size(pass_.gt_areas, pass_.gt_found),
        criterion=primary.label,
        # Per frame, not per second: frame rate is a property of the run, not of
        # the scoring, and a rate in Hz would silently change meaning the moment
        # the same JSONL were re-scored after a --stride change.
        far=n_fp / max(len(frames), 1),
        loc_err=float(offsets.mean()) if len(offsets) else float("nan"),
        loc_err_p90=(float(np.percentile(offsets, 90)) if len(offsets)
                     else float("nan")),
        loc_by_size=_loc_error_by_size(pass_.match_gt_areas, pass_.match_offsets),
    )


def _score_condition(label: str, frames: list[EvalFrame],
                     primary: MatchCriterion,
                     iou_sweep: np.ndarray,
                     axis: str = LEGACY_KEY) -> ConditionScore:
    """Score one bucket of one axis.

    A bucket with no ground truth is reported with NaN scores rather than
    skipped: an empty bucket usually means the conditions file and the run
    disagree about which frames exist, and that is worth seeing.
    """
    n_gt = sum(len(f.gt_boxes) for f in frames)
    if n_gt == 0:
        return ConditionScore(label, len(frames), 0, float("nan"), float("nan"),
                              float("nan"), float("nan"), axis, float("nan"))

    scored = _score(frames, primary, iou_sweep)
    return ConditionScore(
        label=label,
        n_frames=scored.n_frames,
        n_gt=scored.n_gt,
        precision=scored.precision,
        recall=scored.recall,
        f1=f1_score(scored.precision, scored.recall),
        ap50=scored.ap50,
        axis=axis,
        far=scored.far,
    )


def evaluate(frames: list[EvalFrame], primary: float | MatchCriterion,
             iou_sweep: np.ndarray = IOU_SWEEP,
             conditions: dict[str, str] | list[Axis] | None = None) -> Metrics:
    """Score a run, optionally broken down along one or more condition axes.

    `primary` is the matching criterion; a bare float means IoU at that
    threshold. `conditions` is a list of `Axis`, or a bare video -> category map
    for the single-axis case. Every bucket reuses the same scorer on its own
    subset, so a bucket's numbers are computed exactly as the headline ones are
    -- there is no second implementation to drift.

    All axes are returned concatenated in `by_condition`, each score naming its
    axis, so a reader that predates the multi-axis split still sees a flat list.
    """
    primary = as_criterion(primary)
    metrics = _score(frames, primary, iou_sweep)
    axes = as_axes(conditions)
    if not axes:
        return metrics

    return replace(metrics, by_condition=[
        _score_condition(label, subset, primary, iou_sweep, axis.name)
        for axis in axes
        for label, subset in group_by_axis(frames, axis).items()
    ])