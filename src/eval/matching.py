"""Deciding which prediction claims which target.

Detection is a *matching* problem before it is a scoring problem: the model
emits an arbitrary number of boxes and nothing says which one corresponds to
which ground-truth drone. Everything in this module answers that question and
stops there -- the counting and the averaging live in `metrics.py`.

Split out because the answer is needed in three places that have nothing to do
with scoring. `records` writes an `outcome` column from it, `curves` bins on it,
and `output/overlay` colours a box on a rendered video by it. A renderer should
not have to import the AP machinery to find out whether a box was a hit, and
before this split it did.

`MatchCriterion` is the one place a match is decided. Nothing reimplements it,
so a dump's `outcome`, the colour of a box on a rendered frame, and the metric
block that explains them cannot disagree.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from .labels import EvalFrame

EPS = 1e-12  # floor for divisions whose denominator can legitimately be zero
INELIGIBLE = -np.inf  # affinity sentinel for a candidate that may not be matched

IOU = "iou"
CENTER = "center"


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


def box_centres(boxes: np.ndarray) -> np.ndarray:
    """The (x, y) centre of each box, shaped (len(boxes), 2)."""
    return (boxes[:, :2] + boxes[:, 2:]) / 2


def center_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise distance between box centres, shaped (len(a), len(b))."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    return np.linalg.norm(box_centres(a)[:, None, :] - box_centres(b)[None, :, :],
                          axis=2)


def nearest_target(pred_boxes: np.ndarray,
                   gt_boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each prediction, the closest ground-truth box and how far off it is.

    Nearest by centre distance, *ignoring whether the pair was matched*. That is
    the point: a false alarm has no matched target by definition, but it still
    has a nearest one, and the distance to it is what separates "a second box on
    the drone we already found" from "a box on a rooftop".

    Returns `(index, distance)`; the index is -1 and the distance NaN for every
    prediction in a frame holding no targets at all, which is a real answer --
    an alarm on an empty frame has no distance, and filling in a zero or a large
    number would invent one.

    The one definition of the quantity. `records` writes it into the dump and
    `alarms` re-derives it from dumps written before that column existed; both
    call this, so the two cannot drift.
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return (np.full(len(pred_boxes), -1, dtype=int),
                np.full(len(pred_boxes), np.nan))
    distances = center_distance(pred_boxes, gt_boxes)
    index = distances.argmin(axis=1)
    return index, distances[np.arange(len(pred_boxes)), index]


def box_size(boxes: np.ndarray) -> np.ndarray:
    """The side of the equal-area square, sqrt(w*h), per box.

    The same notion of "size" `AREA_BUCKETS` bins on, so a tolerance expressed in
    target sizes means the same thing here as in the recall-by-size table.
    """
    wh = np.clip(boxes[:, 2:] - boxes[:, :2], 0, None)
    return np.sqrt(wh[:, 0] * wh[:, 1])


@dataclass(frozen=True)
class MatchCriterion:
    """How a prediction is judged to have claimed a ground-truth box.

    Two criteria, because one number cannot serve both jobs at these target
    sizes:

    `iou` — overlap ratio, threshold `value` (COCO's rule). Correct for asking
    *how well is this box placed*, and the only thing comparable to published
    mAP. But it collapses on tiny targets: at 12 px a 2 px centre offset drops
    IoU below 0.5 while the detection is plainly correct, so a perfectly good
    detector is scored as both a miss and a false alarm.

    `center` — centre-to-centre distance, matched when it is within `value`
    target sizes (`sqrt(w*h)` of the ground-truth box). Answers *did the detector
    find the drone*, which is the question a false-alarm rate is really asking.
    Size-relative by construction, so it is equally strict on a 10 px and a
    100 px target — where a fixed IoU threshold is not.

    Both expose the same interface: an affinity that is **≥ 0 exactly when the
    pair is an acceptable match**, and larger for a better one, so greedy
    matching is identical under either.
    """

    kind: str
    value: float

    @property
    def label(self) -> str:
        """How this criterion is named in the printed report and the JSON."""
        if self.kind == IOU:
            return f"IoU@{self.value:.2f}"
        return f"centre@{self.value:g}x target size"

    def affinity(self, pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
        """(n_pred, n_gt) match quality; negative means "not a match".

        Shifting each rule so its accept region starts at zero is what lets one
        matching loop serve both.
        """
        if self.kind == IOU:
            return iou_matrix(pred_boxes, gt_boxes) - self.value

        sizes = box_size(gt_boxes)
        allowed = np.maximum(self.value * sizes, EPS)
        proximity = 1.0 - center_distance(pred_boxes, gt_boxes) / allowed
        # A degenerate ground-truth box has no scale to normalise by, so nothing
        # matches it rather than everything. Masking the result rather than the
        # divisor matters: a prediction landing exactly on a zero-size box is at
        # distance 0, which no amount of shrinking the tolerance would reject.
        return np.where(sizes[None, :] > 0, proximity, INELIGIBLE)


def as_criterion(spec: float | MatchCriterion) -> MatchCriterion:
    """Coerce a bare threshold to a criterion.

    A float has always meant "IoU at this threshold" throughout this module, and
    still does, so callers that only care about IoU need not know this type
    exists.
    """
    return spec if isinstance(spec, MatchCriterion) else MatchCriterion(IOU, float(spec))


def match_frame(frame: EvalFrame,
                criterion: float | MatchCriterion) -> tuple[np.ndarray, np.ndarray,
                                                            np.ndarray]:
    """Greedily match predictions to ground truth in confidence order.

    Returns (is_true_positive per prediction, matched-GT index per prediction or
    -1, IoU of each match). Each ground-truth box can be claimed once; extra
    predictions on an already-matched target become false positives, which is
    what penalises duplicate boxes.

    **The reported IoU is always IoU**, whatever criterion decided the match.
    That keeps localisation quality a real measurement rather than a restatement
    of the matching rule -- under `center` matching, mean IoU is the only thing
    still saying how well the boxes are placed.
    """
    criterion = as_criterion(criterion)
    n_pred = len(frame.preds)
    tp = np.zeros(n_pred, dtype=bool)
    matched_gt = np.full(n_pred, -1, dtype=int)
    match_iou = np.zeros(n_pred)
    if n_pred == 0 or len(frame.gt_boxes) == 0:
        return tp, matched_gt, match_iou

    affinity = criterion.affinity(frame.preds.boxes, frame.gt_boxes)
    ious = iou_matrix(frame.preds.boxes, frame.gt_boxes)
    claimed = np.zeros(len(frame.gt_boxes), dtype=bool)

    for pred_idx in np.argsort(-frame.preds.scores):
        candidates = affinity[pred_idx].copy()
        candidates[claimed] = INELIGIBLE
        # Class-aware: a drone box may not be satisfied by a bird prediction.
        candidates[frame.gt_classes != frame.preds.classes[pred_idx]] = INELIGIBLE

        best = int(np.argmax(candidates))
        if candidates[best] >= 0:
            tp[pred_idx] = True
            matched_gt[pred_idx] = best
            match_iou[pred_idx] = ious[pred_idx, best]
            claimed[best] = True
    return tp, matched_gt, match_iou



def criterion_from_args(args: argparse.Namespace) -> MatchCriterion | None:
    """The matching rule the `--match` choices select.

    Here rather than in a CLI because two of them build it, and a criterion
    assembled two ways is a criterion that can disagree with itself -- the exact
    failure `MatchCriterion` exists to prevent. The *flags* stay local to each
    parser: `src.evaluate` explains at length why `center` is the default, and
    `src.render_video` defers to it in three lines. Two help strings cannot
    disagree about a number; two builders can.

    `None` when the caller passed `--no-labels`: with no ground truth there is
    nothing to match, and the renderer draws boxes without judging them.
    """
    if getattr(args, "no_labels", False):
        return None
    return (MatchCriterion(IOU, args.iou) if args.match == IOU
            else MatchCriterion(CENTER, args.match_tol))
