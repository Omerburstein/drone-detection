"""Ground truth and predictions drawn on the same frame.

`annotate.py` draws what the detector said while a run is in progress. This
draws what the detector said *against what was true*, after scoring — so every
box carries its match outcome and the picture cannot disagree with the metric
block: the verdict comes from `match_frame`, the same function `src.eval.records`
and `src.eval.curves` call.

Two things this module does that a rectangle-per-box would not, both forced by
target size:

- **Boxes are inflated before drawing.** A border on a 9 px drone covers the
  drone. The rectangle is pushed outward instead, and ground truth is pushed
  further out than the prediction so the two stay distinguishable when they
  nearly coincide — which, on a true positive, they always do.
- **A magnified inset.** At 10-30 px against 1920x1080 the target is a few
  screen pixels; the inset is what makes the frame readable as evidence rather
  than as a green speck.

Nothing here touches the filesystem — a canvas goes in, a canvas comes out.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..eval.labels import EvalFrame
from ..eval.metrics import MatchCriterion, as_criterion, match_frame

# BGR, matching the caption legend below.
GT_COLOUR = (90, 220, 90)  # green: where the drone actually is
TP_COLOUR = (255, 190, 40)  # blue: a prediction that claimed a target
FP_COLOUR = (60, 60, 255)  # red: a prediction that claimed nothing
MISS_COLOUR = (40, 170, 255)  # orange: a target nothing claimed
INK = (255, 255, 255)
STRIP = (24, 24, 24)

GT_PAD = 5  # px pushed outward from the true box before drawing
PRED_PAD = 2
THICKNESS = 1
FONT = cv2.FONT_HERSHEY_SIMPLEX
CAPTION_SCALE = 0.55
CAPTION_HEIGHT = 58  # two text lines plus padding
INSET_MARGIN = 16
LEGEND = "green  ground truth      blue  matched prediction      red  false alarm      orange  missed target"


@dataclass(frozen=True)
class Style:
    """How much of the frame the overlay spends on making a speck visible.

    A `zoom` of 0 disables the inset entirely, for footage whose targets are
    large enough to read unaided.
    """

    zoom: int = 5
    span: int = 110  # side of the source window, in original pixels
    caption: bool = True


@dataclass(frozen=True)
class View:
    """A mapping from original-frame pixels to canvas pixels.

    The full frame is the identity view and the inset is the same drawing code
    under a shifted, scaled one. That is what keeps one `_draw_box` serving both
    instead of two near-copies free to drift apart.
    """

    origin: tuple[float, float] = (0.0, 0.0)
    scale: float = 1.0
    offset: tuple[int, int] = (0, 0)

    def point(self, x: float, y: float) -> tuple[int, int]:
        """Map one original-frame point onto the canvas."""
        return (int(round((x - self.origin[0]) * self.scale + self.offset[0])),
                int(round((y - self.origin[1]) * self.scale + self.offset[1])))


@dataclass(frozen=True)
class Verdict:
    """One frame's match outcome, in the shape the overlay needs to colour it."""

    tp: np.ndarray  # per prediction: did it claim a target
    match_iou: np.ndarray  # per prediction: IoU with the target it claimed
    found: np.ndarray  # per target: was it claimed by any prediction

    @property
    def summary(self) -> str:
        """`2 TP, 1 FP, 1 missed` — this frame's contribution to the totals."""
        tp, fp = int(self.tp.sum()), int((~self.tp).sum())
        missed = int((~self.found).sum())
        return f"{tp} TP, {fp} FP, {missed} missed"


def judge(frame: EvalFrame, criterion: float | MatchCriterion) -> Verdict:
    """Score one frame with the shared matcher, keyed for drawing."""
    tp, matched_gt, match_iou = match_frame(frame, criterion)
    found = np.zeros(len(frame.gt_boxes), dtype=bool)
    found[matched_gt[tp]] = True
    return Verdict(tp=tp, match_iou=match_iou, found=found)


def _draw_box(canvas: np.ndarray, box: np.ndarray, colour: tuple[int, int, int],
              view: View, pad: float, label: str = "", below: bool = False) -> None:
    """One outward-inflated rectangle, with an optional label beside it.

    `below` puts the text under the box instead of over it: a true positive
    sits its prediction inside the ground truth, and two labels on the same
    edge overprint each other.
    """
    x1, y1, x2, y2 = box
    p1 = view.point(x1 - pad, y1 - pad)
    p2 = view.point(x2 + pad, y2 + pad)
    # Not anti-aliased: on a 9 px target an AA edge blends the box into the
    # thing it is pointing at, and the exact colour is the whole legend.
    cv2.rectangle(canvas, p1, p2, colour, THICKNESS)
    if label:
        y = min(canvas.shape[0] - 4, p2[1] + 16) if below else max(12, p1[1] - 6)
        cv2.putText(canvas, label, (p1[0], y), FONT, 0.45, colour, 1, cv2.LINE_AA)


def _pred_label(index: int, frame: EvalFrame, verdict: Verdict) -> str:
    """`pred 0.87 TP IoU 0.48` — confidence, outcome, and how well it sits."""
    outcome = "TP" if verdict.tp[index] else "FP"
    iou = f" IoU {verdict.match_iou[index]:.2f}" if verdict.tp[index] else ""
    return f"pred {frame.preds.scores[index]:.2f} {outcome}{iou}"


def _draw_frame_boxes(canvas: np.ndarray, frame: EvalFrame, verdict: Verdict,
                      view: View, labels: bool) -> None:
    """Every ground-truth and predicted box for one frame, in one view."""
    for i, box in enumerate(frame.gt_boxes):
        found = bool(verdict.found[i])
        text = ("gt" if found else "gt missed") if labels else ""
        _draw_box(canvas, box, GT_COLOUR if found else MISS_COLOUR, view, GT_PAD, text)
    for i, box in enumerate(frame.preds.boxes):
        colour = TP_COLOUR if verdict.tp[i] else FP_COLOUR
        _draw_box(canvas, box, colour, view, PRED_PAD,
                  _pred_label(i, frame, verdict) if labels else "", below=True)


def crop_window(frame: EvalFrame, shape: tuple[int, int],
                span: int) -> tuple[int, int, int]:
    """Square source window for the inset: `(x0, y0, span)`, clamped to the frame.

    Centred on the ground truth where there is any and on the prediction
    otherwise — a false alarm on an empty frame is exactly the case worth
    looking at, and centring on the frame would hide it.
    """
    height, width = shape
    span = int(min(span, width, height))
    boxes = frame.gt_boxes if len(frame.gt_boxes) else frame.preds.boxes
    if len(boxes):
        centres = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2,
                                   (boxes[:, 1] + boxes[:, 3]) / 2])
        cx, cy = float(centres[:, 0].mean()), float(centres[:, 1].mean())
    else:
        cx, cy = width / 2, height / 2
    x0 = int(min(max(cx - span / 2, 0), width - span))
    y0 = int(min(max(cy - span / 2, 0), height - span))
    return x0, y0, span


def fit_zoom(shape: tuple[int, int], span: int, zoom: int) -> int:
    """The largest magnification up to `zoom` whose inset still fits the canvas.

    Asked for on every frame rather than validated once: `span` is clamped to the
    frame, so a small frame can silently demand an inset larger than the picture
    it is pasted into. Returns 0 when even 1x will not fit, which suppresses the
    inset instead of raising.
    """
    height, width = shape
    room = min((height - 2 * INSET_MARGIN) // span, (width - 2 * INSET_MARGIN) // span)
    return max(0, min(zoom, int(room)))


def _draw_inset(canvas: np.ndarray, image: np.ndarray, frame: EvalFrame,
                verdict: Verdict, style: Style) -> None:
    """Paste a magnified copy of the target's neighbourhood, top right."""
    x0, y0, span = crop_window(frame, image.shape[:2], style.span)
    zoom = fit_zoom(canvas.shape[:2], span, style.zoom)
    if zoom == 0:
        return
    crop = image[y0:y0 + span, x0:x0 + span]
    # Nearest-neighbour: interpolation invents detail on a target a few pixels
    # across, and the point of the inset is to show what is actually there.
    inset = cv2.resize(crop, (span * zoom, span * zoom),
                       interpolation=cv2.INTER_NEAREST)

    top = INSET_MARGIN
    left = canvas.shape[1] - inset.shape[1] - INSET_MARGIN
    canvas[top:top + inset.shape[0], left:left + inset.shape[1]] = inset
    _draw_frame_boxes(canvas, frame, verdict,
                      View(origin=(x0, y0), scale=zoom, offset=(left, top)),
                      labels=True)

    cv2.rectangle(canvas, (left - 1, top - 1),
                  (left + inset.shape[1], top + inset.shape[0]), INK, 1)
    # The window the inset came from, so the magnification is locatable.
    cv2.rectangle(canvas, (x0, y0), (x0 + span, y0 + span), INK, 1)
    cv2.putText(canvas, f"{zoom}x", (left + 8, top + inset.shape[0] - 10),
                FONT, 0.5, INK, 1, cv2.LINE_AA)


def _draw_caption(canvas: np.ndarray, frame: EvalFrame, verdict: Verdict,
                  criterion: MatchCriterion) -> None:
    """Bottom strip: which frame this is, what the run recorded, what it scored."""
    height = canvas.shape[0]
    strip = canvas[height - CAPTION_HEIGHT:, :]
    cv2.addWeighted(np.full_like(strip, STRIP), 0.7, strip, 0.3, 0, strip)

    extras = "   ".join(f"{k}: {v}" for k, v in frame.extras.items())
    line = (f"{frame.key}    {len(frame.gt_boxes)} gt   {len(frame.preds)} pred    "
            f"{verdict.summary}    matched by {criterion.label}")
    cv2.putText(canvas, f"{line}    {extras}".rstrip(),
                (14, height - CAPTION_HEIGHT + 24), FONT, CAPTION_SCALE, INK, 1,
                cv2.LINE_AA)
    cv2.putText(canvas, LEGEND, (14, height - 16), FONT, 0.45, (190, 190, 190), 1,
                cv2.LINE_AA)


def render_frame(image: np.ndarray, frame: EvalFrame,
                 criterion: float | MatchCriterion,
                 style: Style = Style()) -> np.ndarray:
    """One annotated canvas: both boxes, the magnified inset and the caption."""
    criterion = as_criterion(criterion)
    verdict = judge(frame, criterion)
    canvas = image.copy()
    _draw_frame_boxes(canvas, frame, verdict, View(), labels=False)
    if style.zoom:
        _draw_inset(canvas, image, frame, verdict, style)
    if style.caption:
        _draw_caption(canvas, frame, verdict, criterion)
    return canvas
