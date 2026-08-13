"""Score detections against ground truth with IoU-matched detection metrics.

Detection is a *matching* problem before it is a regression problem: the model
emits an arbitrary number of boxes and nothing says which one corresponds to
which ground-truth drone. So the pipeline is always

    match predictions to ground truth by IoU  ->  count TP / FP / FN  ->  AP

Everything below follows the COCO protocol: greedy matching in descending
confidence order, 101-point interpolated AP, and mAP averaged over IoU
thresholds 0.50:0.05:0.95.

Consumes the `detections.jsonl` written by baseline_detect.py and YOLO-format
label files (`<cls> <xc> <yc> <w> <h>`, normalised).

Example
-------
    py -3.13 scripts/evaluate.py \
        --pred runs/exp001/detections.jsonl \
        --labels data/processed/ARD-MAV/labels/val
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

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

IOU_SWEEP = np.arange(0.5, 0.96, 0.05)


@dataclass
class Frame:
    """One image's ground truth and predictions, in absolute xyxy pixels."""

    key: str
    gt_boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    gt_classes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    pred_boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    pred_scores: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pred_classes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))


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
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def yolo_to_xyxy(rows: np.ndarray, width: int, height: int) -> np.ndarray:
    """Normalised YOLO cx,cy,w,h -> absolute xyxy."""
    if len(rows) == 0:
        return np.zeros((0, 4))
    cx, cy = rows[:, 0] * width, rows[:, 1] * height
    half_w, half_h = rows[:, 2] * width / 2, rows[:, 3] * height / 2
    return np.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], axis=1)


def load_label_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a YOLO .txt label file into (cxcywh rows, class ids)."""
    if not path.exists():
        return np.zeros((0, 4)), np.zeros(0, dtype=int)

    classes, rows = [], []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 5:
            sys.exit(f"{path}:{line_no}: expected '<cls> <cx> <cy> <w> <h>', got {line!r}")
        classes.append(int(float(parts[0])))
        rows.append([float(v) for v in parts[1:5]])
    return np.array(rows).reshape(-1, 4), np.array(classes, dtype=int)


def load_frames(pred_path: Path, labels_dir: Path,
                frame_size: tuple[int, int] | None) -> list[Frame]:
    """Pair each prediction record with its label file.

    Records keyed by `image` resolve their size from the image itself; records
    keyed by `frame` (video runs) need --frame-size, since the JSONL does not
    carry frame dimensions.
    """
    frames = []
    for line in pred_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)

        if "image" in record:
            image_path = Path(record["image"])
            stem = image_path.stem
            if frame_size:
                width, height = frame_size
            elif image_path.exists():
                with Image.open(image_path) as img:
                    width, height = img.size
            else:
                sys.exit(f"Cannot determine size for {image_path} (missing). "
                         f"Pass --frame-size W H.")
        else:
            stem = str(record["frame"])
            if not frame_size:
                sys.exit("Video-keyed predictions need --frame-size W H.")
            width, height = frame_size

        rows, gt_classes = load_label_file(labels_dir / f"{stem}.txt")
        detections = record.get("detections", [])

        frames.append(Frame(
            key=stem,
            gt_boxes=yolo_to_xyxy(rows, width, height),
            gt_classes=gt_classes,
            pred_boxes=np.array([d["bbox"] for d in detections]).reshape(-1, 4),
            pred_scores=np.array([d["conf"] for d in detections]),
            pred_classes=np.array([d["cls"] for d in detections], dtype=int),
        ))
    return frames


def match_frame(frame: Frame, iou_thres: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedily match predictions to ground truth in confidence order.

    Returns (is_true_positive per prediction, matched-GT index per prediction or
    -1, IoU of each match). Each ground-truth box can be claimed once; extra
    predictions on an already-matched target become false positives, which is
    what penalises duplicate boxes.
    """
    n_pred = len(frame.pred_boxes)
    tp = np.zeros(n_pred, dtype=bool)
    matched_gt = np.full(n_pred, -1, dtype=int)
    match_iou = np.zeros(n_pred)
    if n_pred == 0 or len(frame.gt_boxes) == 0:
        return tp, matched_gt, match_iou

    ious = iou_matrix(frame.pred_boxes, frame.gt_boxes)
    claimed = np.zeros(len(frame.gt_boxes), dtype=bool)

    for pred_idx in np.argsort(-frame.pred_scores):
        candidates = ious[pred_idx].copy()
        candidates[claimed] = -1.0
        # Class-aware: a drone box may not be satisfied by a bird prediction.
        candidates[frame.gt_classes != frame.pred_classes[pred_idx]] = -1.0

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
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

    # Make precision monotonically decreasing, then sample at 101 recall points.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.mean(np.interp(np.linspace(0, 1, 101), recall, precision,
                                   left=precision[0], right=0.0)))


def evaluate(frames: list[Frame], primary_iou: float) -> dict:
    """Full metric sweep over already-paired frames."""
    n_gt = sum(len(f.gt_boxes) for f in frames)
    if n_gt == 0:
        sys.exit("No ground-truth boxes found -- check --labels points at the right split.")

    aps = []
    for thres in IOU_SWEEP:
        all_tp, all_scores = [], []
        for frame in frames:
            tp, _, _ = match_frame(frame, float(thres))
            all_tp.append(tp)
            all_scores.append(frame.pred_scores)
        aps.append(average_precision(np.concatenate(all_tp) if all_tp else np.zeros(0, bool),
                                     np.concatenate(all_scores) if all_scores else np.zeros(0),
                                     n_gt))

    # Per-frame detail at the primary threshold, for the breakdowns.
    tp_all, score_all, iou_all = [], [], []
    gt_areas, gt_found = [], []
    frames_with_miss = 0
    for frame in frames:
        tp, matched_gt, match_iou = match_frame(frame, primary_iou)
        tp_all.append(tp)
        score_all.append(frame.pred_scores)
        iou_all.append(match_iou[tp])

        found = np.zeros(len(frame.gt_boxes), dtype=bool)
        found[matched_gt[matched_gt >= 0]] = True
        wh = np.clip(frame.gt_boxes[:, 2:] - frame.gt_boxes[:, :2], 0, None)
        gt_areas.append(wh[:, 0] * wh[:, 1])
        gt_found.append(found)
        if len(frame.gt_boxes) and not found.all():
            frames_with_miss += 1

    tp_cat = np.concatenate(tp_all) if tp_all else np.zeros(0, bool)
    n_tp = int(tp_cat.sum())
    n_fp = int((~tp_cat).sum())

    areas = np.concatenate(gt_areas) if gt_areas else np.zeros(0)
    found = np.concatenate(gt_found) if gt_found else np.zeros(0, bool)
    by_size = []
    for label, lo, hi in AREA_BUCKETS:
        sel = (areas >= lo) & (areas < hi)
        by_size.append((label, int(sel.sum()),
                        float(found[sel].mean()) if sel.any() else float("nan")))

    matched_ious = np.concatenate(iou_all) if iou_all else np.zeros(0)
    return {
        "n_frames": len(frames),
        "n_gt": n_gt,
        "n_pred": int(sum(len(f.pred_boxes) for f in frames)),
        "tp": n_tp,
        "fp": n_fp,
        "fn": n_gt - n_tp,
        "precision": n_tp / max(n_tp + n_fp, 1),
        "recall": n_tp / n_gt,
        "ap50": float(aps[0]),
        "map": float(np.nanmean(aps)),
        "mean_iou": float(matched_ious.mean()) if len(matched_ious) else float("nan"),
        "frames_with_miss": frames_with_miss,
        "by_size": by_size,
    }


def report(metrics: dict, primary_iou: float) -> None:
    """Print the metric block. Recall and precision are shown together because
    they need opposite fixes and a single number hides which one is failing."""
    f1 = (2 * metrics["precision"] * metrics["recall"]
          / max(metrics["precision"] + metrics["recall"], 1e-12))
    print(f"\n{'=' * 52}")
    print(f"{metrics['n_frames']} frames | {metrics['n_gt']} ground-truth boxes "
          f"| {metrics['n_pred']} predictions")
    print(f"{'=' * 52}")
    print(f"  AP@0.50          {metrics['ap50']:.4f}")
    print(f"  mAP@0.50:0.95    {metrics['map']:.4f}")
    print(f"\n  at IoU {primary_iou:.2f}:")
    print(f"  precision        {metrics['precision']:.4f}   (TP {metrics['tp']} / "
          f"FP {metrics['fp']})")
    print(f"  recall           {metrics['recall']:.4f}   (missed {metrics['fn']})")
    print(f"  F1               {f1:.4f}")
    print(f"  mean IoU         {metrics['mean_iou']:.4f}   (matched boxes only -- "
          f"localisation quality)")
    print(f"  frames w/ a miss {metrics['frames_with_miss']}/{metrics['n_frames']}")

    print(f"\n  recall by target size:")
    for label, count, recall in metrics["by_size"]:
        if count == 0:
            continue
        print(f"    {label}  n={count:<7} recall={recall:.4f}")
    print()


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for evaluation."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, type=Path,
                    help="detections.jsonl from baseline_detect.py")
    ap.add_argument("--labels", required=True, type=Path,
                    help="Directory of YOLO-format .txt label files.")
    ap.add_argument("--iou", type=float, default=0.5,
                    help="Primary IoU threshold for P/R breakdowns (default 0.5).")
    ap.add_argument("--frame-size", type=int, nargs=2, metavar=("W", "H"),
                    default=None,
                    help="Frame dimensions. Required for video-keyed predictions.")
    ap.add_argument("--json-out", type=Path, default=None,
                    help="Also write metrics as JSON, for the experiment ledger.")
    return ap


def main() -> None:
    """Load predictions and labels, score them, and report."""
    args = build_parser().parse_args()
    if not args.labels.is_dir():
        sys.exit(f"--labels must be a directory, got {args.labels}")

    frames = load_frames(args.pred, args.labels,
                         tuple(args.frame_size) if args.frame_size else None)
    metrics = evaluate(frames, args.iou)
    report(metrics, args.iou)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
