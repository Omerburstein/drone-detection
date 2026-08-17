"""Score detections against ground truth with IoU-matched detection metrics.

Detection is a *matching* problem before it is a regression problem: the model
emits an arbitrary number of boxes and nothing says which one corresponds to
which ground-truth drone. So the pipeline is always

    match predictions to ground truth by IoU  ->  count TP / FP / FN  ->  AP

Everything below follows the COCO protocol: greedy matching in descending
confidence order, 101-point interpolated AP, and mAP averaged over IoU
thresholds 0.50:0.05:0.95.

Consumes the `detections.jsonl` written by `src.baseline_detect` and YOLO-format
label files (`<cls> <xc> <yc> <w> <h>`, normalised).

Example
-------
    py -3.13 -m src.evaluate \
        --pred runs/exp001/detections.jsonl \
        --labels data/processed/ARD-MAV/labels/val
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .eval.conditions import load_conditions
from .eval.labels import load_frames
from .eval.metrics import CENTER, IOU, MatchCriterion, evaluate
from .eval.report import report


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
    ap.add_argument("--match", choices=(IOU, CENTER), default=IOU,
                    help="How a prediction claims a target. 'iou' is COCO's overlap "
                         "rule and the only thing comparable to published mAP. "
                         "'center' matches on centre-to-centre distance within "
                         "--match-tol target sizes, which asks 'did it find the drone' "
                         "rather than 'is the box well placed' -- the right question "
                         "for a false-alarm rate on 10-30 px targets.")
    ap.add_argument("--match-tol", type=float, default=1.0, metavar="SIZES",
                    help="Centre-distance tolerance for --match center, in multiples of "
                         "the target's own size, sqrt(w*h). Default 1.0: a prediction "
                         "more than one drone-width off centre is a miss.")
    ap.add_argument("--frame-size", type=int, nargs=2, metavar=("W", "H"),
                    default=None,
                    help="Frame dimensions. Required for video-keyed predictions.")
    ap.add_argument("--conditions", type=Path, default=None,
                    help="conditions.json from prepare_ardmav; adds a per-scene-category "
                         "breakdown. Needed to compare against papers that report by "
                         "category.")
    ap.add_argument("--json-out", type=Path, default=None,
                    help="Also write metrics as JSON, for the experiment ledger.")
    return ap


def main() -> None:
    """Load predictions and labels, score them, and report."""
    args = build_parser().parse_args()
    if not args.labels.is_dir():
        sys.exit(f"--labels must be a directory, got {args.labels}")

    criterion = (MatchCriterion(IOU, args.iou) if args.match == IOU
                 else MatchCriterion(CENTER, args.match_tol))

    frames = load_frames(args.pred, args.labels,
                         tuple(args.frame_size) if args.frame_size else None)
    conditions = load_conditions(args.conditions) if args.conditions else None
    metrics = evaluate(frames, criterion, conditions=conditions)
    report(metrics)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
