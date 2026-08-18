"""Score detections against ground truth with matched detection metrics.

Detection is a *matching* problem before it is a regression problem: the model
emits an arbitrary number of boxes and nothing says which one corresponds to
which ground-truth drone. So the pipeline is always

    match predictions to ground truth  ->  count TP / FP / FN  ->  AP

Matching defaults to **centre distance within one target size** (`--match
center`): on a 10-30 px drone a 2 px centre offset drops IoU below 0.5, so a
plainly correct detection is scored as a miss *and* a false alarm. Pass
`--match iou` for COCO's overlap rule, the only setting comparable to a
published mAP; under it, greedy confidence-ordered matching, 101-point
interpolated AP and mAP@0.50:0.95 all follow the COCO protocol exactly.

The two criteria are different measurements and their P/R/F1 must never be
compared, which is why the criterion is printed, written to `--json-out`, and
recorded on every `--save` line.

Consumes the `detections.jsonl` written by `src.baseline_detect` and YOLO-format
label files (`<cls> <xc> <yc> <w> <h>`, normalised).

Example
-------
    py -3.13 -m src.evaluate \
        --pred runs/exp001/detections.jsonl \
        --labels data/processed/ARD-MAV/labels/val \
        --save runs/exp001/results.jsonl

    # The same run under COCO's rule; both land in the one log, each line
    # carrying the settings that produced it.
    py -3.13 -m src.evaluate \
        --pred runs/exp001/detections.jsonl \
        --labels data/processed/ARD-MAV/labels/val \
        --match iou --iou 0.5 --save runs/exp001/results.jsonl
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
from .eval.records import write_dump
from .eval.results import EvalSettings, append_result


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
    ap.add_argument("--match", choices=(IOU, CENTER), default=CENTER,
                    help="How a prediction claims a target. Default 'center': "
                         "centre-to-centre distance within --match-tol target sizes, "
                         "which asks 'did it find the drone' rather than 'is the box "
                         "well placed' -- the right question for a false-alarm rate on "
                         "10-30 px targets, and equally strict at every target size. "
                         "'iou' is COCO's overlap rule and the only thing comparable to "
                         "published mAP; pass it, and state the threshold, for that.")
    ap.add_argument("--match-tol", type=float, default=1.0, metavar="SIZES",
                    help="Centre-distance tolerance for --match center, in multiples of "
                         "the target's own size, sqrt(w*h). Default 1.0: a prediction "
                         "more than one drone-width off centre is a miss.")
    ap.add_argument("--frame-size", type=int, nargs=2, metavar=("W", "H"),
                    default=None,
                    help="Frame dimensions. Required for video-keyed predictions.")
    ap.add_argument("--conditions", type=Path, default=None,
                    help="conditions.json from prepare_ardmav; adds a breakdown along "
                         "every axis the file declares -- scene_category (needed to "
                         "compare against papers that report by category), plus the "
                         "measured lighting and relative_range axes.")
    ap.add_argument("--json-out", type=Path, default=None,
                    help="Also write metrics as JSON, for the experiment ledger. "
                         "Overwritten each run -- one snapshot, not a history.")
    ap.add_argument("--save", type=Path, default=None, metavar="JSONL",
                    help="Append this result, together with the settings that produced "
                         "it, to a results log. Re-score the same run with different "
                         "--match/--iou/--match-tol into the same file and the log "
                         "accumulates instead of overwriting, so the settings can be "
                         "read back and compared.")
    ap.add_argument("--dump", type=Path, default=None, metavar="CSV",
                    help="Write one row per true positive, false positive and missed "
                         "target: both boxes, their sizes, the offset between them, "
                         "every condition-axis label and every per-frame field the run "
                         "recorded. The metric block is a GROUP BY over this file, so "
                         "any other cut of the same run -- precision against predicted "
                         "box size, false alarms per video -- can be taken later "
                         "without re-scoring.")
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

    if args.dump:
        rows = write_dump(args.dump, frames, criterion, conditions)
        print(f"Wrote {args.dump} ({rows} rows)")

    if args.save:
        settings = EvalSettings.from_args(args, criterion)
        total = append_result(args.save, settings, metrics)
        print(f"Appended to {args.save} ({total} result{'s' * (total != 1)} logged)")


if __name__ == "__main__":
    main()
