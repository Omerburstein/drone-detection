"""Re-render one source video with both boxes on it: ground truth and prediction.

The metric block says a detector found 52% of the small targets. It cannot say
*what the misses look like* — whether the drone was invisible, whether the box
was two pixels off, whether the false alarms are a cloud edge or an AC unit.
This draws a scored run back onto the footage it came from so that question has
an answer you can watch.

Nothing is re-inferred. The boxes come from a run's persisted `detections.jsonl`
and the match outcome from `src.eval.metrics.match_frame`, so the video cannot
disagree with the numbers in the ledger — and rendering costs a video decode
rather than an inference pass.

Frames are keyed `<video stem>_<1-based frame number, 4 digits>`, which is what
`src.data.prepare_ardmav` named them, so the video file and the run's JSONL line
up without a manifest. Frames the run never recorded (unannotated ones, which
`src.glad_detect` processes but does not score) are dropped rather than drawn
blank: the output is exactly the frames that were measured.

Example
-------
    py -3.13 -m src.render_video \
        --video data/raw/ARD-MAV/videos/phantom19.mp4 \
        --pred runs/exp004_glad/detections.jsonl \
        --labels data/processed/ARD-MAV/labels/test \
        --out runs/exp004_glad/examples/phantom19_overlay.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from .eval.labels import load_frames
from .eval.metrics import CENTER, IOU, MatchCriterion
from .output.overlay import Style, render_frame
from .output.video import LazyVideoWriter

PROGRESS_EVERY = 200


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for the overlay renderer."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path,
                    help="Source .mp4. Its stem is the frame-key prefix unless "
                         "--key-prefix says otherwise.")
    ap.add_argument("--pred", required=True, type=Path,
                    help="detections.jsonl from a recorded run.")
    ap.add_argument("--labels", required=True, type=Path,
                    help="Directory of YOLO-format .txt label files for the split.")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output .mp4. Parent directories are created.")
    ap.add_argument("--key-prefix", default=None,
                    help="Frame-key prefix to select from --pred. Defaults to the "
                         "video's stem, e.g. 'phantom19'.")
    ap.add_argument("--match", choices=(IOU, CENTER), default=CENTER,
                    help="How a prediction claims a target, as in src.evaluate. "
                         "Default 'center'; it decides only the colour of a box, "
                         "never whether it is drawn.")
    ap.add_argument("--iou", type=float, default=0.5,
                    help="IoU threshold for --match iou (default 0.5).")
    ap.add_argument("--match-tol", type=float, default=1.0, metavar="SIZES",
                    help="Centre-distance tolerance for --match center, in target "
                         "sizes (default 1.0).")
    ap.add_argument("--zoom", type=int, default=Style.zoom, metavar="N",
                    help=f"Magnification of the inset (default {Style.zoom}). 0 "
                         f"draws no inset, for targets large enough to read unaided.")
    ap.add_argument("--zoom-span", type=int, default=Style.span, metavar="PX",
                    help=f"Side of the source window the inset magnifies, in original "
                         f"pixels (default {Style.span}).")
    ap.add_argument("--no-caption", action="store_true",
                    help="Omit the bottom caption strip and the colour legend.")
    ap.add_argument("--fps", type=float, default=None,
                    help="Output frame rate. Defaults to the source video's.")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Stop after N rendered frames — a contiguous prefix, for "
                         "checking the overlay without decoding the whole video.")
    return ap


def open_video(path: Path) -> tuple[cv2.VideoCapture, int, int, float]:
    """Open a video and read back the geometry the labels must be scaled by."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        sys.exit(f"Could not open {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    return capture, width, height, fps


def criterion_from_args(args: argparse.Namespace) -> MatchCriterion:
    """The matching rule the two --match choices select, as in src.evaluate."""
    return (MatchCriterion(IOU, args.iou) if args.match == IOU
            else MatchCriterion(CENTER, args.match_tol))


def render(args: argparse.Namespace) -> tuple[int, int]:
    """Write the overlay video; return (frames rendered, frames skipped)."""
    capture, width, height, source_fps = open_video(args.video)
    prefix = args.key_prefix or args.video.stem

    frames = {f.key: f for f in load_frames(args.pred, args.labels, (width, height),
                                            key_filter=lambda k: k.startswith(prefix))}
    if not frames:
        sys.exit(f"No frames keyed '{prefix}_*' in {args.pred}")
    print(f"{len(frames)} scored frames for {prefix}; {width}x{height} "
          f"@ {source_fps:.2f} fps")

    style = Style(zoom=args.zoom, span=args.zoom_span, caption=not args.no_caption)
    criterion = criterion_from_args(args)
    rendered = skipped = 0
    try:
        with LazyVideoWriter(args.out, args.fps or source_fps) as writer:
            index = 0
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                index += 1  # frame keys are one-based, as prepare_ardmav wrote them
                frame = frames.get(f"{prefix}_{index:04d}")
                if frame is None:
                    skipped += 1
                    continue
                writer.write(render_frame(image, frame, criterion, style))
                rendered += 1
                if rendered % PROGRESS_EVERY == 0:
                    print(f"  {rendered} frames", flush=True)
                if args.max_frames and rendered >= args.max_frames:
                    break
    finally:
        capture.release()
    return rendered, skipped


def main() -> None:
    """Render one video's overlay and report what was drawn."""
    args = build_parser().parse_args()
    if not args.labels.is_dir():
        sys.exit(f"--labels must be a directory, got {args.labels}")

    rendered, skipped = render(args)
    print(f"Wrote {args.out} — {rendered} frames"
          + (f", {skipped} unscored frames skipped" if skipped else ""))


if __name__ == "__main__":
    main()
