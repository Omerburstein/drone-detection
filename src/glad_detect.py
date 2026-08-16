"""Run GLAD's released pipeline over ARD-MAV videos and record what it finds.

This is M4a, and it is **a test of our harness, not of GLAD**. GLAD's weights
were trained on ARD-MAV's other 45 videos and its architecture was tuned against
this very split, so the result is optimistic by construction and must never be
reported as "GLAD scores X for us". What it buys is the only independent check
available on M1's evaluation math and M2's VOC->YOLO conversion: if reproducing
the paper's per-category rows lands far off, we have a bug.

Two things separate this from `src.baseline_detect`, and both are why it is a
second CLI rather than a flag:

**Frames must be contiguous.** Both motion branches difference the current frame
against the previous one, so this reads the original `.mp4`s in order rather
than M3's stride-10 stills. There is no `--stride`.

**Frames are recorded, not just detections.** Output is keyed by the processed
image path, so `src.evaluate` scores it against the same labels as EXP-001–003
with no extra flags. Each row also carries the `branch` that produced it, which
is what makes the paper's ablation visible in our own numbers.

Example
-------
    py -3.13 -m src.glad_detect --out runs/exp004_glad
    py -3.13 -m src.evaluate --pred runs/exp004_glad/detections.jsonl \
        --labels data/processed/ARD-MAV/labels/test \
        --conditions data/processed/ARD-MAV/conditions.json \
        --frame-size 1920 1080 --json-out runs/exp004_glad/metrics.json
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2

from .algo.glad.pipeline import GladPipeline
from .algo.glad.vendor import GLAD_DIR
from .algo.glad.yolo import PAD_STYLES
from .data.prepare_ardmav import OFFICIAL_TEST_VIDEOS
from .output.recording import RunRecorder

# GLAD emits no confidence -- see `StepResult.as_detections`. Every box is
# recorded at the same value, which is a single operating point, not a ranking.
GLAD_CONFIDENCE = 1.0
PROGRESS_EVERY = 250


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for the GLAD reproduction run."""
    ap = argparse.ArgumentParser(prog="src.glad_detect", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", type=Path, default=Path("data/raw/ARD-MAV/videos"),
                    help="Directory of source .mp4 files.")
    ap.add_argument("--labels", type=Path,
                    default=Path("data/processed/ARD-MAV/labels/test"),
                    help="Label directory for the split. Frames with no label were "
                         "never annotated and are processed but not recorded, which "
                         "keeps the scored frame set identical to M2's.")
    ap.add_argument("--images", type=Path,
                    default=Path("data/processed/ARD-MAV/images/test"),
                    help="Image directory the JSONL rows are keyed by, so "
                         "src.evaluate resolves labels exactly as for a stills run.")
    ap.add_argument("--video-names", nargs="*", default=None,
                    help="Videos to run. Defaults to the official 15-video test split.")
    ap.add_argument("--out", type=Path, default=Path("runs/glad"),
                    help="Output directory (default: runs/glad).")
    ap.add_argument("--max-frames-per-video", type=int, default=None,
                    help="Stop each video after N frames. A contiguous prefix, so the "
                         "motion branches still work -- for smoke tests, not results.")
    ap.add_argument("--glad-repo", type=Path, default=GLAD_DIR,
                    help="Clone of the GLAD release holding weights/ (default: "
                         "third_party/GLAD).")
    ap.add_argument("--pad", choices=sorted(PAD_STYLES), default="trained",
                    help="Letterbox fill for the global detector, which sees 44%% "
                         "padding on a 1080p frame. 'trained' (114) is what yolov5 "
                         "trained these weights against and is the default. "
                         "**'released' (black) reproduces upstream, bug and all, and "
                         "is what a comparison against the paper must use.** "
                         "'tensorrtx' (128) is what upstream intended.")
    return ap


def run_video(pipeline: GladPipeline, video: Path, stem: str, args: argparse.Namespace,
              recorder: RunRecorder, branches: Counter) -> None:
    """Step the pipeline over one video, recording every annotated frame.

    Unannotated frames are still *processed* -- skipping them would break the
    frame-to-frame differencing -- but are not recorded, because a detection
    scored against a frame nobody labelled would understate precision.
    """
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        sys.exit(f"Could not open {video}")

    pipeline.reset()
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1  # label numbering is one-based

            result = pipeline.step(frame)
            branches[result.branch] += 1

            name = f"{stem}_{frame_index:04d}"
            if (args.labels / f"{name}.txt").exists():
                recorder.record({"image": str(args.images / f"{name}.jpg"),
                                 "branch": result.branch},
                                result.as_detections(GLAD_CONFIDENCE))
            if frame_index % PROGRESS_EVERY == 0:
                recorder.print_progress(f"{stem} {frame_index}")
            if args.max_frames_per_video and frame_index >= args.max_frames_per_video:
                break
    finally:
        capture.release()


def print_branches(branches: Counter) -> None:
    """Report which branch produced each frame's outcome.

    This is the paper's ablation table measured on our own run: how much of the
    recall is appearance and how much is motion.
    """
    total = max(sum(branches.values()), 1)
    print("\nBranch that handled each frame:")
    for branch, count in branches.most_common():
        print(f"  {branch:<14} {count:>7}  {100 * count / total:5.1f}%")


def main() -> None:
    """Load the pipeline once and run it over every video in the split."""
    args = build_parser().parse_args()
    names = list(args.video_names or OFFICIAL_TEST_VIDEOS)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading GLAD from {args.glad_repo} ...")
    print(f"Letterbox fill: {args.pad} ({PAD_STYLES[args.pad]})")
    pipeline = GladPipeline.from_release(args.glad_repo, PAD_STYLES[args.pad])

    branches: Counter = Counter()
    with RunRecorder(args.out / "detections.jsonl") as recorder:
        for n, stem in enumerate(names, 1):
            print(f"[{n}/{len(names)}] {stem}", flush=True)
            run_video(pipeline, args.videos / f"{stem}.mp4", stem, args,
                      recorder, branches)
        recorder.print_summary()
    print_branches(branches)


if __name__ == "__main__":
    main()
