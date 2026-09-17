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
than M3's stride-10 stills. **There is still no `--stride`**, and there will not
be: handing the motion module two frames a third of a second apart measures a
different algorithm.

`--sample` is not that. It selects frames whose *differencing stays coherent* --
`nth` lowers the rate of an otherwise contiguous stream, `burst` differences
inside a short contiguous run and sleeps between runs -- so what it costs is
detection opportunity rather than the motion branch itself. See
`src.data.sampling` for the two failure modes that buys, and section 3 of
`docs/edge-budget.md` for why this is a separate lever from striding.

**Frames are recorded, not just detections.** Output is keyed by the processed
image path, so `src.evaluate` scores it against the same labels as EXP-001-003
with no extra flags. Each row also carries the `branch` that produced it, which
is what makes the paper's ablation visible in our own numbers. A duty-cycled run
records only the frames it processed, and `src.eval.labels` reads the prediction
rows rather than the label directory, so such a run is scored on exactly those
frames -- the ones it skipped are absent, not counted as misses. Compare it
against a full-rate run with `src.evaluate --keys-from`, never against a
different frame set.

`--dataset` picks which prepared tree to run over. `ARD-MAV` is M4a above;
`ARD100` is M4b, the same pipeline and the same 15-video count over videos GLAD
has never seen, which is the run this is a control for. Nothing else may differ
between the two -- same code path, same thresholds, same 1920x1080 -- or the
gap stops being about generalisation.

Example
-------
    py -3.13 -m src.glad_detect --out runs/exp004_glad
    py -3.13 -m src.glad_detect --dataset ARD100 --pad released \
        --out runs/exp005_glad_ard100
    py -3.13 -m src.glad_detect --sample nth --sample-n 2 \
        --out runs/exp006_half_rate
    py -3.13 -m src.glad_detect --sample burst --burst-length 2 \
        --burst-period 60 --out runs/exp007_burst_pairs
    py -3.13 -m src.glad_detect --videos data/raw/FIELD/videos \
        --video-names captured_raw_20260616_040253_004 --record-all \
        --images data/processed/FIELD/images/test --out runs/exp010_field
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
from .data.crop import Crop
from .data.datasets import SPECS, spec_for
from .data.sampling import BURST, EVERY, NTH, Schedule, build_schedule
from .output.recording import RunRecorder

# GLAD emits no confidence -- see `StepResult.as_detections`. Every box is
# recorded at the same value, which is a single operating point, not a ranking.
GLAD_CONFIDENCE = 1.0
PROGRESS_EVERY = 250


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for the GLAD reproduction run."""
    ap = argparse.ArgumentParser(prog="src.glad_detect", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(SPECS), default="ARD-MAV",
                    help="Which prepared dataset to run over (default: ARD-MAV). Sets "
                         "the defaults for --videos, --labels, --images and "
                         "--video-names. 'ARD100' is M4b: the same pipeline over 15 "
                         "videos GLAD has never seen.")
    ap.add_argument("--split", default="test",
                    help="Split whose labels are scored (default: test).")
    ap.add_argument("--videos", type=Path, default=None,
                    help="Directory of source .mp4 files. Defaults to the dataset's.")
    ap.add_argument("--labels", type=Path, default=None,
                    help="Label directory for the split. Frames with no label were "
                         "never annotated and are processed but not recorded, which "
                         "keeps the scored frame set identical to M2's.")
    ap.add_argument("--images", type=Path, default=None,
                    help="Image directory the JSONL rows are keyed by, so "
                         "src.evaluate resolves labels exactly as for a stills run. "
                         "Nothing is read from it, so it need not exist -- a "
                         "labels-only tree (prepare --no-images) runs fine.")
    ap.add_argument("--video-names", nargs="*", default=None,
                    help="Videos to run. Defaults to the dataset's 15-video test split.")
    ap.add_argument("--crop", type=Crop.parse, default=None, metavar="X,Y,W,H",
                    help="Detect on this rectangle of each frame instead of the whole "
                         "one, for sources that are not all picture -- a goggles "
                         "screen recording is pillarboxed, and the bars cost the target "
                         "resolution because the global detector letterboxes the "
                         "longest side. Recorded boxes are in **cropped** coordinates; "
                         "pass the same --crop to src.render_video. Applied at decode, "
                         "so nothing is re-encoded.")
    ap.add_argument("--record-all", action="store_true",
                    help="Record every processed frame, including ones with no label "
                         "file. For **unlabelled footage only** -- field capture that "
                         "nobody has annotated yet. On a labelled split this would "
                         "silently add unannotated frames to the scored set and "
                         "understate precision, which is the whole reason recording "
                         "is gated on a label by default. A run made this way cannot "
                         "be scored until labels exist, so key it (--images) at the "
                         "tree those labels will land in.")
    ap.add_argument("--out", type=Path, default=Path("runs/glad"),
                    help="Output directory (default: runs/glad).")
    ap.add_argument("--max-frames-per-video", type=int, default=None,
                    help="Stop each video after N frames. A contiguous prefix, so the "
                         "motion branches still work -- for smoke tests, not results. "
                         "Counts decoded frames, not processed ones, so it covers the "
                         "same span of video under every --sample mode.")
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
    ap.add_argument("--sample", choices=(EVERY, NTH, BURST), default=EVERY,
                    help="Duty-cycle policy (default: every frame, which is what "
                         "EXP-004 ran). 'nth' runs the whole pipeline at 1/N of the "
                         "source rate; 'burst' processes a short contiguous run of "
                         "frames and sleeps between runs, resetting the pipeline each "
                         "time so the motion branches never difference across a gap. "
                         "**Not a stride** -- see src.data.sampling.")
    ap.add_argument("--sample-n", type=int, default=2, metavar="N",
                    help="For --sample nth: process every Nth frame (default 2, i.e. "
                         "15 fps from a 30 fps source).")
    ap.add_argument("--burst-length", type=int, default=2, metavar="K",
                    help="For --sample burst: frames per burst (default 2). The first "
                         "frame of a burst follows a reset and so can never detect, "
                         "leaving K-1 chances per burst.")
    ap.add_argument("--burst-period", type=int, default=60, metavar="P",
                    help="For --sample burst: frames between burst starts (default 60, "
                         "i.e. every 2 s at 30 fps). The period sets detection "
                         "*latency*; it does not change the per-burst detection "
                         "probability, so measure with a dense period and deploy a "
                         "sparse one.")
    return ap


def run_video(pipeline: GladPipeline, video: Path, stem: str, args: argparse.Namespace,
              recorder: RunRecorder, branches: Counter,
              schedule: Schedule) -> tuple[int, int]:
    """Step the pipeline over one video, recording every annotated frame.

    Returns `(decoded, processed)`. Every frame is decoded -- that cost is real
    on a live camera and is not what a duty cycle saves -- but only the frames
    `schedule` asks for reach the detector.

    Unannotated frames are still *processed* -- skipping them would break the
    frame-to-frame differencing -- but are not recorded, because a detection
    scored against a frame nobody labelled would understate precision.
    `--record-all` lifts that gate for footage where *nothing* is labelled, so
    the run is a record of what the detector fired on rather than a score.
    """
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        sys.exit(f"Could not open {video}")
    if args.crop is not None:
        shape = (int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                 int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        if not args.crop.fits(shape):
            sys.exit(f"--crop {args.crop.label} does not fit {video.name} "
                     f"({shape[1]}x{shape[0]})")

    pipeline.reset()
    frame_index = decoded = processed = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1  # label numbering is one-based
            decoded += 1
            if args.crop is not None:
                frame = args.crop.apply(frame)

            if schedule.wants(frame_index):
                # Clearing state is what keeps a burst honest: without it the
                # motion branches would difference against a frame from the
                # previous burst, seconds of ego-motion ago.
                if schedule.resets(frame_index):
                    pipeline.reset()
                processed += 1

                result = pipeline.step(frame)
                branches[result.branch] += 1

                name = f"{stem}_{frame_index:04d}"
                if args.record_all or (args.labels / f"{name}.txt").exists():
                    recorder.record({"image": str(args.images / f"{name}.jpg"),
                                     "branch": result.branch,
                                     **schedule.position(frame_index)},
                                    result.as_detections(GLAD_CONFIDENCE))

            if frame_index % PROGRESS_EVERY == 0:
                recorder.print_progress(f"{stem} {frame_index}")
            if args.max_frames_per_video and frame_index >= args.max_frames_per_video:
                break
    finally:
        capture.release()
    return decoded, processed


def print_branches(branches: Counter) -> None:
    """Report which branch produced each frame's outcome.

    This is the paper's ablation table measured on our own run: how much of the
    recall is appearance and how much is motion.
    """
    total = max(sum(branches.values()), 1)
    print("\nBranch that handled each frame:")
    for branch, count in branches.most_common():
        print(f"  {branch:<14} {count:>7}  {100 * count / total:5.1f}%")


def print_duty_cycle(decoded: int, processed: int, schedule: Schedule) -> None:
    """Report how much of the source the detector actually saw.

    Printed for every run, not only duty-cycled ones, so the full-rate baseline
    carries the same line and the two are read the same way.
    """
    share = 100 * processed / max(decoded, 1)
    print(f"\nSampling: {schedule.label}")
    print(f"  {processed}/{decoded} frames processed ({share:.2f}% duty cycle)")


def main() -> None:
    """Load the pipeline once and run it over every video in the split."""
    args = build_parser().parse_args()
    spec = spec_for(args.dataset)
    args.videos = args.videos or spec.videos_dir
    args.labels = args.labels or spec.out / "labels" / args.split
    args.images = args.images or spec.out / "images" / args.split
    names = list(args.video_names or spec.videos)
    if args.record_all:
        print("Recording every processed frame (--record-all): the output is a "
              "record of what fired, not a score, until labels exist.")
    args.out.mkdir(parents=True, exist_ok=True)

    schedule = build_schedule(args.sample, args.sample_n,
                              args.burst_length, args.burst_period)

    print(f"Loading GLAD from {args.glad_repo} ...")
    print(f"Letterbox fill: {args.pad} ({PAD_STYLES[args.pad]})")
    print(f"Sampling: {schedule.label}")
    if args.crop is not None:
        print(f"Crop: {args.crop.label} -- boxes are recorded in cropped coordinates")
    pipeline = GladPipeline.from_release(args.glad_repo, PAD_STYLES[args.pad])

    branches: Counter = Counter()
    decoded = processed = 0
    with RunRecorder(args.out / "detections.jsonl") as recorder:
        for n, stem in enumerate(names, 1):
            print(f"[{n}/{len(names)}] {stem}", flush=True)
            video_decoded, video_processed = run_video(
                pipeline, args.videos / f"{stem}.mp4", stem, args, recorder,
                branches, schedule)
            decoded += video_decoded
            processed += video_processed
        recorder.print_summary()
    print_branches(branches)
    print_duty_cycle(decoded, processed, schedule)


if __name__ == "__main__":
    main()
