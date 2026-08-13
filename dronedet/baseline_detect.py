"""Run a pretrained detector over video or images and record what it finds.

This is step 1 of the plan: establish a baseline with off-the-shelf weights
before training anything. See docs/research-notes.md.

The important switch is --tile. Air-to-air drone targets are often 10-30 px in a
1080p or 4K frame; feeding that through a 640 px letterbox destroys the target
before the detector ever sees it. Tiled inference runs the detector over
overlapping crops at native resolution and merges the results, which separates
"the detector is weak on small targets" from "we threw the target away in a
resize". Both numbers are worth having.

Examples
--------
    # Whole-frame baseline over every 5th frame of a video
    py -3.13 -m dronedet.baseline_detect --weights weights/yolov8s_eo_drone.pt \
        --source data/ARD-MAV/video01.mp4 --stride 5

    # Tiled inference on 4K Det-Fly stills
    py -3.13 -m dronedet.baseline_detect --weights weights/yolov8s_eo_drone.pt \
        --source data/Det-Fly/images --tile --tile-size 640 --conf 0.15
"""

from __future__ import annotations

import argparse
from contextlib import closing
from pathlib import Path

from .algo.config import InferenceConfig
from .algo.detector import detect_frame, load_model
from .data.frames import FrameSource, open_source
from .data.sources import resolve_sources
from .output.annotate import AnnotationSink, open_sink
from .output.recording import RunRecorder


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for the baseline run."""
    ap = argparse.ArgumentParser(prog="dronedet.baseline_detect", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True,
                    help="Path to a .pt file, or an ultralytics model name.")
    ap.add_argument("--source", required=True, type=Path,
                    help="Video file, image file, or directory of images.")
    ap.add_argument("--out", type=Path, default=Path("runs/baseline"),
                    help="Output directory (default: runs/baseline).")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="Confidence threshold. Drop to ~0.1 to see whether the "
                         "target is being found weakly rather than not at all.")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--classes", type=int, nargs="*", default=None,
                    help="Class ids to keep. Single-class drone weights need no filter.")
    ap.add_argument("--stride", type=int, default=1,
                    help="Process every Nth video frame. Raise this on CPU.")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--tile", action="store_true",
                    help="Overlapping tiled inference at native resolution.")
    ap.add_argument("--tile-size", type=int, default=640)
    ap.add_argument("--tile-overlap", type=float, default=0.2)
    ap.add_argument("--tile-batch", type=int, default=8)
    ap.add_argument("--no-save-frames", action="store_true",
                    help="Write only the JSONL, skip annotated output.")
    return ap


def run(source: FrameSource, sink: AnnotationSink, recorder: RunRecorder,
        model, cfg: InferenceConfig, names) -> None:
    """Detect over every frame the source yields, recording and annotating each."""
    for frame in source:
        dets = detect_frame(model, frame.image, cfg)
        recorder.record(frame.key, dets)
        sink.write(frame.image, frame.name, dets, names)
        if frame.progress_label:
            recorder.print_progress(frame.progress_label)


def main() -> None:
    """Parse arguments, run the detector over the source, and report."""
    args = build_parser().parse_args()

    kind, paths = resolve_sources(args.source)
    args.out.mkdir(parents=True, exist_ok=True)

    model, names = load_model(args.weights, args.classes)
    cfg = InferenceConfig.from_args(args)

    with RunRecorder(args.out / "detections.jsonl") as recorder:
        source = open_source(kind, paths, args.stride, args.max_frames)
        with closing(source), open_sink(kind, args.out, not args.no_save_frames,
                                        source.output_fps) as sink:
            run(source, sink, recorder, model, cfg, names)
        # Only after a clean run: an aborted one has no meaningful headline number.
        recorder.print_summary()


if __name__ == "__main__":
    main()
