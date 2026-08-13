"""Run a pretrained detector over video or images and record what it finds.

This is step 1 of the plan: establish a baseline with off-the-shelf weights before
training anything. See docs/research-notes.md.

The important switch is --tile. Air-to-air drone targets are often 10-30 px in a
1080p or 4K frame; feeding that through a 640 px letterbox destroys the target
before the detector ever sees it. Tiled inference runs the detector over
overlapping crops at native resolution and merges the results, which separates
"the detector is weak on small targets" from "we threw the target away in a
resize". Both numbers are worth having.

Examples
--------
    # Whole-frame baseline over every 5th frame of a video
    py -3.13 scripts/baseline_detect.py --weights weights/yolov8s_eo_drone.pt \
        --source data/ARD-MAV/video01.mp4 --stride 5

    # Tiled inference on 4K Det-Fly stills
    py -3.13 scripts/baseline_detect.py --weights weights/yolov8s_eo_drone.pt \
        --source data/Det-Fly/images --tile --tile-size 640 --conf 0.15
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms
from ultralytics import YOLO

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}

BOX_COLOUR = (0, 255, 0)  # BGR
PROGRESS_EVERY = 25  # frames between progress lines


@dataclass(frozen=True)
class InferenceConfig:
    """Detector settings for one run, decoupled from argparse so callers can
    construct a run without a command line."""

    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.5
    classes: list[int] | None = None
    tile: bool = False
    tile_size: int = 640
    tile_overlap: float = 0.2
    tile_batch: int = 8

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> InferenceConfig:
        """Build a config from parsed CLI arguments."""
        return cls(
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            classes=args.classes,
            tile=args.tile,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            tile_batch=args.tile_batch,
        )

    def predict_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for `YOLO.predict`."""
        kw: dict[str, Any] = dict(imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                                  verbose=False)
        if self.classes:
            kw["classes"] = self.classes
        return kw


def tile_origins(total: int, tile: int, overlap: float) -> list[int]:
    """Start coordinates for overlapping tiles covering `total` pixels."""
    if total <= tile:
        return [0]
    step = max(1, int(tile * (1.0 - overlap)))
    origins = list(range(0, total - tile + 1, step))
    if origins[-1] != total - tile:
        origins.append(total - tile)
    return origins


def merge_boxes(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray,
                iou_thres: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Class-aware NMS over detections pooled from several tiles."""
    if len(boxes) == 0:
        return boxes, scores, classes

    keep = batched_nms(
        torch.from_numpy(boxes).float(),
        torch.from_numpy(scores).float(),
        torch.from_numpy(classes).long(),
        iou_thres,
    ).numpy()
    return boxes[keep], scores[keep], classes[keep]


def _unpack(result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy().astype(int))


def _crop_grid(frame: np.ndarray, tile: int,
               overlap: float) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Overlapping tiles of `frame` alongside each tile's (x, y) offset."""
    h, w = frame.shape[:2]
    crops, offsets = [], []
    for y0 in tile_origins(h, tile, overlap):
        for x0 in tile_origins(w, tile, overlap):
            crops.append(frame[y0:y0 + tile, x0:x0 + tile])
            offsets.append((x0, y0))
    return crops, offsets


def detect_tiled(model, frame: np.ndarray,
                 cfg: InferenceConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect on overlapping native-resolution crops and merge back to frame
    coordinates, so small targets survive the detector's input resize."""
    crops, offsets = _crop_grid(frame, cfg.tile_size, cfg.tile_overlap)
    predict_kw = cfg.predict_kwargs()

    all_boxes, all_scores, all_classes = [], [], []
    # Chunked so a 4K frame's ~32 tiles don't balloon peak memory on a 16 GB box.
    for i in range(0, len(crops), cfg.tile_batch):
        chunk = crops[i:i + cfg.tile_batch]
        chunk_offsets = offsets[i:i + cfg.tile_batch]
        for result, (x0, y0) in zip(model.predict(chunk, **predict_kw), chunk_offsets):
            boxes, scores, classes = _unpack(result)
            if len(boxes) == 0:
                continue
            boxes[:, [0, 2]] += x0
            boxes[:, [1, 3]] += y0
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_classes.append(classes)

    if not all_boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

    return merge_boxes(np.concatenate(all_boxes), np.concatenate(all_scores),
                       np.concatenate(all_classes), cfg.iou)


def detect_frame(model, frame: np.ndarray,
                 cfg: InferenceConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (xyxy, conf, cls) for one frame, tiled or whole-frame."""
    if cfg.tile:
        return detect_tiled(model, frame, cfg)
    return _unpack(model.predict(frame, **cfg.predict_kwargs())[0])


def draw(frame: np.ndarray, boxes, scores, classes, names) -> np.ndarray:
    """Copy of `frame` with labelled detection boxes burned in."""
    out = frame.copy()
    for (x1, y1, x2, y2), score, cls in zip(boxes, scores, classes):
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, BOX_COLOUR, 2)
        label = f"{names.get(int(cls), int(cls))} {score:.2f}"
        cv2.putText(out, label, (p1[0], max(12, p1[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOUR, 1, cv2.LINE_AA)
    return out


def resolve_sources(source: Path) -> tuple[str, list[Path]]:
    """Classify `source` as "video" or "images" and list the files to process.

    Exits with a message rather than raising: this is a CLI-level check.
    """
    if source.is_dir():
        images = sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            sys.exit(f"No images found under {source}")
        return "images", images
    if source.suffix.lower() in VIDEO_SUFFIXES:
        return "video", [source]
    if source.suffix.lower() in IMAGE_SUFFIXES:
        return "images", [source]
    sys.exit(f"Unrecognised source type: {source}")


class RunRecorder:
    """Owns the per-frame JSONL output and the run's headline counters.

    Both the video and image paths funnel through here so the record schema and
    the statistics stay defined in exactly one place.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = path.open("w", encoding="utf-8")
        self.n_frames = 0
        self.n_dets = 0
        self.n_empty = 0
        self._t0 = time.perf_counter()

    def __enter__(self) -> RunRecorder:
        return self

    def __exit__(self, *_exc) -> None:
        self._file.close()

    def record(self, key: dict[str, Any], boxes, scores, classes) -> None:
        """Append one frame's detections and fold them into the counters."""
        record = dict(key)
        record["detections"] = [
            {"bbox": [round(float(v), 1) for v in b],
             "conf": round(float(s), 4),
             "cls": int(c)}
            for b, s, c in zip(boxes, scores, classes)
        ]
        self._file.write(json.dumps(record) + "\n")
        self.n_frames += 1
        self.n_dets += len(boxes)
        self.n_empty += len(boxes) == 0

    @property
    def elapsed(self) -> float:
        """Seconds since the recorder was created."""
        return time.perf_counter() - self._t0

    def print_progress(self, label: str) -> None:
        """One in-flight progress line, prefixed with a caller-chosen position."""
        rate = self.n_frames / max(self.elapsed, 1e-9)
        print(f"  {label}  |  {self.n_dets} dets  |  {rate:.2f} fps")

    def print_summary(self) -> None:
        """Final throughput and detection-rate report."""
        elapsed = self.elapsed
        frames = max(self.n_frames, 1)
        print(f"\n{self.n_frames} frames in {elapsed:.1f}s "
              f"({self.n_frames / max(elapsed, 1e-9):.2f} fps)")
        print(f"{self.n_dets} detections, {self.n_dets / frames:.2f} per frame")
        # Most air-to-air frames contain exactly one drone, so on those datasets the
        # empty-frame rate is a rough miss rate -- the headline baseline number.
        print(f"Frames with no detection: {self.n_empty}/{self.n_frames} "
              f"({100 * self.n_empty / frames:.1f}%)")
        print(f"Wrote {self.path}")


def load_model(weights: str, class_filter: list[int] | None) -> tuple[Any, dict[int, str]]:
    """Load the detector and warn if the weights look like generic COCO."""
    print(f"Loading {weights} ...")
    model = YOLO(weights)
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    print(f"Classes: {names}")
    if len(names) > 10 and not class_filter:
        print("  ! These look like generic COCO classes -- there is no 'drone' class in "
              "COCO. Use single-class drone weights, or pass --classes.", file=sys.stderr)
    return model, names


def run_video(model, path: Path, cfg: InferenceConfig, recorder: RunRecorder,
              names, args: argparse.Namespace) -> None:
    """Decode `path`, detect on every --stride'th frame, write JSONL and video."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f"Could not open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    idx = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % args.stride:
                continue

            boxes, scores, classes = detect_frame(model, frame, cfg)
            recorder.record({"frame": idx}, boxes, scores, classes)

            if not args.no_save_frames:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(args.out / "annotated.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps / max(1, args.stride), (w, h))
                writer.write(draw(frame, boxes, scores, classes, names))

            if recorder.n_frames % PROGRESS_EVERY == 0:
                recorder.print_progress(f"frame {idx}")
            if args.max_frames and recorder.n_frames >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def run_images(model, paths: list[Path], cfg: InferenceConfig, recorder: RunRecorder,
               names, args: argparse.Namespace) -> None:
    """Detect over a list of stills, writing JSONL and annotated copies."""
    if args.max_frames:
        paths = paths[::args.stride][:args.max_frames]
    frames_dir = args.out / "annotated"
    if not args.no_save_frames:
        frames_dir.mkdir(exist_ok=True)

    for i, path in enumerate(paths):
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"  ! unreadable: {path}", file=sys.stderr)
            continue

        boxes, scores, classes = detect_frame(model, frame, cfg)
        recorder.record({"image": str(path)}, boxes, scores, classes)

        if not args.no_save_frames:
            cv2.imwrite(str(frames_dir / path.name),
                        draw(frame, boxes, scores, classes, names))
        if (i + 1) % PROGRESS_EVERY == 0:
            recorder.print_progress(f"{i + 1}/{len(paths)}")


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for the baseline run."""
    ap = argparse.ArgumentParser(description=__doc__,
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


def main() -> None:
    """Parse arguments, run the detector over the source, and report."""
    args = build_parser().parse_args()

    kind, paths = resolve_sources(args.source)
    args.out.mkdir(parents=True, exist_ok=True)

    model, names = load_model(args.weights, args.classes)
    cfg = InferenceConfig.from_args(args)

    with RunRecorder(args.out / "detections.jsonl") as recorder:
        if kind == "video":
            run_video(model, paths[0], cfg, recorder, names, args)
        else:
            run_images(model, paths, cfg, recorder, names, args)
        # Only on a clean run: an aborted one has no meaningful headline number.
        recorder.print_summary()


if __name__ == "__main__":
    main()
