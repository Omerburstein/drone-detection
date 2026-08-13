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
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}


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

    import torch
    from torchvision.ops import batched_nms

    keep = batched_nms(
        torch.from_numpy(boxes).float(),
        torch.from_numpy(scores).float(),
        torch.from_numpy(classes).long(),
        iou_thres,
    ).numpy()
    return boxes[keep], scores[keep], classes[keep]


def detect_frame(model, frame: np.ndarray, args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (xyxy, conf, cls) for one frame, tiled or whole-frame."""
    predict_kw = dict(imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)
    if args.classes:
        predict_kw["classes"] = args.classes

    if not args.tile:
        r = model.predict(frame, **predict_kw)[0]
        return (r.boxes.xyxy.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
                r.boxes.cls.cpu().numpy().astype(int))

    h, w = frame.shape[:2]
    tile = args.tile_size
    crops, offsets = [], []
    for y0 in tile_origins(h, tile, args.tile_overlap):
        for x0 in tile_origins(w, tile, args.tile_overlap):
            crops.append(frame[y0:y0 + tile, x0:x0 + tile])
            offsets.append((x0, y0))

    all_boxes, all_scores, all_classes = [], [], []
    # Chunked so a 4K frame's ~32 tiles don't balloon peak memory on a 16 GB box.
    for i in range(0, len(crops), args.tile_batch):
        chunk = crops[i:i + args.tile_batch]
        for r, (x0, y0) in zip(model.predict(chunk, **predict_kw), offsets[i:i + args.tile_batch]):
            b = r.boxes.xyxy.cpu().numpy()
            if len(b) == 0:
                continue
            b[:, [0, 2]] += x0
            b[:, [1, 3]] += y0
            all_boxes.append(b)
            all_scores.append(r.boxes.conf.cpu().numpy())
            all_classes.append(r.boxes.cls.cpu().numpy().astype(int))

    if not all_boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

    return merge_boxes(np.concatenate(all_boxes), np.concatenate(all_scores),
                       np.concatenate(all_classes), args.iou)


def draw(frame: np.ndarray, boxes, scores, classes, names) -> np.ndarray:
    out = frame.copy()
    for (x1, y1, x2, y2), s, c in zip(boxes, scores, classes):
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, (0, 255, 0), 2)
        label = f"{names.get(int(c), int(c))} {s:.2f}"
        cv2.putText(out, label, (p1[0], max(12, p1[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def resolve_sources(source: Path) -> tuple[str, list[Path]]:
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


def main() -> None:
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
    args = ap.parse_args()

    kind, paths = resolve_sources(args.source)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.weights} ...")
    model = YOLO(args.weights)
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    print(f"Classes: {names}")
    if len(names) > 10 and not args.classes:
        print("  ! These look like generic COCO classes -- there is no 'drone' class in "
              "COCO. Use single-class drone weights, or pass --classes.", file=sys.stderr)

    jsonl = (args.out / "detections.jsonl").open("w", encoding="utf-8")
    writer = None
    n_frames = n_dets = n_empty = 0
    t0 = time.perf_counter()

    try:
        if kind == "video":
            cap = cv2.VideoCapture(str(paths[0]))
            if not cap.isOpened():
                sys.exit(f"Could not open video {paths[0]}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            idx = -1
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                idx += 1
                if idx % args.stride:
                    continue

                boxes, scores, classes = detect_frame(model, frame, args)
                jsonl.write(json.dumps({
                    "frame": idx,
                    "detections": [
                        {"bbox": [round(float(v), 1) for v in b],
                         "conf": round(float(s), 4),
                         "cls": int(c)}
                        for b, s, c in zip(boxes, scores, classes)
                    ],
                }) + "\n")
                n_frames += 1
                n_dets += len(boxes)
                n_empty += len(boxes) == 0

                if not args.no_save_frames:
                    if writer is None:
                        h, w = frame.shape[:2]
                        writer = cv2.VideoWriter(
                            str(args.out / "annotated.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps / max(1, args.stride), (w, h))
                    writer.write(draw(frame, boxes, scores, classes, names))

                if n_frames % 25 == 0:
                    rate = n_frames / (time.perf_counter() - t0)
                    print(f"  frame {idx}  |  {n_dets} dets  |  {rate:.2f} fps")
                if args.max_frames and n_frames >= args.max_frames:
                    break
            cap.release()
        else:
            if args.max_frames:
                paths = paths[::args.stride][:args.max_frames]
            frames_dir = args.out / "annotated"
            if not args.no_save_frames:
                frames_dir.mkdir(exist_ok=True)
            for i, p in enumerate(paths):
                frame = cv2.imread(str(p))
                if frame is None:
                    print(f"  ! unreadable: {p}", file=sys.stderr)
                    continue

                boxes, scores, classes = detect_frame(model, frame, args)
                jsonl.write(json.dumps({
                    "image": str(p),
                    "detections": [
                        {"bbox": [round(float(v), 1) for v in b],
                         "conf": round(float(s), 4),
                         "cls": int(c)}
                        for b, s, c in zip(boxes, scores, classes)
                    ],
                }) + "\n")
                n_frames += 1
                n_dets += len(boxes)
                n_empty += len(boxes) == 0

                if not args.no_save_frames:
                    cv2.imwrite(str(frames_dir / p.name), draw(frame, boxes, scores, classes, names))
                if (i + 1) % 25 == 0:
                    rate = n_frames / (time.perf_counter() - t0)
                    print(f"  {i + 1}/{len(paths)}  |  {n_dets} dets  |  {rate:.2f} fps")
    finally:
        jsonl.close()
        if writer is not None:
            writer.release()

    elapsed = time.perf_counter() - t0
    print(f"\n{n_frames} frames in {elapsed:.1f}s ({n_frames / max(elapsed, 1e-9):.2f} fps)")
    print(f"{n_dets} detections, {n_dets / max(n_frames, 1):.2f} per frame")
    # Most air-to-air frames contain exactly one drone, so on those datasets the
    # empty-frame rate is a rough miss rate -- the headline baseline number.
    print(f"Frames with no detection: {n_empty}/{n_frames} "
          f"({100 * n_empty / max(n_frames, 1):.1f}%)")
    print(f"Wrote {args.out / 'detections.jsonl'}")


if __name__ == "__main__":
    main()
