"""Loading the detector and running it over a frame."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
from ultralytics import YOLO

from .config import InferenceConfig
from .detections import Detections
from .tiling import crop_grid, merge_boxes

# Above this many classes the weights are almost certainly generic COCO.
COCO_CLASS_COUNT_HINT = 10


def load_model(weights: str, class_filter: list[int] | None) -> tuple[Any, dict[int, str]]:
    """Load the detector and warn if the weights look like generic COCO.

    COCO has no `drone` class, so a COCO checkpoint will quietly score ~100%
    misses on air-to-air data. Catching that at load time saves a wasted run.
    """
    print(f"Loading {weights} ...")
    model = YOLO(weights)
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    print(f"Classes: {names}")
    if len(names) > COCO_CLASS_COUNT_HINT and not class_filter:
        print("  ! These look like generic COCO classes -- there is no 'drone' class in "
              "COCO. Use single-class drone weights, or pass --classes.", file=sys.stderr)
    return model, names


def unpack_result(result) -> Detections:
    """Convert one ultralytics result into `Detections`."""
    return Detections(
        boxes=result.boxes.xyxy.cpu().numpy(),
        scores=result.boxes.conf.cpu().numpy(),
        classes=result.boxes.cls.cpu().numpy().astype(int),
    )


def detect_tiled(model, frame: np.ndarray, cfg: InferenceConfig) -> Detections:
    """Detect on overlapping native-resolution crops, merged to frame coordinates."""
    crops, offsets = crop_grid(frame, cfg.tile_size, cfg.tile_overlap)
    predict_kw = cfg.predict_kwargs()

    all_boxes, all_scores, all_classes = [], [], []
    # Chunked so a 4K frame's ~32 tiles don't balloon peak memory on a 16 GB box.
    for i in range(0, len(crops), cfg.tile_batch):
        chunk = crops[i:i + cfg.tile_batch]
        chunk_offsets = offsets[i:i + cfg.tile_batch]
        for result, (x0, y0) in zip(model.predict(chunk, **predict_kw), chunk_offsets):
            dets = unpack_result(result)
            if len(dets) == 0:
                continue
            boxes = dets.boxes
            boxes[:, [0, 2]] += x0
            boxes[:, [1, 3]] += y0
            all_boxes.append(boxes)
            all_scores.append(dets.scores)
            all_classes.append(dets.classes)

    if not all_boxes:
        return Detections.empty()

    return Detections(*merge_boxes(np.concatenate(all_boxes), np.concatenate(all_scores),
                                   np.concatenate(all_classes), cfg.iou))


def detect_frame(model, frame: np.ndarray, cfg: InferenceConfig) -> Detections:
    """Detections for one frame, tiled or whole-frame per `cfg`."""
    if cfg.tile:
        return detect_tiled(model, frame, cfg)
    return unpack_result(model.predict(frame, **cfg.predict_kwargs())[0])
