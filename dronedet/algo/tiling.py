"""Cutting a frame into overlapping tiles and stitching the results back.

Air-to-air targets are often 10-30 px in a 1080p or 4K frame, and a 640 px
letterbox destroys them before the detector sees them. Running the detector over
native-resolution crops avoids that; the cost is that boxes come back in tile
coordinates and duplicated across seams, which is what this module resolves.
"""

from __future__ import annotations

import numpy as np
import torch
from torchvision.ops import batched_nms


def tile_origins(total: int, tile: int, overlap: float) -> list[int]:
    """Start coordinates for overlapping tiles covering `total` pixels."""
    if total <= tile:
        return [0]
    step = max(1, int(tile * (1.0 - overlap)))
    origins = list(range(0, total - tile + 1, step))
    if origins[-1] != total - tile:
        origins.append(total - tile)
    return origins


def crop_grid(frame: np.ndarray, tile: int,
              overlap: float) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Overlapping tiles of `frame`, each paired with its (x, y) offset.

    Overlap exists so a drone straddling a tile boundary lands whole inside at
    least one crop.
    """
    h, w = frame.shape[:2]
    crops, offsets = [], []
    for y0 in tile_origins(h, tile, overlap):
        for x0 in tile_origins(w, tile, overlap):
            crops.append(frame[y0:y0 + tile, x0:x0 + tile])
            offsets.append((x0, y0))
    return crops, offsets


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
