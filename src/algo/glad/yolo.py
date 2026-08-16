"""CPU replacements for GLAD's three TensorRT detectors.

Upstream ships `detector1_trt.py`, `detector2_trt.py` and `detector3_trt.py`.
They are the same class three times over, differing only in a confidence
threshold and in which box they pick out of the survivors — and none of them run
here, because they deserialise TensorRT 7.2 engines onto `cuda.Device(1)`.

The two PyTorch checkpoints ship in the same folder, so the network is not the
problem; only the runtime is. Each class below pairs one loaded checkpoint with
one upstream operating point:

| Class | Checkpoint | conf | Selection rule |
| --- | --- | --- | --- |
| `GlobalDetector` (was Detector1) | `yolov5s_GLAD.pt` | 0.5 | highest confidence |
| `TrackingDetector` (was Detector2) | `yolov5s_GLAD-crop.pt` | 0.1 | nearest *and* most confident within 50 px of the last box |
| `AcquireDetector` (was Detector3) | `yolov5s_GLAD-crop.pt` | 0.5 | nearest within 10 px of the motion candidate |

**Preprocessing follows tensorrtx, not yolov5.** The engines were built with
`wang-xinyu/tensorrtx`, which letterboxes to a fixed 640x640 rather than
yolov5's own pad-to-a-stride-multiple. Matching the engine, not the framework,
is what makes these weights behave as they were run — down to the padding
colour, which is not the one upstream thinks it is (see `PAD_VALUE`).

The confidence is likewise the tensorrtx convention, `objectness x class score`,
which is also what yolov5's own NMS uses. Two remaining differences from the
engines are unavoidable and are not expected to move a detection: fp32 here
against the engine's build precision, and tensorrtx's 1000-box output cap, which
a single-drone frame never approaches.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import torch

from .vendor import add_import_roots, trusted_torch_load

INPUT_W = 640
INPUT_H = 640
IOU_THRESHOLD = 0.4  # identical across all three upstream detectors

# What fills the letterbox bars. Only the *global* detector is affected: the
# local detectors take square 320x320 crops, which scale to 640x640 with no
# padding at all. A 1080p frame, by contrast, is 44% padding.
#
# Upstream means to use tensorrtx's 128 and writes
# `copyMakeBorder(img, t, b, l, r, BORDER_CONSTANT, (128, 128, 128))`. That
# function's seventh positional parameter is `dst`, not `value`, so the tuple is
# discarded and the bars come out black. The weights themselves were trained by
# yolov5 v6.0, which fills with 114 in `letterbox`, in the mosaic base image and
# in the warp border alike -- so 114 is what the network actually learned to
# ignore, and TRAINED is the correct value rather than merely the intended one.
RELEASED_PAD = 0  # what GLAD's released code produces, bug included
TRAINED_PAD = 114  # what yolov5 v6.0 trained these weights against
TENSORRTX_PAD = 128  # what upstream intended, and what its C++ preprocessing uses

PAD_STYLES = {"released": RELEASED_PAD, "trained": TRAINED_PAD,
              "tensorrtx": TENSORRTX_PAD}


class Candidate(NamedTuple):
    """One surviving box in original-image pixels, as upstream leaves it.

    Corners are integers because upstream casts them before the selection loop,
    and that truncation is visible in the final coordinates.
    """

    x1: int
    y1: int
    x2: int
    y2: int
    conf: float

    @property
    def center(self) -> tuple[float, float]:
        """Box centre, the anchor for both distance-based selection rules."""
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    def as_xywh(self) -> np.ndarray:
        """Top-left plus size — the `[x, y, w, h]` every caller downstream expects."""
        return np.array([self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1])


class Yolov5Backend:
    """One loaded yolov5 v6.0 checkpoint, run on CPU.

    Shared between the two local detectors, which differ only in threshold and
    selection rule — loading `yolov5s_GLAD-crop.pt` twice would just cost memory.
    """

    def __init__(self, weights: Path, pad_value: int = TRAINED_PAD) -> None:
        add_import_roots()
        from models.experimental import attempt_load  # noqa: PLC0415

        with trusted_torch_load():
            self.model = attempt_load(str(weights), map_location="cpu")
        self.model.eval()
        self.pad_value = pad_value

    def raw_predictions(self, image: np.ndarray) -> tuple[np.ndarray, int, int]:
        """Decoded boxes for one image, plus its original height and width.

        Rows are `[cx, cy, w, h, conf, cls]` in letterboxed 640x640 space, which
        is the layout tensorrtx's plugin emits and what the upstream
        post-processing expects.
        """
        height, width = image.shape[:2]
        tensor = torch.from_numpy(_letterbox(image, self.pad_value)).unsqueeze(0)
        with torch.no_grad():
            pred = self.model(tensor)[0][0].numpy()

        conf = pred[:, 4] * pred[:, 5:].max(axis=1)
        classes = pred[:, 5:].argmax(axis=1)
        return np.column_stack([pred[:, :4], conf, classes]), height, width


def _letterbox(image: np.ndarray, pad_value: int) -> np.ndarray:
    """tensorrtx's preprocessing: RGB, fit inside 640x640, pad, scale to [0,1]."""
    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ratio_w, ratio_h = INPUT_W / width, INPUT_H / height
    if ratio_h > ratio_w:
        target_w, target_h = INPUT_W, int(ratio_w * height)
        pad_x1 = pad_x2 = 0
        pad_y1 = int((INPUT_H - target_h) / 2)
        pad_y2 = INPUT_H - target_h - pad_y1
    else:
        target_w, target_h = int(ratio_h * width), INPUT_H
        pad_x1 = int((INPUT_W - target_w) / 2)
        pad_x2 = INPUT_W - target_w - pad_x1
        pad_y1 = pad_y2 = 0

    resized = cv2.resize(rgb, (target_w, target_h))
    # Two things this call has to get right, both of which fail silently:
    # `value=` by keyword, because the positional slot is `dst` (the upstream
    # bug), and a full triple, because a bare scalar widens to (v, 0, 0) and
    # tints the bars.
    padded = cv2.copyMakeBorder(resized, pad_y1, pad_y2, pad_x1, pad_x2,
                                cv2.BORDER_CONSTANT, value=(pad_value,) * 3)
    chw = np.transpose(padded.astype(np.float32) / 255.0, [2, 0, 1])
    return np.ascontiguousarray(chw)


def _to_original(boxes: np.ndarray, height: int, width: int) -> np.ndarray:
    """Undo the letterbox: 640-space `cx,cy,w,h` -> original-pixel `x,y,w,h`.

    A port of upstream's `xywh2xyxy`, which despite the name returns xywh.
    """
    out = np.zeros_like(boxes)
    ratio_w, ratio_h = INPUT_W / width, INPUT_H / height
    if ratio_h > ratio_w:
        offset = (INPUT_H - ratio_w * height) / 2
        out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2 - offset
        out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2 - offset
        out /= ratio_w
    else:
        offset = (INPUT_W - ratio_h * width) / 2
        out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2 - offset
        out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2 - offset
        out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        out /= ratio_h

    out[:, 2] -= out[:, 0]
    out[:, 3] -= out[:, 1]
    return out


class GladDetector(ABC):
    """One upstream detector: a threshold, NMS, and a rule for picking one box.

    GLAD tracks a single target throughout, so every detector collapses its
    survivors to at most one box. `detect` returns `[x, y, w, h]` or None.
    """

    CONF_THRESH: float

    def __init__(self, backend: Yolov5Backend) -> None:
        self.backend = backend

    def candidates(self, image: np.ndarray) -> list[Candidate]:
        """Thresholded, NMS'd boxes in original-image pixels."""
        pred, height, width = self.backend.raw_predictions(image)
        keep = pred[:, 4] > self.CONF_THRESH
        if not keep.any():
            return []

        boxes = _to_original(pred[keep, :4], height, width).tolist()
        scores = pred[keep, 4].tolist()
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=self.CONF_THRESH,
                                   nms_threshold=IOU_THRESHOLD)
        survivors = []
        for i in np.array(indices).flatten():
            x, y, box_w, box_h = boxes[int(i)]
            survivors.append(Candidate(int(x), int(y), int(x + box_w), int(y + box_h),
                                       float(scores[int(i)])))
        return survivors

    def detect(self, image: np.ndarray, *anchor: float) -> np.ndarray | None:
        """Run the detector and apply its selection rule."""
        return self._select(self.candidates(image), *anchor)

    @abstractmethod
    def _select(self, candidates: list[Candidate], *anchor: float) -> np.ndarray | None:
        ...


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two points."""
    return float(np.hypot(x2 - x1, y2 - y1))


class GlobalDetector(GladDetector):
    """GAD — full-frame appearance detection (upstream `Detector1`).

    Takes the most confident box. With `ref_score` starting at the same 0.5 as
    the threshold this is just an argmax, but it is spelled out as upstream has
    it so the two cannot drift if either constant is ever revisited.
    """

    CONF_THRESH = 0.5

    def _select(self, candidates: list[Candidate]) -> np.ndarray | None:
        best, ref_score = None, self.CONF_THRESH
        for candidate in candidates:
            if candidate.conf > ref_score:
                best, ref_score = candidate, candidate.conf
        return best.as_xywh() if best else None


class TrackingDetector(GladDetector):
    """LAD in the locked regime — search-region detection (upstream `Detector2`).

    Runs at a much lower threshold than the global detector: inside a 320x320
    region already known to contain the target, a weak response is evidence.

    The rule tightens *both* references as it goes, so a box only wins if it is
    more confident **and** closer than the current best — which makes the
    outcome order-dependent. Preserved as upstream wrote it.
    """

    CONF_THRESH = 0.1
    MAX_DISTANCE = 50  # px from the previous box centre, in region coordinates

    def _select(self, candidates: list[Candidate],
                x_prev: float, y_prev: float) -> np.ndarray | None:
        best, ref_score, ref_dist = None, self.CONF_THRESH, self.MAX_DISTANCE
        for candidate in candidates:
            x_now, y_now = candidate.center
            distance = _distance(x_prev, y_prev, x_now, y_now)
            if candidate.conf > ref_score and distance < ref_dist:
                best, ref_score, ref_dist = candidate, candidate.conf, distance
        return best.as_xywh() if best else None


class AcquireDetector(GladDetector):
    """LAD confirming a global motion candidate (upstream `Detector3`).

    Motion has already proposed a location, so this only has to agree with it:
    confidence beyond the 0.5 threshold is ignored and the nearest box within
    10 px wins. That tight radius is what makes the motion branch's precision
    hold up — it refines a blob into a box rather than finding a new target.
    """

    CONF_THRESH = 0.5
    MAX_DISTANCE = 10  # px from the motion candidate's centre

    def _select(self, candidates: list[Candidate],
                x_prev: float, y_prev: float) -> np.ndarray | None:
        best, ref_dist = None, self.MAX_DISTANCE
        for candidate in candidates:
            x_now, y_now = candidate.center
            distance = _distance(x_prev, y_prev, x_now, y_now)
            if distance < ref_dist:
                best, ref_dist = candidate, distance
        return best.as_xywh() if best else None
