"""GLAD's state machine, as a stepper instead of a display loop.

Upstream `GLAD.py` is a module-level `while cap.isOpened()` with `imshow` calls
threaded through it, so it cannot be called or tested. The logic below is that
loop, verbatim, with the drawing removed and the state moved onto an object.

The machine alternates between two regimes:

    global   GAD over the full frame; on a miss, GMD over the full frame, whose
             candidate must then be confirmed by LAD inside a search region
             before the target counts as acquired.
    local    LAD inside a 320x320 region around the last box; on a miss, LMD
             inside the same region. 30 consecutive misses drop back to global.

**This is the `GAD+GMD+LAD+LMD` ablation row of the paper, not full GLAD.** The
released code has no Kalman filter and a fixed `a = 160` region rather than the
paper's adaptive `L = 300 + 4*T_lost`. The honest reproduction target is
therefore P 0.91 / R 0.81 / F1 0.86, not the headline 0.92 / 0.82 / 0.87.

At most one box comes out per frame: `MOD2_global` breaks on its first accepted
candidate and the machine tracks a single `init_rect`. That is architectural,
not a threshold — GLAD cannot report two drones.

The detectors and the motion module are constructor arguments, so the machine
can be stepped with stubs in a test; `from_release` is the wiring that loads the
real thing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..detections import Detections
from .classifier import load_gate
from .vendor import (GLAD_DIR, GLOBAL_WEIGHTS, LOCAL_WEIGHTS, add_import_roots,
                     import_motion, weights_dir)
from .yolo import (TRAINED_PAD, AcquireDetector, GlobalDetector, TrackingDetector,
                   Yolov5Backend)

REGION_HALF = 160  # upstream `a`; the search region is 2a x 2a = 320x320
MAX_LOCAL_MISSES = 30  # consecutive local failures before falling back to global

# Which branch produced a frame's result. Recorded per frame because it is the
# only way to see the paper's ablation in our own run: how much of the recall
# comes from appearance and how much from motion.
FIRST_FRAME = "first frame"
GLOBAL_YOLO = "global yolo"
GLOBAL_MOD = "global mod"
LOCAL_YOLO = "local yolo"
LOCAL_MOD = "local mod"
GLOBAL_MISS = "global miss"
LOCAL_MISS = "local miss"


class MotionModule(Protocol):
    """The two frame-differencing entry points of upstream `MOD2`.

    Both return `[]` for "nothing found" and an `(x, y, w, h)` tuple otherwise.
    """

    def MOD2_global(self, previous: np.ndarray,  # noqa: N802 -- upstream's name
                    current: np.ndarray) -> Any:
        ...

    def MOD2_local(self, previous: np.ndarray, current: np.ndarray,  # noqa: N802
                   x_prev: float, y_prev: float) -> Any:
        ...


def search_region(x: float, y: float, half: int, width: int,
                  height: int) -> tuple[int, int, int, int]:
    """The 2*half square search region anchored on a box, clamped to the frame.

    A verbatim port of upstream's `enlarge_region2`, quirks included, because
    both of them are load-bearing and both read like typos:

    * the box **top-left** is treated as the region centre, not the box centre;
    * clamping at the right or bottom edge **shifts** the region back inside the
      frame rather than shrinking it, so the region is always exactly square.

    `tests/unit/test_glad_geometry.py` pins this against the vendored original.
    """
    region_x = x - half
    region_y = y - half
    region_w = half * 2
    region_h = half * 2

    if region_x < 0:
        region_x = 0
    if region_y < 0:
        region_y = 0
    if region_x + region_w >= width:
        region_x = width - region_w
    if region_y + region_h >= height:
        region_y = height - region_h
    return int(region_x), int(region_y), int(region_w), int(region_h)


@dataclass(frozen=True)
class StepResult:
    """One frame's outcome: the box if there was one, and the branch that found it."""

    box: np.ndarray | None  # [x, y, w, h] in absolute frame pixels
    branch: str

    def as_detections(self, confidence: float) -> Detections:
        """Convert to the project's per-frame detection record.

        GLAD emits no confidence — its output rows are `[frame, x, y, w, h]` and
        the branches are not on a common scale — so every box is recorded at the
        same value. A ranking-based metric over a constant score is degenerate;
        score this run on precision/recall/F1 and disregard its AP.
        """
        if self.box is None:
            return Detections.empty()
        x, y, width, height = self.box
        return Detections(
            boxes=np.array([[x, y, x + width, y + height]], dtype=float),
            scores=np.array([confidence]),
            classes=np.zeros(1, dtype=int),
        )


class GladPipeline:
    """The global/local state machine over a contiguous frame sequence.

    Frames must arrive **in order and without gaps**: both motion branches
    difference the current frame against the previous one, so a strided sample
    would measure something else entirely. Call `reset()` between videos.
    """

    def __init__(self, global_detector: GlobalDetector,
                 tracking_detector: TrackingDetector,
                 acquire_detector: AcquireDetector,
                 motion: MotionModule) -> None:
        self.global_detector = global_detector
        self.tracking_detector = tracking_detector
        self.acquire_detector = acquire_detector
        self.motion = motion
        self.reset()

    @classmethod
    def from_release(cls, glad_dir: Path = GLAD_DIR,
                     pad_value: int = TRAINED_PAD) -> GladPipeline:
        """Build the pipeline from the checkpoints in a GLAD clone.

        `pad_value` reaches only the global detector — the local ones take square
        crops and never pad. Pass `RELEASED_PAD` to reproduce upstream exactly,
        bug included; see `src.algo.glad.yolo.PAD_STYLES`.
        """
        add_import_roots(glad_dir)  # first call wins, so this fixes the clone in use
        weights = weights_dir(glad_dir)
        global_backend = Yolov5Backend(weights / GLOBAL_WEIGHTS, pad_value)
        # Detector2 and Detector3 upstream load the same crop-trained engine.
        local_backend = Yolov5Backend(weights / LOCAL_WEIGHTS, pad_value)
        return cls(
            global_detector=GlobalDetector(global_backend),
            tracking_detector=TrackingDetector(local_backend),
            acquire_detector=AcquireDetector(local_backend),
            motion=import_motion(load_gate(weights), glad_dir),
        )

    def reset(self) -> None:
        """Clear all state, as if starting a new video."""
        self._prev: np.ndarray | None = None
        self._locked = False
        self._local_misses = 0
        self._region = (0, 0, 0, 0)  # search region (x, y, w, h) in frame pixels
        self._relative = (0, 0, 0, 0)  # last box (x, y, w, h) within that region

    def step(self, frame: np.ndarray) -> StepResult:
        """Process the next frame in sequence and return its result.

        The first frame of a video can only ever be a miss: there is no previous
        frame to difference against, so upstream skips it outright. It is
        returned as an empty result rather than swallowed, so that the frame
        still counts against recall.
        """
        if self._prev is None:
            self._prev = frame
            return StepResult(None, FIRST_FRAME)

        height, width = frame.shape[:2]
        result = (self._local_step(frame, width, height) if self._locked
                  else self._global_step(frame, width, height))
        self._prev = frame
        return result

    def _global_step(self, frame: np.ndarray, width: int, height: int) -> StepResult:
        """GAD over the full frame, falling back to GMD plus LAD confirmation."""
        box = self.global_detector.detect(frame)
        if box is not None:
            self._lock_on(box, width, height, rebase=True)
            return StepResult(box, GLOBAL_YOLO)

        candidate = self.motion.MOD2_global(self._prev, frame)
        if len(candidate) == 0:
            self._locked = False
            return StepResult(None, GLOBAL_MISS)

        # The motion candidate only proposes a location; LAD has to confirm it
        # at native scale inside the region before the target counts as found.
        x, y, box_w, box_h = candidate
        region_x, region_y, region_w, region_h = search_region(
            x, y, REGION_HALF, width, height)
        region = frame[region_y:region_y + region_h, region_x:region_x + region_w, :]

        confirmed = self.acquire_detector.detect(
            region, (x - region_x) + box_w / 2, (y - region_y) + box_h / 2)
        if confirmed is None:
            self._locked = False
            return StepResult(None, GLOBAL_MISS)

        absolute = _to_absolute(confirmed, region_x, region_y)
        self._lock_on(absolute, width, height, rebase=True)
        return StepResult(absolute, GLOBAL_MOD)

    def _local_step(self, frame: np.ndarray, width: int, height: int) -> StepResult:
        """LAD inside the search region, falling back to LMD in the same region."""
        region_x, region_y, region_w, region_h = self._region
        rows = slice(region_y, region_y + region_h)
        cols = slice(region_x, region_x + region_w)

        last_x, last_y, last_w, last_h = self._relative
        x_prev = last_x + last_w / 2
        y_prev = last_y + last_h / 2

        box = self.tracking_detector.detect(frame[rows, cols, :], x_prev, y_prev)
        branch = LOCAL_YOLO
        if box is None:
            found = self.motion.MOD2_local(self._prev[rows, cols, :],
                                           frame[rows, cols, :], x_prev, y_prev)
            box = np.array(found) if len(found) else None
            branch = LOCAL_MOD

        if box is None:
            self._local_misses += 1
            if self._local_misses == MAX_LOCAL_MISSES:
                self._locked = False
            return StepResult(None, LOCAL_MISS)

        absolute = _to_absolute(box, region_x, region_y)
        # `rebase=False`: upstream recentres the region on the new box but leaves
        # the stored relative position pointing into the *old* region. That makes
        # the next frame's anchor stale by one frame of target motion — a few
        # pixels against thresholds of 50 and 200, so it barely bites. Kept
        # because M4a measures the released code, not a corrected version.
        self._lock_on(absolute, width, height, rebase=False, relative=box)
        return StepResult(absolute, branch)

    def _lock_on(self, box: np.ndarray, width: int, height: int, *, rebase: bool,
                 relative: np.ndarray | None = None) -> None:
        """Enter the local regime with the search region centred on this box."""
        x, y, box_w, box_h = box
        region = search_region(x, y, REGION_HALF, width, height)
        self._region = region
        self._relative = (tuple(int(v) for v in (x - region[0], y - region[1],
                                                 box_w, box_h))
                          if rebase else tuple(int(v) for v in relative))
        self._locked = True
        self._local_misses = 0


def _to_absolute(box: np.ndarray, region_x: int, region_y: int) -> np.ndarray:
    """Shift a box from search-region coordinates into frame coordinates."""
    x, y, box_w, box_h = box
    return np.array([x + region_x, y + region_y, box_w, box_h])
