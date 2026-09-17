"""GLAD's motion branches, ported so their constants can be moved off 1080p.

Upstream `MOD2` is tuned in absolute pixels for gentle 1920x1080 air-to-air
footage. On our O4 intercept clips it does not merely perform worse -- it
**switches itself off**:

    if len(rect_merge) > 50:      # 30 in the local variant
        print('too much bboxes')
        return []                  # third_party/GLAD/MOD2.py:59, :182

| Run | Frames | `too much bboxes` |
| --- | --- | --- |
| EXP-010 FIELD | 3,600 | 23 |
| EXP-011 SOFA-O4 | 7,386 | **3,595** |

Low-altitude flight over close, textured ground through a wide lens leaves a
large residual after a single-homography compensation, so the difference image
fills with blobs and the guard fires. GLAD's published ablation puts recall at
0.51 without the motion branches and 0.81 with them, and acquisition runs
through `MOD2_global`, so roughly half our frames were being handled by a
pipeline with its acquisition path disabled.

**What this port changes, and only when asked.** `MotionConfig.UPSTREAM` is
byte-for-byte upstream behaviour and is the default; `tests/integration/
test_motion_equivalence.py` pins it against the vendored module on real frames.
`MotionConfig.CLUTTER` keeps the candidates instead of discarding them.

The guard exists for a real reason -- the per-candidate loop runs corner
detection, optical flow and a CNN gate, so hundreds of candidates are genuinely
unaffordable. Ranking preserves that: score every candidate, keep the best
`max_candidates`, and spend exactly the budget upstream was willing to spend.

Only `MOD2` itself is ported. `motion_compensate`, `enlargebox` and the CNN gate
are imported from the vendored `Functions` unchanged, which keeps the surface
that could silently diverge as small as possible.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import cv2
import numpy as np

FEATURE_PARAMS = dict(maxCorners=30, qualityLevel=0.15, minDistance=3, blockSize=3)
LK_PARAMS = dict(winSize=(15, 15), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03))
DEGREES = 57.3  # upstream's hand-rolled radians-to-degrees, kept for equivalence


@dataclass(frozen=True)
class MotionConfig:
    """The absolute-pixel assumptions in `MOD2`, made explicit.

    Every default is upstream's value. A run that changes one is a documented
    variant of GLAD and must never be reported as GLAD.
    """

    blur_kernel: int = 11
    # Global thresholds at 5 + mean(frame difference); local at a flat 4.
    global_threshold_base: int = 5
    local_threshold: int = 4
    blob_area: tuple[int, int] = (30, 3000)
    global_blob_ratio: tuple[float, float] = (0.6, 3.0)
    local_blob_ratio: tuple[float, float] = (0.5, 3.0)
    # The bail-out. Upstream returns nothing at all above these counts.
    max_candidates_global: int = 50
    max_candidates_local: int = 30
    rank_when_crowded: bool = False
    enlarge: int = 2
    global_min_dist: float = 1.0
    global_max_ratio: float = 0.8
    local_min_dist: float = 0.6
    local_max_ratio: float = 1.0
    dist_ref: float = 200.0

    @property
    def label(self) -> str:
        """One line naming what differs from upstream, for a run's log."""
        if self == UPSTREAM:
            return "upstream (MOD2 verbatim)"
        parts = []
        if self.rank_when_crowded:
            parts.append("rank when crowded")
        if self.blob_area != UPSTREAM.blob_area:
            parts.append(f"blob area {self.blob_area[0]}-{self.blob_area[1]}")
        if (self.max_candidates_global, self.max_candidates_local) != (
                UPSTREAM.max_candidates_global, UPSTREAM.max_candidates_local):
            parts.append(f"candidates {self.max_candidates_global}/"
                         f"{self.max_candidates_local}")
        return "variant: " + ", ".join(parts or ["unnamed change"])


UPSTREAM = MotionConfig()

# For low-altitude flight over ground clutter. Two changes, both measured
# problems rather than guesses: keep ranked candidates instead of discarding
# every one of them, and raise the blob-area ceiling, because a target at the
# moment of a catch exceeds 3000 px^2 (~55x55) and becomes invisible to the
# motion path at any threshold. The confirmed drone in EXP-011 is 80x34 = 2720,
# already within a few percent of the old ceiling.
CLUTTER = replace(UPSTREAM, rank_when_crowded=True, blob_area=(30, 12000))

PROFILES = {"upstream": UPSTREAM, "clutter": CLUTTER}


def candidate_score(contour_area: float, width: int, height: int) -> float:
    """How target-like a blob's shape is, 0-1. Higher is kept first.

    Used only to choose which candidates to spend the expensive per-candidate
    loop on when there are more than the budget allows. A drone is compact and
    roughly square; residual from ground parallax tends to be thin, ragged and
    elongated. So: how much of its bounding box the contour fills, times how
    close to square that box is.

    This decides *order*, never acceptance -- every kept candidate still faces
    the same optical-flow and CNN tests upstream applies.
    """
    if width <= 0 or height <= 0:
        return 0.0
    fill = contour_area / (width * height)
    aspect = width / height
    squareness = min(aspect, 1 / aspect)
    return float(fill * squareness)


def find_candidates(binary: np.ndarray, area_range: tuple[int, int],
                    ratio_range: tuple[float, float]
                    ) -> list[tuple[tuple[int, int, int, int], float]]:
    """Blobs in a thresholded difference image, with their shape scores."""
    contours, _ = cv2.findContours(binary.copy(), cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if h == 0:
            continue
        ratio = w / h
        if area_range[0] < area < area_range[1] and ratio_range[0] < ratio < ratio_range[1]:
            found.append(((x, y, w, h), candidate_score(area, w, h)))
    return found


def select(candidates: list[tuple[tuple[int, int, int, int], float]],
           limit: int, rank_when_crowded: bool
           ) -> list[tuple[int, int, int, int]]:
    """Apply the crowding policy: upstream discards everything, we rank.

    Order is preserved when the list is within budget, so the upstream path is
    bit-identical rather than merely equivalent -- `MOD2_global` breaks on its
    first accepted candidate, so order decides the answer.
    """
    if len(candidates) <= limit:
        return [box for box, _ in candidates]
    if not rank_when_crowded:
        return []
    ranked = sorted(candidates, key=lambda item: item[1], reverse=True)
    return [box for box, _ in ranked[:limit]]


def _moving_coherently(previous_crop: np.ndarray, current_crop: np.ndarray,
                       min_dist: float, max_ratio: float) -> bool:
    """Upstream's motion classifier: does this patch move as one rigid thing?

    Corners are tracked from the compensated previous crop into the current
    one. A real target moves a consistent distance in a consistent direction;
    mis-compensated background scatters.
    """
    points = cv2.goodFeaturesToTrack(current_crop, mask=None, **FEATURE_PARAMS)
    if points is None:
        return False
    moved, status, _ = cv2.calcOpticalFlowPyrLK(previous_crop, current_crop,
                                                points, None, **LK_PARAMS)
    good_new = moved[status == 1]
    good_old = points[status == 1]
    if len(good_new) < 1:
        return False

    distances, angles = [], []
    for new, old in zip(good_new, good_old):
        ax, ay = new.ravel()
        cx, cy = old.ravel()
        distances.append(np.sqrt((ax - cx) ** 2 + (ay - cy) ** 2))
        angles.append(DEGREES * np.arctan2(cy - ay, cx - ax))
    distances = np.array(distances)
    angles = np.array(angles)

    ratio_theta = np.std(angles) / np.mean(angles)
    ratio_dist = np.std(distances) / np.mean(distances)
    return not (np.mean(distances) < min_dist or ratio_theta > max_ratio
                or ratio_dist > max_ratio)


class MotionPort:
    """`MOD2_global` / `MOD2_local` with their constants exposed.

    Satisfies `src.algo.glad.pipeline.MotionModule`, so it drops in wherever
    the vendored module does.
    """

    def __init__(self, gate: Callable[[np.ndarray], int],
                 compensate: Callable[..., Any],
                 compensate_local: Callable[..., Any],
                 enlargebox: Callable[..., tuple[int, int, int, int]],
                 config: MotionConfig = UPSTREAM,
                 hud_mask: np.ndarray | None = None) -> None:
        self.gate = gate
        self.compensate = compensate
        self.compensate_local = compensate_local
        self.enlargebox = enlargebox
        self.config = config
        # Overlay pixels are removed from the difference image before contours
        # are found, which is the same mechanism upstream already uses for the
        # warp border (`thresh1 = thresh - mask`).
        self.hud_mask = hud_mask

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        """Blur and grey a frame, exactly as upstream does before differencing."""
        kernel = self.config.blur_kernel
        return cv2.cvtColor(cv2.GaussianBlur(frame, (kernel, kernel), 0),
                            cv2.COLOR_BGR2GRAY)

    def _binary(self, difference: np.ndarray, threshold: int,
                border: np.ndarray, median: bool) -> np.ndarray:
        """Threshold, drop the warp border and any overlay, then clean up."""
        _, binary = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)
        binary = binary - border
        # Only the global branch differences a whole frame; the local one is
        # handed a 320x320 crop and does not know where in the frame it came
        # from. Matching on shape applies the mask exactly where it means
        # something. Overlay inside a search region is still caught, one step
        # later, by the pipeline's veto on the box itself.
        if self.hud_mask is not None and self.hud_mask.shape == binary.shape:
            binary[self.hud_mask] = 0
        if median:
            binary = cv2.medianBlur(binary, 5)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, element, iterations=1)
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, element, iterations=3)

    def MOD2_global(self, frame1: np.ndarray,  # noqa: N802 -- upstream's name
                    frame2: np.ndarray) -> Any:
        """Full-frame motion detection; the first accepted candidate wins."""
        config = self.config
        height, width = frame1.shape[:2]
        previous = self._prepare(frame1)
        current = self._prepare(frame2)
        compensated, border, _ = self.compensate(previous, current)

        difference = cv2.absdiff(current, compensated)
        threshold = config.global_threshold_base + int(np.mean(difference))
        binary = self._binary(difference, threshold, border, median=True)

        candidates = find_candidates(binary, config.blob_area,
                                     config.global_blob_ratio)
        for x0, y0, w0, h0 in select(candidates, config.max_candidates_global,
                                     config.rank_when_crowded):
            box = self.enlargebox(x0, y0, w0, h0, config.enlarge, width, height)
            x1, y1, w1, h1 = box
            if not config.global_blob_ratio[0] < w1 / h1 < config.global_blob_ratio[1]:
                continue
            if not _moving_coherently(compensated[y1:y1 + h1, x1:x1 + w1],
                                      current[y1:y1 + h1, x1:x1 + w1],
                                      config.global_min_dist, config.global_max_ratio):
                continue
            if self.gate(frame1[y1:y1 + h1, x1:x1 + w1, :]) == 1:
                return box
        return []

    def MOD2_local(self, frame1: np.ndarray, frame2: np.ndarray,  # noqa: N802
                   x_prev: float, y_prev: float) -> Any:
        """Motion detection inside the search region; nearest candidate wins."""
        config = self.config
        height, width = frame1.shape[:2]
        previous = self._prepare(frame1)
        current = self._prepare(frame2)
        compensated, border, _ = self.compensate_local(previous, current)

        difference = cv2.absdiff(current, compensated)
        binary = self._binary(difference, config.local_threshold, border,
                              median=False)

        candidates = find_candidates(binary, config.blob_area,
                                     config.local_blob_ratio)
        best: tuple[int, int, int, int] | list = []
        nearest = config.dist_ref
        for x0, y0, w0, h0 in select(candidates, config.max_candidates_local,
                                     config.rank_when_crowded):
            box = self.enlargebox(x0, y0, w0, h0, config.enlarge, width, height)
            x1, y1, w1, h1 = box
            if not config.global_blob_ratio[0] < w1 / h1 < config.global_blob_ratio[1]:
                continue
            if not _moving_coherently(compensated[y1:y1 + h1, x1:x1 + w1],
                                      current[y1:y1 + h1, x1:x1 + w1],
                                      config.local_min_dist, config.local_max_ratio):
                continue
            if self.gate(frame1[y1:y1 + h1, x1:x1 + w1, :]) == 1:
                distance = np.sqrt((x1 + w1 / 2 - x_prev) ** 2
                                   + (y1 + h1 / 2 - y_prev) ** 2)
                if distance < nearest:
                    best = box
                    nearest = distance
        return best
