"""Following one box from frame to frame by template correlation.

Lifted out of `seed_track.py`, which was simultaneously a documented CLI and the
tracking library that `annotate.py` and `annotation.py` imported from. Importing
a command-line tool to get at a class is how a CLI's argument parsing ends up
running at import time; this module has no parser and no `__main__`.

Deliberately not frame differencing. The camera is on a moving drone, and after
compensation the difference image carries ~100 terrain blobs per frame besides
the target (EXP-012b) -- it says something moved, not which thing is the box the
user drew. Correlation against that box knows what the target looks like.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DEFAULT_SEARCH = 4.0  # search window, in target sizes
# Measured on first_catch: 0.45 tracks the drone and stops when it is gone;
# 0.35 walks onto trees and rock and keeps going.
DEFAULT_MIN_SCORE = 0.45
DEFAULT_SCALES = (0.92, 1.0, 1.09)  # per-frame scale search, for a closing target
# Template blending is OFF by default, and that is a finding rather than a
# preference. With blending on, a track that slips off the drone onto terrain
# adopts the terrain as its template and then matches it at **0.99** -- the
# highest scores on the review sheet were the worst proposals, which is the
# opposite of what the sheet is for. Pinned to the seed appearance, a lost track
# dies within a few frames and says so. Raise it only for a target whose
# appearance changes slowly.
DEFAULT_UPDATE = 0.0


@dataclass(frozen=True)
class TrackPoint:
    """One frame's proposed box and how well it matched.

    `score` is the peak normalised correlation, 0-1. It is carried all the way
    to the contact sheet because it is the review order: the lowest scores are
    where the tracker let go.
    """

    frame: int
    box: tuple[float, float, float, float]  # x, y, w, h in frame pixels
    score: float



def search_window(box: tuple[float, float, float, float], search: float,
                  shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Search window around a box: `(x, y, w, h)`, clamped to the frame."""
    height, width = shape
    x, y, w, h = box
    pad_x, pad_y = w * search, h * search
    x0 = int(max(x - pad_x, 0))
    y0 = int(max(y - pad_y, 0))
    x1 = int(min(x + w + pad_x, width))
    y1 = int(min(y + h + pad_y, height))
    return x0, y0, max(x1 - x0, 1), max(y1 - y0, 1)


def greyscale(frame: np.ndarray) -> np.ndarray:
    """Greyscale view of a frame that may already be single channel."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame



class TemplateTracker:
    """Follows one patch by normalised cross-correlation in a local window.

    Deliberately small and inspectable. Three behaviours earn their place:

    * **a local search window**, so a repeated texture elsewhere in the frame
      cannot capture the track;
    * **a coarse scale search**, because a target being closed on grows, and a
      fixed-size template loses it within a second or two;
    * **an optional template blend, off by default**. Blending sounds like the
      obvious way to tolerate a changing target, and on this footage it is a
      trap: a track that slips onto terrain adopts the terrain and then matches
      it at 0.99, so the drift arrives wearing the highest confidence in the
      run. Pinned, the score is a real signal.
    """

    def __init__(self, frame: np.ndarray, box: tuple[float, float, float, float],
                 search: float = DEFAULT_SEARCH,
                 scales: tuple[float, ...] = DEFAULT_SCALES,
                 min_score: float = DEFAULT_MIN_SCORE,
                 update: float = DEFAULT_UPDATE) -> None:
        self.box = box
        self.search = search
        self.scales = scales
        self.min_score = min_score
        self.update = update
        template = self._patch(frame, box)
        if template is None:
            raise ValueError(f"seed box {box} does not lie inside the frame")
        self._template = template

    @staticmethod
    def _patch(frame: np.ndarray,
               box: tuple[float, float, float, float]) -> np.ndarray | None:
        """The greyscale patch under a box, or None if it is degenerate."""
        x, y, w, h = (int(round(v)) for v in box)
        if w < 2 or h < 2:
            return None
        patch = frame[max(y, 0):y + h, max(x, 0):x + w]
        if patch.shape[0] < 2 or patch.shape[1] < 2:
            return None
        return greyscale(patch)

    def step(self, frame: np.ndarray) -> tuple[tuple[float, float, float, float],
                                               float] | None:
        """Match the template in the next frame; None once it is lost."""
        grey = greyscale(frame)
        x0, y0, win_w, win_h = search_window(self.box, self.search, grey.shape[:2])
        window = grey[y0:y0 + win_h, x0:x0 + win_w]

        best: tuple[float, tuple[int, int, int, int]] | None = None
        for scale in self.scales:
            tw = max(int(round(self._template.shape[1] * scale)), 2)
            th = max(int(round(self._template.shape[0] * scale)), 2)
            if tw > window.shape[1] or th > window.shape[0]:
                continue
            template = cv2.resize(self._template, (tw, th),
                                  interpolation=cv2.INTER_LINEAR)
            result = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(result)
            if best is None or score > best[0]:
                best = (float(score), (x0 + loc[0], y0 + loc[1], tw, th))

        if best is None or best[0] < self.min_score:
            return None

        score, found = best
        self.box = (float(found[0]), float(found[1]), float(found[2]), float(found[3]))
        patch = self._patch(frame, self.box)
        if patch is not None and self.update > 0:
            resized = cv2.resize(patch, (self._template.shape[1],
                                         self._template.shape[0]))
            self._template = cv2.addWeighted(self._template, 1 - self.update,
                                             resized, self.update, 0)
        return self.box, score


# --- CLI -------------------------------------------------------------------
