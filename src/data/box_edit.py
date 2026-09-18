"""Box geometry for the interactive annotator: grabbing, dragging, zooming.

Kept apart from the window code so the parts that decide where a box ends up --
and therefore what lands in a label file -- are testable without a display.

Every box here is `(x, y, w, h)` in **original-frame pixels**, the same
convention `src.data.seed_track` writes labels from. The display is only ever a
view onto the frame; nothing is stored in display coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Box = tuple[float, float, float, float]

MOVE = "move"
MIN_SIZE = 2.0  # below this a template has no structure to correlate against


def corners(box: Box) -> tuple[float, float, float, float]:
    """`(x, y, w, h)` -> `(x0, y0, x1, y1)`."""
    x, y, w, h = box
    return x, y, x + w, y + h


def from_corners(x0: float, y0: float, x1: float, y1: float) -> Box:
    """Two opposite corners in any order -> a box with positive size.

    Dragging a corner past its opposite one inverts the box; normalising here
    means the caller never has to care which way the user dragged.
    """
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    return left, top, right - left, bottom - top


def clamp_box(box: Box, width: int, height: int) -> Box:
    """Clip a box to the frame, keeping it at least `MIN_SIZE` on each side."""
    x0, y0, x1, y1 = corners(box)
    x0, x1 = min(max(x0, 0.0), width - MIN_SIZE), min(max(x1, MIN_SIZE), float(width))
    y0, y1 = min(max(y0, 0.0), height - MIN_SIZE), min(max(y1, MIN_SIZE), float(height))
    x1, y1 = max(x1, x0 + MIN_SIZE), max(y1, y0 + MIN_SIZE)
    return x0, y0, x1 - x0, y1 - y0


def hit_test(box: Box, point: tuple[float, float], tol: float) -> str | None:
    """Which part of `box` the point grabs: a compass handle, `MOVE`, or None.

    Handles are named by the edges they move (`"nw"`, `"e"`, ...). Corners win
    over edges, edges over the interior. The tolerance shrinks with the box so
    the centre of even a 6 px target still grabs as `MOVE` -- a fixed
    tolerance would turn a tiny box into nothing but handles.
    """
    x0, y0, x1, y1 = corners(box)
    px, py = point
    tol = min(tol, (x1 - x0) / 4, (y1 - y0) / 4)
    if not (x0 - tol <= px <= x1 + tol and y0 - tol <= py <= y1 + tol):
        return None
    vertical = "n" if abs(py - y0) <= tol else "s" if abs(py - y1) <= tol else ""
    horizontal = "w" if abs(px - x0) <= tol else "e" if abs(px - x1) <= tol else ""
    return (vertical + horizontal) or MOVE


def drag(box: Box, handle: str, dx: float, dy: float) -> Box:
    """The box after dragging `handle` by `(dx, dy)` image pixels.

    Always applied to the box as it was when the button went down, with the
    *total* offset -- not incrementally per mouse event, which would accumulate
    rounding into a box that creeps.
    """
    x0, y0, x1, y1 = corners(box)
    if handle == MOVE:
        return x0 + dx, y0 + dy, x1 - x0, y1 - y0
    if "w" in handle:
        x0 += dx
    if "e" in handle:
        x1 += dx
    if "n" in handle:
        y0 += dy
    if "s" in handle:
        y1 += dy
    return from_corners(x0, y0, x1, y1)


@dataclass
class Viewport:
    """Which part of the frame the window shows, and at what magnification.

    `zoom` is display pixels per image pixel; `cx, cy` is the image point at
    the centre of the window. Zooming matters more here than in most
    annotators: a 12 px drone fitted into a laptop window is 9 px across, and
    a box edge placed at that scale is off by a large fraction of the target.
    """

    frame_size: tuple[int, int]    # image (width, height)
    display_size: tuple[int, int]  # window (width, height)
    zoom: float = 0.0
    cx: float = 0.0
    cy: float = 0.0

    MAX_ZOOM = 16.0

    def __post_init__(self) -> None:
        if self.zoom <= 0:
            self.reset()

    @property
    def fit_zoom(self) -> float:
        """The zoom at which the whole frame fits the window."""
        return min(self.display_size[0] / self.frame_size[0],
                   self.display_size[1] / self.frame_size[1])

    @property
    def zoomed(self) -> bool:
        """True once the view is magnified beyond the whole-frame fit."""
        return self.zoom > self.fit_zoom * 1.01

    def reset(self) -> None:
        """Back to the whole frame, centred."""
        self.zoom = self.fit_zoom
        self.cx, self.cy = self.frame_size[0] / 2, self.frame_size[1] / 2

    def to_image(self, px: float, py: float) -> tuple[float, float]:
        """Window pixel -> image pixel."""
        return (self.cx + (px - self.display_size[0] / 2) / self.zoom,
                self.cy + (py - self.display_size[1] / 2) / self.zoom)

    def to_display(self, x: float, y: float) -> tuple[float, float]:
        """Image pixel -> window pixel."""
        return ((x - self.cx) * self.zoom + self.display_size[0] / 2,
                (y - self.cy) * self.zoom + self.display_size[1] / 2)

    def matrix(self) -> np.ndarray:
        """The 2x3 affine taking image pixels to window pixels, for `warpAffine`."""
        tx, ty = self.to_display(0.0, 0.0)
        return np.array([[self.zoom, 0.0, tx], [0.0, self.zoom, ty]])

    def zoom_at(self, px: float, py: float, factor: float) -> None:
        """Zoom by `factor`, keeping the image point under the cursor fixed."""
        anchor = self.to_image(px, py)
        self.zoom = min(max(self.zoom * factor, self.fit_zoom), self.MAX_ZOOM)
        # Solve for the centre that puts `anchor` back under (px, py).
        self.cx = anchor[0] - (px - self.display_size[0] / 2) / self.zoom
        self.cy = anchor[1] - (py - self.display_size[1] / 2) / self.zoom

    def pan(self, dx_display: float, dy_display: float) -> None:
        """Move the view by a drag of `(dx, dy)` window pixels."""
        self.cx -= dx_display / self.zoom
        self.cy -= dy_display / self.zoom

    def focus(self, box: Box, fraction: float = 0.2) -> None:
        """Centre on a box, zoomed so it spans about `fraction` of the window."""
        x, y, w, h = box
        self.cx, self.cy = x + w / 2, y + h / 2
        want = fraction * min(self.display_size) / max(w, h, 1.0)
        self.zoom = min(max(want, self.fit_zoom), self.MAX_ZOOM)

    def keep_in_view(self, box: Box, margin: float = 0.2) -> bool:
        """Recentre on a box that has left the middle of a zoomed view.

        What keeps a zoomed-in session usable while playing: without it the
        target walks out of the window within a second and the user has to
        chase it by hand. Returns whether the view moved.
        """
        if not self.zoomed:
            return False
        x, y, w, h = box
        px, py = self.to_display(x + w / 2, y + h / 2)
        width, height = self.display_size
        inside = (margin * width <= px <= (1 - margin) * width
                  and margin * height <= py <= (1 - margin) * height)
        if inside:
            return False
        self.cx, self.cy = x + w / 2, y + h / 2
        return True
