"""Laying a run's detections out as one image, so many of them can be judged at once.

A rendered video answers "what happened"; a contact sheet answers "what is it
finding". On unlabelled footage the second question is the only one available,
and it is answered by looking -- 800 boxes is far too many to open one at a
time, and far too few to summarise as a number nobody can check.

Two decisions do the work here:

**Crop tight, scale by the box.** A fixed window makes a 6 px blob and a 30 px
drone both unreadable, one because it is lost and the other because it is
clipped. The window is a multiple of the box's longest side, floored so a
sub-10 px target still gets context around it, and the box is drawn in so a
viewer can tell what the detector claimed from what happened to be nearby.

**Group by whatever the run recorded.** GLAD writes `branch`, and its rare
branches are the interesting ones -- `global yolo` acquiring from cold fires six
times in 3,600 frames. Sorted into blocks and coloured, they are visible; in
frame order they are six cells lost among hundreds.

Nothing here touches the filesystem or decodes video: frames go in, one image
comes out.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

CAPTION = 15          # height of the strip under each cell, in canvas pixels
BORDER = 2
BACKDROP = (18, 18, 18)
UNGROUPED = (150, 150, 150)

# Appearance branches cool, motion branches warm, so the two families separate
# at a glance before the legend is read. Anything not listed falls back to grey.
GROUP_COLOURS = {
    "global yolo": (240, 220, 100),
    "local yolo": (120, 240, 120),
    "global mod": (200, 120, 250),
    "local mod": (90, 170, 250),
}
GROUP_ORDER = ("global yolo", "local yolo", "global mod", "local mod")


@dataclass(frozen=True)
class Layout:
    """How big the sheet's cells are and how many sit in a row."""

    cell: int = 104
    cols: int = 26
    window: float = 4.0      # crop side as a multiple of the box's longest side
    min_window: int = 40     # ...but never tighter than this, in source pixels

    def source_window(self, box: np.ndarray, shape: tuple[int, int]) -> int:
        """Side of the square source crop for one box, clamped to the frame."""
        x0, y0, x1, y1 = box
        height, width = shape
        want = max(self.min_window, self.window * max(x1 - x0, y1 - y0))
        return int(round(min(want, width, height)))


def colour_for(group: str) -> tuple[int, int, int]:
    """Border colour for a group label."""
    return GROUP_COLOURS.get(group, UNGROUPED)


def group_sort_key(group: str) -> tuple[int, str]:
    """Rare-and-interesting branches first, then anything else alphabetically."""
    return ((GROUP_ORDER.index(group), "") if group in GROUP_ORDER
            else (len(GROUP_ORDER), group))


def cell(image: np.ndarray, box: np.ndarray, caption: str, group: str,
         layout: Layout = Layout()) -> np.ndarray | None:
    """One crop, centred on `box`, with the box drawn and a caption under it.

    None when the box has no overlap with the frame at all, which is the only
    case that cannot be rendered into something a viewer could read.
    """
    height, width = image.shape[:2]
    x0, y0, x1, y1 = (float(v) for v in box)
    side = layout.source_window(np.array([x0, y0, x1, y1]), (height, width))
    left = int(round(max(0, min(width - side, (x0 + x1) / 2 - side / 2))))
    top = int(round(max(0, min(height - side, (y0 + y1) / 2 - side / 2))))

    crop = image[top:top + side, left:left + side]
    if crop.size == 0:
        return None
    scale = layout.cell / crop.shape[0]
    crop = cv2.resize(crop, (layout.cell, layout.cell),
                      interpolation=cv2.INTER_NEAREST)

    colour = colour_for(group)
    cv2.rectangle(crop, (int((x0 - left) * scale), int((y0 - top) * scale)),
                  (int((x1 - left) * scale), int((y1 - top) * scale)), colour, 1)

    tile = np.full((layout.cell + 2 * BORDER + CAPTION,
                    layout.cell + 2 * BORDER, 3), BACKDROP, dtype=np.uint8)
    tile[:BORDER + layout.cell + BORDER, :] = colour
    tile[BORDER:BORDER + layout.cell, BORDER:BORDER + layout.cell] = crop
    tile[BORDER + layout.cell:, :] = BACKDROP
    cv2.putText(tile, caption, (BORDER + 1, layout.cell + BORDER + 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, colour, 1, cv2.LINE_AA)
    return tile


def grid(tiles: list[np.ndarray], layout: Layout = Layout()) -> np.ndarray:
    """Tiles laid out row-major, the last row padded with backdrop."""
    if not tiles:
        raise ValueError("no tiles to lay out")
    height, width = tiles[0].shape[:2]
    rows = []
    for start in range(0, len(tiles), layout.cols):
        row = list(tiles[start:start + layout.cols])
        while len(row) < layout.cols:
            row.append(np.full((height, width, 3), BACKDROP, dtype=np.uint8))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def legend(width: int, counts: dict[str, int], title: str) -> np.ndarray:
    """Title band with one swatch per group and its share of the total."""
    total = max(sum(counts.values()), 1)
    band = np.full((64, width, 3), BACKDROP, dtype=np.uint8)
    cv2.putText(band, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (235, 235, 235), 1, cv2.LINE_AA)
    x = 8
    for group in sorted(counts, key=group_sort_key):
        n = counts[group]
        if not n:
            continue
        cv2.rectangle(band, (x, 36), (x + 22, 52), colour_for(group), -1)
        text = f"{group}  {n} ({100 * n / total:.1f}%)"
        cv2.putText(band, text, (x + 28, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (215, 215, 215), 1, cv2.LINE_AA)
        x += 34 + int(9.2 * len(text))
    note = f"{total} detections"
    cv2.putText(band, note, (max(x + 20, width - 150), 49),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)
    return band


def sheet(tiles: list[np.ndarray], counts: dict[str, int], title: str,
          layout: Layout = Layout()) -> np.ndarray:
    """Legend band above the grid."""
    body = grid(tiles, layout)
    return np.vstack([legend(body.shape[1], counts, title), body])
