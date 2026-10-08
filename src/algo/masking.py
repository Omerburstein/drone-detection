"""Rejecting detections that landed on something painted onto the video.

EXP-011 ran GLAD over goggles screen recordings whose OSD is burned into the
picture. The appearance detector fired on the glyphs and the tracker then held a
battery digit for hundreds of frames: of 24 randomly sampled detections, 21 were
HUD and none were drones.

A mask of where the HUD lives turns that into a decision the pipeline can make.
Nothing here alters a pixel -- inpainting would put invented texture into frames
that numbers are taken from. The box is simply not emitted, and more importantly
not locked onto.

**Why a fraction rather than the centre point.** The reticle sits at the middle
of the frame, which is exactly where a target being flown at tends to be. A
centre test would veto a real drone crossing a ladder dash. Measured on
EXP-011's 1,347 detections, the fraction separates the two populations cleanly:
at 0.5, **83% of detections are rejected while the one confirmed drone --
`first_catch` frame 962 -- scores 0.000** and is untouched.
"""

from __future__ import annotations

import cv2
import numpy as np

# Measured on EXP-011. The curve is flat here (0.4 -> 88%, 0.5 -> 83%,
# 0.6 -> 82%), so the threshold is not balanced on a knife edge.
HUD_VETO_FRACTION = 0.5


def overlap_fraction(box: np.ndarray | tuple[float, float, float, float],
                     mask: np.ndarray) -> float:
    """How much of `(x, y, w, h)` lies on `mask`, as 0-1.

    A box entirely outside the frame overlaps nothing, which is the right
    answer: it is off the mask as surely as it is off the HUD.
    """
    x, y, w, h = (float(v) for v in box)
    x1 = max(int(round(x)), 0)
    y1 = max(int(round(y)), 0)
    x2 = min(int(round(x + w)), mask.shape[1])
    y2 = min(int(round(y + h)), mask.shape[0])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(mask[y1:y2, x1:x2].mean())


def is_masked(box: np.ndarray | tuple[float, float, float, float],
              mask: np.ndarray | None,
              threshold: float = HUD_VETO_FRACTION) -> bool:
    """Is this box mostly drawn-on furniture rather than scene?

    `mask=None` means no mask was supplied and nothing is ever vetoed, which is
    what keeps EXP-004, EXP-005 and EXP-010 reproducing exactly.
    """
    if mask is None:
        return False
    return overlap_fraction(box, mask) >= threshold


# --- HUD that moves -------------------------------------------------------
#
# A static mask cannot hold an element that moves. On the analog OSD (EXP-013)
# the artificial horizon is a row of identical dashes that slides with pitch
# and roll, and one of them held a `local yolo` lock for 43 frames. At 10 px a
# dash -- bright line over a dark outline -- is not separable from a sunlit
# quadcopter by appearance. What separates it is that a dash has **twins**:
# identical copies one OSD character column to either side. A drone does not.
# This is the "second identical unboxed object" tell from the /inspect skill,
# made mechanical.

# Analog OSD chips (MAX7456 and its clones) draw a 30-column character grid
# across the picture's width, so a column is `width / 30` whatever the capture
# resolution. EXP-013's dashes step 596 -> 563 -> 530 px at 960 wide: 32 px.
OSD_COLUMNS = 30
# Measured on EXP-013's 82 detections: the 15 horizon-dash boxes score
# 0.58-0.95 (13 of them >= 0.70); the compass glyph and telemetry boxes, whose
# neighbours are *different* characters, mostly 0.3-0.6.
TWIN_SCORE = 0.7
# A flat patch correlates with any other flat patch after normalisation -- the
# same trap measured in `src.data.annotate` -- so featureless boxes never twin.
TWIN_MIN_STD = 8.0
TWIN_REACH = (0.5, 2.5)  # in columns: nearest and farthest twin searched for
# The OSD is drawn into the picture, so its geometry scales with the picture: every length
# below is a fraction of one character column, measured at 960 wide (a 32 px column) and
# exact there. TWIN_ROWS is 16 px and TWIN_PAD 2 px at 960 wide.
TWIN_ROWS = 0.5  # vertical slack for a horizon tilted by roll, in columns
TWIN_PAD = 0.0625  # context kept around the box, in columns


def _cols(fraction: float, column: float) -> int:
    """A length given in OSD columns, in whole pixels of this picture."""
    return int(round(fraction * column))


def twin_score(gray: np.ndarray,
               box: np.ndarray | tuple[float, float, float, float],
               columns: int = OSD_COLUMNS) -> float:
    """Best match for `(x, y, w, h)` elsewhere on its own row, as 0-1.

    Searches 0.5-2.5 OSD columns to the left and to the right, with a little
    vertical slack, by normalised correlation. The box's own position is
    excluded, so a lone object scores low however distinctive it is.
    """
    height, width = gray.shape[:2]
    column = width / columns
    pad, slack = _cols(TWIN_PAD, column), _cols(TWIN_ROWS, column)
    x, y, w, h = (float(v) for v in box)
    x1 = max(int(round(x)) - pad, 0)
    y1 = max(int(round(y)) - pad, 0)
    x2 = min(int(round(x + w)) + pad, width)
    y2 = min(int(round(y + h)) + pad, height)
    template = gray[y1:y2, x1:x2].astype(np.float32)
    if template.size == 0 or template.std() < TWIN_MIN_STD:
        return 0.0

    near, far = (int(column * reach) for reach in TWIN_REACH)
    rows = slice(max(y1 - slack, 0), min(y2 + slack, height))
    best = 0.0
    for lo, hi in ((x1 - far, x2 - near), (x1 + near, x2 + far)):
        strip = gray[rows, max(lo, 0):min(hi, width)].astype(np.float32)
        if strip.shape[0] < template.shape[0] or strip.shape[1] < template.shape[1]:
            continue
        match = cv2.matchTemplate(strip, template, cv2.TM_CCOEFF_NORMED)
        best = max(best, float(match.max()))
    return best


def has_twin(gray: np.ndarray,
             box: np.ndarray | tuple[float, float, float, float],
             threshold: float = TWIN_SCORE) -> bool:
    """Is this box one of a row of identical OSD characters?"""
    return twin_score(gray, box) >= threshold


# --- The same tell, locked to the grid --------------------------------------
#
# `twin_score` looks anywhere 0.5-2.5 columns away. On GLAD's boxes that was
# enough, but on the sky branch's 8-14 px blobs it is not: a small dark spot on
# plain sky correlates with any other dark spot, and EXP-027 measured the drone
# "twinning" in 32 of the 53 frames it was shown at #1. The dashes, though, sit
# on the character grid, so their copies are *exactly* whole columns away. Asking
# for copies at two of the four positions +-1 and +-2 columns -- with a few
# pixels of slack, not a 64 px strip -- kept all 53 and still took 93 of the 120
# false alarms the dashes made.

GRID_STEPS = (-2, -1, 1, 2)  # in columns
# Fractions of a column, as TWIN_ROWS: 3 px and 8 px at 960 wide.
GRID_SLACK = 0.09375  # horizontal slack around each grid position, in columns
GRID_ROWS = 0.25  # vertical slack per column stepped, for roll, in columns
GRID_MATCHES = 2  # grid positions that must hold a copy


def grid_twins(gray: np.ndarray,
               box: np.ndarray | tuple[float, float, float, float],
               threshold: float = TWIN_SCORE,
               columns: int = OSD_COLUMNS) -> int:
    """How many of the positions +-1 and +-2 OSD columns away hold a copy of the box.

    A copy is a normalised correlation of at least `threshold` within
    `GRID_SLACK` columns of the grid position horizontally and `GRID_ROWS`
    columns per column stepped vertically. Featureless boxes count none, as in
    `twin_score`.
    """
    height, width = gray.shape[:2]
    column = width / columns
    pad, slack = _cols(TWIN_PAD, column), _cols(GRID_SLACK, column)
    x, y, w, h = (float(v) for v in box)
    x1 = max(int(round(x)) - pad, 0)
    y1 = max(int(round(y)) - pad, 0)
    x2 = min(int(round(x + w)) + pad, width)
    y2 = min(int(round(y + h)) + pad, height)
    template = gray[y1:y2, x1:x2].astype(np.float32)
    if template.size == 0 or template.std() < TWIN_MIN_STD:
        return 0

    copies = 0
    for step in GRID_STEPS:
        dx = int(round(step * column))
        dy = _cols(GRID_ROWS, column) * abs(step)
        strip = gray[max(y1 - dy, 0):min(y2 + dy, height),
                     max(x1 + dx - slack, 0):min(x2 + dx + slack, width)]
        if strip.shape[0] < template.shape[0] or strip.shape[1] < template.shape[1]:
            continue
        match = cv2.matchTemplate(strip.astype(np.float32), template,
                                  cv2.TM_CCOEFF_NORMED)
        copies += float(match.max()) >= threshold
    return copies


def on_osd_grid(gray: np.ndarray,
                box: np.ndarray | tuple[float, float, float, float]) -> bool:
    """Is this box a character in a row on the OSD grid, such as a horizon dash?"""
    return grid_twins(gray, box) >= GRID_MATCHES
