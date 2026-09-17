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
