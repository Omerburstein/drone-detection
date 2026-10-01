"""EXP-017 stage 0: the masks, and specifically the *structural* ones.

Why this module exists
----------------------
EXP-015/016 already mask three things: the burned-in HUD (a fixed painting, 1.80% of the
O4 frame), an edge margin, and a `screen_fixed` map -- the pixels where the plain
difference's top-50 peaks landed in >= 5% of *that clip's own* empty frames.

Measured 2026-09-28, the third one buys nothing. Adding it to the others moves the share
of the top-200 motion peaks that sit in the real scene from 40.2% to 41.2%, and all 200
requested peaks were still available in every frame. That is structural, not a tuning
miss: when the candidate budget is a fixed top-N *by rank*, masking 4.36% of the frame
does not remove 4.36% of the candidates -- it promotes the next 4.36% of clutter into the
budget. Removing clutter only reduces load if the budget is a threshold.

It has a second defect EXP-016 recorded: it is fitted on the same clip it is applied to.
On analog it masks a patch of tree line the drone later flies across (frame 585) and costs
recall there, and a fielded system could not build it at all.

So this module builds two masks that do not have either defect:

  * `prop_mask`  -- where the host's own blades sweep. Fixed relative to the airframe
                    forever, so it is calibrated once over EVERY clip from that airframe
                    and is out-of-sample for any one clip's drone.
  * `ladder_mask`-- the vertical band the pitch ladder sweeps through. The ladder is
                    rarely white at any one position, which is why a frequency threshold
                    on a single position misses it; the swept *extent* per column is what
                    holds it. This is `docs/todo.md`'s own proposal.

Both are built by `calibrate_masks.py` and loaded here. Neither looks at the clip under
test, so neither can launder in-sample information into a result.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from clipcfg import CLIP
from src.data.hud_mask import load_mask

W, H = CLIP["width"], CLIP["height"]
S = W / 1440.0                       # pixel scale against the O4 reference


def px(v: float) -> float:
    """A constant quoted at 1440 px wide, read on this clip."""
    return v * S


EDGE_MARGIN = max(8, int(round(px(24.0))))
HUD_DILATE = max(3, int(round(px(15.0))) | 1)


def _load_or_empty(path: str | None, what: str) -> np.ndarray:
    """A calibrated mask, or an explicit all-false with a warning if it was never built.

    Returning empty rather than raising keeps the pipeline runnable on footage that has no
    such feature at all -- a raw camera capture has no ladder and a ground camera has no
    props -- but it says so, because silently masking nothing looks identical to a mask
    that works.
    """
    if path and os.path.exists(path):
        m = np.load(path)
        if m.shape != (H, W):
            raise ValueError(f"{what} mask is {m.shape}, clip is {(H, W)} -- rebuild it")
        return m.astype(bool)
    print(f"[masks] no {what} mask at {path!r}; nothing masked for it. "
          f"Run calibrate_masks.py to build one.")
    return np.zeros((H, W), bool)


def hud_mask() -> np.ndarray:
    """The burned-in overlay, dilated to cover what an 11x11 blur spreads it to."""
    if not CLIP.get("hud"):
        return np.zeros((H, W), bool)
    m = load_mask(CLIP["hud"])
    return cv2.dilate(m.astype(np.uint8), np.ones((HUD_DILATE, HUD_DILATE), np.uint8)) > 0


def prop_mask() -> np.ndarray:
    return _load_or_empty(CLIP.get("prop_mask"), "prop")


def ladder_mask() -> np.ndarray:
    return _load_or_empty(CLIP.get("ladder_mask"), "ladder")


def edge_mask() -> np.ndarray:
    """True where a pixel is too close to the picture edge to be trusted."""
    m = np.zeros((H, W), bool)
    m[:EDGE_MARGIN] = m[-EDGE_MARGIN:] = True
    m[:, :EDGE_MARGIN] = m[:, -EDGE_MARGIN:] = True
    return m


def warp_border(homography: np.ndarray, erode: int | None = None) -> np.ndarray:
    """True where the compensated previous frame has no pixel to offer.

    The warp leaves an empty wedge whose width is the whole camera motion; differencing
    against it manufactures an edge that is pure artefact.
    """
    e = erode if erode is not None else max(9, int(round(px(17.0))) | 1)
    inside = cv2.warpPerspective(np.full((H, W), 255, np.uint8), homography, (W, H),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP) > 0
    return ~(cv2.erode(inside.astype(np.uint8), np.ones((e, e), np.uint8)) > 0)


class Stage0:
    """Every static mask, composed once and reused.

    Held as separate named layers rather than one boolean so `debug_video.py` can colour
    each source differently and so a run can report which layer vetoed what. A single
    pre-ORed array would make the pipeline faster and the failure reports useless.
    """

    def __init__(self) -> None:
        self.layers = {
            "hud": hud_mask(),
            "prop": prop_mask(),
            "ladder": ladder_mask(),
            "edge": edge_mask(),
        }
        self.static = np.zeros((H, W), bool)
        for m in self.layers.values():
            self.static |= m

    def coverage(self) -> dict[str, float]:
        """Percent of the frame each layer covers, and the union. For the run log."""
        out = {k: 100.0 * m.mean() for k, m in self.layers.items()}
        out["union"] = 100.0 * self.static.mean()
        return out

    def valid(self, homography: np.ndarray | None = None) -> np.ndarray:
        """True where a pixel may be used this frame."""
        if homography is None:
            return ~self.static
        return ~(self.static | warp_border(homography))
