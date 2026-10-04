"""EXP-017 stage 1: split each frame into sky and scene, and find the horizon.

What this is for
----------------
It is *not* semantic segmentation and it is not a detector. It exists because the local
ring test needs to know where the depth discontinuities are.

EXP-016 measured that 57.5% of analog's confirmed false tracks confirm at the sky / tree
line boundary. The ring test assumes the surroundings 25-120 px away share the
hypothesis's parallax. At a horizon that is false by construction: half the ring is sky
with no parallax and half is near tree line with several px, so the ring median describes
neither and the leftover reads as differential motion. The fix named in EXP-016's own
"next" is a ring that respects depth. This module is the cheapest usable depth prior
available -- a two-level one, sky (infinite) against everything else.

The rule
--------
Sky is bright and untextured; scene is textured at every depth. Both hold on CVBS and on a
digital link, which is why texture leads and colour is only a tie-breaker.

  1. luminance, blurred
  2. local texture energy: box-filtered Sobel magnitude
  3. blue excess, b - (r+g)/2, on colour footage only. CVBS is close to greyscale and
     carries chroma crawl, so `use_colour=False` drops this term rather than feeding it
     noise.
  4. threshold each on a percentile of THIS frame, so nothing is a fixed grey level and a
     frame that is all sky or all ground still behaves
  5. keep components connected to the top edge, close small holes

Then the horizon is read off the mask: per column, the lowest sky row.

Deliberate limits, because a wrong prior is worse than none
-----------------------------------------------------------
  * A drone against sky is a small dark hole in it. `fill_holes` closes holes up to
    `MAX_HOLE_PX` so the target is not carved out of the sky region and then treated as
    scene. Anything bigger is genuinely not sky.
  * Where the split is uncertain -- near the boundary, or where the frame is too dark to
    call -- `Skyline.confidence` is low and the caller should widen rather than guess.
    `uncertain_band` is what the ring test must refuse to straddle.
  * Sun, bloom and thin cloud read as sky. That is correct for parallax purposes: they are
    at infinity too.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from clipcfg import CLIP

W, H = CLIP["width"], CLIP["height"]
S = W / 1440.0
USE_COLOUR = bool(CLIP.get("use_colour", True))


def px(v: float) -> float:
    return v * S


BLUR = max(3, int(round(px(9.0))) | 1)
TEXTURE_WIN = max(5, int(round(px(21.0))) | 1)
CLOSE = max(3, int(round(px(15.0))) | 1)
MAX_HOLE_PX = int(round(px(60.0)) ** 2)      # a 60x60 target must not punch out of sky
UNCERTAIN_PX = max(4, int(round(px(24.0))))  # half-width of the band around the boundary

# Percentiles, not grey levels. A frame decides its own thresholds. A clip may override
# both: on the FIELD capture deep blue sky is darker than sunlit ground, so "sky is bright"
# is false there (EXP-025e sets luma 0, texture 60; colour carries the call instead).
TEXTURE_PCTL = float(CLIP.get("sky_texture_pctl", 45.0))  # below this texture pctl: may be sky
LUMA_PCTL = float(CLIP.get("sky_luma_pctl", 55.0))        # and above this luminance pctl
MIN_SKY_FRAC = 0.004      # below this, call the frame skyless rather than invent a region


@dataclass(frozen=True)
class Skyline:
    """One frame's sky/scene split.

    `sky` and `scene` are not complements: `uncertain` is carved out of both, so a caller
    that wants a clean background sample can take `scene & ~uncertain` and know it has not
    quietly included the boundary that breaks the ring test.
    """
    sky: np.ndarray            # bool (H, W)
    uncertain: np.ndarray      # bool (H, W): within UNCERTAIN_PX of the boundary
    horizon: np.ndarray        # int (W,): lowest sky row per column, -1 where no sky
    sky_fraction: float
    has_sky: bool

    @property
    def scene(self) -> np.ndarray:
        return ~self.sky & ~self.uncertain

    def label(self, x: float, y: float) -> str:
        """'sky', 'scene' or 'uncertain' for one point, for per-candidate reporting."""
        xi, yi = int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1))
        if self.uncertain[yi, xi]:
            return "uncertain"
        return "sky" if self.sky[yi, xi] else "scene"

    def straddles(self, x: float, y: float, r_in: float, r_out: float) -> bool:
        """True if a ring at (x, y) spans both labels -- i.e. the ring test is invalid here.

        This is the EXP-016 horizon failure, made checkable before the test runs rather
        than diagnosed afterwards.
        """
        yy, xx = np.ogrid[:H, :W]
        d2 = (xx - x) ** 2 + (yy - y) ** 2
        ring = (d2 >= r_in ** 2) & (d2 <= r_out ** 2)
        if not ring.any():
            return True
        in_sky = (self.sky & ring).sum()
        in_scene = (self.scene & ring).sum()
        return bool(in_sky and in_scene)


def _texture(grey: np.ndarray) -> np.ndarray:
    f = grey.astype(np.float32)
    g = np.hypot(cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3),
                 cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)) / 8.0
    return cv2.boxFilter(g, -1, (TEXTURE_WIN, TEXTURE_WIN))


def _fill_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    """Close holes in `mask` up to `max_area` px. A drone is a hole; a field is not."""
    inv = (~mask).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, 8)
    out = mask.copy()
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] <= max_area:
            x, y, w_, h_ = stats[i, :4]
            # A hole touching the frame edge is open ground, not an enclosed hole.
            if x > 0 and y > 0 and x + w_ < mask.shape[1] and y + h_ < mask.shape[0]:
                out[lab == i] = True
    return out


def _top_seed(valid: np.ndarray, depth: int = 3) -> np.ndarray:
    """The topmost usable rows of each column.

    Not literally row 0: the edge margin and the HUD mask make the true top rows invalid,
    so anchoring on row 0 finds nothing and the whole frame comes back skyless. This walks
    down each column to the first valid row instead.
    """
    seed = np.zeros_like(valid)
    cols = np.nonzero(valid.any(axis=0))[0]
    first = valid.argmax(axis=0)
    for x in cols:
        seed[first[x]:first[x] + depth, x] = True
    return seed & valid


def _top_connected(mask: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Keep only components reaching the topmost usable row of some column."""
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), 8)
    keep = set(np.unique(lab[_top_seed(valid)])) - {0}
    if not keep:
        return np.zeros_like(mask)
    return np.isin(lab, list(keep))


def split(frame_bgr: np.ndarray, valid: np.ndarray | None = None) -> Skyline:
    """Split one BGR frame. `valid` (from Stage0) keeps masked pixels out of the statistics."""
    grey = cv2.GaussianBlur(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), (BLUR, BLUR), 0)
    tex = _texture(grey)
    ok = np.ones((H, W), bool) if valid is None else valid

    # Thresholds from this frame's own distribution, over unmasked pixels only -- the HUD
    # and the props are bright and textured and would drag a global percentile.
    if ok.sum() < 1000:
        empty = np.zeros((H, W), bool)
        return Skyline(empty, empty, np.full(W, -1, int), 0.0, False)
    t_thr = np.percentile(tex[ok], TEXTURE_PCTL)
    l_thr = np.percentile(grey[ok], LUMA_PCTL)

    cand = (tex <= t_thr) & (grey >= l_thr) & ok
    if USE_COLOUR:
        b, g, r = (frame_bgr[..., i].astype(np.float32) for i in range(3))
        blue = b - 0.5 * (g + r)
        cand &= blue >= np.percentile(blue[ok], 40.0)

    k = np.ones((CLOSE, CLOSE), np.uint8)
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
    sky = _fill_holes(_top_connected(cand, ok), MAX_HOLE_PX)

    frac = float(sky.mean())
    if frac < MIN_SKY_FRAC:
        empty = np.zeros((H, W), bool)
        return Skyline(empty, empty, np.full(W, -1, int), frac, False)

    d = max(3, UNCERTAIN_PX | 1)
    grown = cv2.dilate(sky.astype(np.uint8), np.ones((d, d), np.uint8)) > 0
    shrunk = cv2.erode(sky.astype(np.uint8), np.ones((d, d), np.uint8)) > 0
    uncertain = grown & ~shrunk

    rows = np.arange(H)[:, None] * sky
    horizon = np.where(sky.any(axis=0), rows.max(axis=0), -1).astype(int)
    return Skyline(sky, uncertain, horizon, frac, True)
