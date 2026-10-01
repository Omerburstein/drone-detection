"""EXP-016 shared pieces: masks, seeds, background flow, and the local-ring hypothesis test.

Imported by `collect.py`, `verify.py`, `longbase.py`, `score.py` and `draw_video.py`.
The clip is whichever `clipcfg` is first on `PYTHONPATH`, so the same code runs O4 and
analog with no branch on the clip name.

What this module is for
-----------------------
EXP-015 established that two frames cannot find the long-range drone: every map ranked it
outside the top 10 over `first_catch` 708-800. Chaining those peaks over time does not fix
it either -- parallax edges, the host's own props and the HUD persist exactly as well as
the drone. What separates them is *differential* motion measured against the target's own
local background: the 3-5 px of parallax that swamps a whole-frame map is shared by the
surroundings 25-120 px away, so it cancels in the difference.

So: seed many candidates, then test each one over time against its own ring.

Every pixel constant below is quoted at 1440 px wide (O4) and scaled by `S` for analog.
The speed cap is the exception -- it is derived from physical units, and the ratio
px-per-frame to apparent-size is lens-, range- and resolution-independent because both
terms scale together.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np

import common          # EXP-015's module: homography, warp, grad, prep, valid_mask, peaks
from clipcfg import CLIP
from src.data.hud_mask import load_mask

# --- clip geometry --------------------------------------------------------
W, H_PX = CLIP["width"], CLIP["height"]
S = W / 1440.0                       # pixel scale against the O4 reference
FPS = CLIP["fps"]
OUT = CLIP["out"]


def px(v: float) -> float:
    """A constant quoted at 1440 px wide, read on this clip."""
    return v * S


# --- the physical speed cap -----------------------------------------------
# A 0.6 m quadrotor flying no faster than 30 m/s (user, 2026-09-23), worst case against a
# pursuer closing at the same speed. Image motion per frame is bounded by the target's own
# apparent size, and the bound does not depend on the lens, the range or the resolution:
#
#     max px/frame = v_rel / (fps * target_size_m) * apparent_size_px
#
# Nothing here is a pixel number; change the airframe or the frame rate and one of these
# three inputs moves, not a threshold scattered through the code.
TARGET_SIZE_M = 0.6
V_MAX_MS = 30.0
CLOSING_FACTOR = 2.0                 # both aircraft at v_max, head on
SPEED_CAP_RATIO = CLOSING_FACTOR * V_MAX_MS / (FPS * TARGET_SIZE_M)   # = 3.33 per px of size


def speed_cap(size_px: float) -> float:
    """Largest believable image motion for a target this big, in px per frame."""
    return SPEED_CAP_RATIO * size_px


# --- hypothesis-test parameters (1440-px units) ---------------------------
SEEDS_PER_FRAME = 200
ANNULUS = (px(25.0), px(120.0))      # ring the local background is read from
K_MIN, K_MAX = 3, 15                 # user's latency ceiling: 15 frames = 0.5 s
D_MIN = px(6.0)                      # accumulated differential motion before any confirm
FB_MAX = px(1.0)                     # forward-backward agreement, px
FB_STRIKES = 2                       # kill after this many FB failures while unconfirmed
NOISE_FLOOR = px(0.3)                # per-frame noise floor, px; keeps z finite on clean ring
MAD_TO_SIGMA = 1.48
SCREEN_FIXED_VMAX = px(0.5)          # "not moving on the screen at all"
RING_FLOW_MIN = px(3.0)              # "...while its surroundings plainly are"
GRID = max(8, int(round(px(24.0))))  # background LK grid spacing
MIN_RING_PTS = 8
SEED_SIZE_RANGE = (px(8.0), px(120.0))
KLT_WIN = max(9, int(round(px(15.0))) | 1)
LK = dict(winSize=(KLT_WIN, KLT_WIN), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
# After confirmation the track is held rather than re-seeded every 15 frames.
HOLD_FB_STRIKES = 3
HOLD_TEMPLATE_MIN = 0.45             # same floor `src.data.annotate.Follower` uses

# --- masks ----------------------------------------------------------------
# A clip with no burned-in overlay (a raw camera capture rather than a recording of a
# goggles display) passes `hud=None` and gets an empty mask, so nothing is masked out.
HUD = load_mask(CLIP["hud"]) if CLIP.get("hud") else np.zeros((H_PX, W), bool)
common.HUD_DIL = cv2.dilate(HUD.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
common.MARGIN = max(8, int(round(px(24.0))))
common.PEAK_RADIUS = max(6, int(round(px(15.0))))
HUD_DIL = common.HUD_DIL


def screen_fixed(path: str | None = None) -> np.ndarray:
    """Boolean map of screen-fixed clutter: props, ladder dashes, burned-in digits.

    EXP-015's rule, unchanged -- where the plain difference's top-50 peaks land in >= 5%
    of the clip's *empty* frames, dilated by 9. In-sample for the overlay layout on both
    clips; out-of-sample for the drone, which never appears in an empty frame.
    """
    frac = np.load(path or CLIP["screen_fixed"])
    return cv2.dilate((frac >= 0.05).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0


# --- labels ---------------------------------------------------------------
# An unlabelled clip (`labels=None`) leaves BOXES empty. `on_drone` is then always False
# and `rank_on_drone` always None, so every number this module can produce about the
# *target* disappears rather than quietly becoming zero -- which is the honest outcome,
# because without ground truth there is nothing to be right or wrong about.
if CLIP.get("labels"):
    _L = json.load(open(CLIP["labels"]))["frames"]
    BOXES = {int(k): v["box"] for k, v in _L.items() if v and v.get("box")}
    SOURCE = {int(k): v.get("source") for k, v in _L.items() if v and v.get("box")}
else:
    BOXES, SOURCE = {}, {}
DRONE_FRAMES = sorted(BOXES)

# Frames kept out of the screen-fixed calibration set on top of the labelled ones. A
# labelled clip names its drone frames one by one and needs none of this; an unlabelled
# clip can still say "the target is somewhere in here" from its provenance, and that is
# enough to keep those frames out of a map that is meant to describe the empty scene.
# Default empty, so a clip that carries labels is unaffected.
_EXCLUDED = {f for lo, hi in CLIP.get("drone_ranges", []) for f in range(lo, hi + 1)}
EMPTY_FRAMES = [f for f in range(2, CLIP["n_frames"] + 1)
                if f not in BOXES and f not in _EXCLUDED]


def label_spans() -> list[tuple[int, int]]:
    if not DRONE_FRAMES:
        return []
    spans, start, prev = [], DRONE_FRAMES[0], DRONE_FRAMES[0]
    for f in DRONE_FRAMES[1:]:
        if f != prev + 1:
            spans.append((start, prev))
            start = f
        prev = f
    spans.append((start, prev))
    return spans


def on_drone(x: float, y: float, frame: int) -> bool:
    """EXP-015's match criterion: inside the label box grown by max(10 px, 25% of its longest side)."""
    box = BOXES.get(frame)
    if box is None:
        return False
    x0, y0, x1, y1 = common.drone_region(box)
    return x0 <= x <= x1 and y0 <= y <= y1


# --- the seed map ---------------------------------------------------------
def win_map(prev_bgr, cur_bgr, Hm, eps: float = 4.0, k: int = 5, win: int = 9):
    """`win_b5_e4` from EXP-015's `common.maps`, computed on its own.

    Reproduced rather than called so a frame pair costs one map instead of eleven;
    `collect.py` asserts it is bit-identical to `maps()['win_b5_e4']` on a sample frame.
    """
    p, c = common.prep(prev_bgr, k), common.prep(cur_bgr, k)
    comp = common.warp(p, Hm)
    d = cv2.absdiff(c, comp).astype(np.float32)
    gmax = np.maximum(common.grad(c), common.grad(comp))
    num = cv2.boxFilter(d * d, -1, (win, win))
    den = cv2.boxFilter(gmax * gmax, -1, (win, win)) + eps * eps
    return np.sqrt(num / den), d


def seed_sizes(diff: np.ndarray, pk: np.ndarray) -> np.ndarray:
    """Apparent size of each peak, from the extent of its own difference blob.

    A peak carries no extent, and the speed cap needs one. Take the difference image's
    half-maximum region around the peak inside a window twice the largest target we will
    entertain, and report the longer side of its bounding box. Over-estimating is the
    permissive direction -- the cap only ever rejects -- so this cannot manufacture recall.
    """
    lo, hi = SEED_SIZE_RANGE
    r = int(round(hi))
    h, w = diff.shape
    out = np.empty(len(pk), np.float32)
    for i, (_, x, y) in enumerate(pk):
        xi, yi = int(x), int(y)
        x0, y0 = max(xi - r, 0), max(yi - r, 0)
        patch = diff[y0:min(yi + r + 1, h), x0:min(xi + r + 1, w)]
        peak = float(diff[yi, xi])
        if peak <= 0:
            out[i] = lo
            continue
        mask = (patch >= 0.5 * peak).astype(np.uint8)
        n, lab = cv2.connectedComponents(mask)
        cid = lab[yi - y0, xi - x0]
        if cid == 0:
            out[i] = lo
            continue
        ys, xs = np.nonzero(lab == cid)
        out[i] = np.clip(max(xs.max() - xs.min(), ys.max() - ys.min()) + 1.0, lo, hi)
    return out


# --- background flow on a regular grid ------------------------------------
@dataclass
class Flow:
    """One frame pair's background motion: the homography plus what it leaves behind.

    `resid` is the grid point's tracked motion minus the homography's prediction for it --
    local parallax, in px. `raw` is its uncompensated image motion, used by the
    screen-fixed veto.
    """
    Hm: np.ndarray
    pts: np.ndarray        # (rows, cols, 2) grid positions in the previous frame
    resid: np.ndarray      # (rows, cols, 2)
    raw: np.ndarray        # (rows, cols)
    ok: np.ndarray         # (rows, cols) bool
    rows: int
    cols: int
    origin: tuple[float, float]

    def cell(self, x: float, y: float) -> tuple[int, int]:
        return (int(round((y - self.origin[1]) / GRID)), int(round((x - self.origin[0]) / GRID)))

    def ring(self, x: float, y: float):
        """Median residual, residual scatter and median raw flow in the 25-120 px ring.

        Returns `(median_resid[2], sigma_px, raw_flow_px, n)`; `n < MIN_RING_PTS` means the
        ring is unusable and the hypothesis takes no evidence from this frame.
        """
        inner, outer = ANNULUS
        span = int(np.ceil(outer / GRID))
        r0, c0 = self.cell(x, y)
        rs = slice(max(r0 - span, 0), min(r0 + span + 1, self.rows))
        cs = slice(max(c0 - span, 0), min(c0 + span + 1, self.cols))
        p = self.pts[rs, cs]
        d = np.hypot(p[..., 0] - x, p[..., 1] - y)
        sel = (d >= inner) & (d <= outer) & self.ok[rs, cs]
        n = int(sel.sum())
        if n < MIN_RING_PTS:
            return np.zeros(2, np.float32), 0.0, 0.0, n
        r = self.resid[rs, cs][sel]
        med = np.median(r, axis=0)
        sigma = MAD_TO_SIGMA * float(np.median(np.linalg.norm(r - med, axis=1)))
        return med.astype(np.float32), sigma, float(np.median(self.raw[rs, cs][sel])), n


def _grid_points() -> tuple[np.ndarray, int, int, tuple[float, float]]:
    m = common.MARGIN
    xs = np.arange(m + GRID / 2, W - m, GRID, np.float32)
    ys = np.arange(m + GRID / 2, H_PX - m, GRID, np.float32)
    pts = np.stack(np.meshgrid(xs, ys), -1).astype(np.float32)
    return pts, len(ys), len(xs), (float(xs[0]), float(ys[0]))


GRID_PTS, GRID_ROWS, GRID_COLS, GRID_ORIGIN = _grid_points()


def track(prev_gray, cur_gray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward-backward LK. Returns the tracked points and the FB error in px."""
    if len(pts) == 0:
        return pts.copy(), np.zeros(0, np.float32)
    p0 = pts.reshape(-1, 1, 2).astype(np.float32)
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, p0, None, **LK)
    p0b, st0, _ = cv2.calcOpticalFlowPyrLK(cur_gray, prev_gray, p1, None, **LK)
    fb = np.linalg.norm(p0b.reshape(-1, 2) - pts, axis=1).astype(np.float32)
    bad = (st1.ravel() == 0) | (st0.ravel() == 0) | ~np.isfinite(fb)
    fb[bad] = np.inf
    return p1.reshape(-1, 2), fb


def predict(Hm: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Where static background at `pts` (previous frame) lands in the current frame.

    `common.homography` returns H mapping current-frame coordinates to previous-frame
    coordinates -- that is what `warp(prev, H)` with `WARP_INVERSE_MAP` consumes -- so the
    forward prediction is H inverse.
    """
    if len(pts) == 0:
        return pts.copy()
    Hi = np.linalg.inv(Hm)
    p = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), Hi.astype(np.float32))
    return p.reshape(-1, 2)


def background_flow(prev_gray, cur_gray, Hm, valid: np.ndarray):
    """Grid flow for one pair, plus the three reasons a grid point is dropped.

    The reasons are kept apart because they mean different things: a **forward-backward**
    failure is the tracker not agreeing with itself, which is the number the plan asks for
    on analog; a **mask** rejection is deliberate; a **wild** residual is a track that ran
    away. Pooling them would hide whether analog grain is breaking KLT.
    """
    pts = GRID_PTS.reshape(-1, 2)
    p1, fb = track(prev_gray, cur_gray, pts)
    pred = predict(Hm, pts)
    resid = p1 - pred
    raw = np.linalg.norm(p1 - pts, axis=1)
    fb_ok = np.isfinite(fb) & (fb <= FB_MAX)
    ix = np.clip(pts[:, 0].astype(int), 0, W - 1)
    iy = np.clip(pts[:, 1].astype(int), 0, H_PX - 1)
    in_mask = valid[iy, ix]
    tame = np.linalg.norm(resid, axis=1) < px(60.0)      # a wild track is not background
    ok = fb_ok & in_mask & tame
    shape = (GRID_ROWS, GRID_COLS)
    flow = Flow(Hm, GRID_PTS, resid.reshape(*shape, 2).astype(np.float32),
                raw.reshape(shape).astype(np.float32), ok.reshape(shape),
                GRID_ROWS, GRID_COLS, GRID_ORIGIN)
    why = dict(fb=float(1 - fb_ok.mean()), mask=float(1 - in_mask.mean()),
               wild=float(1 - tame.mean()), ok=float(ok.mean()),
               fb_med=float(np.median(fb[np.isfinite(fb)])) if np.isfinite(fb).any() else np.inf)
    return flow, why


# --- the hypothesis -------------------------------------------------------
@dataclass
class Hypothesis:
    """One seed being tested over time by its motion relative to its own ring."""
    hid: int
    born: int
    size: float
    x: float
    y: float
    seed_val: float
    x0: float = 0.0
    y0: float = 0.0
    k: int = 0
    sum_d: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float64))
    var: float = 0.0                 # accumulated noise variance, px^2
    raw_path: float = 0.0            # raw image motion travelled, px
    ring_path: float = 0.0           # raw image motion of its surroundings, px
    fb_fail: int = 0
    z: float = 0.0
    confirmed: int | None = None     # frame it confirmed on
    dead: str | None = None
    template: np.ndarray | None = None
    trail: list = field(default_factory=list)

    def __post_init__(self):
        self.x0, self.y0 = self.x, self.y
        self.trail.append((self.born, self.x, self.y))

    @property
    def cap(self) -> float:
        return speed_cap(self.size)

    @property
    def disp(self) -> float:
        return float(np.linalg.norm(self.sum_d))

    def update_z(self):
        self.z = self.disp / np.sqrt(self.var) if self.var > 0 else 0.0
