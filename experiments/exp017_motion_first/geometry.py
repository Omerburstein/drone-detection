"""EXP-017 stage 2, part 1: the homography, the residual field, and the epipole.

Everything here works in **current-frame pixel coordinates**, which is not what
EXP-015/016 did and is the reason the two tests in `criteria.py` can be written down at
all. EXP-015's `common.homography(f1, f2)` returns H mapping *frame2 -> frame1*, because
that is the direction upstream's `motion_compensate` warps in. Plane+parallax is stated
about where a point lands in the frame you are looking at, so this module inverts it once
and then stays in frame 2.

The one relation the rest of stage 2 rests on
---------------------------------------------
Compensate a frame pair with the homography of some plane. For any **static** point, the
displacement left over is

    mu  =  gamma * (e - x)

where `x` is the point's compensated position, `e` is the epipole (the focus of expansion,
where the translation direction pierces the image), and `gamma` is a scalar that depends
only on the point's depth relative to the plane and on the baseline of that frame pair.

Two things follow, and they are the whole of steps 2 and 3:

  * `mu` is **parallel to the line joining x and e**, for every static point, always. It
    does not matter how deep the point is or how much parallax it has. EXP-016 measured
    only |Sum mu| and so could not use this.
  * `gamma` is **one scalar per point**, constant across the window up to the per-pair
    baseline. A static point's whole track is explained by it; a moving point's is not.

Where it breaks, stated here so the callers can refuse rather than guess
-----------------------------------------------------------------------
  * **Pure rotation.** No translation means no epipole. The residual field is then noise
    and its directions are uniform, so `Epipole.reliable` goes false and both tests must
    be skipped for that pair, not run on a fitted-to-noise epipole.
  * **Near the epipole.** `|x - e|` small makes the direction ill-conditioned: every
    direction is nearly radial. This is exactly where an intercept target sits, so it is
    reported (`Epipole.degenerate_for`) rather than silently scored.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

import common          # EXP-015's module, so H is bit-identical to EXP-015/016's
from clipcfg import CLIP

W, H_PX = CLIP["width"], CLIP["height"]
S = W / 1440.0


def px(v: float) -> float:
    return v * S


GRID = max(8, int(round(px(24.0))))
FB_MAX = px(1.0)
KLT_WIN = max(9, int(round(px(15.0))) | 1)
LK = dict(winSize=(KLT_WIN, KLT_WIN), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# An epipole fitted to fewer than this many directed residuals is not a measurement.
MIN_EPIPOLE_PTS = 24
# Residuals shorter than this carry no usable direction and are dropped from the fit.
MIN_RESIDUAL_PX = px(0.6)
# Below this, the residual directions are too close to uniform to locate an epipole.
MIN_DIRECTION_ANISOTROPY = 0.15
# Within this distance of the epipole, direction is ill-conditioned.
DEGENERATE_RADIUS = px(90.0)


def homography(prev_grey: np.ndarray, cur_grey: np.ndarray) -> np.ndarray:
    """GLAD's grid-KLT homography, mapping CURRENT -> PREVIOUS. Unchanged from EXP-015."""
    return common.homography(prev_grey, cur_grey)


def grid_points() -> np.ndarray:
    """The fixed sampling grid, (N, 2). One definition, so a window can reseed the same one."""
    return np.array([(np.float32(i * GRID + GRID / 2), np.float32(j * GRID + GRID / 2))
                     for i in range(W // GRID) for j in range(H_PX // GRID)], np.float32)


def track_grid(prev_grey: np.ndarray, cur_grey: np.ndarray,
               valid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Forward-backward LK on a fixed grid. Returns (prev_pts, cur_pts), both (N, 2)."""
    p = grid_points().reshape(-1, 1, 2)
    fwd, st, _ = cv2.calcOpticalFlowPyrLK(prev_grey, cur_grey, p, None, **LK)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(cur_grey, prev_grey, fwd, None, **LK)
    ok = (st.ravel() == 1) & (st2.ravel() == 1)
    ok &= np.linalg.norm((back - p).reshape(-1, 2), axis=1) < FB_MAX
    a, b = p.reshape(-1, 2)[ok], fwd.reshape(-1, 2)[ok]
    if valid is not None and len(a):
        keep = valid[np.clip(a[:, 1].astype(int), 0, H_PX - 1),
                     np.clip(a[:, 0].astype(int), 0, W - 1)]
        a, b = a[keep], b[keep]
    return a, b


def apply_h(hmat: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Map points through a 3x3 homography. (N, 2) in, (N, 2) out."""
    if not len(pts):
        return pts.reshape(-1, 2)
    return cv2.perspectiveTransform(np.asarray(pts, np.float32).reshape(-1, 1, 2),
                                    hmat).reshape(-1, 2)


def residual_field(prev_pts: np.ndarray, cur_pts: np.ndarray,
                   h_cur_to_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Where each tracked point actually landed, minus where the plane says it should have.

    Returns (x, mu) in CURRENT-frame coordinates: `x` is the homography's prediction of the
    point's current position and `mu` is the leftover displacement. For a static point
    `mu` is the parallax, and it points along the line through `x` and the epipole.
    """
    h_prev_to_cur = np.linalg.inv(h_cur_to_prev)
    x = apply_h(h_prev_to_cur, prev_pts)
    return x, cur_pts - x


@dataclass(frozen=True)
class Epipole:
    """The focus of expansion for one frame pair, with its own reliability attached.

    `reliable` false means the pair carries too little translation to locate an epipole at
    all -- a rotation-dominated pair. Callers must skip the direction test rather than use
    `point`, because a fit to noise still returns coordinates.
    """
    point: np.ndarray          # (2,) in current-frame pixels
    reliable: bool
    n_points: int
    anisotropy: float          # 0 = directions uniform (no epipole), 1 = perfectly radial
    inlier_fraction: float

    def degenerate_for(self, x: float, y: float,
                       radius: float = DEGENERATE_RADIUS) -> bool:
        """True where direction cannot separate anything: too close to the epipole.

        An intercept target sits near the focus of expansion, so this fires on exactly the
        case the project cares about. It is reported, not silently treated as a pass.
        """
        return bool(np.hypot(x - self.point[0], y - self.point[1]) < radius)


def estimate_epipole(x: np.ndarray, mu: np.ndarray,
                     min_residual: float = MIN_RESIDUAL_PX) -> Epipole:
    """Least-squares intersection of the lines through each x along its own mu.

    Every static point's residual points at the epipole, so each (x, mu) pair votes for a
    line and the epipole is where the lines meet. Solved in closed form: minimising
    sum((n_i . (p - x_i))^2) over p, with n_i the unit normal to mu_i, is a 2x2 system.

    Weighted by residual length, because a 0.3 px residual has a direction dominated by
    tracking noise and a 5 px one does not.

    `min_residual` scales with the window when this is called on an ACCUMULATED field: a
    0.6 px residual is noise over one frame pair and is even more clearly noise over five,
    so a caller summing k steps should raise the floor rather than inherit the per-pair one.
    """
    if len(x) < MIN_EPIPOLE_PTS:
        return Epipole(np.array([W / 2.0, H_PX / 2.0]), False, len(x), 0.0, 0.0)

    mag = np.linalg.norm(mu, axis=1)
    keep = mag >= min_residual
    if keep.sum() < MIN_EPIPOLE_PTS:
        return Epipole(np.array([W / 2.0, H_PX / 2.0]), False, int(keep.sum()), 0.0, 0.0)
    xs, ms, mg = x[keep], mu[keep], mag[keep]

    d = ms / mg[:, None]                      # unit residual directions
    n = np.stack([-d[:, 1], d[:, 0]], axis=1)  # unit normals
    w = mg

    # Anisotropy: if the directions are uniformly distributed there is no epipole to find.
    # The doubled-angle mean is the standard axial (mod 180 deg) concentration.
    ang2 = 2.0 * np.arctan2(d[:, 1], d[:, 0])
    anis = float(np.abs(np.mean(np.exp(1j * ang2))))

    a = np.einsum("i,ij,ik->jk", w, n, n)
    b = np.einsum("i,ij,i->j", w, n, np.einsum("ij,ij->i", n, xs))
    try:
        p = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return Epipole(np.array([W / 2.0, H_PX / 2.0]), False, len(xs), anis, 0.0)

    # How many of the voting lines the solution actually explains, measured as an ANGLE.
    #
    # This was a fixed 3 px tolerance on |n . (p - x)|, which is the epipole's offset from
    # the line through x along mu. That offset grows with |p - x| for a fixed angular
    # error, so a fixed pixel tolerance is only meaningful next to the epipole and reads
    # ~0.01 everywhere else -- it was measuring distance, not agreement. Measured 2026-09-29
    # over frames 700-758: 0.009 by the old rule, and a RANSAC fit maximising the same
    # quantity landed 282 px away and hopped four times as much, which is what a
    # meaningless objective looks like when you optimise it harder.
    #
    # The angular form is also the quantity the direction test itself thresholds, so this
    # number now says directly what fraction of the background would be called static by
    # the epipole it voted for.
    v = p[None, :] - xs
    nv = np.linalg.norm(v, axis=1)
    ok = nv > 1e-6
    sin_a = np.abs(np.einsum("ij,ij->i", n[ok], v[ok] / nv[ok, None]))
    inl = float((sin_a <= 0.35).mean()) if ok.any() else 0.0

    reliable = bool(anis >= MIN_DIRECTION_ANISOTROPY and len(xs) >= MIN_EPIPOLE_PTS
                    and np.all(np.isfinite(p)))
    return Epipole(p, reliable, len(xs), anis, inl)


def background_gain(x: np.ndarray, mu: np.ndarray, epipole: Epipole) -> float:
    """This pair's typical parallax per unit distance from the epipole.

    Used only to normalise `gamma` in the structure-consistency test, so that a window
    whose frame pairs have different baselines can still be compared against itself. It is
    a median over grid points at many depths, so it is a scale normaliser and NOT a
    calibrated baseline -- no physical meaning should be read into its value.
    """
    if not len(x):
        return 0.0
    v = epipole.point[None, :] - x
    r = np.linalg.norm(v, axis=1)
    ok = r > max(1.0, px(10.0))
    if ok.sum() < 8:
        return 0.0
    u = v[ok] / r[ok][:, None]
    along = np.einsum("ij,ij->i", mu[ok], u)
    return float(np.median(np.abs(along) / r[ok]))
