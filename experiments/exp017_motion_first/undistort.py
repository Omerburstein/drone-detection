"""Radial undistortion for EXP-017, and the honest account of where lambda comes from.

The model
---------
One-parameter division model, on coordinates:

    p_u = c + (p_d - c) / (1 + lam * r_d^2),   r normalised so r = 1 at the corner

One parameter, not five, because we have **no calibration target** -- this is a screen
recording of a goggles display, the lens is behind a digital link that already applies its
own correction, and nothing in the repo can be pointed at a checkerboard. A single
division term is what a plumb-line fit can actually identify from footage; fitting k1..k3
to a tree line would fit the tree line.

Which direction lambda runs
---------------------------
`lam < 0` is barrel (the fisheye look: straight lines bow away from centre), `lam > 0` is
pincushion. `undistort_points(lam < 0)` pushes points outward with radius, straightening
a barrel-bowed line.

Two entry points, and they are not interchangeable
--------------------------------------------------
  * `undistort_points` -- coordinates only. This is what the geometry wants: it changes
    no pixel and invents no interpolation, so a residual measured after it is measured on
    real samples. `distcheck.py` (EXP-017 scratch) used this to ask whether the
    post-homography residual is radial.
  * `undistort_maps` / `remap` -- an actual resampled image, needed when a *statistic over
    pixels* has to be computed in undistorted space, which is stage 1's case: it
    thresholds texture and luminance and then walks connected components, and none of that
    can be carried through a coordinate transform after the fact.

The remap costs a bilinear interpolation per pixel per frame and it is lossy. Do not put
it in front of the difference image without re-measuring: EXP-015's `win_b5_e4` normal-flow
map reads a 1.4 px displacement on the drone, and bilinear resampling blurs at that scale.
"""
from __future__ import annotations

import numpy as np
import cv2

from clipcfg import CLIP

W, H = CLIP["width"], CLIP["height"]

# The optical axis. The crop is a symmetric window out of the goggles picture (MANIFEST:
# 1440x1080 at x=540, equal pillarbox either side), so the picture centre is the best
# available estimate. `check_fisheye.py` refits it rather than trusting this.
CENTRE = np.array(CLIP.get("optical_centre", (W / 2.0, H / 2.0)), float)
SCALE = 0.5 * float(np.hypot(W, H))      # r = 1 at the corner


def undistort_points(pts: np.ndarray, lam: float,
                     centre: np.ndarray | None = None) -> np.ndarray:
    """Distorted -> undistorted. (N, 2) in, (N, 2) out."""
    c = CENTRE if centre is None else np.asarray(centre, float)
    d = (np.asarray(pts, float) - c) / SCALE
    r2 = (d ** 2).sum(axis=1, keepdims=True)
    return c + (d / (1.0 + lam * r2)) * SCALE


def distort_points(pts: np.ndarray, lam: float,
                   centre: np.ndarray | None = None) -> np.ndarray:
    """Undistorted -> distorted, in closed form.

    r_u = r_d / (1 + lam r_d^2) inverts to lam r_u r_d^2 - r_d + r_u = 0, and the root
    that tends to r_d = r_u as lam -> 0 is the one below. The other root runs off to
    infinity and would fold the image.
    """
    c = CENTRE if centre is None else np.asarray(centre, float)
    d = (np.asarray(pts, float) - c) / SCALE
    ru = np.linalg.norm(d, axis=1)
    if abs(lam) < 1e-12:
        return np.asarray(pts, float)
    disc = 1.0 - 4.0 * lam * ru ** 2
    # Beyond the discriminant the model has no real inverse: that radius is not imaged at
    # all. Clamping rather than producing NaN keeps the map finite; `remap` masks it.
    disc = np.clip(disc, 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        rd = np.where(ru > 1e-9, (1.0 - np.sqrt(disc)) / (2.0 * lam * np.maximum(ru, 1e-9)), 0.0)
    scale = np.where(ru > 1e-9, rd / np.maximum(ru, 1e-9), 1.0)[:, None]
    return c + d * scale * SCALE


def undistort_maps(lam: float, centre: np.ndarray | None = None):
    """(map_x, map_y) for cv2.remap, plus the validity mask of the undistorted frame.

    Built by *distorting* the destination grid, which is the direction cv2.remap wants:
    for every undistorted pixel, where did it come from in the source.
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    src = distort_points(grid, lam, centre)
    mx = src[:, 0].reshape(H, W).astype(np.float32)
    my = src[:, 1].reshape(H, W).astype(np.float32)
    inside = (mx >= 0) & (mx <= W - 1) & (my >= 0) & (my <= H - 1)
    return mx, my, inside


def fov_retained(lam: float, centre: np.ndarray | None = None) -> float:
    """Fraction of the SOURCE frame still visible after undistorting to the same canvas.

    `undistort_maps`'s `inside` flag answers the other question -- which destination
    pixels have a source -- and for a barrel correction the answer is "all of them",
    because that map zooms in. The picture being thrown away is at the *source* edges,
    and only the forward map sees it. Reporting `inside` alone would call a 40% crop
    "keeps 100%", which is how a crop gets mistaken for an improvement.
    """
    return float(visible_source(lam, centre).mean())


def visible_source(lam: float, centre: np.ndarray | None = None) -> np.ndarray:
    """Bool (H, W) over the SOURCE frame: True where that pixel survives the undistortion.

    Needed to compare a statistic before and after on equal ground. A sky fraction over
    the whole source against one over a cropped destination is not a comparison.
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    u = undistort_points(np.column_stack([xx.ravel(), yy.ravel()]), lam, centre)
    vis = (u[:, 0] >= 0) & (u[:, 0] <= W - 1) & (u[:, 1] >= 0) & (u[:, 1] <= H - 1)
    return vis.reshape(H, W)


def remap(img: np.ndarray, mx: np.ndarray, my: np.ndarray) -> np.ndarray:
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=0)


def _selfcheck() -> int:
    """Round-trip and sign checks. The closed-form inverse is where a sign error hides.

        PYTHONPATH="experiments/exp017_motion_first;." py -3.13 -m undistort
    """
    rng = np.random.default_rng(3)
    pts = np.column_stack([rng.uniform(0, W, 4000), rng.uniform(0, H, 4000)])
    ok = True
    for lam in (-0.60, -0.35, -0.16, 0.0, 0.16, 0.35):
        back = distort_points(undistort_points(pts, lam), lam)
        err = float(np.abs(back - pts).max())
        good = err < 1e-6
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] round trip at lam {lam:+.2f}: "
              f"max error {err:.2e} px")

    # Sign: lam < 0 must push points AWAY from centre (that is what straightens barrel).
    far = np.array([[W - 1.0, H - 1.0]])
    r0 = np.linalg.norm(far - CENTRE)
    r_neg = float(np.linalg.norm(undistort_points(far, -0.2) - CENTRE))
    r_pos = float(np.linalg.norm(undistort_points(far, +0.2) - CENTRE))
    good = r_neg > r0 > r_pos
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] lam<0 pushes out ({r0:.0f} -> {r_neg:.0f} px), "
          f"lam>0 pulls in ({r0:.0f} -> {r_pos:.0f} px)")

    # A barrel correction on a fixed canvas must LOSE field of view, never gain it.
    f_neg, f_pos = fov_retained(-0.2), fov_retained(0.0)
    good = f_neg < f_pos == 1.0
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] barrel correction crops: "
          f"{f_neg * 100:.1f}% retained against {f_pos * 100:.1f}% at lam=0")

    print("ALL PASS" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selfcheck())
