"""Stage 2b, the scene branch: two planes instead of one, and the rejections that need them.

What this adds to `geometry.py`
--------------------------------
`geometry.py` fits **one** homography and calls everything left over "parallax". EXP-012b
measured that leftover at ~1 px median, and the 5-frame window run then measured that its
direction is not radial -- the field is a near-uniform shift, which is what a *plane
misfit* looks like, not what depth-induced parallax looks like.

A single plane is the obvious suspect. The scene below the horizon is not one plane: there
is ground at tens of metres and a tree line or skyline at hundreds. One homography fitted
across both splits the difference, and every point gets a residual that is mostly
"the plane is wrong here" rather than "this point has a different depth".

So this module fits two:

  * `h_dominant` -- RANSAC over every tracked point, which takes whichever plane owns the
    most of the frame. EXP-017 measured that this is the ground; the code does not assume
    it, it reports where the layer's points sit so the run can say.
  * `h_second`   -- refit on the **outliers** of the first. That is the other plane, if
    there is one.

Each candidate is then assigned to whichever layer explains it better, and its residual is
measured against *that* layer.

The guard that makes this honest
---------------------------------
A RANSAC fit to 40 outlier points always returns a homography. It returns one when the
outliers are pure tracking noise, and the result is a second "layer" that is an artefact
of the fitting. That is precisely the failure the epipole made in EXP-019 -- a fit to noise
still returns coordinates -- and it cost a ledger entry to find.

`LayerFit.separation` is the defence: the median distance, over a grid spanning the frame,
between where the two homographies send the same point. Two genuinely different planes
disagree by pixels; two fits to the same plane plus noise agree to a fraction of one.
Below `MIN_SEPARATION_PX` the fit reports `two_layers = False` and the caller must treat
the frame as single-plane rather than assign anything. **The number is reported every
frame**, so "two planes" is a measurement in this run and not an assumption.

Where this is a rejector and never a confirmer
-----------------------------------------------
Stated in the plan and repeated here because the code cannot enforce it: near the focus of
expansion every direction is near-radial, and an intercept target sits near the FOE. So a
candidate *passing* the direction test means nothing, and only a rejection carries
information. `epipolar_reject` is named for what it does. Nothing here should ever be used
to promote a candidate.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

import geometry
from clipcfg import CLIP

W, H = CLIP["width"], CLIP["height"]
S = W / 1440.0


def px(v: float) -> float:
    return v * S


RANSAC_PX = px(3.0)          # the tolerance `common.homography` already uses
MIN_LAYER_PTS = 20           # fewer outliers than this is not a plane, it is noise
# Two fits to the SAME plane agree to well under a pixel. Two real planes, separated by
# tens of metres at these baselines, disagree by several. 1.5 px sits between the two and
# is quoted at 1440 px wide like every other constant here.
MIN_SEPARATION_PX = px(1.5)
# A candidate whose two layer residuals differ by less than this cannot be assigned: both
# planes explain it equally well, which happens wherever the two homographies agree.
ASSIGN_MARGIN_PX = px(0.5)


@dataclass(frozen=True)
class LayerFit:
    """Two planes for one frame pair, with the evidence that there are in fact two.

    `two_layers` false means only `h_dominant` may be used. It is not an error: most of
    the sky-heavy frames in this footage genuinely contain one usable plane.
    """
    h_dominant: np.ndarray           # current -> previous, the plane owning most points
    h_second: np.ndarray | None      # the same, refitted on the first fit's outliers
    n_dominant: int
    n_second: int
    separation: float                # median px disagreement between the two, over a grid
    dominant_mean_y: float           # where the dominant layer's points sit, for the report
    second_mean_y: float
    two_layers: bool

    def describe(self) -> str:
        if not self.two_layers:
            return (f"1 plane ({self.n_dominant} pts, mean y {self.dominant_mean_y:.0f}); "
                    f"second {'refused' if self.h_second is None else 'too close'} "
                    f"(sep {self.separation:.2f} px)")
        return (f"2 planes: dominant {self.n_dominant} pts @ y {self.dominant_mean_y:.0f}, "
                f"second {self.n_second} pts @ y {self.second_mean_y:.0f}, "
                f"sep {self.separation:.1f} px")


def _grid_for_separation(step: int = 80) -> np.ndarray:
    """A coarse grid spanning the frame, for measuring how far apart two homographies are."""
    xs = np.arange(step // 2, W, step, dtype=np.float32)
    ys = np.arange(step // 2, H, step, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], 1)


def homography_separation(h_a: np.ndarray, h_b: np.ndarray) -> float:
    """Median px between where two homographies send the same point, over the frame.

    Measured over a spanning grid rather than over the tracked points, because the tracked
    points are exactly where each fit was optimised and would flatter both.
    """
    g = _grid_for_separation()
    a, b = geometry.apply_h(h_a, g), geometry.apply_h(h_b, g)
    d = np.linalg.norm(a - b, axis=1)
    ok = np.isfinite(d)
    return float(np.median(d[ok])) if ok.any() else float("inf")


def fit_layers(prev_pts: np.ndarray, cur_pts: np.ndarray) -> LayerFit | None:
    """Fit the dominant plane, then refit on its outliers to find a second.

    Returns None when there are not even enough points for one plane -- the caller must
    then skip the frame rather than warp with an identity that silently means "no motion".
    """
    if len(prev_pts) < MIN_LAYER_PTS or len(cur_pts) != len(prev_pts):
        return None
    a = np.asarray(prev_pts, np.float32)
    b = np.asarray(cur_pts, np.float32)

    h1, inl = cv2.findHomography(b, a, cv2.RANSAC, RANSAC_PX)
    if h1 is None:
        return None
    inl = inl.ravel().astype(bool)
    out = ~inl
    y1 = float(b[inl, 1].mean()) if inl.any() else float("nan")

    h2, n2, y2 = None, 0, float("nan")
    if out.sum() >= MIN_LAYER_PTS:
        cand, inl2 = cv2.findHomography(b[out], a[out], cv2.RANSAC, RANSAC_PX)
        if cand is not None and np.all(np.isfinite(cand)):
            inl2 = inl2.ravel().astype(bool)
            if inl2.sum() >= MIN_LAYER_PTS:
                h2, n2 = cand, int(inl2.sum())
                y2 = float(b[out][inl2, 1].mean())

    sep = homography_separation(h1, h2) if h2 is not None else float("nan")
    two = bool(h2 is not None and np.isfinite(sep) and sep >= MIN_SEPARATION_PX)
    return LayerFit(h1, h2, int(inl.sum()), n2, sep, y1, y2, two)


def assign_layer(prev_pt: np.ndarray, cur_pt: np.ndarray, fit: LayerFit):
    """Which layer explains this point's motion, and its residual under that layer.

    Returns (layer, residual_vector, x_predicted) with `layer` one of "dominant",
    "second" or "ambiguous". Ambiguous is returned where both planes fit equally well,
    which is not a failure -- it is what happens wherever the two homographies agree, and
    a caller that forces a choice there is inventing a depth it did not measure.
    """
    p = np.asarray(prev_pt, np.float32).reshape(1, 2)
    c = np.asarray(cur_pt, float).reshape(2)

    def residual(hmat):
        x = geometry.apply_h(np.linalg.inv(hmat), p)[0]
        return x, c - x

    x1, r1 = residual(fit.h_dominant)
    if not fit.two_layers:
        return "dominant", r1, x1
    x2, r2 = residual(fit.h_second)
    n1, n2 = float(np.linalg.norm(r1)), float(np.linalg.norm(r2))
    if abs(n1 - n2) < ASSIGN_MARGIN_PX:
        return "ambiguous", r1, x1
    return ("dominant", r1, x1) if n1 < n2 else ("second", r2, x2)


def layered_residual_field(prev_pts: np.ndarray, cur_pts: np.ndarray, fit: LayerFit):
    """Every tracked point's residual, each measured against its OWN layer.

    This is the field the epipole should be fitted to. `geometry.residual_field` measures
    everything against one plane, so on a two-plane scene it hands the fit a systematic
    offset for every point of the minority plane -- which is one concrete mechanism for
    the uniform shift the window run measured.

    Returns (x, mu, layer_index) with layer_index 0 dominant, 1 second, -1 ambiguous.
    """
    n = len(prev_pts)
    x = np.zeros((n, 2))
    mu = np.zeros((n, 2))
    lab = np.full(n, -1, int)
    code = {"dominant": 0, "second": 1, "ambiguous": -1}
    for i in range(n):
        which, r, xi = assign_layer(prev_pts[i], cur_pts[i], fit)
        x[i], mu[i], lab[i] = xi, r, code[which]
    return x, mu, lab


def epipolar_reject(displacement: np.ndarray, position: np.ndarray,
                    epipole: geometry.Epipole, *, sin_threshold: float = 0.35,
                    min_displacement: float | None = None):
    """A REJECTOR. True means "this is static parallax, drop it"; False never confirms.

    Deliberately not symmetrical with `criteria.epipolar_direction`, which returns a
    `moving` verdict. Near the FOE -- where an intercept target sits -- a passing direction
    carries no information at all, so a function that can return "moving" invites reading
    a pass as evidence. This one can only ever say "reject" or "no opinion".

    Returns **(reject, judged, reason)**. The middle field is not decoration: there are two
    different reasons this returns `reject=False`, and conflating them corrupts every rate
    computed from the result. "The test looked and the direction is off-epipolar" is a
    judgement; "there is no usable epipole / the candidate sits at the FOE / it barely
    moved" is a refusal to look. A caller that counts refusals as passes dilutes its
    denominator with candidates the test never judged, and its lift-over-chance then
    measures how often the test declined rather than how well it discriminates.
    """
    md = px(2.0) if min_displacement is None else min_displacement
    d = np.asarray(displacement, float).reshape(2)
    x = np.asarray(position, float).reshape(2)
    if not epipole.reliable:
        return False, False, "unjudged: epipole unreliable"
    if epipole.degenerate_for(float(x[0]), float(x[1])):
        return False, False, "unjudged: near the FOE"
    nd = float(np.linalg.norm(d))
    if nd < md:
        return False, False, "unjudged: displacement too small"
    v = epipole.point - x
    nv = float(np.linalg.norm(v))
    if nv < 1e-6:
        return False, False, "unjudged: on the epipole"
    sin_a = float(min(1.0, abs(d[0] * v[1] - d[1] * v[0]) / (nd * nv)))
    if sin_a < sin_threshold:
        return True, True, f"radial (|sin| {sin_a:.2f})"
    return False, True, f"off-epipolar (|sin| {sin_a:.2f})"


def depth_aware_ring(score: np.ndarray, x: float, y: float, r_in: float, r_out: float,
                     same_label: np.ndarray, min_px: int = 40):
    """Ring median and robust scatter using ONLY pixels sharing the candidate's label.

    The plan's rule: never let a ring straddle the stage-1 boundary. A ring half in sky
    and half in near tree line describes neither, and its inflated scatter is what makes a
    target at the horizon read as low contrast. `same_label` is the stage-1 mask for the
    candidate's own side.

    Returns (median, sigma, n_used) or None when too little of the ring survives -- which
    is itself the answer at a boundary, and the caller must refuse rather than widen.

    One caveat on the scatter: MAD is degenerate on an exactly two-valued ring split 50/50,
    where it can return 0 for a ring that straddles a large step. Real imagery carries
    noise on both sides so this does not arise in the run, but a synthetic step scene will
    hit it, and a future caller reading `sigma == 0` as "clean ring" would be wrong.
    """
    h, w = score.shape
    r = int(np.ceil(r_out)) + 2
    x0, x1 = max(0, int(x) - r), min(w, int(x) + r + 1)
    y0, y1 = max(0, int(y) - r), min(h, int(y) + r + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    yy, xx = np.ogrid[y0:y1, x0:x1]
    d2 = (xx - x) ** 2 + (yy - y) ** 2
    ring = (d2 >= r_in ** 2) & (d2 <= r_out ** 2) & same_label[y0:y1, x0:x1]
    v = score[y0:y1, x0:x1][ring]
    if v.size < min_px:
        return None
    med = float(np.median(v))
    return med, float(1.4826 * np.median(np.abs(v - med))), int(v.size)


def _selfcheck() -> int:
    """Synthetic checks, run with `py -3.13 -m layers`.

    The one that matters most is 3: a scene with ONE plane must not come back with two.
    Every other failure here shows up as a bad number on video; that one shows up as a
    confident number that is about nothing, which is how EXP-019's epipole survived to the
    ledger.
    """
    rng = np.random.default_rng(23)
    fails = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal fails
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
        fails += 0 if ok else 1

    f, c = 900.0, np.array([W / 2.0, H / 2.0])

    def project(p3, cam):
        p = p3 - cam
        return (c + f * p[:, :2] / p[:, 2:3]).astype(np.float32)

    def plane(n, z, seed_rng):
        return np.column_stack([seed_rng.uniform(-30, 30, n), seed_rng.uniform(-20, 20, n),
                                np.full(n, z)])

    cam0, cam1 = np.zeros(3), np.array([0.35, 0.0, 0.0])

    print("layers self-check")

    # 1 & 2. Two genuine planes: they must be found, separated, and points assigned right.
    near, far = plane(220, 40.0, rng), plane(140, 260.0, rng)
    pts3 = np.vstack([near, far])
    truth = np.array([0] * len(near) + [1] * len(far))
    a, b = project(pts3, cam0), project(pts3, cam1)
    a += rng.normal(0, 0.05, a.shape).astype(np.float32)
    fit = fit_layers(a, b)
    check("two planes are found and separated",
          fit is not None and fit.two_layers,
          fit.describe() if fit else "no fit")

    if fit is not None and fit.two_layers:
        _, _, lab = layered_residual_field(a, b, fit)
        judged = lab >= 0
        # Which fitted layer is which true plane is arbitrary, so score both pairings.
        acc = max(float((lab[judged] == truth[judged]).mean()),
                  float((1 - lab[judged] == truth[judged]).mean()))
        check("points are assigned to the right plane", acc >= 0.80,
              f"{acc * 100:.0f}% correct on {int(judged.sum())} of {len(lab)} judged")
    else:
        check("points are assigned to the right plane", False, "no two-layer fit to test")

    # 3. ONE plane plus noise must NOT produce two. This is the guard.
    one = plane(320, 40.0, rng)
    a1, b1 = project(one, cam0), project(one, cam1)
    a1 += rng.normal(0, 0.3, a1.shape).astype(np.float32)
    f1 = fit_layers(a1, b1)
    check("a single plane does not come back as two",
          f1 is not None and not f1.two_layers,
          f1.describe() if f1 else "no fit")

    # 4. A moving object must have a large residual under whichever layer claims it.
    obj_prev = np.array([[3.0, 1.0, 45.0]])
    obj_cur = obj_prev + np.array([[1.2, 0.35, 0.0]])
    pa = project(np.vstack([pts3, obj_prev]), cam0)
    pb = project(np.vstack([pts3, obj_cur]), cam1)
    f2 = fit_layers(pa[:-1], pb[:-1])
    if f2 is not None:
        _, r_obj, _ = assign_layer(pa[-1], pb[-1], f2)
        _, mu_bg, _ = layered_residual_field(pa[:-1], pb[:-1], f2)
        bg = float(np.median(np.linalg.norm(mu_bg, axis=1)))
        obj = float(np.linalg.norm(r_obj))
        check("a moving object's residual exceeds the background's", obj > 5.0 * bg,
              f"object {obj:.2f} px vs background median {bg:.2f} px")
    else:
        check("a moving object's residual exceeds the background's", False, "no fit")

    # 5. `epipolar_reject` never confirms. Off-epipolar must be "no opinion", not a pass.
    ep = geometry.Epipole(np.array([100.0, 100.0]), True, 99, 0.5, 0.5)
    perp, perp_j, _ = epipolar_reject(np.array([0.0, 20.0]), np.array([600.0, 100.0]), ep)
    rad, rad_j, _ = epipolar_reject(np.array([-20.0, 0.0]), np.array([600.0, 100.0]), ep)
    check("the direction test rejects radially and never confirms",
          rad and not perp, f"radial -> {rad}, perpendicular -> {perp} (must be True, False)")
    check("an off-epipolar candidate counts as JUDGED, not as a refusal",
          rad_j and perp_j,
          f"judged flags: radial {rad_j}, off-epipolar {perp_j} (both must be True)")

    # 6. An unreliable epipole yields no rejections at all, rather than a fit to noise --
    #    and must report itself UNJUDGED, so it never lands in a rate's denominator.
    bad = geometry.Epipole(np.array([100.0, 100.0]), False, 3, 0.01, 0.0)
    r, judged, why = epipolar_reject(np.array([-20.0, 0.0]), np.array([600.0, 100.0]), bad)
    check("an unreliable epipole rejects nothing and is not counted as judged",
          not r and not judged, why)
    # The FOE is the project's own case: an intercept target sits there. It must refuse.
    near_foe, foe_j, foe_why = epipolar_reject(np.array([-20.0, 0.0]),
                                               np.array([105.0, 102.0]), ep)
    check("a candidate at the FOE is refused, not passed",
          not near_foe and not foe_j, foe_why)

    # 7. The depth-aware ring excludes the other label, and refuses when too little is left.
    score = rng.normal(0, 1.0, (300, 400)).astype(np.float32)
    score[150:, :] += 50.0
    lab_sky = np.zeros((300, 400), bool)
    lab_sky[:150] = True
    # Probed ON the boundary, which is the only place the two rings can differ. The
    # property is the SCATTER: a ring straddling the step has its sigma set by the step,
    # and that inflated sigma is what makes a target at the horizon read as low contrast.
    got = depth_aware_ring(score, 200, 150, 10, 30, lab_sky)
    whole = depth_aware_ring(score, 200, 150, 10, 30, np.ones((300, 400), bool))
    check("the ring excludes the other label's step from its scatter",
          got is not None and whole is not None and whole[1] > 5.0 * got[1],
          f"sky-only sigma {got[1]:.1f} (median {got[0]:.1f}), "
          f"whole-ring sigma {whole[1]:.1f} (median {whole[0]:.1f})"
          if got and whole else "ring unusable")
    deep = depth_aware_ring(score, 200, 295, 10, 30, lab_sky)
    check("the ring refuses when its own label is nearly absent", deep is None,
          "refused" if deep is None else f"returned {deep}")

    print(f"\n{'all checks passed' if not fails else str(fails) + ' CHECK(S) FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_selfcheck())
