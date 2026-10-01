"""Accumulate a candidate's geometry over a short window, and drop the ones that flash.

Why a window changes the answer rather than just smoothing it
-------------------------------------------------------------
EXP-019 measured two things that both point here.

  * The **epipole hops 99-127 px between consecutive frames.** That is not a bad
    estimator -- a RANSAC fit over the same points was four times worse -- it is that a
    single pair's residuals are ~1 px, so each point's direction is mostly tracking noise
    and the fit wanders. Pooling every pair's votes in the window fits **one** epipole from
    five times the evidence.
  * The direction test ran at **x1.74 chance on O4 and x1.40 on analog**. A static point's
    residuals all lie along its epipolar line, so they add **coherently** across the
    window while noise adds as sqrt(k). Testing the accumulated `Sum mu` instead of one
    step is the same test with a better signal-to-noise ratio, not a different one.

And the persistence rule is the cheap half: a peak that appears in one frame and not the
next four is noise in the difference image, whatever direction it happened to point.

The window
----------
`k` frames, each paired with its own predecessor, so `k` frames give `k` candidate maps
and "appears in 4 of 5 frames" means what it says. That needs `k + 1` raw frames.

Everything is accumulated in the **reference frame** -- the newest frame in the window --
by chaining the per-pair homographies. A point's residual in frame j is carried forward by
transforming both its predicted position and its landing point, and differencing there, so
the sum is of comparable vectors rather than of vectors in five different coordinate
systems.

The assumption this makes, stated
---------------------------------
Pooling the epipole over the window assumes the **translation direction is roughly
constant across it**. At 30 fps, k = 5 is 1/6 s, which for an airframe that is not
snap-rolling is reasonable. It is not free, though: `WindowFit.epipole_spread` reports how
far the per-pair epipoles sit from the pooled one, so a window that violates the
assumption says so instead of quietly averaging across a manoeuvre.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

import criteria
import geometry
from clipcfg import CLIP

W, H = CLIP["width"], CLIP["height"]
S = W / 1440.0


def px(v: float) -> float:
    return v * S


# A tracked point counts as "appearing" in a frame if a candidate peak of that frame sits
# within this distance of it. Generous enough to match the same feature after a peak
# shifts by a pixel or two, tight enough that a different blob nearby is not credited.
APPEAR_RADIUS = px(8.0)
DEFAULT_K = 5
DEFAULT_MIN_APPEAR = 4


@dataclass(frozen=True)
class Pair:
    """One frame pair's worth of everything the window needs. Built once, reused k times."""
    frame: int                 # the pair's CURRENT frame number
    grey: np.ndarray           # the current frame, prepped
    prev_grey: np.ndarray
    hmat: np.ndarray           # current -> previous
    cands: np.ndarray          # (N, 3) of (value, x, y), current-frame coords
    grid_x: np.ndarray         # background residual field, current-frame coords
    grid_mu: np.ndarray
    epipole: geometry.Epipole  # this pair alone, kept for the comparison


@dataclass(frozen=True)
class WindowFit:
    """The window's pooled geometry."""
    epipole: geometry.Epipole
    n_pairs: int
    epipole_spread: float      # median distance of the per-pair epipoles from the pooled one
    reliable: bool


@dataclass(frozen=True)
class Track:
    """One candidate carried across the window."""
    x: float                   # position in the reference frame
    y: float
    appearances: int           # in how many of the k frames a candidate peak was there
    steps: int                 # how many pairs produced a usable residual
    total_mu: np.ndarray       # accumulated residual, reference-frame coords
    mean_sin: float            # mean per-step |sin|, as a cross-check on the accumulated one
    direction: criteria.DirectionVerdict

    @property
    def persistent(self) -> bool:
        return self.appearances >= DEFAULT_MIN_APPEAR

    def verdict(self, min_appear: int = DEFAULT_MIN_APPEAR) -> str:
        """'flash', 'rejected', 'unjudged' or 'survives'. Order matters: persistence first.

        A candidate that fails persistence is dropped without asking the direction test
        anything, because the direction of a one-frame blob is not a measurement.
        """
        if self.appearances < min_appear:
            return "flash"
        if not self.direction.usable:
            return "unjudged"
        return "survives" if self.direction.moving else "rejected"


def forward_maps(pairs: list[Pair]) -> list[np.ndarray]:
    """For each pair, the homography carrying its current frame to the newest one.

    `Pair.hmat` runs current -> previous, so stepping *forward* in time inverts it, and
    reaching the reference frame composes every inverse above this pair.
    """
    k = len(pairs)
    out = [np.eye(3) for _ in range(k)]
    for t in range(k - 2, -1, -1):
        out[t] = out[t + 1] @ np.linalg.inv(pairs[t + 1].hmat)
    return out


def _carry(hmat: np.ndarray, x: np.ndarray, mu: np.ndarray):
    """Move a residual field into another frame by transforming both of its endpoints.

    A residual is a difference of two positions, so it is carried by transforming the
    positions and differencing there. Transforming the vector itself would be wrong: a
    homography is not linear on directions.
    """
    a = geometry.apply_h(hmat, x)
    b = geometry.apply_h(hmat, x + mu)
    return a, b - a


def accumulate_grid(pairs: list[Pair], valid: np.ndarray | None = None,
                    min_steps: int | None = None):
    """Track the background grid across the window and SUM each point's residual.

    This is the part that a first version got wrong, and the self-test caught it. Pooling
    every pair's votes into one fit gives k times as many votes -- but each one is still a
    single pair's ~1 px residual whose direction is mostly noise, so the *directional
    concentration* of the set does not improve, and `Epipole.reliable` is gated on exactly
    that concentration. Measured on the synthetic: at 0.8 px of tracking noise, 9 of 24
    single pairs came back reliable and only 5 of 20 pooled windows did. More noise, no
    better.

    Accumulating **per point** is different in kind. A static point's residuals lie along
    its own epipolar line at every step, so they add coherently while the noise adds as
    sqrt(k); each surviving point then votes once, with a direction that is genuinely
    sharper. That raises anisotropy as well as accuracy, which is what the gate needs.

    Returns (x_ref, sum_mu, steps) in reference-frame coordinates.
    """
    k = len(pairs)
    need = (k - 1) if min_steps is None else min_steps
    seeds = geometry.grid_points()
    if valid is not None and len(seeds):
        inside = valid[np.clip(seeds[:, 1].astype(int), 0, H - 1),
                       np.clip(seeds[:, 0].astype(int), 0, W - 1)]
        seeds = seeds[inside]
    if not len(seeds):
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0, int)

    pos = _track_back(pairs, seeds)
    fwd = forward_maps(pairs)
    total = np.zeros((len(seeds), 2))
    steps = np.zeros(len(seeds), int)

    for t in range(k):
        cur_i = k - 1 - t
        p_cur, p_prev = pos[cur_i], pos[cur_i + 1]
        ok = np.isfinite(p_cur).all(axis=1) & np.isfinite(p_prev).all(axis=1)
        if not ok.any():
            continue
        x, mu = geometry.residual_field(p_prev[ok], p_cur[ok], pairs[t].hmat)
        _, d = _carry(fwd[t], x, mu)
        total[ok] += d
        steps[ok] += 1

    keep = steps >= need
    return pos[0][keep].astype(float), total[keep], steps[keep]


def fit_window(pairs: list[Pair], x_ref: np.ndarray, mu_ref: np.ndarray) -> WindowFit:
    """One epipole from the ACCUMULATED background field, in the reference frame.

    `x_ref` / `mu_ref` come from `accumulate_grid`. They are kept as arguments rather than
    computed here so this stays a pure function of geometry -- `accumulate_grid` needs real
    images for LK, and this does not, which is what lets the self-test drive it directly.
    """
    if not len(x_ref):
        return WindowFit(geometry.estimate_epipole(np.zeros((0, 2)), np.zeros((0, 2))),
                         len(pairs), float("nan"), False)

    # The floor scales with the window: 0.6 px is noise over one pair and still noise
    # after five, so inheriting the per-pair threshold would admit exactly the votes the
    # accumulation is meant to suppress.
    ep = geometry.estimate_epipole(x_ref, mu_ref,
                                   min_residual=geometry.MIN_RESIDUAL_PX * len(pairs))
    fwd = forward_maps(pairs)

    # How far each pair's own epipole sits from the pooled answer, carried to the same
    # frame. Large spread means the translation direction turned inside the window and the
    # pooling assumption does not hold there.
    offs = []
    for p, hmat in zip(pairs, fwd):
        if not p.epipole.reliable:
            continue
        e = geometry.apply_h(hmat, p.epipole.point.reshape(1, 2))[0]
        offs.append(float(np.hypot(e[0] - ep.point[0], e[1] - ep.point[1])))
    spread = float(np.median(offs)) if offs else float("nan")
    return WindowFit(ep, len(pairs), spread, bool(ep.reliable))


def _track_back(pairs: list[Pair], seeds: np.ndarray) -> np.ndarray:
    """Positions of each seed in every frame of the window, newest first.

    Returns (k + 1, N, 2) with NaN where the track was lost: index 0 is the reference
    frame, index k is the frame before the oldest pair's current frame -- which the oldest
    pair needs in order to form a residual at all.
    """
    k = len(pairs)
    out = np.full((k + 1, len(seeds), 2), np.nan, np.float32)
    out[0] = seeds
    cur = seeds.reshape(-1, 1, 2).astype(np.float32)
    alive = np.ones(len(seeds), bool)

    for step in range(k):
        p = pairs[k - 1 - step]
        if not alive.any():
            break
        src = cur[alive]
        back, st, _ = cv2.calcOpticalFlowPyrLK(p.grey, p.prev_grey, src, None, **geometry.LK)
        fwd2, st2, _ = cv2.calcOpticalFlowPyrLK(p.prev_grey, p.grey, back, None, **geometry.LK)
        ok = (st.ravel() == 1) & (st2.ravel() == 1)
        ok &= np.linalg.norm((fwd2 - src).reshape(-1, 2), axis=1) < geometry.FB_MAX

        nxt = np.full((len(seeds), 1, 2), np.nan, np.float32)
        idx = np.nonzero(alive)[0]
        nxt[idx[ok]] = back[ok]
        out[step + 1] = nxt.reshape(-1, 2)
        alive = np.zeros(len(seeds), bool)
        alive[idx[ok]] = True
        cur = nxt
    return out


def _appears(cands: np.ndarray, pt: np.ndarray, radius: float) -> bool:
    """Is one of this frame's candidate peaks within `radius` of the tracked point?"""
    if not len(cands) or not np.all(np.isfinite(pt)):
        return False
    d = np.hypot(cands[:, 1] - pt[0], cands[:, 2] - pt[1])
    return bool(d.min() <= radius)


def build_tracks(pairs: list[Pair], fit: WindowFit, *,
                 sin_threshold: float = 0.35,
                 appear_radius: float = APPEAR_RADIUS) -> list[Track]:
    """Carry every candidate of the newest frame across the window and judge it there."""
    k = len(pairs)
    ref = pairs[-1]
    if not len(ref.cands):
        return []

    seeds = ref.cands[:, 1:3].astype(np.float32)
    pos = _track_back(pairs, seeds)          # (k + 1, N, 2), index 0 = reference frame
    fwd = forward_maps(pairs)

    tracks = []
    for i in range(len(seeds)):
        appearances = 0
        total = np.zeros(2)
        sins, steps = [], 0
        for t in range(k):
            # pair t's current frame is window index (k - 1 - t) counting back from the
            # reference, and its previous frame is one older still.
            cur_i = k - 1 - t
            p_cur, p_prev = pos[cur_i, i], pos[cur_i + 1, i]
            if _appears(pairs[t].cands, p_cur, appear_radius):
                appearances += 1
            if not (np.all(np.isfinite(p_cur)) and np.all(np.isfinite(p_prev))):
                continue
            x, mu = geometry.residual_field(p_prev.reshape(1, 2), p_cur.reshape(1, 2),
                                            pairs[t].hmat)
            a, d = _carry(fwd[t], x, mu)
            total += d[0]
            steps += 1
            if pairs[t].epipole.reliable:
                v = criteria.epipolar_direction(
                    d[0], a[0], fit.epipole.point,
                    epipole_reliable=fit.reliable,
                    degenerate=fit.epipole.degenerate_for(*a[0]),
                    sin_threshold=sin_threshold, min_displacement=px(1.0))
                if v.usable:
                    sins.append(v.sin_angle)

        ref_pt = pos[0, i]
        verdict = criteria.epipolar_direction(
            total, ref_pt, fit.epipole.point,
            epipole_reliable=fit.reliable,
            degenerate=fit.epipole.degenerate_for(*ref_pt),
            sin_threshold=sin_threshold,
            # Accumulated over k steps, so the floor scales with the window. A track that
            # has drifted under 2 px in five frames has no direction worth testing.
            min_displacement=px(2.0))
        tracks.append(Track(float(ref_pt[0]), float(ref_pt[1]), appearances, steps,
                            total, float(np.mean(sins)) if sins else float("nan"),
                            verdict))
    return tracks


def _selfcheck() -> int:
    """Synthetic checks for the two things the window adds.

    Run it with the EXP-017 and EXP-015 directories on PYTHONPATH:
    `py -3.13 -m window`.

    `accumulate_grid` and `build_tracks` need real images for LK and are exercised on
    video. What is checked here is what would be wrong silently: the homography chaining,
    and whether accumulating actually buys a sharper epipole. The accumulated field is
    built from exact projections rather than from LK, so a failure here is geometry, not
    tracking.
    """
    import selftest as sr

    print("window.py self-check (synthetic, no video)")
    print()
    ok_all = True
    rng = np.random.default_rng(11)
    plane, off = sr.SCENE
    static = np.vstack([plane, off])
    dummy = np.zeros((4, 4), np.uint8)
    # Forward motion, but slow enough that the camera never reaches the nearest scene
    # point over the longest run below. The first version flew to z = 40 through a plane
    # at Z = 40 and divided by zero.
    vel = np.array([0.1969, 0.0525, 0.70])
    truth = sr.C + sr.F * vel[:2] / vel[2]

    def noisy_frames(cams, noise):
        """One noisy projection per camera position, REUSED by both pairs that touch it.

        This matters and a first version got it wrong by drawing fresh noise for each pair.
        A real frame is measured once: its error enters pair t as the "current" endpoint
        and pair t+1 as the "previous" one, with opposite sign, so the intermediate frames'
        noise largely cancels when residuals are summed across a window. Independent draws
        destroy that cancellation and make accumulation look worse than it is -- which is
        exactly what this test reported before the fix.
        """
        return [sr.project(static, c) + rng.normal(0, noise, (len(static), 2)) for c in cams]

    def build(cams, seen, t0, k):
        """k consecutive pairs ending at cams[t0 + k], plus the accumulated field."""
        pairs = []
        for t in range(t0, t0 + k):
            hm = sr.fit_plane_homography(sr.project(plane, cams[t]),
                                         sr.project(plane, cams[t + 1]))
            x, mu = geometry.residual_field(seen[t], seen[t + 1], hm)
            pairs.append(Pair(t, dummy, dummy, hm, np.zeros((0, 3)), x, mu,
                              geometry.estimate_epipole(x, mu)))
        fwd = forward_maps(pairs)
        total = np.zeros((len(static), 2))
        x_ref = np.zeros((len(static), 2))
        for t, (pr, hm) in enumerate(zip(pairs, fwd)):
            a_c, d = _carry(hm, pr.grid_x, pr.grid_mu)
            total += d
            if t == len(pairs) - 1:
                x_ref = a_c
        return pairs, x_ref, total

    def jump(pts):
        pts = np.asarray(pts, float).reshape(-1, 2)
        if len(pts) < 2:
            return float("nan")
        return float(np.median(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    # --- 1. the chaining is right ----------------------------------------------
    k = 5
    cams = [vel * j for j in range(k + 1)]
    pairs, _, _ = build(cams, noisy_frames(cams, 0.35), 0, k)
    fwd = forward_maps(pairs)
    worst = 0.0
    for t in range(k):
        carried = geometry.apply_h(fwd[t], sr.project(plane, cams[t + 1]))
        worst = max(worst, float(np.abs(carried - sr.project(plane, cams[k])).max()))
    good = worst < 1.0
    ok_all &= good
    print(f"  [{'PASS' if good else 'FAIL'}] forward_maps carries frame t to the reference"
          f"  -- worst {worst:.3f} px")

    # --- 2. accumulating sharpens the epipole, across three noise regimes -------
    #
    # A first version pooled each pair's votes instead of accumulating per point, and
    # FAILED: k times as many votes of the same poor quality do not concentrate, so the
    # anisotropy never improved and the reliability gate still refused. At 0.8 px of noise
    # only 5 of 20 pooled windows came back reliable against 9 of 24 single pairs. What
    # follows is the same measurement after the fix.
    n_long = 14
    cams_l = [vel * j for j in range(n_long + 2)]
    for noise in (0.35, 0.8, 1.6):
        seen = noisy_frames(cams_l, noise)
        per, pooled, errs_pair, errs_pool = [], [], [], []
        n_rel_pair = n_rel_pool = 0
        for t0 in range(n_long - k):
            prs, xr, mr = build(cams_l, seen, t0, k)
            fit = fit_window(prs, xr, mr)
            last = prs[-1].epipole
            # Errors and jumps are collected on EVERY window, not only the reliable ones.
            # Filtering each estimator by its own reliability compares them on different
            # subsets -- the per-pair fit would be scored only on the windows it found
            # easy -- which is how the first version of this test made accumulation look
            # worse while it was in fact usable twice as often.
            per.append(last.point)
            pooled.append(fit.epipole.point)
            errs_pair.append(float(np.linalg.norm(last.point - truth)))
            errs_pool.append(float(np.linalg.norm(fit.epipole.point - truth)))
            n_rel_pair += int(last.reliable)
            n_rel_pool += int(fit.reliable)
        n_win = n_long - k
        med = lambda v: float(np.median(v)) if v else float("nan")
        print()
        print(f"  noise {noise:.2f} px:  (all {n_win} windows, both estimators)")
        print(f"    reliable  per-pair {n_rel_pair:2d}/{n_win}   "
              f"accumulated {n_rel_pool:2d}/{n_win}")
        print(f"    error     per-pair {med(errs_pair):6.1f} px   "
              f"accumulated {med(errs_pool):6.1f} px")
        print(f"    jump      per-pair {jump(per):6.1f} px   "
              f"accumulated {jump(pooled):6.1f} px")
        if noise == 0.8:
            # Asserted: usability and accuracy, which is what this synthetic can see.
            # NOT asserted: frame-to-frame jump. Per-pair jump here is a few px, while
            # EXP-019 measured 126.8 px on video -- this scene is nowhere near that
            # scatter regime, so a jump assertion would be claiming something the test
            # cannot observe. The jump comparison is made on video instead, and the
            # numbers are printed above only as context.
            good = n_rel_pool >= n_rel_pair and med(errs_pool) < med(errs_pair)
            ok_all &= good
            print(f"    [{'PASS' if good else 'FAIL'}] accumulating is more often usable "
                  f"and closer to truth (jump is asserted on video, not here)")

    # --- 3. persistence is applied before the direction test --------------------
    e = np.array([900.0, 10.0])

    def mk(n, d):
        return Track(10.0, 10.0, n, n, d, 0.1,
                     criteria.epipolar_direction(d, np.array([10.0, 10.0]), e,
                                                 epipole_reliable=True, degenerate=False))

    steady, flash = mk(5, np.array([9.0, 0.0])), mk(2, np.array([9.0, 9.0]))
    good = steady.verdict() == "rejected" and flash.verdict() == "flash"
    ok_all &= good
    print()
    print(f"  [{'PASS' if good else 'FAIL'}] persistence is applied before direction"
          f"  -- steady '{steady.verdict()}', 2/5 flash '{flash.verdict()}'")

    print()
    print("ALL PASS" if ok_all else "FAILURES ABOVE")
    return 0 if ok_all else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selfcheck())
