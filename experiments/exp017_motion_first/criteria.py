"""EXP-017 stage 2, steps 2 and 3: the two tests that replace `z >= z*`.

EXP-016 confirmed a hypothesis on `z = |Sum d| / (sqrt(k) * sigma)` -- a pure magnitude.
That cannot work, and the reason is geometric rather than statistical. A static point at
depth Z_t, while the homography fits a plane at Z_g, has leftover motion of about

    parallax ~ f * T_perp * (1/Z_t - 1/Z_g)

which with a steadily translating camera accumulates **linearly and in a consistent
direction** -- indistinguishable from a translating object by magnitude alone. Measured:
clutter reached z = 62 against the drone's 15-frame ceiling of ~14.

So magnitude is demoted to a recall gate and the decision moves to two tests a static
point must pass and a target must fail.

Both are pure functions of a track's accumulated geometry. Nothing here reads a frame, a
file or a config, so `selftest.py` can check them against synthetic scenes where the right
answer is known by construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DirectionVerdict:
    """Step 2. `moving` is only meaningful when `usable` -- see `reason`."""
    moving: bool
    usable: bool
    sin_angle: float          # |sin| between the track's displacement and the line to e
    reason: str

    @property
    def static_like(self) -> bool:
        return self.usable and not self.moving


def epipolar_direction(displacement: np.ndarray, position: np.ndarray,
                       epipole: np.ndarray, *,
                       epipole_reliable: bool,
                       degenerate: bool,
                       sin_threshold: float = 0.35,
                       min_displacement: float = 2.0) -> DirectionVerdict:
    """Is this track's motion along the line to the epipole, as static parallax must be?

    For any static point the leftover displacement after plane compensation satisfies
    `mu = gamma * (e - x)`, so it is **collinear with the line joining x and e** whatever
    its depth. An independently moving object has no such constraint.

    Returns `moving` when the perpendicular component is large enough that parallax cannot
    explain it. `sin_threshold` is |sin(angle)|, so 0.35 is about 20 degrees off the
    epipolar line.

    Three ways this refuses to answer, and each is a real situation rather than an edge
    case to paper over:
      * the pair is rotation-dominated, so there is no epipole to be collinear with;
      * the track sits near the epipole, where every direction is nearly radial -- which
        is where an intercept target sits, so this is the project's own case;
      * the track has barely moved, so its direction is noise.
    """
    d = np.asarray(displacement, float).reshape(2)
    x = np.asarray(position, float).reshape(2)
    e = np.asarray(epipole, float).reshape(2)

    if not epipole_reliable:
        return DirectionVerdict(False, False, float("nan"), "epipole unreliable (rotation?)")
    if degenerate:
        return DirectionVerdict(False, False, float("nan"), "near the epipole (FOE)")
    nd = float(np.linalg.norm(d))
    if nd < min_displacement:
        return DirectionVerdict(False, False, float("nan"), "displacement too small")
    v = e - x
    nv = float(np.linalg.norm(v))
    if nv < 1e-6:
        return DirectionVerdict(False, False, float("nan"), "on the epipole")

    cross = abs(d[0] * v[1] - d[1] * v[0]) / (nd * nv)
    cross = float(min(1.0, cross))
    return DirectionVerdict(cross >= sin_threshold, True, cross,
                            "off-epipolar" if cross >= sin_threshold else "epipolar")


@dataclass(frozen=True)
class StructureVerdict:
    """Step 3. `moving` is only meaningful when `usable`."""
    moving: bool
    usable: bool
    drift_sigma: float        # second-half deviation, in units of first-half scatter
    gamma_cv: float           # coefficient of variation of gamma over the whole window
    reason: str


def structure_consistency(gammas: np.ndarray, *,
                          min_steps: int = 6,
                          drift_threshold: float = 3.0,
                          floor: float = 1e-3) -> StructureVerdict:
    """Is the whole track explained by ONE depth, as a static point's must be?

    `gammas[j]` is the track's relative projective structure at step j -- the along-epipole
    component of its residual divided by its distance to that pair's epipole, normalised by
    the same quantity for the background (`geometry.background_gain`) so that pairs with
    different baselines are comparable.

    For a static point this is one number repeated, up to noise: its depth relative to the
    plane does not change. For a moving object it drifts, because the residual is its own
    motion and that is not tied to any fixed depth.

    Fit a constant on the first half, then ask how far the second half departs from it in
    units of the first half's own scatter. Fitting and testing on disjoint halves is what
    makes this independent of the magnitude gate -- a large but *consistent* gamma is
    static, and a small but drifting one is not.

    This test does NOT degenerate near the epipole the way the direction test does, which
    is why both are run.
    """
    g = np.asarray(gammas, float)
    g = g[np.isfinite(g)]
    if len(g) < min_steps:
        return StructureVerdict(False, False, float("nan"), float("nan"),
                                f"only {len(g)} usable steps, need {min_steps}")

    half = len(g) // 2
    a, b = g[:half], g[half:]
    mu_a = float(np.mean(a))
    sd_a = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
    scale = max(sd_a, floor * max(abs(mu_a), floor))
    drift = float(abs(np.mean(b) - mu_a) / scale)

    denom = max(abs(float(np.mean(g))), floor)
    cv = float(np.std(g) / denom)
    return StructureVerdict(drift >= drift_threshold, True, drift, cv,
                            "structure drifts" if drift >= drift_threshold else "one depth fits")


def gamma_step(mu: np.ndarray, x: np.ndarray, epipole: np.ndarray,
               gain: float, *, min_radius: float = 10.0) -> float:
    """One step's relative projective structure, normalised by the frame's background gain.

    Returns NaN where it cannot be formed -- too near the epipole, or a pair with no
    measurable parallax at all. NaN rather than 0.0 on purpose: `structure_consistency`
    drops non-finite steps, and a zero would be read as a real measurement of "no depth".
    """
    v = np.asarray(epipole, float).reshape(2) - np.asarray(x, float).reshape(2)
    r = float(np.linalg.norm(v))
    if r < min_radius or gain <= 0:
        return float("nan")
    along = float(np.dot(np.asarray(mu, float).reshape(2), v / r))
    return along / (r * gain)


def confirm(direction: DirectionVerdict, structure: StructureVerdict, *,
            require_both: bool = False) -> tuple[bool, str]:
    """Combine the two tests into one decision, and say which test decided.

    Default is OR over whichever tests could answer, because the two fail in different
    places: direction dies near the focus of expansion, structure needs enough steps. A
    track near the epipole is therefore still judgeable, on structure alone.

    `require_both` is the strict arm, for the false-alarm end of the operating curve.
    If neither test could answer, the answer is no -- an unjudgeable track is not a
    detection.
    """
    usable = [v for v in (direction, structure) if v.usable]
    if not usable:
        return False, "no test could answer"
    votes = [v.moving for v in usable]
    decided = all(votes) if (require_both and len(usable) == 2) else any(votes)
    who = []
    if direction.usable:
        who.append(f"direction({direction.sin_angle:.2f})")
    if structure.usable:
        who.append(f"structure({structure.drift_sigma:.1f}sig)")
    return decided, " ".join(who)
