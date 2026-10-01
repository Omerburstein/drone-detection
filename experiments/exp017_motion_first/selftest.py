"""Check the stage-2 geometry against synthetic scenes where the answer is known.

    PYTHONPATH="experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
        py -3.13 -m selftest

No video, no weights, no labels. A 3D scene is built, a camera is flown through it, points
are projected exactly, and the two tests are asked to separate static points from a moving
one. If the maths in `geometry.py` or `criteria.py` is wrong, it is wrong here first and
cheaply, rather than after a two-hour run on a clip where a null result is
indistinguishable from a bug.

This is the check EXP-012a did not have. Its retracted claim was about what was being
compared with what, and it survived long enough to reach the ledger.

It has already earned its place: the first version of the structure test passed a static
point and **also** passed a moving one, because the synthetic flew the camera in a
straight line at constant speed. That is a real degeneracy of plane+parallax, not a bug,
and it is now a named test (`5.`) rather than a surprise on real footage.
"""
from __future__ import annotations

import sys

import cv2
import numpy as np

import criteria
import geometry

RNG = np.random.default_rng(17)
F = 900.0                      # focal length, px
C = np.array([720.0, 540.0])   # principal point
PLANE_Z = 40.0


def build_scene(n_plane: int = 300, n_off: int = 160):
    """One fixed scene, reused across a window: a dominant ground plane plus other depths."""
    plane = np.column_stack([RNG.uniform(-30, 30, n_plane),
                             RNG.uniform(-20, 20, n_plane),
                             np.full(n_plane, PLANE_Z)])
    off = np.column_stack([RNG.uniform(-30, 30, n_off),
                           RNG.uniform(-20, 20, n_off),
                           RNG.uniform(12.0, 120.0, n_off)])
    return plane, off


SCENE = build_scene()


def project(pts3: np.ndarray, cam_t: np.ndarray) -> np.ndarray:
    p = pts3 - cam_t
    return C + F * p[:, :2] / p[:, 2:3]


def fit_plane_homography(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """H mapping CURRENT -> PREVIOUS, fitted on the plane points, as RANSAC would find."""
    h, _ = cv2.findHomography(b.astype(np.float32), a.astype(np.float32), cv2.RANSAC, 1.0)
    return h


def run_pair(t_prev, t_cur, obj_prev, obj_cur, noise: float = 0.0):
    """One frame pair over the fixed scene. Returns epipole, background field, object geometry."""
    plane, off = SCENE
    static = np.vstack([plane, off])
    a_s, b_s = project(static, t_prev), project(static, t_cur)
    a_p, b_p = project(plane, t_prev), project(plane, t_cur)
    if noise:
        a_s = a_s + RNG.normal(0, noise, a_s.shape)
        b_s = b_s + RNG.normal(0, noise, b_s.shape)

    hmat = fit_plane_homography(a_p, b_p)
    x, mu = geometry.residual_field(a_s, b_s, hmat)
    ep = geometry.estimate_epipole(x, mu)
    xo, muo = geometry.residual_field(project(obj_prev[None, :], t_prev),
                                      project(obj_cur[None, :], t_cur), hmat)
    return ep, x, mu, xo[0], muo[0], geometry.background_gain(x, mu, ep)


def gamma_window(path, obj0, vel, noise: float = 0.05) -> list[float]:
    """Relative structure per step, for an object at obj0 moving at `vel` along `path`."""
    g, pos = [], np.asarray(obj0, float)
    for j in range(len(path) - 1):
        nxt = pos + vel
        ep, _, _, xo, muo, gain = run_pair(path[j], path[j + 1], pos, nxt, noise)
        if ep.reliable:
            g.append(criteria.gamma_step(muo, xo, ep.point, gain))
        pos = nxt
    return g


def straight_path(n: int = 13):
    """Constant velocity in a straight line -- the plane+parallax degenerate case."""
    return [np.array([0.05 * j, 0.0, 0.20 * j]) for j in range(n)]


def banking_path(n: int = 13):
    """A turning, climbing flight: the translation direction changes across the window.

    This is the realistic case. An FPV airframe on an intercept is not on a rail, and the
    epipole moves across the frame as it manoeuvres -- which is what gives the structure
    test something to bite on.
    """
    out = []
    for j in range(n):
        a = 0.18 * j
        out.append(np.array([1.2 * np.sin(a), 0.25 * a, 0.22 * j]))
    return out


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def main() -> int:
    print("EXP-017 stage-2 self-test (synthetic geometry, no video)\n")
    ok_all = True

    # --- 1. the epipole is recoverable from the residual field alone -----------
    print("1. epipole estimation")
    t_prev, t_cur = np.array([0.0, 0.0, 0.0]), np.array([0.6, 0.0, 2.0])
    still = np.array([3.0, 1.0, 60.0])
    ep, x, mu, xm, mum, gain = run_pair(t_prev, t_cur, still, still)
    truth = C + F * np.array([0.6, 0.0]) / 2.0
    err = float(np.linalg.norm(ep.point - truth))
    ok_all &= check("epipole is reliable", ep.reliable,
                    f"anisotropy {ep.anisotropy:.2f}, {ep.n_points} pts")
    ok_all &= check("epipole within 25 px of truth", err < 25.0,
                    f"error {err:.1f} px, truth {truth.round(1)}")

    # --- 2. static points read as epipolar, a moving one does not -------------
    print("\n2. direction test (step 2)")
    strong = np.linalg.norm(mu, axis=1) > 1.0
    sins = []
    for xi, mi in zip(x[strong], mu[strong]):
        v = criteria.epipolar_direction(mi, xi, ep.point, epipole_reliable=True,
                                        degenerate=ep.degenerate_for(*xi),
                                        min_displacement=1.0)
        if v.usable:
            sins.append(v.sin_angle)
    frac = float(np.mean(np.asarray(sins) < 0.35)) if sins else 0.0
    ok_all &= check("static parallax reads epipolar in >90% of points", frac > 0.90,
                    f"{frac * 100:.1f}% of {len(sins)} points")

    cross = still + np.array([0.9, 0.5, 0.0])
    _, _, _, xm2, mum2, _ = run_pair(t_prev, t_cur, still, cross)
    vm = criteria.epipolar_direction(mum2, xm2, ep.point, epipole_reliable=True,
                                     degenerate=ep.degenerate_for(*xm2))
    ok_all &= check("a crossing target reads as moving", vm.moving,
                    f"sin {vm.sin_angle:.2f} ({vm.reason})")

    # --- 3. the refusals actually refuse --------------------------------------
    print("\n3. refusals (the honesty guards)")
    v_rot = criteria.epipolar_direction(np.array([5.0, 5.0]), np.array([100.0, 100.0]),
                                        ep.point, epipole_reliable=False, degenerate=False)
    ok_all &= check("rotation-dominated pair is refused, not scored", not v_rot.usable,
                    v_rot.reason)
    v_foe = criteria.epipolar_direction(np.array([5.0, 5.0]), ep.point + 5.0,
                                        ep.point, epipole_reliable=True, degenerate=True)
    ok_all &= check("a track at the FOE is refused, not scored", not v_foe.usable,
                    v_foe.reason)
    zero = np.zeros(3)
    ep_rot, *_ = run_pair(zero, zero + 1e-6, still, still, noise=0.15)
    ok_all &= check("no translation => epipole reports itself unreliable", not ep_rot.reliable,
                    f"anisotropy {ep_rot.anisotropy:.2f}")

    # --- 4. structure consistency over a manoeuvring window (step 3) ----------
    print("\n4. structure test (step 3), banking flight")
    path = banking_path()
    vel = np.array([0.22, 0.13, -0.30])
    g_static = gamma_window(path, still, np.zeros(3))
    g_moving = gamma_window(path, still, vel)
    v_s = criteria.structure_consistency(np.asarray(g_static))
    v_m = criteria.structure_consistency(np.asarray(g_moving))
    ok_all &= check("a static point fits one depth", v_s.usable and not v_s.moving,
                    f"drift {v_s.drift_sigma:.1f} sigma, cv {v_s.gamma_cv:.2f}")
    ok_all &= check("a moving point does not", v_m.usable and v_m.moving,
                    f"drift {v_m.drift_sigma:.1f} sigma, cv {v_m.gamma_cv:.2f}")

    # --- 5. the named degeneracy, asserted rather than discovered later -------
    print("\n5. the known plane+parallax degeneracy (straight flight, constant velocity)")
    sp = straight_path()
    v_sm = criteria.structure_consistency(np.asarray(gamma_window(sp, still, vel)))
    ok_all &= check("structure test CANNOT see a collinear constant-velocity target",
                    v_sm.usable and not v_sm.moving,
                    f"drift {v_sm.drift_sigma:.1f} sigma -- direction test must carry this case")

    # --- 6. combining -----------------------------------------------------------
    print("\n6. confirm()")
    decided, who = criteria.confirm(v_foe, v_m)
    ok_all &= check("decides at the FOE on structure alone", decided, who)
    decided2, who2 = criteria.confirm(v_rot, criteria.structure_consistency(
        np.asarray(g_static[:3])))
    ok_all &= check("says no when neither test can answer", not decided2, who2)

    print("\n" + ("ALL PASS" if ok_all else "FAILURES ABOVE"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
