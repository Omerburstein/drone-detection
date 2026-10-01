"""What the direction test does to the real candidate load, per frame pair.

    PYTHONPATH="experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
        py -3.13 -m measure_tests [--tau 1.661] [--frames 40]

This is a **single-pair preview of step 2**, not the full tracker. Each candidate is
tracked one frame with LK, its residual after the plane homography is formed, and the
epipolar-direction test is asked whether that residual could be static parallax. The
structure test needs a window and is not exercised here -- `selftest.py` covers its maths,
and it lands with the tracker.

Two numbers matter and they pull against each other:

  * **rejection rate on empty frames** -- how much of the 150-candidate load the test
    removes for free. This is the whole promise of step 2.
  * **survival on drone frames** -- whether the labelled target is among the survivors. A
    test that removes 90% of clutter and the drone with it is worse than no test.

The `unusable` column is not a failure. It is the test refusing: near the focus of
expansion, or on a rotation-dominated pair. Those candidates pass through unjudged, which
is what the tracker will hand to the structure test.
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

import budget
import common
import criteria
import geometry
import masks
from clipcfg import CLIP

W, H = CLIP["width"], CLIP["height"]
LK = geometry.LK


def labelled_boxes() -> dict[int, list]:
    p = CLIP.get("labels")
    if not p or not os.path.exists(p):
        return {}
    return {int(k): v["box"] for k, v in json.load(open(p))["frames"].items()
            if v and v.get("box")}


def on_drone(x: float, y: float, box) -> bool:
    """EXP-015's match criterion: inside the box grown by max(10 px, 25% of longest side)."""
    bx, by, bw, bh = box
    m = max(10.0, 0.25 * max(bw, bh))
    return (bx - m) <= x <= (bx + bw + m) and (by - m) <= y <= (by + bh + m)


def analyse(a, b, valid_static, tau: float):
    """One frame pair: candidates, their residuals, and the direction verdict for each."""
    ga, gb = common.prep(a, 11), common.prep(b, 11)
    hmat = common.homography(ga, gb)
    valid = valid_static & ~masks.warp_border(hmat)

    gp, gc = geometry.track_grid(ga, gb, valid)
    if len(gp) < geometry.MIN_EPIPOLE_PTS:
        return None
    gx, gmu = geometry.residual_field(gp, gc, hmat)
    ep = geometry.estimate_epipole(gx, gmu)

    cands = budget.candidates(common.maps(a, b, hmat)["win_b5_e4"], valid, tau)
    if not len(cands):
        return None

    pts = cands[:, 1:3].astype(np.float32).reshape(-1, 1, 2)
    fwd, st, _ = cv2.calcOpticalFlowPyrLK(ga, gb, pts, None, **LK)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(gb, ga, fwd, None, **LK)
    good = (st.ravel() == 1) & (st2.ravel() == 1)
    good &= np.linalg.norm((back - pts).reshape(-1, 2), axis=1) < geometry.FB_MAX

    prev = pts.reshape(-1, 2)[good]
    cur = fwd.reshape(-1, 2)[good]
    kept = cands[good]
    if not len(prev):
        return None
    cx, cmu = geometry.residual_field(prev, cur, hmat)

    rows = []
    for pos, res in zip(cx, cmu):
        v = criteria.epipolar_direction(res, pos, ep.point,
                                        epipole_reliable=ep.reliable,
                                        degenerate=ep.degenerate_for(*pos))
        rows.append((pos[0], pos[1], v.usable, v.moving))
    return ep, kept, np.asarray(rows, float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tau", type=float, default=1.661)
    ap.add_argument("--frames", type=int, default=40)
    a = ap.parse_args()

    boxes = labelled_boxes()
    stage0 = masks.Stage0()
    valid_static = ~stage0.static

    pool_empty = [f for f in range(3, CLIP["n_frames"] + 1) if f not in boxes]
    pool_drone = sorted(boxes)
    pick = lambda p, n: [p[i] for i in np.linspace(0, len(p) - 1, min(n, len(p))).astype(int)]

    cap = cv2.VideoCapture(CLIP["video"])

    def sweep(frames, label):
        tot = rej = uns = 0
        n_frames = 0
        drone_seen = drone_kept = 0
        for f in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f - 2)
            ok, p = cap.read()
            ok2, c = cap.read()
            if not (ok and ok2):
                continue
            out = analyse(p, c, valid_static, a.tau)
            if out is None:
                continue
            ep, kept, rows = out
            n_frames += 1
            tot += len(rows)
            usable = rows[:, 2] > 0
            uns += int((~usable).sum())
            rej += int((usable & (rows[:, 3] < 1)).sum())   # usable and NOT moving => cut
            if f in boxes:
                hits = [i for i, r in enumerate(rows) if on_drone(r[0], r[1], boxes[f])]
                if hits:
                    drone_seen += 1
                    if any(rows[i, 2] < 1 or rows[i, 3] > 0 for i in hits):
                        drone_kept += 1
        if not n_frames:
            print(f"  {label}: no usable pairs")
            return
        print(f"  {label}: {n_frames} pairs, {tot / n_frames:6.1f} candidates/frame")
        print(f"      rejected as epipolar (static parallax): {rej / max(tot,1) * 100:5.1f}%")
        print(f"      unjudged (FOE / rotation, passed on):   {uns / max(tot,1) * 100:5.1f}%")
        print(f"      surviving load:                         "
              f"{(tot - rej) / n_frames:6.1f} candidates/frame")
        if drone_seen:
            print(f"      drone present among candidates in {drone_seen} frames, "
                  f"survives in {drone_kept} ({drone_kept / drone_seen * 100:.0f}%)")

    print(f"tau={a.tau}  stage-0 union {stage0.coverage()['union']:.2f}%\n")
    print("EMPTY FRAMES (the false-alarm load)")
    sweep(pick(pool_empty, a.frames), "empty")
    print("\nDRONE FRAMES (does the target survive?)")
    sweep(pick(pool_drone, a.frames), "drone")
    cap.release()


if __name__ == "__main__":
    main()
