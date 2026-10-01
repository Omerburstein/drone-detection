"""Step 1 on real footage: calibrate the threshold budget, and check that masking now pays.

    PYTHONPATH="experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
        py -3.13 -m calibrate_budget [--frames 60] [--target 200]

Two things happen here, and the second is the point.

1. The threshold is fitted on **empty frames only**, two-fold, and frozen at the more
   conservative fold -- the same discipline EXP-016 used for z*.

2. The same empty frames are then counted under three mask sets. Under EXP-016's top-N
   budget, adding masks moved the share of candidates in the real scene from 40.2% to
   41.2% and every frame still returned all 200 peaks. If the threshold budget is doing
   what it is supposed to, the **count itself** should now fall as masks are added, which
   is what makes stage-0 work worth doing at all.

A null result here kills step 1 cheaply, before either geometric test is run on video.
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

import budget
import common
import masks
from clipcfg import CLIP

W, H = CLIP["width"], CLIP["height"]


def empty_frames(limit: int) -> list[int]:
    """Frames with no labelled target, spread evenly through the clip."""
    labelled: set[int] = set()
    p = CLIP.get("labels")
    if p and os.path.exists(p):
        labelled = {int(k) for k, v in json.load(open(p))["frames"].items()
                    if v and v.get("box")}
    pool = [f for f in range(2, CLIP["n_frames"] + 1) if f not in labelled]
    if not pool:
        raise SystemExit("no empty frames in this clip")
    idx = np.linspace(0, len(pool) - 1, min(limit, len(pool))).astype(int)
    return [pool[i] for i in idx]


def score_maps(frames: list[int]):
    """(win_b5_e4 map, homography) per frame pair. The map is EXP-015's, unchanged."""
    cap = cv2.VideoCapture(CLIP["video"])
    out = []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f - 2)
        ok, a = cap.read()
        ok2, b = cap.read()
        if not (ok and ok2):
            continue
        hmat = common.homography(common.prep(a, 11), common.prep(b, 11))
        out.append((common.maps(a, b, hmat)["win_b5_e4"], hmat))
    cap.release()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--target", type=float, default=200.0,
                    help="candidates per empty frame at the operating point")
    a = ap.parse_args()

    frames = empty_frames(a.frames)
    print(f"[budget] {len(frames)} empty frames from {CLIP['name']}")
    pairs = score_maps(frames)
    maps_ = [m for m, _ in pairs]
    print(f"[budget] {len(pairs)} usable frame pairs")

    stage0 = masks.Stage0()
    cov = stage0.coverage()
    print("[stage 0] " + ", ".join(f"{k} {v:.2f}%" for k, v in cov.items()))

    # Three mask sets, weakest first. Each adds to the one before it.
    hud_only = ~(stage0.layers["hud"] | stage0.layers["edge"])
    full = ~stage0.static
    sets = {
        "edge + HUD only (EXP-015 baseline)": hud_only,
        "+ prop + ladder (EXP-017 structural)": full,
    }

    # Calibrate on the weakest set, so the threshold is NOT chosen to flatter the masks.
    valids = [hud_only & ~masks.warp_border(h) for _, h in pairs]
    b = budget.calibrate(np.asarray(maps_, dtype=object), np.asarray(valids, dtype=object),
                         target_per_frame=a.target)
    print(f"\n[budget] {b.describe()}")

    print("\ncandidates per empty frame at that ONE frozen threshold:")
    base = None
    for name, m in sets.items():
        v = [m & ~masks.warp_border(h) for _, h in pairs]
        n = [len(budget.candidates(mp, vv, b.tau)) for mp, vv in zip(maps_, v)]
        mean = float(np.mean(n))
        base = base if base is not None else mean
        print(f"  {name:38s} {mean:7.1f}   ({mean / base * 100:5.1f}% of baseline)")

    print("\nFor comparison, the same mask sets under a top-200 budget:")
    for name, m in sets.items():
        v = [m & ~masks.warp_border(h) for _, h in pairs]
        n = [len(np.asarray(common.peaks(mp, vv, top=200), float))
             for mp, vv in zip(maps_, v)]
        print(f"  {name:38s} {float(np.mean(n)):7.1f}")


if __name__ == "__main__":
    main()
