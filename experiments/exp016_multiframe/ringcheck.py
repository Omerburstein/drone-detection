"""EXP-016: re-derive the local-ring measurement the plan rests on, from the labels.

The design pass claimed the drone moves ~5.2 px/frame against a ring whose scatter is
0.54 px -- about 9 sigma -- over `first_catch` 708-800. That is the whole premise, so it
is re-derived here rather than trusted: take the label box centre in consecutive frames,
subtract the homography's prediction for static background at that point, subtract the
median residual of the grid tracks in the 25-120 px ring, and compare what is left with
the ring's own scatter.

`hand` restricts both frames to hand-placed labels, which is the honest version -- a
follower box carries the follower's own error into the displacement.
"""
import pickle

import numpy as np

import mf
from mf import CLIP, OUT

col = pickle.load(open(OUT + "collect.pkl", "rb"))


def centre(f):
    x, y, w, h = mf.BOXES[f]
    return np.array([x + w / 2, y + h / 2], np.float32)


def rows(pairs):
    out = []
    for f0, f1 in pairs:
        rec = col.get(f1)
        if rec is None:
            continue
        flow = mf.Flow(rec["H"].astype(np.float64), mf.GRID_PTS, rec["resid"], rec["raw"],
                       rec["ok"], mf.GRID_ROWS, mf.GRID_COLS, mf.GRID_ORIGIN)
        p0, p1 = centre(f0), centre(f1)
        pred = mf.predict(flow.Hm, p0.reshape(1, 2))[0]
        med, sigma, ring_raw, n = flow.ring(float(p0[0]), float(p0[1]))
        if n < mf.MIN_RING_PTS:
            continue
        d = (p1 - pred) - med
        out.append((f1, float(np.linalg.norm(p1 - p0)), float(np.linalg.norm(p1 - pred)),
                    float(np.linalg.norm(d)), sigma, ring_raw, n,
                    max(mf.BOXES[f1][2:])))
    return np.array(out, np.float64)


print(f"{CLIP['name']}: ring {mf.ANNULUS[0]:.0f}-{mf.ANNULUS[1]:.0f} px, grid {mf.GRID} px\n")
hand = sorted(f for f in mf.DRONE_FRAMES if mf.SOURCE.get(f) == "human")
for what, pairs in (
        ("consecutive labels (follower boxes included)",
         [(f, f + 1) for f in mf.DRONE_FRAMES if f + 1 in mf.BOXES]),
        ("hand-placed labels one frame apart",
         [(a, b) for a, b in zip(hand, hand[1:]) if b == a + 1]),
        ("hand-placed labels <= 3 frames apart (per-frame rate)",
         [(a, b) for a, b in zip(hand, hand[1:]) if b - a <= 3])):
    r = rows(pairs)
    if not len(r):
        print(f"{what}: no usable pairs\n")
        continue
    print(what)
    print(f"{'stretch':>12} {'n':>4} {'raw':>7} {'vs H':>7} {'vs ring':>8} "
          f"{'sigma':>7} {'z1':>7} {'ringpts':>8}")
    for lo, hi in CLIP["stretches"]:
        m = (r[:, 0] >= lo) & (r[:, 0] <= hi)
        if not m.any():
            continue
        s = r[m]
        gap = np.array([1.0] * len(s))
        if "<= 3" in what:
            gap = np.array([b - a for a, b in pairs if lo <= b <= hi and b - a <= 3],
                           float)[:len(s)]
        z1 = s[:, 3] / np.maximum(s[:, 4] + mf.NOISE_FLOOR, 1e-6)
        print(f"{f'{lo}-{hi}':>12} {len(s):4d} {np.median(s[:, 1] / gap):7.2f} "
              f"{np.median(s[:, 2] / gap):7.2f} {np.median(s[:, 3] / gap):8.2f} "
              f"{np.median(s[:, 4]):7.2f} {np.median(z1):7.1f} {np.median(s[:, 6]):8.0f}")
    print("  raw = image motion, vs H = after the homography, vs ring = after the ring median")
    print("  sigma = 1.48 x ring MAD (px), z1 = single-frame differential / (sigma + floor)\n")
