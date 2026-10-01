"""EXP-016 comparison arm: the cheap multi-frame baseline, a long-baseline difference.

Frame n against frame n-k through chained homographies, k = 3 and 6, scored by the same
`win_b5_e4` map and the same drone-rank criterion EXP-015 used. This is what the
hypothesis test has to beat: if simply widening the baseline recovers the long-range drone,
none of the tracking machinery is worth building.

`common.homography(prev, cur)` returns H mapping *current*-frame coordinates to
*previous*-frame coordinates, so the composite from n back to n-k is
H_{n-k+1} @ ... @ H_{n-1} @ H_n, and `warp(frame_{n-k}, composite)` lands it on frame n.
"""
import pickle
from collections import deque

import cv2
import numpy as np

import common
import mf
from mf import CLIP, OUT

KS = (1, 3, 6)


def main():
    col = pickle.load(open(OUT + "collect.pkl", "rb"))
    fixed = mf.screen_fixed()
    buf = deque(maxlen=max(KS) + 1)
    cap = cv2.VideoCapture(CLIP["video"])
    n = 0
    out = {k: {} for k in KS}
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        n += 1
        buf.append((n, fr))
        if n not in mf.BOXES:
            continue
        for k in KS:
            if len(buf) <= k or (n - k) != buf[-(k + 1)][0]:
                continue
            chain = np.eye(3)
            bad = False
            for j in range(n - k + 1, n + 1):
                if j not in col:
                    bad = True
                    break
                chain = chain @ col[j]["H"].astype(np.float64)
            if bad:
                continue
            wm, _ = mf.win_map(buf[-(k + 1)][1], fr, chain)
            valid = common.valid_mask(chain, fr.shape[:2]) & ~fixed
            pk = common.peaks(wm, valid, top=mf.SEEDS_PER_FRAME)
            x0, y0, x1, y1 = common.drone_region(mf.BOXES[n])
            on = (pk[:, 1] >= x0) & (pk[:, 1] <= x1) & (pk[:, 2] >= y0) & (pk[:, 2] <= y1)
            out[k][n] = (1 + int((pk[~on, 0] > pk[on, 0].max()).sum())) if on.any() else np.inf
        if n % 100 == 0:
            print(n, flush=True)
    cap.release()
    print(f"\n{'k':>3} {'stretch':>12} {'n':>4} {'seen':>6} {'top1':>6} {'top5':>6} "
          f"{'top10':>6} {'medrank':>8}")
    for k in KS:
        for lo, hi in CLIP["stretches"]:
            fs = [f for f in out[k] if lo <= f <= hi]
            if not fs:
                continue
            r = np.array([out[k][f] for f in fs])
            print(f"{k:3d} {f'{lo}-{hi}':>12} {len(fs):4d} {np.mean(np.isfinite(r)):6.2f} "
                  f"{np.mean(r <= 1):6.2f} {np.mean(r <= 5):6.2f} {np.mean(r <= 10):6.2f} "
                  f"{np.median(r):8.0f}")
    with open(OUT + "longbase.pkl", "wb") as fh:
        pickle.dump(out, fh)


if __name__ == "__main__":
    main()
