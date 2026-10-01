"""EXP-016: rebuild EXP-015's screen-fixed clutter map, so the analog clip can have one.

EXP-015 kept `screen_fixed_frac.npy` but not the script that made it. The rule is recorded
in its `score.py`: *where the plain difference's top-50 peaks land in >= 5% of the clip's
empty frames*. Terrain sweeps across the screen as the host manoeuvres; the host's own prop
blades, the pitch ladder and the burned-in digits do not.

Reimplemented here with a 7 px stamp, which reproduces the 14-15 px blobs in EXP-015's
file. Running it on O4 prints the agreement with that file; the O4 pipeline keeps using
EXP-015's original so the two experiments stay directly comparable, and only analog uses
the output of this script.

Calibrated on empty frames only -- in-sample for the overlay layout, out-of-sample for the
drone, which never appears in one.
"""
import sys
import time

import cv2
import numpy as np

import common
import mf
from mf import CLIP, OUT

TOP = 50
STAMP = max(3, int(round(mf.px(7.0))))
FRAC = 0.05


def main():
    want = set(mf.EMPTY_FRAMES)
    acc = np.zeros((mf.H_PX, mf.W), np.float32)
    cap = cv2.VideoCapture(CLIP["video"])
    prev, n, used = None, 0, 0
    t0 = time.time()
    disc = cv2.circle(np.zeros((2 * STAMP + 1,) * 2, np.uint8), (STAMP, STAMP), STAMP, 1, -1)
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        n += 1
        if prev is not None and n in want:
            p, c = common.prep(prev, 11), common.prep(fr, 11)
            Hm = common.homography(p, c)
            valid = common.valid_mask(Hm, fr.shape[:2])
            d = cv2.absdiff(c, common.warp(p, Hm)).astype(np.float32)
            pk = common.peaks(d, valid, top=TOP)
            stamp = np.zeros_like(acc)
            for _, x, y in pk:
                x0, y0 = int(x) - STAMP, int(y) - STAMP
                xs, ys = slice(max(x0, 0), min(x0 + 2 * STAMP + 1, mf.W)), \
                    slice(max(y0, 0), min(y0 + 2 * STAMP + 1, mf.H_PX))
                sub = disc[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]
                stamp[ys, xs] = np.maximum(stamp[ys, xs], sub)
            acc += stamp
            used += 1
            if used % 100 == 0:
                print(f"{n}  {used} empty frames  {time.time() - t0:.0f}s", flush=True)
        prev = fr
    cap.release()
    acc /= max(used, 1)
    path = OUT + "screen_fixed_frac.npy"
    np.save(path, acc)
    mask = acc >= FRAC
    print(f"{used} empty frames -> {path}; mask covers {mask.mean():.4f} of the frame")
    ref = CLIP.get("screen_fixed")
    if ref and "exp015" in ref:
        r = np.load(ref) >= FRAC
        inter = (mask & r).sum()
        print(f"vs EXP-015: theirs {r.mean():.4f}, ours {mask.mean():.4f}, "
              f"IoU {inter / max((mask | r).sum(), 1):.3f}")


if __name__ == "__main__":
    main()
