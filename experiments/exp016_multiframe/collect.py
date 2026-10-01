"""EXP-016 collect: per consecutive frame pair, the homography, 200 seeds, and the grid flow.

EXP-015's `peaks.pkl` samples every second empty frame, so a tracker cannot walk it. This
pass is every frame, consecutively, which is the whole point.

Written per frame n (meaning the pair n-1 -> n):
  * `H`      the GLAD homography, current-frame coords -> previous-frame coords
  * `seeds`  the top 200 `win_b5_e4` peaks inside the valid mask: (value, x, y, size_px)
  * `resid`  (rows, cols, 2) grid residual: tracked motion minus the homography prediction
  * `raw`    (rows, cols)    uncompensated grid motion, for the screen-fixed veto
  * `ok`     (rows, cols)    forward-backward and mask acceptance

Run from the repo root:
    PYTHONPATH="<clip dir>;experiments/exp016_multiframe;experiments/exp015_normalised_motion;." \
    py -3.13 -m collect
"""
import pickle
import time

import cv2
import numpy as np

import common
import mf
from mf import CLIP, OUT

CHECK_FRAME = CLIP["stretches"][0][0] + 20   # one frame where win_map is pinned to EXP-015


def main():
    fixed = mf.screen_fixed()
    cap = cv2.VideoCapture(CLIP["video"])
    prev = None
    n = 0
    out = {}
    fb_fail = []
    t0 = time.time()
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        n += 1
        if prev is not None:
            pb, cb = common.prep(prev, 11), common.prep(fr, 11)
            Hm = common.homography(pb, cb)
            valid = common.valid_mask(Hm, fr.shape[:2]) & ~fixed
            wm, diff = mf.win_map(prev, fr, Hm)
            if n == CHECK_FRAME:
                ref = common.maps(prev, fr, Hm)["win_b5_e4"]
                assert np.array_equal(ref, wm), "win_map drifted from EXP-015's maps()"
                print(f"win_b5_e4 bit-identical to EXP-015 on frame {n}", flush=True)
            pk = common.peaks(wm, valid, top=mf.SEEDS_PER_FRAME)
            sizes = mf.seed_sizes(diff, pk)
            pg, cg = common.prep(prev, 3), common.prep(fr, 3)
            flow, why = mf.background_flow(pg, cg, Hm, valid)
            fb_fail.append(why)
            out[n] = dict(H=Hm.astype(np.float32),
                          seeds=np.column_stack([pk, sizes]).astype(np.float32),
                          resid=flow.resid, raw=flow.raw, ok=flow.ok)
            if n % 100 == 0:
                w = fb_fail[-1]
                print(f"{n}  {time.time() - t0:.0f}s  seeds {len(pk)}  "
                      f"grid fb {w['fb']:.3f} mask {w['mask']:.3f} wild {w['wild']:.3f} "
                      f"ok {w['ok']:.3f}", flush=True)
        prev = fr
    cap.release()
    with open(OUT + "collect.pkl", "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    print(f"done {len(out)} pairs in {time.time() - t0:.0f}s")
    print(f"grid tracks, fraction dropped (mean over pairs)  "
          f"FB-failure {np.mean([w['fb'] for w in fb_fail]):.4f}  "
          f"masked {np.mean([w['mask'] for w in fb_fail]):.4f}  "
          f"wild {np.mean([w['wild'] for w in fb_fail]):.4f}  "
          f"-> usable {np.mean([w['ok'] for w in fb_fail]):.4f}")
    fbm = np.array([w["fb_med"] for w in fb_fail])
    print(f"grid forward-backward error px: median-of-frames {np.median(fbm):.3f}  "
          f"p90 {np.percentile(fbm, 90):.3f}   (FB_MAX = {mf.FB_MAX:.2f} px)")
    with open(OUT + "collect_fb.pkl", "wb") as fh:
        pickle.dump(fb_fail, fh)
    sz = np.concatenate([v["seeds"][:, 3] for v in out.values()])
    print(f"seed size px: median {np.median(sz):.1f}  p10 {np.percentile(sz, 10):.1f}  "
          f"p90 {np.percentile(sz, 90):.1f}")


if __name__ == "__main__":
    main()
