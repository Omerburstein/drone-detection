"""EXP-016: what are the confirmed false tracks, so the verdict names a cause not a number.

Each confirmed track that fired on a frame with no label is sorted into a bucket by where
it confirmed: on or beside the burned-in overlay, at the frame edge, in the lower
(ground / near-parallax) half, or in open sky. Also reports how many would have been
caught by the OSD-twin test and by the screen-fixed veto had they been applied harder, and
the distribution of z and seed size, since a cause that is a missing veto is a
prerequisite rather than a detector failure.
"""
import argparse
import pickle

import cv2
import numpy as np

import common
import mf
from mf import CLIP, OUT
from src.algo.masking import twin_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="frozen")
    a = ap.parse_args()
    res = pickle.load(open(OUT + f"verify_{a.tag}.pkl", "rb"))
    fixed = mf.screen_fixed()
    near_overlay = cv2.dilate((fixed | mf.HUD_DIL).astype(np.uint8),
                              np.ones((2 * int(mf.px(30.0)) + 1,) * 2, np.uint8)) > 0
    edge = np.ones((mf.H_PX, mf.W), bool)
    e = int(mf.px(60.0))
    edge[e:-e, e:-e] = False

    false_tracks = [tr for tr in res["tracks"].values()
                    if int(tr["confirmed"]) not in mf.BOXES]
    print(f"{CLIP['name']} {a.tag}: {len(false_tracks)} confirmed false tracks over "
          f"{len(mf.EMPTY_FRAMES)} empty frames "
          f"({len(mf.EMPTY_FRAMES) / CLIP['fps']:.1f} s)")
    if not false_tracks:
        return

    # where they confirmed
    frames = {}
    for tr in false_tracks:
        frames.setdefault(int(tr["confirmed"]), []).append(tr)
    cap = cv2.VideoCapture(CLIP["video"])
    n = 0
    buckets, twins, zs, sizes, ys = {}, [], [], [], []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        n += 1
        if n not in frames:
            continue
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        for tr in frames[n]:
            row = tr["trail"][0]
            x, y = int(round(row[1])), int(round(row[2]))
            x = min(max(x, 0), mf.W - 1)
            y = min(max(y, 0), mf.H_PX - 1)
            b = ("edge" if edge[y, x] else
                 "beside overlay" if near_overlay[y, x] else
                 "lower half (ground)" if y > mf.H_PX * 0.55 else
                 "upper half (sky/horizon)")
            buckets[b] = buckets.get(b, 0) + 1
            s = float(tr["size"])
            twins.append(twin_score(gray, (row[1] - s / 2, row[2] - s / 2, s, s)))
            zs.append(float(tr["z"]))
            sizes.append(s)
            ys.append(y / mf.H_PX)
    cap.release()
    print("\nwhere they confirmed:")
    for k, v in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"  {k:24s} {v:4d}  {100 * v / len(false_tracks):5.1f}%")
    t = np.array(twins)
    print(f"\nOSD-twin score at the confirming frame: median {np.median(t):.2f}, "
          f"{np.mean(t >= 0.7):.0%} at or over the 0.70 veto "
          f"(applied on this clip: {CLIP['osd_twins']})")
    print(f"z at death: median {np.median(zs):.1f} p90 {np.percentile(zs, 90):.1f} "
          f"max {max(zs):.1f}")
    print(f"seed size px: median {np.median(sizes):.0f} p90 {np.percentile(sizes, 90):.0f}; "
          f"speed cap at the median = {mf.speed_cap(float(np.median(sizes))):.0f} px/frame")
    print(f"vertical position (0 = top): median {np.median(ys):.2f}")


if __name__ == "__main__":
    main()
