"""EXP-016: what the missing moving-overlay veto is worth, measured rather than asserted.

On O4 the burned-in pitch ladder moves with pitch, so the static screen-fixed mask cannot
hold it, and `--osd-twins` (EXP-014) was never applied to this clip. This applies that test
**post hoc** to every confirmed track in every ladder pass -- false tracks and on-drone
tracks alike -- and reports the false-track rate with and without it, plus what it would
have cost the drone. The plan allows a dominant moving-overlay source to be reported as a
missing prerequisite rather than a detector failure; this is the number that decides which.
"""
import argparse
import glob
import pickle
import re

import cv2
import numpy as np

import mf
from mf import CLIP, OUT
from src.algo.masking import TWIN_SCORE, twin_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=TWIN_SCORE)
    a = ap.parse_args()
    runs = {}
    for path in [p for p in glob.glob(OUT + "verify_z*.pkl") if re.fullmatch(r"verify_z[0-9.]+.pkl", p.replace("\\","/").split("/")[-1])]:
        z = float(re.search(r"verify_z([0-9.]+)\.pkl", path).group(1))
        runs[z] = pickle.load(open(path, "rb"))["tracks"]
    if not runs:
        print("no ladder passes found")
        return
    need = {}
    for z, tracks in runs.items():
        for hid, t in tracks.items():
            need.setdefault(int(t["confirmed"]), {})[hid] = t

    scores = {}
    cap = cv2.VideoCapture(CLIP["video"])
    n = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        n += 1
        if n not in need:
            continue
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        for hid, t in need[n].items():
            r = t["trail"][0]
            s = float(t["size"])
            scores[(n, hid)] = twin_score(gray, (r[1] - s / 2, r[2] - s / 2, s, s))
    cap.release()

    minutes = len(mf.EMPTY_FRAMES) / CLIP["fps"] / 60.0
    print(f"{CLIP['name']}: post-hoc OSD-twin veto at {a.threshold:.2f}, "
          f"{len(mf.EMPTY_FRAMES)} empty frames = {minutes * 60:.1f} s\n")
    print(f"{'z*':>5} {'FA':>5} {'FA/min':>8} {'FA kept':>8} {'kept/min':>9} "
          f"{'drone tracks':>13} {'drone kept':>11}")
    for z in sorted(runs):
        fa = [(int(t['confirmed']), hid) for hid, t in runs[z].items()
              if int(t["confirmed"]) not in mf.BOXES]
        dr = []
        for hid, t in runs[z].items():
            if any(mf.on_drone(float(x), float(y), int(m)) for m, x, y in t["trail"]):
                dr.append((int(t["confirmed"]), hid))
        kept = [k for k in fa if scores.get(k, 0.0) < a.threshold]
        dkept = [k for k in dr if scores.get(k, 0.0) < a.threshold]
        print(f"{z:5g} {len(fa):5d} {len(fa) / minutes:8.1f} {len(kept):8d} "
              f"{len(kept) / minutes:9.1f} {len(dr):13d} {len(dkept):11d}")
    ds = [scores[k] for z in runs for hid, t in runs[z].items()
          if any(mf.on_drone(float(x), float(y), int(m)) for m, x, y in t["trail"])
          for k in [(int(t["confirmed"]), hid)] if k in scores]
    if ds:
        print(f"\ntwin score of on-drone confirmed tracks: median {np.median(ds):.2f}  "
              f"p90 {np.percentile(ds, 90):.2f}  max {max(ds):.2f}  (veto at {a.threshold:.2f})")


if __name__ == "__main__":
    main()
