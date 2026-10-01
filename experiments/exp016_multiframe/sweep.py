"""EXP-016: the operating curve -- what each z* on the ladder buys and what it costs.

The pass bar is a single point, but one point cannot say whether the method is weak or
merely mis-tuned. This reads every `verify_z*.pkl` the ladder produced and tabulates, for
each threshold, the false-track rate on the empty frames against the acquisition latency,
the coverage and the rank on the primary span. **The z* column is not a free parameter to
pick from afterwards** -- the frozen value came from `calibrate.py` on empty frames alone,
and this table exists to show the shape of the trade, not to choose a better point.
"""
import argparse
import glob
import pickle
import re

import numpy as np

import mf
from mf import CLIP, OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--span", type=int, nargs=2, default=None,
                    help="the primary span; defaults to the first labelled stretch")
    a = ap.parse_args()
    lo, hi = a.span or CLIP["stretches"][0]
    minutes = len(mf.EMPTY_FRAMES) / CLIP["fps"] / 60.0
    print(f"{CLIP['name']}: primary span {lo}-{hi}, "
          f"{len(mf.EMPTY_FRAMES)} empty frames = {minutes * 60:.1f} s\n")
    print(f"{'z*':>5} {'FA/min':>8} {'acq':>7} {'cover':>7} {'frags':>6} "
          f"{'top1':>6} {'top5':>6} {'top10':>6} {'tracks':>7}")
    rows = []
    for path in [p for p in glob.glob(OUT + "verify_z*.pkl") if re.fullmatch(r"verify_z[0-9.]+.pkl", p.replace("\\","/").split("/")[-1])]:
        z = float(re.search(r"verify_z([0-9.]+)\.pkl", path).group(1))
        res = pickle.load(open(path, "rb"))
        tracks, ranks = res["tracks"], res["ranks"]
        fa = sum(1 for t in tracks.values() if int(t["confirmed"]) not in mf.BOXES)
        hits, frags = set(), set()
        for hid, t in tracks.items():
            for n, x, y in t["trail"]:
                n = int(n)
                if lo <= n <= hi and mf.on_drone(float(x), float(y), n):
                    hits.add(n)
                    frags.add(hid)
        fs = [f for f in mf.DRONE_FRAMES if lo <= f <= hi]
        r = np.array([ranks[f][0] for f in fs if f in ranks]) if fs else np.array([])
        seen = len(r) / max(len(fs), 1)
        acq = (min(hits) - lo) if hits else np.inf
        rows.append((z, fa / minutes, acq, len(hits) / max(len(fs), 1), len(frags),
                     np.mean(r <= 1) * seen if len(r) else 0.0,
                     np.mean(r <= 5) * seen if len(r) else 0.0,
                     np.mean(r <= 10) * seen if len(r) else 0.0, len(tracks)))
    for z, fa, acq, cov, fr, t1, t5, t10, nt in sorted(rows):
        print(f"{z:5g} {fa:8.1f} {acq if np.isfinite(acq) else -1:7.0f} {cov:7.2f} "
              f"{fr:6d} {t1:6.2f} {t5:6.2f} {t10:6.2f} {nt:7d}")
    print("\n  FA/min = confirmed tracks whose confirming frame carries no label")
    print("  acq    = frames from the span's start to the first confirmed on-drone track "
          "(-1 = never)")
    print("  cover  = labelled frames in the span with a confirmed track on the drone")
    print("  top-k  = best on-drone hypothesis in the top k by z, over all live hypotheses")


if __name__ == "__main__":
    main()
