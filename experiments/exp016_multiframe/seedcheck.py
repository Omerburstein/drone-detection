"""EXP-016 sanity check: does a seed exist on the drone, where is it ranked, and how big does it read?

The plan's design pass measured 92% seed coverage of `win_b5_e4`'s top 200 over
`first_catch` 708-800. This re-derives it from `collect.pkl` -- consecutive frames this
time, and with the seed's estimated size beside it, because the speed cap is only as good
as that estimate.
"""
import pickle

import numpy as np

import common
import mf
from mf import CLIP, OUT

col = pickle.load(open(OUT + "collect.pkl", "rb"))
print(f"{CLIP['name']}: {len(col)} pairs, {mf.SEEDS_PER_FRAME} seeds each")
print(f"\n{'stretch':>12} {'n':>4} {'seed':>6} {'medrank':>8} {'top10':>6} "
      f"{'seedsize':>9} {'labelsize':>10}")
for lo, hi in CLIP["stretches"]:
    fs = [f for f in mf.DRONE_FRAMES if lo <= f <= hi and f in col]
    if not fs:
        continue
    have, ranks, sizes, lab = 0, [], [], []
    for f in fs:
        s = col[f]["seeds"]
        x0, y0, x1, y1 = common.drone_region(mf.BOXES[f])
        on = (s[:, 1] >= x0) & (s[:, 1] <= x1) & (s[:, 2] >= y0) & (s[:, 2] <= y1)
        lab.append(max(mf.BOXES[f][2:]))
        if on.any():
            have += 1
            best = int(np.argmax(np.where(on, s[:, 0], -1)))
            ranks.append(1 + int((s[~on, 0] > s[best, 0]).sum()))
            sizes.append(float(s[best, 3]))
    r = np.array(ranks) if ranks else np.array([np.inf])
    print(f"{lo}-{hi:>6} {len(fs):4d} {have / len(fs):6.2f} {np.median(r):8.0f} "
          f"{np.mean(r <= 10):6.2f} {np.median(sizes) if sizes else np.nan:9.0f} "
          f"{np.median(lab):10.0f}")
print("\n  seed   = fraction of labelled frames with a win_b5_e4 peak on the drone (of 200)")
print("  medrank= that peak's rank among the 200 by map value")

allsz = np.concatenate([v["seeds"][:, 3] for v in col.values()])
print(f"\nall seeds: size median {np.median(allsz):.0f}  "
      f"at the {mf.SEED_SIZE_RANGE[1]:.0f} px clamp {np.mean(allsz >= mf.SEED_SIZE_RANGE[1]):.2f}")
print(f"speed cap at the median seed size: {mf.speed_cap(float(np.median(allsz))):.0f} px/frame")
