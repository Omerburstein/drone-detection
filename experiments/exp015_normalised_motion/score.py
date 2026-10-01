"""EXP-015 scoring: drone rank among peaks, top-k, found at a matched false-alarm load, per stretch."""
import csv, pickle, sys
import numpy as np
from common import BOXES, SOURCE, OUT, drone_region

res = pickle.load(open(OUT + "peaks.pkl", "rb"))
STRETCH = [(708, 800), (801, 900), (901, 953), (954, 964), (708, 964)]
LOADS = (10, 3)          # candidates per empty frame the threshold is set to
empty = [f for f in res if f < 708]
drone = [f for f in res if f >= 708]
names = list(res[drone[0]])

# Screen-fixed clutter, calibrated on the empty frames only: where plain_b11's top-50 peaks
# land in >=5% of frames 2-707. Terrain moves across the screen; the host drone's props,
# the pitch ladder and HUD digits do not. MASK=props keeps only the two prop regions.
import os, cv2
MASKMODE = os.environ.get("MASK", "none")
frac = np.load("data/processed/SOFA-O4/screen_fixed_frac.npy")
fixed = cv2.dilate((frac >= 0.05).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
if MASKMODE == "props":
    keep = np.zeros_like(fixed); keep[450:720, :420] = True; keep[450:720, 1100:] = True
    fixed &= keep
if MASKMODE == "strict":   # headroom estimate: also drop 60 px edges and anything within 30 px of an overlay
    from common import HUD_DIL
    near = cv2.dilate((fixed | HUD_DIL).astype(np.uint8), np.ones((61, 61), np.uint8)) > 0
    fixed = fixed | near
    fixed[:60] = fixed[-60:] = True; fixed[:, :60] = fixed[:, -60:] = True
    print("strict mask covers", round(fixed.mean(), 3), "of the frame")
if MASKMODE != "none":
    for f in res:
        for k, pk in res[f].items():
            res[f][k] = pk[~fixed[pk[:, 2].astype(int), pk[:, 1].astype(int)]]


def drone_stats(pk, box):
    x0, y0, x1, y1 = drone_region(box)
    inside = (pk[:, 1] >= x0) & (pk[:, 1] <= x1) & (pk[:, 2] >= y0) & (pk[:, 2] <= y1)
    if not inside.any():
        return 0.0, np.inf, pk[~inside, 0]
    v = pk[inside, 0].max()
    off = pk[~inside, 0]
    return float(v), 1 + int((off > v).sum()), off


rows = []
for name in names:
    pooled = np.sort(np.concatenate([res[f][name][:, 0] for f in empty]))[::-1]
    tau = {L: float(pooled[L * len(empty) - 1]) for L in LOADS}
    per = {f: drone_stats(res[f][name], BOXES[f]) for f in drone}
    for a, b in STRETCH:
        fs = [f for f in drone if a <= f <= b]
        ranks = np.array([per[f][1] for f in fs])
        vals = np.array([per[f][0] for f in fs])
        hand = [f for f in fs if SOURCE[f] == "human"]
        row = dict(map=name, stretch=f"{a}-{b}", frames=len(fs),
                   top1=np.mean(ranks <= 1), top5=np.mean(ranks <= 5), top10=np.mean(ranks <= 10),
                   median_rank=float(np.median(ranks)),
                   drone_val_hand=float(np.median([per[f][0] for f in hand])) if hand else np.nan,
                   top_off_val=float(np.median([per[f][2].max() if len(per[f][2]) else 0 for f in fs])))
        for L in LOADS:
            row[f"tau{L}"] = tau[L]
            row[f"found@{L}"] = float(np.mean(vals >= tau[L]))
            row[f"cands@{L}"] = float(np.mean([(per[f][2] >= tau[L]).sum() for f in fs]))
        rows.append(row)

with open(OUT + f"score_{MASKMODE}.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

only = sys.argv[1:] or None
for r in rows:
    if only and r["map"] not in only: continue
    print(f"{r['map']:22s} {r['stretch']:8s} n{r['frames']:4d} top1 {r['top1']:.2f} top5 {r['top5']:.2f} "
          f"top10 {r['top10']:.2f} medrank {r['median_rank']:6.0f} | tau10 {r['tau10']:6.2f} found {r['found@10']:.2f} "
          f"cands {r['cands@10']:5.1f} | tau3 {r['tau3']:6.2f} found {r['found@3']:.2f} | drone(hand) {r['drone_val_hand']:6.2f} topoff {r['top_off_val']:6.2f}")
