"""Where do the top-10 off-drone peaks sit, after the screen-fixed mask? Edge / next to an overlay / scene."""
import pickle
from common import *

res = pickle.load(open(OUT + "peaks.pkl", "rb"))
frac = np.load("data/processed/SOFA-O4/screen_fixed_frac.npy")
fixed = cv2.dilate((frac >= 0.05).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
overlay = cv2.dilate((fixed | HUD_DIL).astype(np.uint8), np.ones((61, 61), np.uint8)) > 0   # within 30 px
edge = np.ones((1080, 1440), bool); edge[60:-60, 60:-60] = False
for name in ("plain_b11", "norm_b11_gmax_e2", "norm_b5_gmax_e4"):
    for label, frames in (("empty", [f for f in res if f < 708]), ("drone", [f for f in res if f >= 708])):
        c = {"edge": 0, "overlay": 0, "scene": 0}
        for f in frames:
            pk = res[f][name]
            pk = pk[~fixed[pk[:, 2].astype(int), pk[:, 1].astype(int)]]
            if f in BOXES:
                x0, y0, x1, y1 = drone_region(BOXES[f])
                pk = pk[~((pk[:, 1] >= x0) & (pk[:, 1] <= x1) & (pk[:, 2] >= y0) & (pk[:, 2] <= y1))]
            for v, x, y in pk[:10]:
                x, y = int(x), int(y)
                c["edge" if edge[y, x] else "overlay" if overlay[y, x] else "scene"] += 1
        t = sum(c.values())
        print(f"{name:18s} {label:5s} " + "  ".join(f"{k} {100 * v / t:4.1f}%" for k, v in c.items()))
