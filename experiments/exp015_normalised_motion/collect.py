"""EXP-015: every score map's peaks on first_catch. Drone frames 708-964 all; empty frames 2-707 every 2nd."""
import pickle, sys, time
from common import *

dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
want = set(range(708, 965)) | set(range(2, 708, 2))
res = {}   # frame -> {map name -> peaks array}
cap = cv2.VideoCapture(VIDEO)
prev, n, t0 = None, 0, time.time()
while True:
    ok, fr = cap.read()
    if not ok: break
    n += 1
    if prev is not None and n in want:
        H = homography(prep(prev, 11), prep(fr, 11))
        v = valid_mask(H, fr.shape[:2])
        res[n] = {k: peaks(m, v) for k, m in maps(prev, fr, H, dis).items()}
        if len(res) % 50 == 0:
            print(n, len(res), f"{time.time() - t0:.0f}s", flush=True)
    prev = fr
pickle.dump(res, open(OUT + "peaks.pkl", "wb"))
print("done", len(res))
