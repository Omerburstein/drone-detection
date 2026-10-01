"""Re-measure EXP-012a's motion claim on first_catch using the hand labels."""
import csv, json, sys
import cv2, numpy as np
sys.path.insert(0, "third_party/GLAD")
from Functions import motion_compensate  # GLAD's own compensation, unchanged

sess = json.load(open("data/processed/SOFA-O4/annotations/first_catch.json"))["frames"]
boxes = {int(k): v for k, v in sess.items() if v and v.get("box")}
hud = cv2.imread("data/processed/SOFA-O4/hud_mask.png", 0) > 0
cap = cv2.VideoCapture("data/processed/SOFA-O4/videos/first_catch.avi")
lk = dict(winSize=(15, 15), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

def prep(f):  # MOD2's _prepare
    return cv2.cvtColor(cv2.GaussianBlur(f, (11, 11), 0), cv2.COLOR_BGR2GRAY)

def centre(b):
    return np.array([b[0] + b[2] / 2, b[1] + b[3] / 2])

rows, prev, Hs = [], None, {}
first = min(boxes)
n = 0
while True:
    ok, fr = cap.read()
    if not ok:
        break
    n += 1  # 1-based, matching first_catch_0001
    if n < first - 1:
        continue
    cur = prep(fr)
    if prev is not None and n in boxes and n - 1 in boxes:
        comp, border, _ = motion_compensate(prev, cur)
        # Recover H (cur->prev) the same way GLAD does, to measure residuals.
        h, w = cur.shape
        gx, gy = np.meshgrid(np.arange(96 / 2, w - 96, 96), np.arange(72 / 2, h - 72, 72))
        p0 = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32).reshape(-1, 1, 2)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, **lk)
        good0, good1 = p0[st == 1], p1[st == 1]
        H, _ = cv2.findHomography(good1, good0, cv2.RANSAC, 3.0)
        Hs[n] = H
        bp, bc = boxes[n - 1]["box"], boxes[n]["box"]
        x, y, bw, bh = bc
        # background points: not on the target (with margin), not on the HUD
        keep = []
        for a, b in zip(good0, good1):
            inside = (x - bw <= b[0] <= x + 2 * bw) and (y - bh <= b[1] <= y + 2 * bh)
            onhud = hud[min(int(b[1]), h - 1), min(int(b[0]), w - 1)]
            keep.append(not inside and not onhud)
        keep = np.array(keep)
        g0, g1 = good0[keep], good1[keep]
        raw = np.linalg.norm(g1 - g0, axis=1)
        back = cv2.perspectiveTransform(g1.reshape(-1, 1, 2), H).reshape(-1, 2)
        resid = np.linalg.norm(back - g0, axis=1)
        cp, cc = centre(bp), centre(bc)
        pred = cv2.perspectiveTransform(cc.reshape(1, 1, 2).astype(np.float64), H).ravel()
        diff_motion = np.linalg.norm(pred - cp)  # target vs where background says it'd be
        # pixel level, as MOD2_global sees it
        d = cv2.absdiff(cur, comp)
        thr = 5 + int(d.mean())
        valid = (border == 0) & ~hud
        tmask = np.zeros_like(valid); xi, yi = int(x), int(y)
        tmask[max(yi, 0):yi + int(bh) + 1, max(xi, 0):xi + int(bw) + 1] = True
        tv = d[tmask & valid]
        bg = d[valid & ~tmask]
        rows.append(dict(
            frame=n, source=boxes[n]["source"], size=max(bw, bh),
            tgt_raw=np.linalg.norm(cc - cp), bg_raw_med=np.median(raw), bg_raw_p90=np.percentile(raw, 90),
            diff_motion=diff_motion, resid_med=np.median(resid), resid_p90=np.percentile(resid, 90),
            thr=thr, tgt_frac_over=(tv > thr).mean() if tv.size else np.nan,
            tgt_p90=np.percentile(tv, 90) if tv.size else np.nan,
            bg_frac_over=(bg > thr).mean(), bg_p99=np.percentile(bg, 99)))
    prev = cur

out = sys.argv[1]
with open(out, "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader(); wr.writerows(rows)

# Jitter-free check: differential motion between consecutive *hand-placed* boxes, chaining H.
human = sorted(k for k, v in boxes.items() if v["source"] == "human")
chain = []
for a, b in zip(human, human[1:]):
    if not all(k in Hs for k in range(a + 1, b + 1)):
        continue
    pt = centre(boxes[b]["box"]).reshape(1, 1, 2).astype(np.float64)
    for k in range(b, a, -1):  # map b's centre back to frame a via background
        pt = cv2.perspectiveTransform(pt, Hs[k])
    chain.append((a, b, np.linalg.norm(pt.ravel() - centre(boxes[a]["box"])) / (b - a)))
np.save(out + ".chain.npy", np.array(chain))
print(len(rows), "pairs;", len(chain), "hand-to-hand spans")
