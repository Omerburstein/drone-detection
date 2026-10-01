"""EXP-015 stills: plain difference vs a normalised map, top-10 peaks of each, with the label."""
import sys
from common import *

NORM = sys.argv[1]                      # e.g. win_b5_e2
FRAMES = [int(a) for a in sys.argv[2:]] or [600, 750, 850, 925, 962]
PLAIN = "plain_b11"
_frac = np.load("data/processed/SOFA-O4/screen_fixed_frac.npy")   # same screen-fixed mask as score.py MASK=fixed
FIXED = cv2.dilate((_frac >= 0.05).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
cap = cv2.VideoCapture(VIDEO)
prev, n = None, 0
WHITE, GREEN = (255, 255, 255), (0, 255, 0)


def heat(m, valid, vmax, title):
    v = np.clip(np.where(valid, m, 0) / vmax * 255, 0, 255).astype(np.uint8)
    img = cv2.applyColorMap(v, cv2.COLORMAP_INFERNO)
    img[~valid] = (img[~valid] * 0.25 + 40).astype(np.uint8)
    cv2.putText(img, title, (10, 40), 0, 1.0, WHITE, 2)
    return img


def mark(img, pk, box, col, label):
    x0, y0, x1, y1 = drone_region(box) if box else (0, 0, -1, -1)
    for i, (v, x, y) in enumerate(pk[:10]):
        on = x0 <= x <= x1 and y0 <= y <= y1
        c = GREEN if on else col
        cv2.circle(img, (int(x), int(y)), 22, c, 3)
        cv2.putText(img, f"{label}{i + 1}", (int(x) + 24, int(y) + 8), 0, 0.8, c, 2)


while True:
    ok, fr = cap.read()
    if not ok: break
    n += 1
    if prev is not None and n in FRAMES:
        H = homography(prep(prev, 11), prep(fr, 11))
        valid = valid_mask(H, fr.shape[:2]) & ~FIXED
        ms = maps(prev, fr, H, cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM))
        box = BOXES.get(n)
        a = fr.copy()
        pp, pn = peaks(ms[PLAIN], valid), peaks(ms[NORM], valid)
        if box:
            x, y, w, h = [int(round(v)) for v in box]
            cv2.rectangle(a, (x, y), (x + w, y + h), WHITE, 2)
        a[~valid] = (a[~valid] * 0.6).astype(np.uint8)
        b = a.copy()
        mark(a, pp, box, (0, 165, 255), "P")
        mark(b, pn, box, (255, 0, 255), "N")
        cv2.putText(a, f"frame {n}: top-10 peaks, plain difference (orange; green = on drone)", (10, 40), 0, 0.9, WHITE, 2)
        cv2.putText(b, f"frame {n}: top-10 peaks, {NORM} (magenta; green = on drone)", (10, 40), 0, 0.9, WHITE, 2)
        vmax_n = 40.0
        top = np.hstack([a, b])
        bot = np.hstack([heat(ms[PLAIN], valid, 40, "plain difference, grey levels (white = 40+)"),
                         heat(ms[NORM], valid, vmax_n, f"{NORM}, px (white = {vmax_n:g}+ px)")])
        sheet = cv2.resize(np.vstack([top, bot]), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imwrite(OUT + f"still_{n}.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if box:   # zoom around the drone, 4 panels
            cx, cy = int(box[0] + box[2] / 2), int(box[1] + box[3] / 2)
            r = max(120, int(1.5 * max(box[2], box[3])))
            ys, xs = slice(max(cy - r, 0), cy + r), slice(max(cx - r, 0), cx + r)
            tiles = [fr[ys, xs].copy(), heat(ms[PLAIN], valid, 40, "")[ys, xs],
                     heat(ms[NORM], valid, vmax_n, "")[ys, xs]]
            tiles = [cv2.resize(t, (400, 400), interpolation=cv2.INTER_NEAREST) for t in tiles]
            for t, s in zip(tiles, ["frame", "plain diff", NORM]):
                cv2.putText(t, s, (8, 28), 0, 0.8, WHITE, 2)
            cv2.imwrite(OUT + f"zoom_{n}.jpg", np.hstack(tiles))
    prev = fr
