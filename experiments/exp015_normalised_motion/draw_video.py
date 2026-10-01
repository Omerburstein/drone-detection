"""EXP-015 overlay video: plain difference vs norm_b11_gmax_e2, top-10 peaks + heatmaps, frames 650-964."""
from common import *

FIRST, LAST, FPS = 650, 964, 10
_frac = np.load("data/processed/SOFA-O4/screen_fixed_frac.npy")          # same screen-fixed mask as score.py MASK=fixed
FIXED = cv2.dilate((_frac >= 0.05).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
WHITE, GREEN, ORANGE, MAGENTA = (255, 255, 255), (0, 255, 0), (0, 165, 255), (255, 0, 255)
STRETCH = [(708, 800, "far, ~43 px"), (801, 900, "~87 px"), (901, 953, "~104 px"), (954, 964, "close, ~112 px")]


def heat(m, valid, vmax, title):
    v = np.clip(np.where(valid, m, 0) / vmax * 255, 0, 255).astype(np.uint8)
    img = cv2.applyColorMap(v, cv2.COLORMAP_INFERNO)
    img[~valid] = (img[~valid] * 0.25 + 40).astype(np.uint8)
    cv2.putText(img, title, (10, 40), 0, 1.0, WHITE, 2)
    return img


def drone_rank(pk, box):
    """1 + off-drone peaks stronger than the best on-drone peak; None if no peak on the drone."""
    x0, y0, x1, y1 = drone_region(box)
    on = (pk[:, 1] >= x0) & (pk[:, 1] <= x1) & (pk[:, 2] >= y0) & (pk[:, 2] <= y1)
    if not on.any():
        return None
    return int((pk[~on, 0] > pk[on, 0].max()).sum()) + 1


def mark(img, pk, box, col, label):
    x0, y0, x1, y1 = drone_region(box) if box else (0, 0, -1, -1)
    for i, (v, x, y) in enumerate(pk[:10]):
        c = GREEN if (x0 <= x <= x1 and y0 <= y <= y1) else col
        cv2.circle(img, (int(x), int(y)), 22, c, 3)
        cv2.putText(img, f"{label}{i + 1}", (int(x) + 24, int(y) + 8), 0, 0.8, c, 2)


def label(img, text, org, col, scale=1.1):
    (w, h), _ = cv2.getTextSize(text, 0, scale, 3)
    cv2.rectangle(img, (org[0] - 6, org[1] - h - 10), (org[0] + w + 6, org[1] + 10), (0, 0, 0), -1)
    cv2.putText(img, text, org, 0, scale, col, 3)


def rank_text(r):
    return "not in map" if r is None else (f"#{r}  (top-10)" if r <= 10 else f"#{r}")


cap = cv2.VideoCapture(VIDEO)
vid, prev, n = None, None, 0
while True:
    ok, fr = cap.read()
    if not ok or n >= LAST: break
    n += 1
    if prev is not None and n >= FIRST:
        p, c = prep(prev, 11), prep(fr, 11)
        H = homography(p, c)
        valid = valid_mask(H, fr.shape[:2]) & ~FIXED
        comp = warp(p, H)
        plain = cv2.absdiff(c, comp).astype(np.float32)
        norm = plain / np.maximum(np.maximum(grad(c), grad(comp)), 2.0)
        box = BOXES.get(n)
        pp, pn = peaks(plain, valid), peaks(norm, valid)
        a = fr.copy()
        if box:
            x, y, w, h = [int(round(v)) for v in box]
            cv2.rectangle(a, (x, y), (x + w, y + h), WHITE, 2)
        a[~valid] = (a[~valid] * 0.6).astype(np.uint8)
        b = a.copy()
        mark(a, pp, box, ORANGE, "P")
        mark(b, pn, box, MAGENTA, "N")
        if box:
            ra, rb = rank_text(drone_rank(pp, box)), rank_text(drone_rank(pn, box))
        else:
            ra = rb = "no drone in frame"
        label(a, "PLAIN difference (GLAD): top-10 peaks", (16, 50), ORANGE)
        label(a, f"drone rank: {ra}", (16, 110), WHITE)
        label(b, "NORMALISED by edge sharpness: top-10 peaks", (16, 50), MAGENTA)
        label(b, f"drone rank: {rb}", (16, 110), WHITE)
        top = np.hstack([a, b])
        bot = np.hstack([heat(plain, valid, 40, "plain difference, grey levels (white = 40+)"),
                         heat(norm, valid, 10, "normalised, ~px moved (white = 10+ px)")])
        sheet = cv2.resize(np.vstack([top, bot]), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        stage = next((f"{lo}-{hi}: {name}" for lo, hi, name in STRETCH if lo <= n <= hi), "no drone yet (false-alarm baseline)")
        bar = np.zeros((60, sheet.shape[1], 3), np.uint8)
        cv2.putText(bar, f"frame {n}  |  {stage}  |  white box = your label, green = peak on drone",
                    (12, 40), 0, 0.75, WHITE, 2)
        sheet = np.vstack([bar, sheet])
        if vid is None:
            vid = cv2.VideoWriter(OUT + "overlay_exp015.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, sheet.shape[1::-1])
        vid.write(sheet)
    prev = fr
vid.release()
print("frames written:", n - FIRST + 1)
