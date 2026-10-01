"""Shared pieces for EXP-015: GLAD's homography (with H returned), the masks, and the score maps."""
import json
import cv2, numpy as np
from src.data.hud_mask import load_mask

VIDEO = "data/processed/SOFA-O4/videos/first_catch.avi"
OUT = "runs/sofa_o4/exp015_normalised_motion/"
LABELS = json.load(open("data/processed/SOFA-O4/annotations/first_catch.json"))["frames"]
BOXES = {int(k): v["box"] for k, v in LABELS.items() if v and v.get("box")}
SOURCE = {int(k): v.get("source") for k, v in LABELS.items() if v and v.get("box")}
HUD = load_mask("data/processed/SOFA-O4/hud_mask.png")
HUD_DIL = cv2.dilate(HUD.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0   # blur-11 spreads the overlay ~5 px
MARGIN = 24          # px dropped at every edge: fisheye rim, encoder edge
PEAK_RADIUS = 15     # local-max NMS radius (px)
EPS = (1.0, 2.0, 4.0)  # gradient floor, grey levels per px
WIN = 9              # window for the windowed (LK-style) normal-flow estimate


def prep(frame, k):
    return cv2.cvtColor(cv2.GaussianBlur(frame, (k, k), 0), cv2.COLOR_BGR2GRAY)


def homography(frame1, frame2):
    """Upstream `Functions.motion_compensate`, verbatim apart from returning H."""
    lk = dict(winSize=(15, 15), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    h, w = frame2.shape
    gw, gh = 32 * 3, 24 * 3
    p1 = np.array([(np.float32(i * gw + gw / 2.0), np.float32(j * gh + gh / 2.0))
                   for i in range(int(w / gw - 1)) for j in range(int(h / gh - 1))]).reshape(-1, 1, 2)
    cur, st, _ = cv2.calcOpticalFlowPyrLK(frame1, frame2, p1, None, **lk)
    gn, go = cur[st == 1], p1[st == 1]
    keep = np.linalg.norm(gn - go, axis=1) <= 50
    if len(go) < 9:
        return np.array([[0.999, 0, 0], [0, 0.999, 0], [0, 0, 1]])
    H, _ = cv2.findHomography(gn, go, cv2.RANSAC, 3.0)   # upstream fits on all tracked points
    return H


def warp(img, H):
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)


def valid_mask(H, shape):
    h, w = shape
    inside = warp(np.full(shape, 255, np.uint8), H) > 0
    inside = cv2.erode(inside.astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
    inside[:MARGIN] = inside[-MARGIN:] = False
    inside[:, :MARGIN] = inside[:, -MARGIN:] = False
    return inside & ~HUD_DIL


def grad(img):
    f = img.astype(np.float32)
    return np.hypot(cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)) / 8.0


def maps(prev_bgr, cur_bgr, H, dis=None):
    """Every score map for one frame pair, keyed by name. Values are float32, masked pixels 0."""
    out = {}
    for k in (11, 5):
        p, c = prep(prev_bgr, k), prep(cur_bgr, k)
        comp = warp(p, H)
        d = cv2.absdiff(c, comp).astype(np.float32)
        out[f"plain_b{k}"] = d
        gc, gp = grad(c), grad(comp)
        gmax = np.maximum(gc, gp)
        for e in EPS:
            out[f"norm_b{k}_gcur_e{e:g}"] = d / np.maximum(gc, e)
            out[f"norm_b{k}_gmax_e{e:g}"] = d / np.maximum(gmax, e)
            num = cv2.boxFilter(d * d, -1, (WIN, WIN))
            den = cv2.boxFilter(gmax * gmax, -1, (WIN, WIN)) + e * e
            out[f"win_b{k}_e{e:g}"] = np.sqrt(num / den)
        if dis is not None and k == 5:
            fl = dis.calc(comp, c, None)
            out["dis_b5"] = np.linalg.norm(fl, axis=2)
    return out


def peaks(m, valid, top=400):
    """Local maxima (radius PEAK_RADIUS) inside `valid`, strongest first: array of (value, x, y)."""
    m = np.where(valid, m, 0).astype(np.float32)
    k = 2 * PEAK_RADIUS + 1
    dil = cv2.dilate(m, np.ones((k, k), np.uint8))
    ys, xs = np.nonzero((m >= dil) & (m > 0))
    v = m[ys, xs]
    o = np.argsort(-v)[:top]
    return np.stack([v[o], xs[o], ys[o]], 1).astype(np.float32)


def drone_region(box):
    x, y, w, h = box
    m = max(10.0, 0.25 * max(w, h))
    return x - m, y - m, x + w + m, y + h + m
