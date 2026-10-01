"""EXP-016 overlay video, in the style of EXP-015's.

Left: the frame, your label in white, every live hypothesis as a ring coloured by its z
(dim blue = no evidence, yellow = at z*, green = confirmed), confirmed tracks carrying
their trail and their id. Right: the `win_b5_e4` seed map the hypotheses are drawn from,
with the frame's 200 seeds. A caption strip carries the frame number, the stretch, the
number of live and confirmed hypotheses, and the best on-drone rank by z.

    py -3.13 -m draw_video --tag frozen --first 400 --last 964
"""
import argparse
import pickle

import cv2
import numpy as np

import common
import mf
from mf import CLIP, OUT

WHITE, GREEN, YELLOW, GREY = (255, 255, 255), (0, 255, 0), (0, 230, 255), (150, 150, 150)
FPS_OUT = 10

# What the overlay may claim depends on whether the clip carries ground truth.
LABELLED = bool(mf.BOXES)
OVERLAY = CLIP.get("overlay_name", "overlay_exp016.mp4")
TITLE = "EXP-016 local-ring test" if LABELLED else "EXP-017 local-ring test (unlabelled)"
KEY = ("white box = your label, green = confirmed track, dim rings = hypotheses under test"
       if LABELLED else
       "green = confirmed track, dim rings = hypotheses under test; NO ground truth here")


def z_colour(z: float, zstar: float):
    """Dim blue below a third of z*, through orange, to yellow at z*."""
    t = float(np.clip(z / max(zstar, 1e-6), 0, 1))
    v = np.uint8([[[int(120 - 100 * t), 200, int(90 + 165 * t)]]])
    return tuple(int(c) for c in cv2.cvtColor(v, cv2.COLOR_HSV2BGR)[0, 0])


def fit(text, scale, width, pad=24):
    """Shrink `scale` until `text` fits `width`. Never grows it."""
    (w, _), _ = cv2.getTextSize(text, 0, scale, 2)
    return scale if w + pad <= width else scale * (width - pad) / max(w, 1)


def label(img, text, org, col, scale=0.9):
    scale = fit(text, scale, img.shape[1] - org[0])
    (w, h), _ = cv2.getTextSize(text, 0, scale, 2)
    cv2.rectangle(img, (org[0] - 5, org[1] - h - 8), (org[0] + w + 5, org[1] + 8), (0, 0, 0), -1)
    cv2.putText(img, text, org, 0, scale, col, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="frozen")
    ap.add_argument("--first", type=int, default=None)
    ap.add_argument("--last", type=int, default=None)
    a = ap.parse_args()
    res = pickle.load(open(OUT + f"verify_{a.tag}.pkl", "rb"))
    col = pickle.load(open(OUT + "collect.pkl", "rb"))
    zstar = res["zstar"]
    frames, tracks, ranks = res["frames"], res["tracks"], res["ranks"]
    trails = {hid: {int(r[0]): (r[1], r[2]) for r in tr["trail"]} for hid, tr in tracks.items()}
    first = a.first or (max(1, min(mf.DRONE_FRAMES) - 120) if mf.DRONE_FRAMES else 1)
    last = a.last or CLIP["n_frames"]
    fixed = mf.screen_fixed()
    scale = 1440.0 / mf.W

    cap = cv2.VideoCapture(CLIP["video"])
    vid, prev, n = None, None, 0
    while True:
        ok, fr = cap.read()
        if not ok or n >= last:
            break
        n += 1
        if prev is None or n < first or n not in col:
            prev = fr
            continue
        Hm = col[n]["H"].astype(np.float64)
        valid = common.valid_mask(Hm, fr.shape[:2]) & ~fixed
        wm, _ = mf.win_map(prev, fr, Hm)
        heat = cv2.applyColorMap(
            np.clip(np.where(valid, wm, 0) / 10.0 * 255, 0, 255).astype(np.uint8),
            cv2.COLORMAP_INFERNO)
        heat[~valid] = (heat[~valid] * 0.25 + 30).astype(np.uint8)
        for _, x, y, _ in col[n]["seeds"][:60]:
            cv2.circle(heat, (int(x), int(y)), 6, GREY, 1)
        label(heat, "win_b5_e4 seed map (white = 10+ px); rings = the frame's top 60 seeds",
              (12, 34), WHITE, 0.7 * scale)

        img = fr.copy()
        img[~valid] = (img[~valid] * 0.55).astype(np.uint8)
        box = mf.BOXES.get(n)
        if box:
            x, y, w, h = [int(round(v)) for v in box]
            cv2.rectangle(img, (x, y), (x + w, y + h), WHITE, 2)

        rows = frames.get(n, np.zeros((0, 7), np.float32))
        nlive = len(rows)
        nconf = 0
        for hid, x, y, z, k, size, conf in rows:
            r = max(8, int(round(0.6 * size)))
            if conf > 0:
                nconf += 1
                cv2.circle(img, (int(x), int(y)), r, GREEN, 2)
                cv2.putText(img, f"#{int(hid)} z{z:.1f}", (int(x) + r + 3, int(y) + 5),
                            0, 0.55 * scale, GREEN, 2)
                t = trails.get(int(hid), {})
                pts = [t[m] for m in range(max(first, n - 40), n + 1) if m in t]
                for p, q in zip(pts, pts[1:]):
                    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), GREEN, 1)
            elif k >= 2:
                cv2.circle(img, (int(x), int(y)), max(5, r // 2), z_colour(z, zstar), 1)

        rk = ranks.get(n)
        label(img, f"{TITLE}   z* = {zstar:g}", (12, 40), YELLOW, 1.0 * scale)
        label(img, f"live {nlive}   confirmed {nconf}", (12, 84), WHITE, 0.8 * scale)
        if LABELLED:
            stretch = next((f"{lo}-{hi}" for lo, hi in CLIP["stretches"][:-1] if lo <= n <= hi),
                           "no drone in frame")
            label(img, "drone rank by z: " + ("no hypothesis on the drone" if rk is None
                                              else f"#{rk[0]}  (z {rk[1]:.1f})"),
                  (12, 124), WHITE, 0.8 * scale)
        else:
            # No labels, so there is no rank to report and no frame that can be called
            # empty. PROVENANCE names episodes where the target was *seen*; it warns in
            # terms that a gap may be a stretch the detector missed. The caption says
            # exactly that and no more -- a viewer must not read "outside" as "no drone".
            stretch = next((f"PROVENANCE episode {lo}-{hi}"
                            for lo, hi in CLIP["stretches"][:-1] if lo <= n <= hi),
                           "outside the known episodes (NOT verified empty)")
            label(img, "unlabelled clip: no rank, no score -- judge by eye",
                  (12, 124), WHITE, 0.8 * scale)

        sheet = np.hstack([img, heat])
        sheet = cv2.resize(sheet, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        bar = np.zeros((52, sheet.shape[1], 3), np.uint8)
        cap_text = f"frame {n}  |  {stretch}  |  {KEY}"
        cv2.putText(bar, cap_text, (12, 34), 0, fit(cap_text, 0.62, bar.shape[1] - 12),
                    WHITE, 2)
        sheet = np.vstack([bar, sheet])
        if vid is None:
            vid = cv2.VideoWriter(OUT + OVERLAY,
                                  cv2.VideoWriter_fourcc(*"mp4v"), FPS_OUT, sheet.shape[1::-1])
        vid.write(sheet)
        prev = fr
    cap.release()
    vid.release()
    print("wrote", OUT + OVERLAY)


if __name__ == "__main__":
    main()
