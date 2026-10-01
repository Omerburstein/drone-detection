"""EXP-016 stills: a magnified crop around the label, so a 17-43 px target can be judged by eye.

The overlay video is 0.5x and a long-range drone is a speck in it. For each requested
frame this writes a three-panel sheet at 4x: the picture, the `win_b5_e4` seed map, and
the picture again with every live hypothesis drawn, cropped to a window around the label
(or around the frame centre when there is none).

    py -3.13 -m stills --tag frozen 515 560 700
"""
import argparse
import pickle

import cv2
import numpy as np

import common
import mf
from mf import CLIP, OUT

WHITE, GREEN = (255, 255, 255), (0, 255, 0)
ZOOM = 4


def crop(img, cx, cy, half):
    x0, y0 = int(cx - half), int(cy - half)
    x0 = min(max(x0, 0), mf.W - 2 * half)
    y0 = min(max(y0, 0), mf.H_PX - 2 * half)
    return img[y0:y0 + 2 * half, x0:x0 + 2 * half], x0, y0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", type=int, nargs="+")
    ap.add_argument("--tag", default="frozen")
    ap.add_argument("--half", type=int, default=90)
    a = ap.parse_args()
    res = pickle.load(open(OUT + f"verify_{a.tag}.pkl", "rb"))
    col = pickle.load(open(OUT + "collect.pkl", "rb"))
    fixed = mf.screen_fixed()
    want = set(a.frames)
    cap = cv2.VideoCapture(CLIP["video"])
    prev, n = None, 0
    while True:
        ok, fr = cap.read()
        if not ok or n > max(want):
            break
        n += 1
        if n in want and prev is not None and n in col:
            Hm = col[n]["H"].astype(np.float64)
            valid = common.valid_mask(Hm, fr.shape[:2]) & ~fixed
            wm, _ = mf.win_map(prev, fr, Hm)
            heat = cv2.applyColorMap(
                np.clip(np.where(valid, wm, 0) / 10.0 * 255, 0, 255).astype(np.uint8),
                cv2.COLORMAP_INFERNO)
            box = mf.BOXES.get(n)
            cx, cy = (box[0] + box[2] / 2, box[1] + box[3] / 2) if box else (mf.W / 2, mf.H_PX / 2)
            marked = fr.copy()
            for hid, x, y, z, k, size, conf in res["frames"].get(n, []):
                c = GREEN if conf > 0 else (60, 200, 255)
                cv2.circle(marked, (int(x), int(y)), max(6, int(0.5 * size)), c, 1)
                if conf > 0:
                    cv2.putText(marked, f"z{z:.0f}", (int(x) + 8, int(y) - 8), 0, 0.5, c, 1)
            if box:
                x, y, w, h = [int(round(v)) for v in box]
                for im in (fr, marked):
                    cv2.rectangle(im, (x, y), (x + w, y + h), WHITE, 1)
            panels = [crop(im, cx, cy, a.half)[0] for im in (fr, heat, marked)]
            sheet = np.hstack([cv2.resize(p, None, fx=ZOOM, fy=ZOOM,
                                          interpolation=cv2.INTER_NEAREST) for p in panels])
            bar = np.zeros((44, sheet.shape[1], 3), np.uint8)
            rk = res["ranks"].get(n)
            cv2.putText(bar, f"{CLIP['name']} frame {n}  |  picture | win_b5_e4 | hypotheses  "
                             f"|  drone rank by z: " + ("none" if rk is None else f"#{rk[0]} z{rk[1]:.1f}")
                             + (f"  |  label {int(box[2])}x{int(box[3])} px" if box else ""),
                        (10, 30), 0, 0.6, WHITE, 2)
            path = OUT + f"still_{n}.png"
            cv2.imwrite(path, np.vstack([bar, sheet]))
            print("wrote", path)
        prev = fr
    cap.release()


if __name__ == "__main__":
    main()
