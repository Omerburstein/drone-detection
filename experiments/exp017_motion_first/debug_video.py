"""Render stage 0 and stage 1 back onto the clip, so a human can check them.

This is the whole point of stopping here. The depth-aware ring in stage 2 is only as good
as this boundary, so the boundary gets looked at before anything is built on it.

    PYTHONPATH="experiments/exp017_motion_first;." py -3.13 -m debug_video
    PYTHONPATH="runs/sofa_analog/exp017_motion_first;experiments/exp017_motion_first;." \
        py -3.13 -m debug_video --start 400 --end 800

What you are looking at
-----------------------
  * blue tint      -- called sky
  * no tint        -- called scene
  * yellow tint    -- uncertain: the band either test must refuse to straddle
  * yellow line    -- the horizon read off the mask, per column
  * red tint       -- masked by stage 0; the legend names which layer
  * green box      -- the labelled drone, when the clip has labels

What to look for, in order of how much it would cost me
-------------------------------------------------------
  1. Is the drone ever swallowed by a stage-0 mask? That is recall gone before anything
     runs, and it is what EXP-016's in-sample screen-fixed map did at analog frame 585.
  2. Is a tree line called sky, or sky called scene? A boundary in the wrong place moves
     the ring test's blind spot somewhere useless.
  3. Does the split flicker between frames on a steady scene? Per-frame percentiles are
     meant to track exposure, not to chatter.
  4. When the drone is against the canopy rather than sky, is it inside `scene` with a
     clean ring? That is the O4 708-800 case, and it is the one I expect to be hardest.
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

import masks
import skyline
from clipcfg import CLIP

W, H = CLIP["width"], CLIP["height"]
TINT = {"sky": (200, 120, 40), "uncertain": (40, 220, 230), "masked": (60, 60, 220)}
LAYER_COLOUR = {"hud": (60, 60, 220), "prop": (200, 60, 220),
                "ladder": (60, 200, 220), "edge": (120, 120, 120)}


def tint(img: np.ndarray, m: np.ndarray, colour, alpha: float) -> None:
    """Blend `colour` into `img` where `m`, in place."""
    if not m.any():
        return
    img[m] = (img[m] * (1 - alpha) + np.asarray(colour, np.float32) * alpha).astype(np.uint8)


def load_boxes() -> dict[int, list]:
    p = CLIP.get("labels")
    if not p or not os.path.exists(p):
        return {}
    frames = json.load(open(p))["frames"]
    return {int(k): v["box"] for k, v in frames.items() if v and v.get("box")}


def legend(img: np.ndarray, stage0: masks.Stage0, sl: skyline.Skyline, frame: int) -> None:
    cov = stage0.coverage()
    lines = [f"frame {frame}",
             f"sky {sl.sky_fraction * 100:.1f}%" + ("" if sl.has_sky else "  (NO SKY)"),
             f"uncertain {sl.uncertain.mean() * 100:.1f}%",
             "masked: " + "  ".join(f"{k} {cov[k]:.1f}%" for k in ("hud", "prop", "ladder"))]
    for i, t in enumerate(lines):
        cv2.putText(img, t, (12, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, t, (12, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=CLIP["n_frames"])
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = a.out or os.path.join(CLIP["out"], f"skyline_{CLIP['name']}_{a.start}_{a.end}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    stage0 = masks.Stage0()
    print("[stage 0] coverage: " + ", ".join(f"{k} {v:.2f}%" for k, v in stage0.coverage().items()))
    boxes = load_boxes()
    valid = stage0.valid()

    cap = cv2.VideoCapture(CLIP["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.start - 1)
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))
    swallowed, n = [], 0

    for f in range(a.start, a.end + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if (f - a.start) % a.stride:
            continue
        sl = skyline.split(frame, valid)
        img = frame.copy()
        tint(img, sl.sky, TINT["sky"], 0.28)
        tint(img, sl.uncertain, TINT["uncertain"], 0.35)
        for name, m in stage0.layers.items():
            tint(img, m, LAYER_COLOUR[name], 0.45)

        xs = np.nonzero(sl.horizon >= 0)[0]
        for x in xs[::4]:
            cv2.circle(img, (int(x), int(sl.horizon[x])), 1, (40, 220, 230), -1)

        box = boxes.get(f)
        if box:
            x, y, w_, h_ = [int(v) for v in box]
            cv2.rectangle(img, (x, y), (x + w_, y + h_), (60, 230, 60), 2)
            cx, cy = x + w_ / 2, y + h_ / 2
            lab = sl.label(cx, cy)
            hit = [k for k, m in stage0.layers.items()
                   if m[int(np.clip(cy, 0, H - 1)), int(np.clip(cx, 0, W - 1))]]
            if hit:
                swallowed.append((f, hit))
            txt = f"drone: {lab}" + (f"  MASKED BY {','.join(hit)}" if hit else "")
            cv2.putText(img, txt, (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, txt, (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (60, 230, 60), 1, cv2.LINE_AA)

        legend(img, stage0, sl, f)
        writer.write(img)
        n += 1
    writer.release()
    cap.release()

    print(f"[debug_video] {n} frames -> {out}")
    if swallowed:
        print(f"[WARNING] stage 0 masks cover the labelled drone in {len(swallowed)} frames, "
              f"first at {swallowed[0][0]} ({','.join(swallowed[0][1])}). "
              f"That is recall lost before the detector runs.")
    elif boxes:
        print("[ok] no labelled drone centre falls inside a stage-0 mask.")


if __name__ == "__main__":
    main()
