"""EXP-025b: the sky branch's top-N per frame, after a persistence gate of m of k frames.

    SOFA-ANALOG catch_2:
    PYTHONPATH="experiments/exp023_sky_branch/analog_catch_2;experiments/exp025_top3;experiments/exp023_sky_branch;experiments/exp017_motion_first;experiments/exp015_normalised_motion;." py -3.13 -m overlay_gate --start 441 --end 800 --out runs/sofa_analog/exp025_top3/gate4of5_top3_catch_2_441_800.mp4

`overlay_top3.py` (EXP-025) is unchanged; this adds one step between EXP-023's threshold
and EXP-025's cap.

The gate
--------
A kept candidate (c >= 6) in frame f is chained backwards through frames f-1 .. f-k+1.
Each step carries the chain's position into the older frame with that pair's camera
homography, then looks for a kept sky candidate within `--radius` of it. A hit counts one
appearance and **moves the chain onto that candidate**, so the chain follows a target that
moves against the background; a miss keeps the carried position. The candidate survives
when it appears in at least m of the k frames, its own frame included. This is the
peak-chaining association EXP-024 recommended: no LK track and no texture, only the
candidates' own positions.

`--coords image` skips the homography and chains in raw picture coordinates. On an
intercept clip the camera follows the target, so the target is steadier in the picture
than the background is: on analog catch_2 the labelled drone's step is a median 4.1 px raw
against 6.7 px camera-compensated, over 9 px in 20% of frames against 40%. Compensation
suits static clutter, not a chased target. `scene` (the default) is what was asked for.

The gate is causal (frame f looks only at f and earlier), and the decode starts k-1
frames before `--start` so the first rendered frame has a full window.

Every appearance threshold 1..k is reported from one pass, as EXP-024 does: `appearances`
does not depend on m, which only decides what is drawn. m = 1 is EXP-025 exactly.

Drawing
-------
EXP-025's clean look: each of the top N survivors as a red circle (at least 10 px across)
tagged `#rank c`, `#1` thicker, and a two-line caption. Nothing is drawn from the labels.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import deque

import cv2
import numpy as np

import common
import geometry
import masks
import silhouette
import skyline
from clipcfg import CLIP
from overlay_sky import _draw_background
from overlay_top3 import draw_ranked
from overlay_video import load_boxes, on_drone, px, text_block

W, H = CLIP["width"], CLIP["height"]
FIELDS = ["frame", "x", "y", "c", "diameter", "appearances", "rank_all", "on_target"]


def appearances(x: float, y: float, window: list, radius: float,
                compensate: bool = True) -> int:
    """In how many of the window's frames a kept candidate lies on this one's chain.

    `window` is oldest-first, the candidate's own frame last; each entry is
    `(hmat, xy)` where `hmat` carries that frame's points into the frame before it.
    With `compensate` off the chain stays in picture coordinates and `hmat` is unused.
    """
    pos, n = np.array([[x, y]], np.float32), 1
    for i in range(len(window) - 1, 0, -1):
        if compensate:
            pos = geometry.apply_h(window[i][0], pos)
        older = window[i - 1][1]
        if len(older):
            d = np.hypot(older[:, 0] - pos[0, 0], older[:, 1] - pos[0, 1])
            j = int(d.argmin())
            if d[j] <= radius:
                n += 1
                pos = older[j:j + 1].copy()
    return n


def ranked_survivors(cands: list[dict], min_appear: int, top: int) -> list[dict]:
    """One frame's candidates at >= min_appear appearances, highest c first, at most `top`."""
    alive = [c for c in cands if c["appearances"] >= min_appear]
    return sorted(alive, key=lambda c: c["c"], reverse=True)[:top]


def drone_rank(shown: list[dict]) -> int | None:
    """The best rank a candidate on the drone holds among those shown, or None."""
    return next((i for i, c in enumerate(shown, 1) if c["on_target"]), None)


class _Shown:
    """Adapter so EXP-025's `draw_ranked` can draw a dump row."""

    def __init__(self, c: dict):
        self.x, self.y, self.contrast, self.diameter = c["x"], c["y"], c["c"], c["diameter"]


def operating_points(frames: dict[int, list[dict]], labelled: set[int], k: int, top: int):
    """Load and drone hits at every appearance threshold 1..k."""
    n, d = max(len(frames), 1), max(len(labelled), 1)
    print(f"\n  OPERATING POINTS (every threshold from this one pass; drone of {len(labelled)}"
          " labelled frames)")
    print(f"    need   survivors/frame  shown/frame   drone among survivors   drone in top {top}"
          "   #1 / #2 / #3")
    for m in range(1, k + 1):
        alive = shown = any_hit = top_hit = 0
        hist = {}
        for f, cands in frames.items():
            surv = [c for c in cands if c["appearances"] >= m]
            ranked = ranked_survivors(cands, m, top)
            alive += len(surv)
            shown += len(ranked)
            if f in labelled:
                any_hit += any(c["on_target"] for c in surv)
                r = drone_rank(ranked)
                top_hit += r is not None
                hist[r] = hist.get(r, 0) + 1
        ranks = " / ".join(str(hist.get(r, 0)) for r in range(1, top + 1))
        print(f"    {m}/{k}   {alive / n:15.2f}  {shown / n:11.2f}   {any_hit:5d} ({any_hit / d * 100:5.1f}%)"
              f"        {top_hit:5d} ({top_hit / d * 100:5.1f}%)   {ranks}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=CLIP["n_frames"])
    ap.add_argument("--top", type=int, default=3, help="at most this many per frame")
    ap.add_argument("--k", type=int, default=5, help="window length, frames")
    ap.add_argument("--min-appear", type=int, default=4,
                    help="appearances needed in the window; the video is drawn at this one")
    ap.add_argument("--radius", type=float, default=px(14),
                    help="chain step radius, px (14 px at 1440 wide: 9 on analog, as EXP-024)")
    ap.add_argument("--coords", choices=("scene", "image"), default="scene",
                    help="chain camera-compensated (scene) or in raw picture coordinates")
    ap.add_argument("--contrast", type=float, default=6.0, help="EXP-023's threshold")
    ap.add_argument("--cap", type=int, default=4000)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--dump", default=None,
                    help="CSV of every kept candidate with its appearances and rank")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    coords = "" if a.coords == "scene" else "_image"
    tag = f"gate{a.min_appear}of{a.k}{coords}_top{a.top}_{CLIP['name']}_{a.start}_{a.end}"
    out = a.out or os.path.join(CLIP["out"], f"{tag}.mp4")
    dump = a.dump or os.path.splitext(out)[0] + ".csv"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    stage0 = masks.Stage0()
    print("[stage 0] " + ", ".join(f"{k} {v:.2f}%" for k, v in stage0.coverage().items()))
    print(f"[gate] {a.min_appear} of {a.k} frames, chain radius {a.radius:g} px in {a.coords} "
          f"coordinates, then at most "
          f"{a.top} per frame ranked by c, among c >= {a.contrast}")
    boxes = load_boxes()

    first = max(2, a.start - (a.k - 1))           # pre-roll so frame `start` has a full window
    cap = cv2.VideoCapture(CLIP["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, first - 2)
    ok, prev = cap.read()
    if not ok:
        raise SystemExit(f"cannot read frame {first - 1}")
    writer = (None if a.no_video else
              cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H)))

    window: deque = deque(maxlen=a.k)
    frames: dict[int, list[dict]] = {}
    for f in range(first, a.end + 1):
        ok, cur = cap.read()
        if not ok:
            break
        hmat = common.homography(common.prep(prev, 11), common.prep(cur, 11))
        hmat = np.eye(3) if hmat is None else hmat
        valid = stage0.valid(hmat)
        prev = cur
        hits = [s for s in silhouette.detect(cur, valid, threshold=a.contrast, cap=a.cap)
                if s.kept]
        window.append((hmat, np.array([[s.x, s.y] for s in hits], np.float32).reshape(-1, 2)))
        if f < a.start:
            continue

        box = boxes.get(f)
        hits.sort(key=lambda s: s.contrast, reverse=True)
        cands = [dict(frame=f, x=round(s.x, 1), y=round(s.y, 1), c=round(s.contrast, 3),
                      diameter=round(s.diameter, 2),
                      appearances=appearances(s.x, s.y, list(window), a.radius,
                                              a.coords == "scene"),
                      rank_all=i, on_target=int(box is not None and on_drone(s.x, s.y, box)))
                 for i, s in enumerate(hits, 1)]
        frames[f] = cands

        if writer is not None:
            shown = ranked_survivors(cands, a.min_appear, a.top)
            img = cur.copy()
            _draw_background(img, skyline.split(cur, valid), stage0, sky=False)
            draw_ranked(img, [_Shown(c) for c in shown])
            n_alive = sum(c["appearances"] >= a.min_appear for c in cands)
            text_block(img, [(f"frame {f}    top {a.top} by c    {a.min_appear} of {a.k} frames"
                              f" ({a.coords})    c >= {a.contrast} sigma"),
                             f"kept {len(cands)}   persistent {n_alive}   shown {len(shown)}"])
            writer.write(img)

    if writer is not None:
        writer.release()
    cap.release()
    with open(dump, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for cands in frames.values():
            w.writerows(cands)

    labelled = {f for f in frames if f in boxes}
    print(f"\n[gate] {len(frames)} frames -> {out}\n[dump] every kept candidate -> {dump}")
    operating_points(frames, labelled, a.k, a.top)
    print("\n  'drone' = a candidate inside the label box grown by max(10 px, 25%); the detector"
          "\n  fires at a fraction of the airframe's size (EXP-023 correction), so read these as"
          "\n  'a top candidate was in the right place', not 'the drone was detected'.")


if __name__ == "__main__":
    main()
