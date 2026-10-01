"""EXP-025: the sky branch with only its top-N candidates per frame, ranked and labelled.

    SOFA-ANALOG catch_2:
    PYTHONPATH="experiments/exp023_sky_branch/analog_catch_2;experiments/exp025_top3;experiments/exp023_sky_branch;experiments/exp017_motion_first;experiments/exp015_normalised_motion;." py -3.13 -m overlay_top3 --start 441 --end 800

    O4 first_catch:
    PYTHONPATH="experiments/exp023_sky_branch;experiments/exp025_top3;experiments/exp017_motion_first;experiments/exp015_normalised_motion;." py -3.13 -m overlay_top3 --start 650 --end 964

The clip config comes from EXP-023's folders, so outputs land in EXP-023's run folder
unless `--out` is given.

What this run answers
---------------------
EXP-023's clean render draws every candidate at c >= 6 -- 5.5 per frame on analog. A
downstream stage (a tracker, a gimbal, an operator) can take only a few. So: **if each
frame hands over only its N highest-contrast candidates, how often is the drone among
them, and how often is it #1?** Nothing about detection changes. It is EXP-023's
detector at EXP-022 defaults (whole ring, c >= 6, uncertain band scored), with a
per-frame cap applied after the threshold.

Drawing
-------
The clean look: stage-0 mask tints, each of the top N as a RED circle (at least 10 px
across, as in EXP-023) with a `#rank c` tag beside it, and a two-line caption. `#1` is
drawn thicker. Nothing is drawn from the labels.

Scoring caveat, carried from EXP-023's correction: "on the drone" means the candidate
landed in the label box grown by max(10 px, 25%), and the detector fires at ~0.16x the
target's true size on analog. A hit says a candidate was in the right place, not that
the airframe was detected.
"""
from __future__ import annotations

import argparse
import csv
import os

import cv2
import numpy as np

import common
import masks
import silhouette
import skyline
from clipcfg import CLIP
from overlay_sky import KEPT, _draw_background, draw_radius
from overlay_video import load_boxes, on_drone, px, text_block

W, H = CLIP["width"], CLIP["height"]


def top_n(hits, n: int) -> list:
    """The kept candidates of one frame, highest contrast first, at most `n` of them."""
    return sorted((s for s in hits if s.kept), key=lambda s: s.contrast, reverse=True)[:n]


def draw_ranked(img, ranked) -> None:
    """A red circle per candidate and a `#rank c` tag to its upper right."""
    for i, s in enumerate(ranked, 1):
        p, r = (int(s.x), int(s.y)), draw_radius(s.diameter)
        cv2.circle(img, p, r, KEPT, max(1, px(1)) + (1 if i == 1 else 0))
        tag = f"#{i} {s.contrast:.1f}"
        org = (p[0] + r + 3, p[1] - r - 2)
        for colour, thick in (((0, 0, 0), 3), ((255, 255, 255), 1)):   # outlined, readable on sky
            cv2.putText(img, tag, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, thick,
                        cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=CLIP["n_frames"])
    ap.add_argument("--top", type=int, default=3, help="at most this many per frame")
    ap.add_argument("--contrast", type=float, default=6.0,
                    help="EXP-023's threshold; ranking happens only among candidates above it")
    ap.add_argument("--cap", type=int, default=4000)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--dump", default=None, help="CSV of every ranked candidate")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tag = f"top{a.top}_{CLIP['name']}_{a.start}_{a.end}"
    out = a.out or os.path.join(CLIP["out"], f"{tag}.mp4")
    dump = a.dump or os.path.splitext(out)[0] + ".csv"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    stage0 = masks.Stage0()
    print("[stage 0] " + ", ".join(f"{k} {v:.2f}%" for k, v in stage0.coverage().items()))
    print(f"[top] at most {a.top} per frame, ranked by c, among c >= {a.contrast} "
          f"(whole ring, uncertain band scored -- EXP-022 defaults)")
    boxes = load_boxes()

    cap = cv2.VideoCapture(CLIP["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.start - 2)
    ok, prev = cap.read()
    if not ok:
        raise SystemExit(f"cannot read frame {a.start - 1}")
    writer = (None if a.no_video else
              cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H)))

    rows, n, kept_total, shown_total = [], 0, 0, 0
    drone_frames, rank_hist, drone_kept_any = 0, {}, 0
    for f in range(a.start, a.end + 1):
        ok, cur = cap.read()
        if not ok:
            break
        hmat = common.homography(common.prep(prev, 11), common.prep(cur, 11))
        valid = stage0.valid(hmat)
        prev = cur
        sl = skyline.split(cur, valid)
        hits = silhouette.detect(cur, valid, threshold=a.contrast, cap=a.cap)
        ranked = top_n(hits, a.top)
        n_kept = sum(1 for s in hits if s.kept)
        n += 1
        kept_total += n_kept
        shown_total += len(ranked)

        box = boxes.get(f)
        best = None
        for i, s in enumerate(ranked, 1):
            hit = box is not None and on_drone(s.x, s.y, box)
            best = i if hit and best is None else best
            rows.append(dict(frame=f, rank=i, x=round(s.x, 1), y=round(s.y, 1),
                             c=round(s.contrast, 3), diameter=round(s.diameter, 2),
                             on_target=int(hit)))
        if box is not None:
            drone_frames += 1
            rank_hist[best] = rank_hist.get(best, 0) + 1
            drone_kept_any += 1 if any(s.kept and on_drone(s.x, s.y, box) for s in hits) else 0

        if writer is not None:
            img = cur.copy()
            _draw_background(img, sl, stage0, sky=False)
            draw_ranked(img, ranked)
            text_block(img, [f"frame {f}    top {a.top} by c    c >= {a.contrast} sigma",
                             f"kept {n_kept}   shown {len(ranked)}"])
            writer.write(img)

    if writer is not None:
        writer.release()
    cap.release()
    with open(dump, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame", "rank", "x", "y", "c", "diameter",
                                           "on_target"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n[top] {n} frames -> {out}\n[dump] {len(rows)} ranked candidates -> {dump}\n")
    print(f"  kept at c >= {a.contrast:<4g}         {kept_total / max(n, 1):6.2f}/frame")
    print(f"  shown (top {a.top})              {shown_total / max(n, 1):6.2f}/frame")
    d = max(drone_frames, 1)
    print(f"\n  labelled drone frames          {drone_frames}")
    print(f"  drone among ALL kept           {drone_kept_any:4d}  ({drone_kept_any / d * 100:5.1f}%)"
          "   <- EXP-023's figure, the ceiling for any cap")
    cum = 0
    for r in range(1, a.top + 1):
        cum += rank_hist.get(r, 0)
        print(f"  drone is #{r}                    {rank_hist.get(r, 0):4d}"
              f"   within top {r}: {cum:4d}  ({cum / d * 100:5.1f}%)")
    print(f"  drone not in top {a.top}             {rank_hist.get(None, 0):4d}")
    print("\n  'drone' = a candidate inside the label box grown by max(10 px, 25%); the detector"
          "\n  fires at a fraction of the airframe's size (EXP-023 correction), so read these as"
          "\n  'a top candidate was in the right place', not 'the drone was detected'.")


if __name__ == "__main__":
    main()
