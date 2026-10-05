"""EXP-029: the sky branch, chained into tracks, then held to a moving factor.

    PYTHONPATH="experiments/exp029_moving_factor/analog_catch_2;experiments/exp029_moving_factor;experiments/exp023_sky_branch;experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
        py -3.13 -m overlay_moving --start 441 --end 800

What this pipeline is
---------------------
EXP-024 measured that the 5-frame motion window is beaten by EXP-023's 2-frame sky branch
at every matched load, because the motion front-end only makes the target a candidate in
77.7% of labelled frames against the sky branch's 99.6%. So this runs the **sky branch**
and adds the two things the window was for, without the window:

  1. **Association by chained peaks**, `src/algo/kinematics.KinematicTracker` (EXP-028).
     EXP-024 measured Lucas-Kanade dying at a median of 0 usable steps on analog, so
     chaining candidate *positions* replaces tracking pixels. It needs no texture, and a
     duplicated frame (1 in 6 of every analog clip) contributes no candidate rather than
     killing a tracker.
  2. **The moving factor**, the limit from below: a track must have travelled far enough
     against the static scene. The tracker carries each past sighting forward through
     every frame's homography, so the travel it reports is already ego-compensated.

`moving.judge` then adds the **epipolar direction test** on that same displacement, with
`--mode` deciding how much authority it gets. See `moving.py` for why magnitude leads and
direction may only veto.

Why the tracker's own `min_move` is left off here
-------------------------------------------------
`KinematicTracker(min_move=...)` implements the magnitude test and is the shipping form of
it. This experiment passes **0** and judges in `moving.judge` instead, because the report
has to count `still` separately from `parallax` and from `unjudged` -- which means it must
see every confirmed track rather than a pre-filtered list. One decision point, four
outcomes, nothing applied twice.

The epipole here is a single pair's
-----------------------------------
EXP-019 measured the per-pair epipole hopping 99-127 px between frames, which is why
EXP-017's window pools k pairs' votes. This renders one frame at a time, so the epipole is
this pair's alone and is the weakest part of the direction half. `agreement` is printed for
exactly that reason: read it before reading any `parallax` count, and if it sits near the
chance floor then the direction test is rejecting at random and `--mode off` is the honest
setting.

Drawing
-------
The clean EXP-023 look: stage-0 mask tints, every kept track a RED circle at least 10 px
across, a two-line caption. `--show-rejected` adds the overruled ones in grey so a video
can show what the factor removed.
"""
from __future__ import annotations

import argparse
import csv
import io
import os

import cv2
import numpy as np

import common
import geometry
import masks
import moving
import silhouette
from clipcfg import CLIP
from overlay_video import LAYER_COLOUR, load_boxes, on_drone, px, text_block, tint

from src.algo.kinematics import OK, KinematicTracker, SpeedLimit

W, H = CLIP["width"], CLIP["height"]
S = W / 1440.0
KEPT_COLOUR = (0, 0, 255)      # the clean look's red
DROPPED = (110, 110, 110)
MIN_DRAW_DIAMETER = 10

FIELDS = ["frame", "track", "x", "y", "c", "diameter", "moved", "span", "sin",
          "outcome", "reason", "on_target"]


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=CLIP["n_frames"])
    ap.add_argument("--contrast", type=float, default=6.0,
                    help="sky-branch threshold for a STRONG candidate (EXP-023's c)")
    ap.add_argument("--c-keep", type=float, default=3.0,
                    help="a confirmed track may be continued by a candidate at or above "
                         "this, weak or strong (EXP-028's --c-keep)")
    ap.add_argument("--min-move", type=float, default=8.0,
                    help="the moving factor, px of ego-compensated travel over the "
                         "window. Default 8 sits between the background field's p90 of "
                         "5.60 and the target's p10 of 13.9 (EXP-024, 2026-10-04)")
    ap.add_argument("--move-window", type=int, default=5,
                    help="frames the travel is measured across")
    ap.add_argument("--mode", default="veto", choices=moving.MODES,
                    help="how much authority the epipolar direction test gets: off "
                         "(magnitude alone), veto (direction may only reject), require "
                         "(direction must affirm -- EXP-021's failure mode, for measuring)")
    ap.add_argument("--sin-threshold", type=float, default=0.35)
    ap.add_argument("--confirm", type=int, default=2)
    ap.add_argument("--max-coast", type=int, default=5)
    ap.add_argument("--decay", type=float, default=0.8)
    ap.add_argument("--v-max", type=float, default=40.0)
    ap.add_argument("--min-range", type=float, default=10.0)
    ap.add_argument("--hfov", type=float, default=130.0)
    ap.add_argument("--ceiling", type=float, default=25.0)
    ap.add_argument("--top", type=int, default=0,
                    help="cap on drawn tracks per frame, 0 for no cap")
    ap.add_argument("--show-rejected", action="store_true",
                    help="also draw what the factor overruled, in grey")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--dump", default=None, help="CSV of every judged track per frame")
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def sky_candidates(cur, valid, threshold: float) -> list[silhouette.Silhouette]:
    """Every sky-branch candidate at or above the weakest contrast we might continue on.

    Detected once at the lower threshold; `--contrast` then decides which are *strong*
    enough to start a track. Running `detect` twice would cost a second DoG pyramid.
    """
    return [s for s in silhouette.detect(cur, valid, threshold=threshold) if s.kept]


def epipole_for(prev_grey, cur_grey, hmat, valid):
    """This pair's focus of expansion, from the background residual field."""
    tp, tc = geometry.track_grid(prev_grey, cur_grey, valid)
    gx, gmu = geometry.residual_field(tp, tc, hmat)
    return geometry.estimate_epipole(gx, gmu), len(gx)


def main() -> None:
    a = parse()
    if a.min_move < 0:
        raise SystemExit("--min-move cannot be negative")
    out = a.out or os.path.join(
        CLIP["out"], f"moving_m{a.min_move:g}_w{a.move_window}_{a.mode}_"
                     f"{CLIP['name']}_{a.start}_{a.end}.mp4")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    stage0 = masks.Stage0()
    print("[stage 0] " + ", ".join(f"{k} {v:.2f}%" for k, v in stage0.coverage().items()))
    limit = SpeedLimit(width_px=W, fps=CLIP["fps"], v_max_ms=a.v_max,
                       min_range_m=a.min_range, hfov_deg=a.hfov, ceiling_px=a.ceiling)
    print(f"[sky] strong at c >= {a.contrast}, continued at c >= {a.c_keep}")
    print(f"[chain] speed limit {limit.px_per_frame:.1f} px/frame, confirm {a.confirm}, "
          f"coast {a.max_coast}")
    print(f"[factor] min move {a.min_move:g} px over {a.move_window} frames, "
          f"direction mode {a.mode}")

    # min_move stays 0 here on purpose: the report must see every confirmed track so it
    # can separate `still` from `parallax`. moving.judge is the single decision point.
    tracker = KinematicTracker(limit, a.confirm, a.max_coast, a.decay,
                               min_move=0.0, move_window=a.move_window)
    boxes = load_boxes()

    cap = cv2.VideoCapture(CLIP["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.start - 2)
    ok, prev = cap.read()
    if not ok:
        raise SystemExit(f"cannot read frame {a.start - 1}")
    writer = (None if a.no_video else
              cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H)))

    st = dict(n=0, cands=0, confirmed=0, outcome={k: 0 for k in
                                                  (moving.STILL, moving.PARALLAX,
                                                   moving.UNJUDGED, moving.MOVING)},
              kept=0, kept_fa=0, drone_frames=set(), drone_cand_frames=set(),
              agree=[], epi_ok=0, moved_on=[], moved_off=[], grid_n=[])
    rows: list[dict] = []

    for f in range(a.start, a.end + 1):
        ok, cur = cap.read()
        if not ok:
            break
        gp, gc = common.prep(prev, 11), common.prep(cur, 11)
        hmat = common.homography(gp, gc)
        valid = stage0.valid(hmat)

        hits = sky_candidates(cur, valid, min(a.contrast, a.c_keep))
        xy = np.array([[s.x, s.y] for s in hits], float).reshape(-1, 2)
        score = np.array([s.contrast for s in hits], float)
        strong = score >= a.contrast
        res = tracker.step(f, xy, score, strong, np.linalg.inv(hmat))

        ep, n_grid = (None, 0)
        if a.mode != "off":
            ep, n_grid = epipole_for(gp, gc, hmat, valid)
            st["agree"].append(ep.inlier_fraction)
            st["epi_ok"] += 1 if ep.reliable else 0
            st["grid_n"].append(n_grid)

        box = boxes.get(f)
        img = (np.zeros((H, W, 3), np.uint8) if a.no_video else cur.copy())
        if not a.no_video:
            for name, m in stage0.layers.items():
                tint(img, m, LAYER_COLOUR[name], 0.35)

        judged = []
        for i, s in enumerate(hits):
            on = bool(box) and on_drone(s.x, s.y, box)
            if on:
                st["drone_cand_frames"].add(f)
            if res.reason[i] != OK:          # not on a confirmed track: nothing to judge
                continue
            st["confirmed"] += 1
            v = moving.judge(
                res.moved[i], int(res.move_span[i]), res.move_vec[i], (s.x, s.y),
                ep.point if ep is not None else np.zeros(2),
                min_move=a.min_move, mode=a.mode,
                epipole_reliable=bool(ep.reliable) if ep is not None else False,
                degenerate=bool(ep.degenerate_for(s.x, s.y)) if ep is not None else False,
                sin_threshold=a.sin_threshold, min_span=a.move_window)
            st["outcome"][v.outcome] += 1
            if np.isfinite(v.moved):
                (st["moved_on"] if on else st["moved_off"]).append(v.moved)
            judged.append((s, v, int(res.track_id[i]), on))
            if a.dump is not None:
                rows.append(dict(frame=f, track=int(res.track_id[i]),
                                 x=round(s.x, 1), y=round(s.y, 1),
                                 c=round(s.contrast, 3), diameter=round(s.diameter, 2),
                                 moved=("" if not np.isfinite(v.moved)
                                        else round(v.moved, 2)),
                                 span=v.span, sin=("" if not np.isfinite(v.sin_angle)
                                                   else round(v.sin_angle, 3)),
                                 outcome=v.outcome, reason=v.reason, on_target=int(on)))

        st["cands"] += len(hits)
        kept = [t for t in judged if t[1].kept]
        kept.sort(key=lambda t: -t[0].contrast)
        if a.top:
            kept = kept[:a.top]
        st["kept"] += len(kept)
        st["kept_fa"] += sum(1 for t in kept if not t[3])
        if any(t[3] for t in kept):
            st["drone_frames"].add(f)

        if not a.no_video:
            if a.show_rejected:
                for s, v, _tid, _on in judged:
                    if not v.kept:
                        cv2.circle(img, (int(s.x), int(s.y)), MIN_DRAW_DIAMETER // 2,
                                   DROPPED, max(1, px(1)))
            for s, _v, _tid, _on in kept:
                r = max(MIN_DRAW_DIAMETER, s.diameter) / 2.0
                cv2.circle(img, (int(s.x), int(s.y)), int(round(r)), KEPT_COLOUR,
                           max(1, px(1)))
            text_block(img, [
                f"frame {f}   sky c>={a.contrast:g}   move >= {a.min_move:g} px"
                f"/{a.move_window}f   dir {a.mode}",
                f"candidates {len(hits)}   confirmed {len(judged)}   kept {len(kept)}",
            ])
            writer.write(img)
        st["n"] += 1
        prev = cur

    if writer is not None:
        writer.release()
    cap.release()
    if a.dump is not None:
        with io.open(a.dump, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"[dump] {len(rows)} judged tracks -> {a.dump}")
    report(st, a, out, sum(1 for k in boxes if a.start <= k <= a.end))


def pctl(v, q):
    return float(np.percentile(v, q)) if len(v) else float("nan")


def report(st, a, out, n_labelled: int) -> None:
    n = max(st["n"], 1)
    print()
    print(f"[moving] {n} frames -> {out}")
    print()
    print("LOAD")
    print(f"  sky candidates                {st['cands'] / n:8.2f}/frame")
    print(f"  on a confirmed track          {st['confirmed'] / n:8.2f}/frame")
    print(f"  KEPT after the factor         {st['kept'] / n:8.2f}/frame")
    print(f"  false alarms kept             {st['kept_fa'] / n:8.2f}/frame "
          f"({st['kept_fa']} total)")
    print()
    print(f"WHAT THE FACTOR DID to {sum(st['outcome'].values())} judged tracks")
    tot = max(sum(st["outcome"].values()), 1)
    for k in (moving.STILL, moving.PARALLAX, moving.UNJUDGED, moving.MOVING):
        print(f"  {k:<10} {st['outcome'][k]:7d}   {st['outcome'][k] / tot * 100:5.1f}%")
    print()
    print("THE MEASUREMENT ITSELF -- ego-compensated travel over the window")
    for name, key in (("on the drone", "moved_on"), ("everything else", "moved_off")):
        v = st[key]
        print(f"  {name:<16} n={len(v):6d}  p10 {pctl(v, 10):7.2f}  "
              f"median {pctl(v, 50):7.2f}  p90 {pctl(v, 90):7.2f} px")
    print("  (EXP-024 from the labels: target p10 13.85, median 28.38; "
          "background median 3.18, p90 5.60)")
    if a.mode != "off":
        print()
        print("EPIPOLE, this pair only -- the weakest part of the direction half")
        print(f"  reliable in {st['epi_ok']} of {n} frames, "
              f"fitted on a median {pctl(st['grid_n'], 50):.0f} background points")
        print(f"  ** agreement with the background it was fitted to: "
              f"{pctl(st['agree'], 50) * 100:.1f}% **")
        print("     Near the chance floor (~23% on analog, EXP-021) the direction test is")
        print("     rejecting at random and --mode off is the honest setting.")
    print()
    seen = len(st["drone_cand_frames"])
    print(f"DRONE, out of {n_labelled} labelled frames on the clip")
    print(f"  a sky candidate in            {seen:4d}")
    print(f"  KEPT in                       {len(st['drone_frames']):4d}"
          f"   ({len(st['drone_frames']) / max(n_labelled, 1) * 100:.1f}% of labelled, "
          f"{len(st['drone_frames']) / max(seen, 1) * 100:.1f}% of those it was a candidate in)")
    print("  Recall is quoted on the LABELLED denominator; EXP-024's 174 is a motion")
    print("  front-end ceiling, not a denominator (see docs/experiments.md EXP-024).")


if __name__ == "__main__":
    main()
