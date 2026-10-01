"""EXP-024: the same k-frame renderer, with the window length as the question.

Forked from EXP-017 so that experiment stays frozen at the numbers EXP-021/022 cite.
Two differences, both recorded in this folder's README: a `--direction` flag whose
default is OFF, and a report that prints every `--min-appear` operating point from one
pass.

    O4 first_catch:
    PYTHONPATH="experiments/exp024_window_length/o4_first_catch;experiments/exp024_window_length;experiments/exp017_motion_first;experiments/exp015_normalised_motion;." py -3.13 -m overlay_window --start 650 --end 964 --appear-radius 14 --k 5 --min-appear 4

    SOFA-ANALOG catch_2:
    PYTHONPATH="experiments/exp024_window_length/analog_catch_2;experiments/exp024_window_length;experiments/exp017_motion_first;experiments/exp015_normalised_motion;." py -3.13 -m overlay_window --start 441 --end 800 --appear-radius 9 --k 5 --min-appear 4

`overlay_video.py` stays as the single-pair baseline. Keeping both is the point: the
window is only worth its cost if the same span comes out better, and "better" has to be
read against the same chance floor.

What is different from the single-pair render
---------------------------------------------
  * **One epipole per window**, fitted from all k pairs' votes pooled into the newest
    frame, instead of one per pair. EXP-019 measured the per-pair epipole hopping 99-127 px
    between frames, which is the largest single thing limiting the direction test.
  * **The accumulated residual `Sum mu`** is tested, not one step's. A static point's
    residuals lie along its epipolar line every step, so they add coherently while noise
    adds as sqrt(k).
  * **Persistence**: a candidate must be a candidate in at least `--min-appear` of the k
    frames. Checked before the direction test, because the direction of a one-frame blob
    is not a measurement.

Colours
-------
The default render is the CLEAN look, matching EXP-023's `clean_catch_2_441_800.mp4`: the
stage-0 mask tints, every kept candidate (survives, or unjudged with `--direction`) as a RED
circle at least 10 px across, and a two-line caption. Nothing else -- no stage-1 tint or
horizon, no epipole, no flash dots, no arrows, no label box. `--diagnostic` restores the full
drawing below. Drawing only: the report is identical either way.

  * DARK GREY dot  -- flash: failed persistence, dropped before being judged
  * RED dot        -- rejected: accumulated residual is along the line to the epipole
  * GREY ring      -- unjudged: near the FOE, or the window carries no reliable epipole
  * GREEN dot      -- survives, with its accumulated displacement drawn from it
"""
from __future__ import annotations

import argparse
import os
from collections import deque

import cv2
import numpy as np

import budget
import common
import geometry
import masks
import skyline
import window as win
from clipcfg import CLIP
from overlay_video import (LAYER_COLOUR, TINT, draw_epipole, load_boxes, on_drone, px,
                           text_block, tint)

W, H = CLIP["width"], CLIP["height"]
S = W / 1440.0
FLASH = (90, 90, 90)
REJECT, SURVIVE, UNJUDGED = (60, 60, 235), (60, 230, 60), (170, 170, 170)
COLOUR = {"flash": FLASH, "rejected": REJECT, "unjudged": UNJUDGED, "survives": SURVIVE}
KEPT = (0, 0, 255)       # clean render: same red as overlay_sky's kept circles
MIN_DRAW_DIAMETER = 10   # px; a track has no scale, so every kept circle is drawn at this


def make_pair(prev_bgr, cur_bgr, frame: int, stage0: masks.Stage0, tau: float) -> win.Pair:
    """Everything one frame pair contributes to the window. Computed once, used k times."""
    gp, gc = common.prep(prev_bgr, 11), common.prep(cur_bgr, 11)
    hmat = common.homography(gp, gc)
    valid = stage0.valid(hmat)
    tp, tc = geometry.track_grid(gp, gc, valid)
    gx, gmu = geometry.residual_field(tp, tc, hmat)
    cands = budget.candidates(common.maps(prev_bgr, cur_bgr, hmat)["win_b5_e4"], valid, tau)
    return win.Pair(frame, gc, gp, hmat, cands, gx, gmu, geometry.estimate_epipole(gx, gmu))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=CLIP["n_frames"])
    ap.add_argument("--k", type=int, default=win.DEFAULT_K, help="frames in the window")
    ap.add_argument("--min-appear", type=int, default=win.DEFAULT_MIN_APPEAR)
    ap.add_argument("--appear-radius", type=float, default=win.APPEAR_RADIUS,
                    help="how near a candidate peak must be to count as the same thing")
    ap.add_argument("--fb-max", type=float, default=None,
                    help="forward-backward LK gate in px. The default is px(1.0), which "
                         "on a noisy CVBS capture rejects nearly every track.")
    ap.add_argument("--direction", action="store_true",
                    help="also apply the epipolar direction test. OFF by default: "
                         "EXP-021 measured it costing 33 points of recall while "
                         "rejecting at chance, and EXP-022/023 default it off too.")
    ap.add_argument("--tau", type=float, default=1.661)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--diagnostic", action="store_true",
                    help="draw everything: stage-1 tint and horizon, epipole, flash dots, "
                         "arrows, the label box. OFF by default -- the default is the "
                         "clean look (red kept circles only), as in EXP-023's clean renders.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = a.out or os.path.join(
        CLIP["out"], f"window{a.k}_{CLIP['name']}_{a.start}_{a.end}.mp4")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    if a.fb_max is not None:
        geometry.FB_MAX = a.fb_max
    stage0 = masks.Stage0()
    print(f"[stage 0] " + ", ".join(f"{k} {v:.2f}%" for k, v in stage0.coverage().items()))
    print(f"[window] k={a.k} frames, need {a.min_appear} appearances, "
          f"appear radius {a.appear_radius:.1f} px, tau={a.tau}, "
          f"fb gate {geometry.FB_MAX:.2f} px")
    boxes = load_boxes()

    cap = cv2.VideoCapture(CLIP["video"])
    # k pairs need k + 1 frames, and the first pair's previous frame is one older again.
    first = max(1, a.start - a.k)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first - 1)
    ok, prev = cap.read()
    if not ok:
        raise SystemExit(f"cannot read frame {first}")

    writer = (None if a.no_video else
              cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H)))
    pairs: deque[win.Pair] = deque(maxlen=a.k)

    st = dict(n=0, counts={"flash": 0, "rejected": 0, "unjudged": 0, "survives": 0},
              seeds=0, epi_ok=0, epi_jump=[], epi_prev=None, spread=[],
              surv_sky=0, surv_scene=0, surv_unc=0, grid_n=[], agree=[],
              steps_hist=[], drone_v={"flash": 0, "rejected": 0, "unjudged": 0,
                                      "survives": 0},
              drone_seen=0, drone_kept=0, drone_flash=0, drone_red=[], appear_hist=[],
              drone_appear=[])

    for f in range(first + 1, a.end + 1):
        ok, cur = cap.read()
        if not ok:
            break
        pairs.append(make_pair(prev, cur, f, stage0, a.tau))
        prev = cur
        if f < a.start or len(pairs) < a.k:
            continue

        plist = list(pairs)
        valid = stage0.valid(plist[-1].hmat)
        gx, gmu, gsteps = win.accumulate_grid(plist, valid)
        fit = win.fit_window(plist, gx, gmu)
        tracks = win.build_tracks(plist, fit, appear_radius=a.appear_radius)
        sl = skyline.split(cur, valid)

        img = np.zeros((H, W, 3), np.uint8) if a.no_video else cur.copy()
        if a.diagnostic:
            tint(img, sl.sky, TINT["sky"], 0.22)
            tint(img, sl.uncertain, TINT["uncertain"], 0.30)
        for name, m in stage0.layers.items():
            tint(img, m, LAYER_COLOUR[name], 0.35)
        if a.diagnostic:
            xs = np.nonzero(sl.horizon >= 0)[0]
            for x in xs[::4]:
                cv2.circle(img, (int(x), int(sl.horizon[x])), px(2), (40, 220, 230), -1)
            draw_epipole(img, fit.epipole)

        per = {"flash": 0, "rejected": 0, "unjudged": 0, "survives": 0}
        for t in tracks:
            v = (t.verdict(a.min_appear) if a.direction else
                 ("flash" if t.appearances < a.min_appear else "survives"))
            per[v] += 1
            st["counts"][v] += 1
            st["appear_hist"].append(t.appearances)
            st["steps_hist"].append(t.steps)
            p = (int(t.x), int(t.y))
            if not a.diagnostic:
                if v in ("survives", "unjudged"):
                    cv2.circle(img, p, MIN_DRAW_DIAMETER // 2, KEPT, max(1, px(1)))
            elif v == "flash":
                cv2.circle(img, p, px(3), FLASH, -1)
            elif v == "unjudged":
                cv2.circle(img, p, px(6), UNJUDGED, max(1, px(2)))
            elif v == "rejected":
                cv2.circle(img, p, px(5), REJECT, -1)
            else:
                cv2.circle(img, p, px(6), SURVIVE, -1)
                g = 4.0
                cv2.arrowedLine(img, p,
                                (int(t.x + t.total_mu[0] * g), int(t.y + t.total_mu[1] * g)),
                                SURVIVE, max(1, px(2)), tipLength=0.3)
            if v == "survives":
                lab = sl.label(t.x, t.y)
                st[f"surv_{'unc' if lab == 'uncertain' else lab}"] += 1
        st["seeds"] += len(tracks)

        box = boxes.get(f)
        if box:
            bx, by, bw, bh = [int(v) for v in box]
            if a.diagnostic:
                cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (60, 230, 60), px(2))
            mine = [t for t in tracks if on_drone(t.x, t.y, box)]
            if mine:
                st["drone_seen"] += 1
                vs = [(t.verdict(a.min_appear) if a.direction else
                       ("flash" if t.appearances < a.min_appear else "survives"))
                      for t in mine]
                best = ("survives" if "survives" in vs else
                        "unjudged" if "unjudged" in vs else
                        "rejected" if "rejected" in vs else "flash")
                st["drone_v"][best] += 1
                st["drone_kept"] += 1 if best in ("survives", "unjudged") else 0
                if best == "rejected":
                    st["drone_red"].append(f)
                if best == "flash":
                    st["drone_flash"] += 1
                appears = max(t.appearances for t in mine)
                st["drone_appear"].append(appears)
                txt = f"drone: {best} ({appears}/{a.k} frames)"
            else:
                txt = "drone: not a candidate"
            if a.diagnostic:
                text_block(img, [txt], x0=bx, y0=max(int(20 * S), by - int(10 * S)),
                           scale=0.55)

        st["n"] += 1
        st["grid_n"].append(len(gx))
        st["agree"].append(fit.epipole.inlier_fraction)
        st["epi_ok"] += 1 if fit.reliable else 0
        if np.isfinite(fit.epipole_spread):
            st["spread"].append(fit.epipole_spread)
        if fit.reliable:
            if st["epi_prev"] is not None:
                st["epi_jump"].append(float(np.hypot(fit.epipole.point[0] - st["epi_prev"][0],
                                                     fit.epipole.point[1] - st["epi_prev"][1])))
            st["epi_prev"] = fit.epipole.point.copy()

        if not a.diagnostic:
            text_block(img, [
                f"frame {f}   window k={a.k}, need {a.min_appear}",
                f"candidates {len(tracks)}   kept {per['survives'] + per['unjudged']}",
            ])
        else:
            text_block(img, [
            f"frame {f}   window k={a.k}, need {a.min_appear}   tau {a.tau}",
            f"stage 1: sky {sl.sky_fraction * 100:.1f}%  uncertain {sl.uncertain.mean() * 100:.1f}%",
            f"stage 2: epipole {'OK' if fit.reliable else 'UNRELIABLE'}"
            f"  agree {fit.epipole.inlier_fraction:.2f}"
            + ("  <-- AT CHANCE (0.23): red dots are not meaningful"
               if fit.epipole.inlier_fraction < 0.28 else
               f"  per-pair spread {fit.epipole_spread:.0f}px"),
            f"seeds {len(tracks)}:  flash {per['flash']}  rejected {per['rejected']}"
            f"  unjudged {per['unjudged']}  SURVIVE {per['survives']}",
            ])
        if writer is not None:
            writer.write(img)

    if writer is not None:
        writer.release()
    cap.release()
    report(st, a, out)


def report(st, a, out) -> None:
    n = st["n"]
    if not n:
        raise SystemExit("no frames rendered -- span shorter than the window?")
    c = st["counts"]
    tot = max(st["seeds"], 1)
    chance = 2.0 * np.arcsin(0.35) / np.pi
    print(f"\n[window] {n} frames -> {out}\n")

    print(f"PERSISTENCE (k={a.k}, need {a.min_appear})")
    hist = np.asarray(st["appear_hist"])
    for v in range(a.k + 1):
        share = float((hist == v).mean()) * 100 if len(hist) else 0.0
        print(f"  appeared in {v}/{a.k} frames: {share:5.1f}%")
    print(f"  -> DROPPED AS FLASH: {c['flash'] / tot * 100:5.1f}% of {tot} seeds")
    sh = np.asarray(st["steps_hist"])
    if len(sh):
        print(f"  usable residual steps per seed: median {np.median(sh):.0f} of {a.k}, "
              f"{float((sh >= a.k - 1).mean()) * 100:.0f}% got {a.k - 1}+")

    # The whole min-appear curve from ONE pass. `appearances` does not depend on the
    # threshold -- `Track.verdict` only compares against it -- so every operating point of
    # this window length is already in these histograms. Sweeping by re-rendering would
    # cost k times as much for the same table, and would invite reading thresholds off
    # different spans.
    if len(hist) and st["drone_appear"]:
        da = np.asarray(st["drone_appear"])
        seen = st["drone_seen"]
        print()
        print("  OPERATING POINTS (persistence alone, every threshold from this one pass)")
        print(f"  {'need':>6} {'load/frame':>11} {'drone kept':>14} {'seeds held':>11}")
        for m in range(1, a.k + 1):
            share = float((hist >= m).mean())
            kept = int((da >= m).sum())
            print(f"  {str(m) + '/' + str(a.k):>6} {share * tot / n:>11.1f} "
                  f"{kept:>6} ({kept / seen * 100:3.0f}%) {share * 100:>10.1f}%")

    print(f"\nWHAT SURVIVES, per frame (seeds {tot / n:.1f}/frame)")
    for kk in ("flash", "rejected", "unjudged", "survives"):
        print(f"  {kk:9s} {c[kk] / n:7.1f}/frame   {c[kk] / tot * 100:5.1f}%")
    judged = (c["rejected"] + c["survives"]) if a.direction else 0
    if judged:
        got = c["rejected"] / judged
        print(f"\n  rejection among JUDGED (persistent) candidates  {got * 100:5.1f}%")
        print(f"  the same figure for random directions          {chance * 100:5.1f}%")
        print(f"  -> lift over chance  x{got / chance:.2f}")
    print(f"  surviving load {c['survives'] / n:.1f}/frame  "
          f"(single-pair baseline is in EXP-019)")

    # Persistence and the direction test are reported apart because they are not equally
    # well supported. Persistence rests only on "a real thing is visible in consecutive
    # frames". The direction test rests on mu = gamma * (e - x), and the epipole agreement
    # printed below is what says whether that holds on this footage at all.
    persist = c["rejected"] + c["unjudged"] + c["survives"]
    print()
    print(f"  PERSISTENCE ALONE (no direction test): {persist / n:.1f}/frame kept")
    print(f"  persistence + direction:               {c['survives'] / n:.1f}/frame kept")

    print(f"\nEPIPOLE, pooled over the window")
    print(f"  reliable in {st['epi_ok']} of {n} windows")
    print(f"  fitted on {np.mean(st['grid_n']):.0f} accumulated background points/window")
    print(f"  ** agreement with the background it was fitted to: "
          f"{np.median(st['agree']) * 100:.1f}% (chance {chance * 100:.1f}%) **")
    print("     If this is near chance the epipole explains nothing and every rejection")
    print("     below is an accident. It is the first number to read on this report.")
    if st["epi_jump"]:
        j = np.asarray(st["epi_jump"])
        print(f"  frame-to-frame jump  median {np.median(j):7.1f} px   "
              f"p90 {np.percentile(j, 90):7.1f} px")
        print(f"    (EXP-019 single-pair: 126.8 px O4 / 98.9 px analog)")
    if st["spread"]:
        print(f"  per-pair epipoles sit a median {np.median(st['spread']):.0f} px from the "
              f"pooled one")
        print("    (large means the translation direction turned inside the window, so")
        print("     pooling is averaging across a manoeuvre -- read the lift with that in mind)")

    s = max(c["survives"], 1)
    print(f"\nWHERE THE SURVIVORS SIT")
    print(f"  sky {st['surv_sky'] / s * 100:5.1f}%   scene {st['surv_scene'] / s * 100:5.1f}%"
          f"   uncertain {st['surv_unc'] / s * 100:5.1f}%   of {c['survives']} survivors")

    if st["drone_seen"]:
        dv = st["drone_v"]
        keep_p = st["drone_seen"] - dv["flash"]
        print()
        print(f"DRONE under PERSISTENCE ALONE: kept in {keep_p} of {st['drone_seen']} "
              f"({keep_p / st['drone_seen'] * 100:.0f}%)")
        print(f"\nDRONE: a candidate in {st['drone_seen']} labelled frames, "
              f"kept in {st['drone_kept']} ({st['drone_kept'] / st['drone_seen'] * 100:.0f}%)")
        if st["drone_flash"]:
            print(f"  LOST TO THE PERSISTENCE RULE in {st['drone_flash']} frames -- "
                  f"the target itself failed {a.min_appear}/{a.k}")
        print("  drone verdicts: " +
              ", ".join(f"{k} {v}" for k, v in st["drone_v"].items() if v))
        if st["drone_red"]:
            print(f"  rejected as parallax in {len(st['drone_red'])} frames, "
                  f"first at {st['drone_red'][0]}")


if __name__ == "__main__":
    main()
