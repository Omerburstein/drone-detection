"""Stage 0, stage 1 and stage 2 on one picture, plus a stage-1 health report over the span.

    PYTHONPATH="experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
        py -3.13 -m overlay_video --start 650 --end 964

    PYTHONPATH="runs/sofa_analog/exp017_motion_first;experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
        py -3.13 -m overlay_video --start 441 --end 800

Why the two stages are drawn together
-------------------------------------
`debug_video.py` draws stage 0 and stage 1 only, which was right while stage 2 did not
exist. It does now, and the thing worth looking at is the **interaction**: EXP-016's
finding was that 57.5% of analog's false tracks confirm at the sky / tree line boundary,
so what matters is whether the candidates stage 2 fails to reject are the ones sitting on
stage 1's boundary. One of those facts on its own does not say that; the two on one frame
do.

What you are looking at
-----------------------
Stage 0/1, as before:

  * blue tint      -- called sky            * yellow tint -- uncertain band
  * yellow dots    -- the horizon per column  * red/magenta/orange tint -- stage-0 masks
  * green box      -- the labelled drone

Stage 2, new:

  * grey arrows    -- the background residual field after plane compensation. These are
                      what locates the epipole; they should fan out from one point.
  * cyan cross     -- the epipole (focus of expansion). Its dashed circle is
                      `DEGENERATE_RADIUS`, inside which the direction test refuses.
                      Drawn at the frame edge with an arrow when it falls outside.
  * RED dot        -- candidate REJECTED: its residual is along the line to the epipole,
                      so static parallax explains it. This is the load stage 2 removes.
  * GREEN dot      -- candidate SURVIVES: off-epipolar, parallax cannot explain it.
  * GREY ring      -- candidate UNJUDGED: rotation-dominated pair, or too near the FOE.
                      Not a pass and not a failure; the tracker must carry it.

A survivor is not a detection. This is one frame pair; the multi-frame tracker is what
turns a survivor into a track, and it does not exist yet.

What to look for
----------------
  1. Do the grey arrows actually fan out from the cyan cross? If they do not, the epipole
     is fitted to noise and every red dot on that frame is an accident.
  2. Are the green survivors clustered on the sky / tree line boundary? That is EXP-016's
     failure reappearing, and it is the argument for the depth-aware ring.
  3. Is the drone red on any frame it is visible in? That is a target being explained away
     as parallax, and it is the one failure that costs recall outright.
  4. Does the split flicker frame to frame on a steady scene? The report at the end
     measures that as sky-mask IoU between consecutive frames, so it does not have to be
     judged by eye.
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

import budget
import common
import criteria
import geometry
import masks
import skyline
from clipcfg import CLIP

W, H = CLIP["width"], CLIP["height"]
S = W / 1440.0
TINT = {"sky": (200, 120, 40), "uncertain": (40, 220, 230)}
LAYER_COLOUR = {"hud": (60, 60, 220), "prop": (200, 60, 220),
                "ladder": (60, 200, 220), "edge": (120, 120, 120)}
REJECT, SURVIVE, UNJUDGED = (60, 60, 235), (60, 230, 60), (170, 170, 170)
EPI = (240, 230, 60)


def px(v: float) -> int:
    return max(1, int(round(v * S)))


def tint(img, m, colour, alpha):
    if not m.any():
        return
    img[m] = (img[m] * (1 - alpha) + np.asarray(colour, np.float32) * alpha).astype(np.uint8)


def load_boxes() -> dict[int, list]:
    p = CLIP.get("labels")
    if not p or not os.path.exists(p):
        return {}
    return {int(k): v["box"] for k, v in json.load(open(p))["frames"].items()
            if v and v.get("box")}


def on_drone(x: float, y: float, box) -> bool:
    """EXP-015's match criterion, unchanged: inside the box grown by max(10 px, 25%)."""
    bx, by, bw, bh = box
    m = max(10.0, 0.25 * max(bw, bh))
    return (bx - m) <= x <= (bx + bw + m) and (by - m) <= y <= (by + bh + m)


def draw_epipole(img, ep: geometry.Epipole) -> None:
    """The focus of expansion, or an arrow towards it when it is off the picture.

    Drawing it only when it happens to be inside the frame would hide exactly the pairs
    where it is furthest away, which are the ones where the direction test works best.
    """
    x, y = float(ep.point[0]), float(ep.point[1])
    colour = EPI if ep.reliable else (120, 120, 120)
    if -W <= x <= 2 * W and -H <= y <= 2 * H and 0 <= x < W and 0 <= y < H:
        r = int(geometry.DEGENERATE_RADIUS)
        for a0 in range(0, 360, 18):          # dashed, so it reads over busy ground
            cv2.ellipse(img, (int(x), int(y)), (r, r), 0, a0, a0 + 9, colour, px(2))
        cv2.drawMarker(img, (int(x), int(y)), colour, cv2.MARKER_CROSS, px(34), px(3))
    else:
        c = np.array([W / 2.0, H / 2.0])
        d = np.array([x, y]) - c
        n = np.linalg.norm(d)
        if n < 1e-6:
            return
        tip = c + d / n * min(n, 0.42 * min(W, H))
        cv2.arrowedLine(img, (int(c[0]), int(c[1])), (int(tip[0]), int(tip[1])),
                        colour, px(3), tipLength=0.12)
        cv2.putText(img, f"epipole off-frame ({x:.0f},{y:.0f})",
                    (int(tip[0]) + 8, int(tip[1])), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * S, colour, max(1, px(1)), cv2.LINE_AA)


def draw_field(img, x: np.ndarray, mu: np.ndarray, every: int = 5, gain: float = 6.0):
    """The background residual field, subsampled. These arrows are the epipole's evidence."""
    for p, d in zip(x[::every], mu[::every]):
        if not np.all(np.isfinite(p)) or np.linalg.norm(d) < 0.25:
            continue
        a = (int(p[0]), int(p[1]))
        b = (int(p[0] + d[0] * gain), int(p[1] + d[1] * gain))
        cv2.arrowedLine(img, a, b, (200, 200, 200), max(1, px(1)), tipLength=0.35)


def text_block(img, lines, x0=12, y0=28, scale=0.6):
    for i, t in enumerate(lines):
        p = (x0, int(y0 + 26 * S * i))
        cv2.putText(img, t, p, cv2.FONT_HERSHEY_SIMPLEX, scale * S, (0, 0, 0),
                    px(3), cv2.LINE_AA)
        cv2.putText(img, t, p, cv2.FONT_HERSHEY_SIMPLEX, scale * S, (255, 255, 255),
                    max(1, px(1)), cv2.LINE_AA)


def analyse_pair(prev_bgr, cur_bgr, stage0: masks.Stage0, tau: float):
    """Everything both stages know about one frame pair."""
    gp, gc = common.prep(prev_bgr, 11), common.prep(cur_bgr, 11)
    hmat = common.homography(gp, gc)
    valid = stage0.valid(hmat)

    sl = skyline.split(cur_bgr, valid)

    tp, tc = geometry.track_grid(gp, gc, valid)
    gx, gmu = geometry.residual_field(tp, tc, hmat)
    ep = geometry.estimate_epipole(gx, gmu)

    cands = budget.candidates(common.maps(prev_bgr, cur_bgr, hmat)["win_b5_e4"], valid, tau)
    rows = np.zeros((0, 4))
    if len(cands):
        pts = cands[:, 1:3].astype(np.float32).reshape(-1, 1, 2)
        fwd, st, _ = cv2.calcOpticalFlowPyrLK(gp, gc, pts, None, **geometry.LK)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(gc, gp, fwd, None, **geometry.LK)
        good = (st.ravel() == 1) & (st2.ravel() == 1)
        good &= np.linalg.norm((back - pts).reshape(-1, 2), axis=1) < geometry.FB_MAX
        if good.any():
            cx, cmu = geometry.residual_field(pts.reshape(-1, 2)[good],
                                              fwd.reshape(-1, 2)[good], hmat)
            out = []
            for pos, res in zip(cx, cmu):
                v = criteria.epipolar_direction(res, pos, ep.point,
                                                epipole_reliable=ep.reliable,
                                                degenerate=ep.degenerate_for(*pos))
                out.append((pos[0], pos[1], float(v.usable), float(v.moving)))
            rows = np.asarray(out, float)
    return sl, ep, gx, gmu, rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=CLIP["n_frames"])
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--tau", type=float, default=1.661)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--no-field", action="store_true",
                    help="drop the grey background arrows; the candidates read more clearly")
    ap.add_argument("--no-video", action="store_true",
                    help="measure the span and print the report without writing frames")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = a.out or os.path.join(CLIP["out"], f"stages_{CLIP['name']}_{a.start}_{a.end}.mp4")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    stage0 = masks.Stage0()
    print("[stage 0] " + ", ".join(f"{k} {v:.2f}%" for k, v in stage0.coverage().items()))
    boxes = load_boxes()

    cap = cv2.VideoCapture(CLIP["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, a.start - 2))
    ok, prev = cap.read()
    if not ok:
        raise SystemExit(f"cannot read frame {a.start - 1}")

    writer = (None if a.no_video else
              cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H)))
    st = dict(n=0, skyless=0, sky=[], unc=[], iou=[], cands=[], rej=[], unj=[],
              surv_sky=0, surv_scene=0, surv_unc=0, surv=0,
              epi_ok=0, epi_anis=[], epi_inl=[], epi_pos=[], epi_jump=[],
              drone_seen=0, drone_kept=0, drone_red=[], swallowed=[],
              drone_label={})
    prev_sky = None

    for f in range(a.start, a.end + 1):
        ok, cur = cap.read()
        if not ok:
            break
        if (f - a.start) % a.stride:
            prev = cur
            continue

        sl, ep, gx, gmu, rows = analyse_pair(prev, cur, stage0, a.tau)
        # Still full size under --no-video: the drawing calls double as bounds checks, and
        # a 1x1 stand-in makes them throw instead of being skipped.
        img = np.zeros((H, W, 3), np.uint8) if a.no_video else cur.copy()
        tint(img, sl.sky, TINT["sky"], 0.22)
        tint(img, sl.uncertain, TINT["uncertain"], 0.30)
        for name, m in stage0.layers.items():
            tint(img, m, LAYER_COLOUR[name], 0.35)

        xs = np.nonzero(sl.horizon >= 0)[0]
        for x in xs[::4]:
            cv2.circle(img, (int(x), int(sl.horizon[x])), px(2), (40, 220, 230), -1)

        if not a.no_field:
            draw_field(img, gx, gmu)
        draw_epipole(img, ep)

        n_c = len(rows)
        n_rej = n_unj = 0
        for rx, ry, usable, moving in rows:
            p = (int(rx), int(ry))
            if usable < 1:
                n_unj += 1
                cv2.circle(img, p, px(6), UNJUDGED, max(1, px(2)))
                continue
            if moving < 1:
                n_rej += 1
                cv2.circle(img, p, px(5), REJECT, -1)
            else:
                cv2.circle(img, p, px(5), SURVIVE, -1)
                lab = sl.label(rx, ry)
                st[f"surv_{'unc' if lab == 'uncertain' else lab}"] += 1
                st["surv"] += 1

        box = boxes.get(f)
        drone_txt = ""
        if box:
            bx, by, bw, bh = [int(v) for v in box]
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (60, 230, 60), px(2))
            cx, cy = bx + bw / 2.0, by + bh / 2.0
            lab = sl.label(cx, cy)
            st["drone_label"][lab] = st["drone_label"].get(lab, 0) + 1
            hit = [k for k, m in stage0.layers.items()
                   if m[int(np.clip(cy, 0, H - 1)), int(np.clip(cx, 0, W - 1))]]
            if hit:
                st["swallowed"].append(f)
            mine = [r for r in rows if on_drone(r[0], r[1], box)]
            if mine:
                st["drone_seen"] += 1
                kept = any(r[2] < 1 or r[3] > 0 for r in mine)
                st["drone_kept"] += 1 if kept else 0
                if not kept:
                    st["drone_red"].append(f)
                drone_txt = ("SURVIVES" if kept else "REJECTED as parallax")
            else:
                drone_txt = "not a candidate"
            cap_txt = f"drone: {lab} | {drone_txt}" + (f" | MASKED BY {','.join(hit)}" if hit else "")
            text_block(img, [cap_txt], x0=bx, y0=max(int(20 * S), by - int(10 * S)), scale=0.55)

        st["n"] += 1
        st["sky"].append(sl.sky_fraction)
        st["unc"].append(float(sl.uncertain.mean()))
        st["cands"].append(n_c)
        st["rej"].append(n_rej)
        st["unj"].append(n_unj)
        st["epi_ok"] += 1 if ep.reliable else 0
        st["epi_anis"].append(ep.anisotropy)
        st["epi_inl"].append(ep.inlier_fraction)
        # A real focus of expansion moves smoothly as the aircraft manoeuvres. One fitted
        # to noise hops. Frame-to-frame jump separates the two without ground truth.
        if st["epi_pos"] and ep.reliable and st["epi_pos"][-1][2]:
            st["epi_jump"].append(float(np.hypot(ep.point[0] - st["epi_pos"][-1][0],
                                                 ep.point[1] - st["epi_pos"][-1][1])))
        st["epi_pos"].append((float(ep.point[0]), float(ep.point[1]), bool(ep.reliable)))
        if not sl.has_sky:
            st["skyless"] += 1
        if prev_sky is not None:
            u = float((prev_sky | sl.sky).sum())
            st["iou"].append(float((prev_sky & sl.sky).sum()) / u if u else 1.0)
        prev_sky = sl.sky

        text_block(img, [
            f"frame {f}   tau {a.tau}",
            f"stage 1: sky {sl.sky_fraction * 100:.1f}%" + ("" if sl.has_sky else " (NO SKY)")
            + f"  uncertain {sl.uncertain.mean() * 100:.1f}%",
            f"stage 2: epipole {'OK' if ep.reliable else 'UNRELIABLE'}"
            f"  anis {ep.anisotropy:.2f}  inl {ep.inlier_fraction:.2f}  n {ep.n_points}",
            f"candidates {n_c}:  rejected {n_rej}  unjudged {n_unj}  "
            f"survive {n_c - n_rej - n_unj}",
        ])
        if writer is not None:
            writer.write(img)
        prev = cur

    if writer is not None:
        writer.release()
    cap.release()
    report(st, out)


def report(st, out) -> None:
    n = st["n"]
    if not n:
        raise SystemExit("no frames rendered")
    tot = max(sum(st["cands"]), 1)
    print(f"\n[overlay] {n} frames -> {out}\n")
    print("STAGE 1 over the span")
    print(f"  sky fraction        median {np.median(st['sky']) * 100:5.1f}%   "
          f"p10 {np.percentile(st['sky'], 10) * 100:5.1f}%   "
          f"p90 {np.percentile(st['sky'], 90) * 100:5.1f}%")
    print(f"  uncertain band      median {np.median(st['unc']) * 100:5.1f}% of frame")
    print(f"  frames with NO sky  {st['skyless']} of {n}")
    if st["iou"]:
        low = int(np.sum(np.asarray(st["iou"]) < 0.90))
        print(f"  frame-to-frame sky IoU  median {np.median(st['iou']):.3f}, "
              f"{low} of {len(st['iou'])} consecutive pairs below 0.90")
        print("    (this is the flicker check: a steady scene should hold well above 0.9)")
    if st["drone_label"]:
        print("  labelled drone sits in: " +
              ", ".join(f"{k} {v}" for k, v in sorted(st["drone_label"].items())))

    print("\nSTAGE 2 over the span")
    print(f"  epipole reliable in {st['epi_ok']} of {n} pairs")
    print(f"  epipole anisotropy   median {np.median(st['epi_anis']):.2f}  "
          f"(threshold {geometry.MIN_DIRECTION_ANISOTROPY})")
    print(f"  epipole inlier frac  median {np.median(st['epi_inl']):.3f}  "
          f"-- NOT part of `reliable`; see the note below")
    if st["epi_jump"]:
        j = np.asarray(st["epi_jump"])
        print(f"  epipole frame-to-frame jump  median {np.median(j):7.1f} px   "
              f"p90 {np.percentile(j, 90):7.1f} px")
        print("    (a real FOE drifts smoothly with the manoeuvre; a fit to noise hops)")
    print(f"  candidates/frame    {tot / n:6.1f}")
    print(f"  rejected as epipolar {sum(st['rej']) / tot * 100:5.1f}%")
    print(f"  unjudged             {sum(st['unj']) / tot * 100:5.1f}%")
    print(f"  surviving load      {(tot - sum(st['rej'])) / n:6.1f} /frame")

    # The number that actually says whether the direction test is doing anything. A
    # candidate is only rejected if |sin| to the epipolar line is under the threshold, so
    # a candidate with a RANDOM direction is rejected with probability 2*asin(thr)/pi.
    # Rejection among judged candidates has to beat that, or the test is a coin.
    judged = tot - sum(st["unj"])
    chance = 2.0 * np.arcsin(0.35) / np.pi
    if judged > 0:
        got = sum(st["rej"]) / judged
        print(f"\n  rejection among JUDGED candidates  {got * 100:5.1f}%")
        print(f"  the same figure for random directions {chance * 100:5.1f}%  "
              f"(= 2*asin(0.35)/pi)")
        print(f"  -> lift over chance  x{got / chance:.2f}" +
              ("   -- at or below chance, the test is not deciding anything"
               if got <= chance * 1.1 else ""))

    print("\nWHERE THE SURVIVORS SIT (stage 1 x stage 2 -- the point of this render)")
    s = max(st["surv"], 1)
    print(f"  sky {st['surv_sky'] / s * 100:5.1f}%   "
          f"scene {st['surv_scene'] / s * 100:5.1f}%   "
          f"uncertain {st['surv_unc'] / s * 100:5.1f}%   of {st['surv']} survivors")
    print("  The uncertain band is the sky/scene boundary. EXP-016 put 57.5% of analog's")
    print("  false tracks there; compare that against the band's share of the frame above.")

    if st["drone_seen"]:
        print(f"\nDRONE: a candidate in {st['drone_seen']} labelled frames, "
              f"survives in {st['drone_kept']} "
              f"({st['drone_kept'] / st['drone_seen'] * 100:.0f}%)")
        if st["drone_red"]:
            print(f"  REJECTED AS PARALLAX in {len(st['drone_red'])} frames, "
                  f"first at {st['drone_red'][0]} -- recall lost by the direction test")
    if st["swallowed"]:
        print(f"\n[WARNING] stage-0 masks cover the labelled drone in "
              f"{len(st['swallowed'])} frames, first at {st['swallowed'][0]}")


if __name__ == "__main__":
    main()
