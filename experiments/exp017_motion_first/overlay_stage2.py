"""Stage 2 with both branches on one picture: the sky silhouette detector and the
parallax-discounted scene motion, measured over a span.

    PYTHONPATH="experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
        py -3.13 -m overlay_stage2 --start 650 --end 964

    PYTHONPATH="runs/sofa_analog/exp017_motion_first;experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
        py -3.13 -m overlay_stage2 --start 441 --end 800

`--no-video` prints the report without writing frames, which is the fast way to re-cut a
threshold. `overlay_window.py` stays as the motion-only baseline to compare against.

Two things this run does differently from the plan it implements, both because a
measurement already in this repo contradicts the plan
--------------------------------------------------------------------------------------
1. **The sky branch is NOT gated on stage 1.** The plan puts silhouette detection on "sky
   pixels". EXP-019 measured stage 1 calling the labelled airborne target **scene in 257
   of 257** O4 frames -- the split separates *blue sky* from everything, while what the
   branch needs is *far* from *near*, and a target at shallow elevation against a distant
   tree line is far. Gated as written, this branch would score zero on O4 and the run
   would say nothing about the detector. So it runs over the whole valid frame and the
   report **cuts the result by stage-1 label**, which measures both the detector and what
   the gate would have cost. Gating becomes a reporting choice rather than a silent loss.

2. **The epipolar direction test is off unless `--direction` is passed.** The 5-frame
   window run measured the pooled epipole agreeing with the background it was fitted to at
   22.1% against a 22.8% chance floor, and `|mu|` uncorrelated with distance from the
   epipole -- the residual field is a near-uniform shift, not a parallax field, so the
   rejections are at chance and cost 33 points of recall. The plan's claim that this test
   "kills the 57.5% of analog false tracks at the horizon" is separately contradicted by
   EXP-019, which found the horizon failure is not at the candidate stage at all.

   It is still implemented and still measured, because this run changes its input: the
   residual field is now built against **two** planes instead of one. If the uniform shift
   was a single-plane misfit, the layered field should be radial where the old one was
   not. That is the question, and `SINGLE-PLANE CONTROL` in the report is what answers it
   -- the same diagnostics on the same frames with one plane, so the comparison is
   internal and not against a number from another run.

Colours
-------
  * CYAN circle, radius = detected scale -- sky branch, kept. Thicker means higher `c`.
  * DARK CYAN circle  -- sky branch, rejected (cloud / bloom / too large)
  * GREEN dot         -- scene branch, survives
  * RED dot           -- scene branch, rejected as radial (only with `--direction`)
  * GREY dot          -- scene branch, dropped as a flash by persistence
  * MAGENTA tick   -- background grid point assigned to the dominant plane
  * BLUE tick      -- assigned to the second plane
  * GREY tick      -- ambiguous: both planes explain it equally well
    Each tick is drawn along that point's own residual, so the field the direction test
    would be judging is visible directly.
  * YELLOW cross   -- the fitted epipole, when it is on screen and reliable
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
import layers
import masks
import silhouette
import skyline
import window as win
from clipcfg import CLIP
from overlay_video import LAYER_COLOUR, TINT, load_boxes, on_drone, px, text_block, tint

W, H = CLIP["width"], CLIP["height"]
S = W / 1440.0

SKY_KEPT, SKY_REJ = (230, 230, 60), (110, 110, 40)
SURVIVE, REJECT, FLASH = (60, 230, 60), (60, 60, 235), (95, 95, 95)
LAYER_TICK = {0: (220, 80, 220), 1: (40, 160, 240), -1: (120, 120, 120)}
CHANCE = 2.0 * np.arcsin(0.35) / np.pi        # the floor every rejection rate is read against


def make_pair(prev_bgr, cur_bgr, frame: int, stage0: masks.Stage0, tau: float):
    """One frame pair's geometry, fitted with TWO planes, plus the single-plane control.

    Returns (win.Pair, LayerFit | None, control) where `control` carries the same
    diagnostics computed against one plane, so the report can put them side by side on
    identical frames rather than against a number from a different run.
    """
    gp, gc = common.prep(prev_bgr, 11), common.prep(cur_bgr, 11)
    hmat = common.homography(gp, gc)                    # the single-plane fit, as before
    valid = stage0.valid(hmat)
    tp, tc = geometry.track_grid(gp, gc, valid)

    # The control: one plane, exactly what EXP-019 and the window run measured.
    cx, cmu = geometry.residual_field(tp, tc, hmat)
    control = (cx, cmu, geometry.estimate_epipole(cx, cmu))

    fit = layers.fit_layers(tp, tc)
    if fit is None:
        gx, gmu, lab = cx, cmu, np.zeros(len(cx), int)
        ep = control[2]
    else:
        gx, gmu, lab = layers.layered_residual_field(tp, tc, fit)
        keep = lab >= 0                                 # ambiguous points vote for nothing
        ep = geometry.estimate_epipole(gx[keep], gmu[keep])

    cands = budget.candidates(common.maps(prev_bgr, cur_bgr, hmat)["win_b5_e4"], valid, tau)
    pair = win.Pair(frame, gc, gp, hmat, cands, gx, gmu, ep)
    return pair, fit, control, lab, valid


def radial_agreement(x: np.ndarray, mu: np.ndarray, ep: geometry.Epipole,
                     sin_threshold: float = 0.35):
    """What fraction of a residual field points at the epipole, and does |mu| grow with r.

    These are the two diagnostics that decide whether a field is parallax at all. The
    first is read against `CHANCE`; the second must be strongly positive for parallax,
    because `mu = gamma * (e - x)` makes `|mu|` proportional to `|e - x|` by construction.
    A field that is a uniform shift scores near chance on the first and near zero on the
    second, which is what the window run measured on the single-plane field.
    """
    if len(x) < 8 or not np.all(np.isfinite(ep.point)):
        return float("nan"), float("nan")
    v = ep.point[None, :] - x
    r = np.linalg.norm(v, axis=1)
    nd = np.linalg.norm(mu, axis=1)
    ok = (r > 1e-6) & (nd > 1e-6)
    if ok.sum() < 8:
        return float("nan"), float("nan")
    u = v[ok] / r[ok][:, None]
    d = mu[ok] / nd[ok][:, None]
    sin_a = np.abs(d[:, 0] * u[:, 1] - d[:, 1] * u[:, 0])
    agree = float((sin_a <= sin_threshold).mean())
    corr = float(np.corrcoef(r[ok], nd[ok])[0, 1]) if ok.sum() > 2 else float("nan")
    return agree, corr


def sample_ring_sigma(grey: np.ndarray, valid: np.ndarray, sl, rng, n: int = 120):
    """sigma_ring at random points, split by stage-1 label. The premise check.

    The sky branch rests on one claim: sky has no texture, so `sigma_ring` there is tiny
    and a small dark speck is a large-sigma event. That claim is about the *background*,
    not about the target, so it can and should be measured without a target anywhere near
    -- and it must be measured on our own footage rather than inherited from the IRST
    literature, which is about thermal imagery of clear sky.

    Measured at a fixed small scale, which is the scale a distant target is detected at.
    Returns {label: [sigma, ...]}.
    """
    out: dict[str, list[float]] = {"sky": [], "scene": [], "uncertain": []}
    ys, xs = np.nonzero(valid)
    if not len(ys):
        return out
    idx = rng.choice(len(ys), size=min(n, len(ys)), replace=False)
    gf = grey.astype(np.float32)
    sigma = float(silhouette.scale_ladder()[1])
    core = cv2.GaussianBlur(gf, (0, 0), sigma)
    for i in idx:
        x, y = float(xs[i]), float(ys[i])
        got = silhouette.contrast_at(core, gf, x, y, sigma, None)
        if got is not None:
            out[sl.label(x, y)].append(got[3])
    return out


def magnified_inset(img, cur, cx: float, cy: float, half: int):
    """A magnified crop pasted top-right. A 6-40 px target is otherwise a speck on a 1440.

    Drawn from the CLEAN frame, not from `img`, so the overlays do not obscure the very
    pixels the inset exists to show; the box is redrawn on top afterwards.
    """
    # The inset must fit on the frame whatever the target's size. A fixed magnification
    # silently produced NO inset at all on the large late-approach boxes -- the computed
    # panel was wider than the frame and the draw bailed out -- which is the case the
    # inset is least needed for but exactly the case that hid the bug. Size the panel
    # first, then pick the magnification that fills it.
    half = int(np.clip(half, 24 * S, 140 * S))
    panel = int(W / 3.2)
    scale = max(2, int(panel / (2 * half)))
    x0, y0 = int(np.clip(cx - half, 0, W - 2 * half)), int(np.clip(cy - half, 0, H - 2 * half))
    crop = cur[y0:y0 + 2 * half, x0:x0 + 2 * half]
    if crop.size == 0:
        return
    big = cv2.resize(crop, (2 * half * scale, 2 * half * scale), interpolation=cv2.INTER_NEAREST)
    bh, bw = big.shape[:2]
    px0, py0 = W - bw - int(16 * S), int(16 * S)
    if px0 < 0 or py0 + bh > H:
        return
    img[py0:py0 + bh, px0:px0 + bw] = big
    cv2.rectangle(img, (px0, py0), (px0 + bw, py0 + bh), (255, 255, 255), max(1, px(2)))
    cv2.putText(img, f"x{scale}", (px0 + px(4), py0 + bh - px(6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * S, (255, 255, 255), max(1, px(1)))
    return x0, y0, scale, px0, py0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=CLIP["n_frames"])
    ap.add_argument("--tau", type=float, default=1.661, help="motion candidate threshold")
    ap.add_argument("--contrast", type=float, default=6.0,
                    help="sky branch threshold on c, in units of sigma_ring")
    ap.add_argument("--cap", type=int, default=4000,
                    help="memory guard on sky-branch peaks per frame, NOT a budget. If it "
                         "binds, the branch has silently become a top-N rank budget -- the "
                         "defect budget.py exists to avoid -- and the report says so.")
    ap.add_argument("--k", type=int, default=win.DEFAULT_K)
    ap.add_argument("--min-appear", type=int, default=win.DEFAULT_MIN_APPEAR)
    ap.add_argument("--appear-radius", type=float, default=px(14.0))
    ap.add_argument("--direction", action="store_true",
                    help="enable the epipolar rejector. Off by default: the window run "
                         "measured it at chance on the single-plane field.")
    ap.add_argument("--depth-ring", action="store_true", default=True,
                    help="restrict rings to the candidate's own stage-1 label")
    ap.add_argument("--fb-max", type=float, default=None)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = a.out or os.path.join(CLIP["out"],
                                f"stage2_{CLIP['name']}_{a.start}_{a.end}.mp4")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if a.fb_max is not None:
        geometry.FB_MAX = a.fb_max

    stage0 = masks.Stage0()
    print("[stage 0] " + ", ".join(f"{k} {v:.2f}%" for k, v in stage0.coverage().items()))
    print(f"[stage 2a] sky branch: c >= {a.contrast} sigma, scales "
          f"{silhouette.SIGMA_MIN:.1f}-{silhouette.SIGMA_MAX:.1f} sigma "
          f"({silhouette.scale_ladder()[0] * 2 * np.sqrt(2):.0f}-"
          f"{silhouette.scale_ladder()[-1] * 2 * np.sqrt(2):.0f} px diameter), UNGATED")
    print(f"[stage 2b] scene branch: two planes, tau={a.tau}, k={a.k} need {a.min_appear}, "
          f"direction test {'ON' if a.direction else 'OFF'}")
    boxes = load_boxes()

    cap = cv2.VideoCapture(CLIP["video"])
    first = max(1, a.start - a.k)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first - 1)
    ok, prev = cap.read()
    if not ok:
        raise SystemExit(f"cannot read frame {first}")
    writer = (None if a.no_video else
              cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H)))

    pairs: deque = deque(maxlen=a.k)
    st = _new_stats()
    rng = np.random.default_rng(7)

    for f in range(first + 1, a.end + 1):
        ok, cur = cap.read()
        if not ok:
            break
        pair, fit, control, lab, valid = make_pair(prev, cur, f, stage0, a.tau)
        pairs.append(pair)
        prev = cur
        if f < a.start or len(pairs) < a.k:
            continue

        plist = list(pairs)
        sl = skyline.split(cur, valid)
        img = np.zeros((H, W, 3), np.uint8) if a.no_video else cur.copy()
        _draw_background(img, sl, stage0, pair.epipole)
        _draw_layers(img, pair, lab)

        sky_hits, n_sky = _run_sky_branch(cur, valid, sl, a, st, img, boxes.get(f))
        scene_hits, n_scene = _run_scene_branch(plist, fit, sl, a, st, img, boxes.get(f))
        _record_geometry(st, pair, fit, control, lab)
        for k, v in sample_ring_sigma(cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY),
                                      valid, sl, rng).items():
            st["bg_sigma"][k] += v
        _draw_target(img, cur, boxes.get(f), sky_hits, scene_hits, st)
        _caption(img, f, a, sl, fit, pair, n_sky, n_scene)

        st["n"] += 1
        if writer is not None:
            writer.write(img)

    if writer is not None:
        writer.release()
    cap.release()
    report(st, a, out)


def _new_stats() -> dict:
    return dict(
        n=0,
        sky_kept=0, sky_rej={}, sky_by_label={"sky": 0, "scene": 0, "uncertain": 0},
        sky_floored=0, sky_total=0, sky_clutter_c=[], sky_drone_c=[], sky_drone_sigma=[],
        sky_drone_hit=0, sky_drone_frames=0, sky_drone_scale=[], cap_bound=0,
        bg_sigma={"sky": [], "scene": [], "uncertain": []},
        two_layers=0, sep=[], lay_share={0: 0, 1: 0, -1: 0}, lay_y={0: [], 1: []},
        agree_lay=[], corr_lay=[], agree_one=[], corr_one=[], epi_ok_lay=0, epi_ok_one=0,
        scene_counts={"flash": 0, "rejected": 0, "unjudged": 0, "survives": 0},
        scene_seeds=0, scene_drone_seen=0, scene_drone_kept=0, scene_drone_v={},
        surv_label={"sky": 0, "scene": 0, "uncertain": 0},
        ring_plain=[], ring_depth=[], ring_refused=0, ring_tried=0,
        both=0, either=0, neither=0, labelled=0)


def _draw_layers(img, pair, lab) -> None:
    """A tick at each background grid point, coloured by the plane it was assigned to.

    This is the only way to see whether the two-plane fit is describing the scene or
    slicing it arbitrarily. A real ground/skyline split shows as a coherent band of one
    colour below and the other above; a fit to noise shows as the two colours interleaved
    everywhere, which is what `LayerFit.separation` catches numerically and what this
    catches by eye. The tick is drawn along the point's own residual, so its direction
    carries the `mu` the direction test would be judging.
    """
    for i in range(0, len(pair.grid_x), 2):
        x, y = pair.grid_x[i]
        if not (0 <= x < W and 0 <= y < H):
            continue
        mu = pair.grid_mu[i]
        n = float(np.hypot(mu[0], mu[1]))
        colour = LAYER_TICK[int(lab[i])]
        if n > 1e-6:
            g = min(px(14.0), 6.0 * n) / n
            cv2.line(img, (int(x), int(y)), (int(x + mu[0] * g), int(y + mu[1] * g)),
                     colour, max(1, px(1)))
        cv2.circle(img, (int(x), int(y)), max(1, px(1)), colour, -1)


def _draw_background(img, sl, stage0, ep) -> None:
    """Stage 0 masks, the stage-1 split and the epipole. Everything a candidate is judged against."""
    tint(img, sl.sky, TINT["sky"], 0.22)
    tint(img, sl.uncertain, TINT["uncertain"], 0.30)
    for name, m in stage0.layers.items():
        tint(img, m, LAYER_COLOUR[name], 0.35)
    xs = np.nonzero(sl.horizon >= 0)[0]
    for x in xs[::4]:
        cv2.circle(img, (int(x), int(sl.horizon[x])), px(2), (40, 220, 230), -1)
    if ep.reliable and np.all(np.isfinite(ep.point)):
        p = (int(np.clip(ep.point[0], -1e4, 1e4)), int(np.clip(ep.point[1], -1e4, 1e4)))
        if 0 <= p[0] < W and 0 <= p[1] < H:
            cv2.drawMarker(img, p, (0, 255, 255), cv2.MARKER_CROSS, px(28), max(1, px(2)))


def _run_sky_branch(cur, valid, sl, a, st, img, box):
    """Stage 2a over the whole valid frame, drawn and counted by stage-1 label."""
    hits = silhouette.detect(cur, valid, threshold=a.contrast, cap=a.cap)
    kept = [s for s in hits if s.kept]
    st["cap_bound"] += 1 if len(hits) >= a.cap else 0
    st["sky_total"] += len(hits)
    st["sky_kept"] += len(kept)
    for s in hits:
        if s.reason and s.reason != "below threshold":
            st["sky_rej"][s.reason] = st["sky_rej"].get(s.reason, 0) + 1
    on_target = []
    for s in kept:
        lab = sl.label(s.x, s.y)
        st["sky_by_label"][lab] += 1
        st["sky_floored"] += 1 if s.floored else 0
        if box is not None and on_drone(s.x, s.y, box):
            on_target.append(s)
        else:
            st["sky_clutter_c"].append(s.contrast)
        r = max(px(3), int(round(s.diameter / 2.0)))
        cv2.circle(img, (int(s.x), int(s.y)), r, SKY_KEPT,
                   max(1, px(1) + int(s.contrast >= 2 * a.contrast)))
    for s in hits:
        if s.reason and s.reason != "below threshold":
            cv2.circle(img, (int(s.x), int(s.y)), max(px(3), int(s.diameter / 2)),
                       SKY_REJ, max(1, px(1)))
    return on_target, len(kept)


def _run_scene_branch(plist, fit, sl, a, st, img, box):
    """Stage 2b: persistence over k frames, then the layered direction rejector."""
    ref = plist[-1]
    if not len(ref.cands):
        return [], 0
    seeds = ref.cands[:, 1:3].astype(np.float32)
    pos = win._track_back(plist, seeds)
    ep = ref.epipole
    out, kept_n = [], 0
    for i in range(len(seeds)):
        appearances = sum(1 for t in range(len(plist))
                          if win._appears(plist[t].cands, pos[len(plist) - 1 - t, i],
                                          a.appear_radius))
        x, y = float(seeds[i][0]), float(seeds[i][1])
        if appearances < a.min_appear:
            verdict = "flash"
        elif a.direction:
            mu = _candidate_residual(pos, i, plist, fit)
            if mu is None:
                verdict = "unjudged"
            else:
                rej, judged, _ = layers.epipolar_reject(mu, np.array([x, y]), ep,
                                                        min_displacement=px(1.0))
                # A refusal is NOT a pass. Counting it as one puts candidates the test
                # never judged into the denominator of the rejection rate, which then
                # measures how often the test declined rather than how well it separates.
                verdict = "rejected" if rej else ("survives" if judged else "unjudged")
        else:
            verdict = "survives"
        st["scene_counts"][verdict] += 1
        st["scene_seeds"] += 1
        colour = {"flash": FLASH, "rejected": REJECT,
                  "unjudged": (170, 170, 170), "survives": SURVIVE}[verdict]
        cv2.circle(img, (int(x), int(y)), px(3) if verdict == "flash" else px(5),
                   colour, -1 if verdict != "unjudged" else max(1, px(2)))
        if verdict == "survives":
            kept_n += 1
            st["surv_label"][sl.label(x, y)] += 1
            if box is not None and on_drone(x, y, box):
                out.append((x, y))
    return out, kept_n


def _candidate_residual(pos, i, plist, fit):
    """This candidate's residual under its OWN layer, for the newest pair only.

    `_track_back` returns positions newest-first: index 0 is the reference frame, which is
    the newest pair's *current* frame, and index 1 is its *previous*. Indexing from the
    other end reaches the oldest pair, and pairing those positions with the newest pair's
    homography and layer fit -- as a first version here did -- silently produces a residual
    measured across five frames against a one-frame plane fit. It is only ever a wrong
    number, never an exception, which is why it is spelled out rather than left to the
    reader of `len(plist) - 1`.
    """
    p_cur, p_prev = pos[0, i], pos[1, i]
    if not (np.all(np.isfinite(p_cur)) and np.all(np.isfinite(p_prev))):
        return None
    if fit is None:
        _, mu = geometry.residual_field(p_prev.reshape(1, 2), p_cur.reshape(1, 2),
                                        plist[-1].hmat)
        return mu[0]
    _, mu, _ = layers.assign_layer(p_prev, p_cur, fit)
    return mu


def _record_geometry(st, pair, fit, control, lab) -> None:
    """The layered-vs-single comparison. This is what the run exists to measure."""
    if fit is not None:
        st["two_layers"] += 1 if fit.two_layers else 0
        if np.isfinite(fit.separation):
            st["sep"].append(fit.separation)
        for v in (0, 1, -1):
            st["lay_share"][v] += int((lab == v).sum())
        if fit.two_layers:
            st["lay_y"][0].append(fit.dominant_mean_y)
            st["lay_y"][1].append(fit.second_mean_y)
    keep = lab >= 0
    ag, co = radial_agreement(pair.grid_x[keep], pair.grid_mu[keep], pair.epipole)
    if np.isfinite(ag):
        st["agree_lay"].append(ag)
        st["corr_lay"].append(co)
    st["epi_ok_lay"] += 1 if pair.epipole.reliable else 0
    cx, cmu, cep = control
    ag1, co1 = radial_agreement(cx, cmu, cep)
    if np.isfinite(ag1):
        st["agree_one"].append(ag1)
        st["corr_one"].append(co1)
    st["epi_ok_one"] += 1 if cep.reliable else 0


def _draw_target(img, cur, box, sky_hits, scene_hits, st) -> None:
    """The ground-truth box, the magnified inset, and which branch found it."""
    if box is None:
        return
    st["labelled"] += 1
    bx, by, bw, bh = [int(v) for v in box]
    got_sky, got_scene = bool(sky_hits), bool(scene_hits)
    st["sky_drone_frames"] += 1
    if got_sky:
        st["sky_drone_hit"] += 1
        best = max(sky_hits, key=lambda s: s.contrast)
        st["sky_drone_c"].append(best.contrast)
        st["sky_drone_sigma"].append(best.ring_sigma)
        st["sky_drone_scale"].append(best.diameter)
    if got_scene:
        st["scene_drone_kept"] += 1
    st["both"] += 1 if (got_sky and got_scene) else 0
    st["either"] += 1 if (got_sky or got_scene) else 0
    st["neither"] += 1 if not (got_sky or got_scene) else 0

    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (60, 230, 60), max(1, px(2)))
    # Enough context around the box to judge the target against its background, which is
    # the whole question on this footage. `magnified_inset` clamps and picks the zoom.
    magnified_inset(img, cur, bx + bw / 2.0, by + bh / 2.0, int(max(bw, bh) * 1.6))
    tag = ("BOTH" if got_sky and got_scene else "sky only" if got_sky else
           "scene only" if got_scene else "MISSED by both")
    c_txt = (f"  c={max(s.contrast for s in sky_hits):.1f}s" if got_sky else "")
    text_block(img, [f"drone: {tag}{c_txt}"], x0=bx,
               y0=max(int(20 * S), by - int(10 * S)), scale=0.55)


def _caption(img, f, a, sl, fit, pair, n_sky, n_scene) -> None:
    lay = fit.describe() if fit is not None else "no fit"
    text_block(img, [
        f"frame {f}    tau {a.tau}   c >= {a.contrast} sigma   k={a.k}/{a.min_appear}",
        f"stage 1: sky {sl.sky_fraction * 100:.1f}%  uncertain {sl.uncertain.mean() * 100:.1f}%"
        + ("  [NO SKY]" if not sl.has_sky else ""),
        f"stage 2b layers: {lay}",
        f"epipole {'OK' if pair.epipole.reliable else 'UNRELIABLE'}"
        f"  agree {pair.epipole.inlier_fraction:.2f} (chance {CHANCE:.2f})"
        + ("   direction test OFF" if not a.direction else ""),
        f"this frame:  sky branch {n_sky} kept   scene branch {n_scene} survive",
    ])


def _pct(v) -> str:
    return "n/a" if not len(v) else f"{float(np.median(v)):.3f}"


def report(st, a, out) -> None:
    n = st["n"]
    if not n:
        raise SystemExit("no frames rendered -- span shorter than the window?")
    print(f"\n[stage 2] {n} frames -> {out}\n")

    print("THE PREMISE: is sigma_ring actually small on sky in OUR footage?")
    print("  The sky branch rests on sky having no texture, so that a dark speck is a")
    print("  large-sigma event. Measured at random background points, by stage-1 label:")
    bg = st["bg_sigma"]
    for k in ("sky", "uncertain", "scene"):
        v = np.asarray(bg[k])
        if not len(v):
            print(f"    {k:<10s}  no samples")
            continue
        print(f"    {k:<10s}  sigma_ring median {np.median(v):6.2f}  "
              f"p90 {np.percentile(v, 90):6.2f}   ({len(v)} samples)")
    if len(bg["sky"]) and len(bg["scene"]):
        sky_med = float(np.median(bg["sky"]))
        ratio = float(np.median(bg["scene"])) / max(sky_med, 1e-6)
        print(f"  -> scene is {ratio:.1f}x noisier than sky. The regime inversion is real")
        print(f"     where the background IS sky; it says nothing about where the target is.")
        # 1.4826 * 1 is what MAD returns when the ring spans exactly one grey level. If
        # sky sits there, `sigma_ring` is measuring the quantiser and not the sky, so `c`
        # on sky is really "contrast in units of 1.5 grey levels" -- still a usable
        # ranking, but the sigma has no distributional meaning and no Gaussian tail
        # argument can be made from it.
        if abs(sky_med - 1.4826) < 0.02:
            print("  ** sky sigma_ring is EXACTLY 1.4826 = MAD of one grey level: sky is")
            print("     flat to the 8-bit quantiser. `c` on sky is contrast in units of")
            print("     1.5 grey levels, not in units of a measured noise distribution. **")

    print("\nSTAGE 2a -- THE SKY BRANCH (silhouette), run UNGATED over the whole frame")
    if st["cap_bound"]:
        print(f"  ** the peak cap bound in {st['cap_bound']} of {n} frames -- the branch was")
        print(f"     a top-N rank budget there, not a threshold. Raise --cap and re-run. **")
    print(f"  candidates before the c threshold   {st['sky_total'] / n:8.1f}/frame")
    print(f"  kept at c >= {a.contrast:<4g}                  {st['sky_kept'] / n:8.1f}/frame")
    for k, v in sorted(st["sky_rej"].items(), key=lambda kv: -kv[1]):
        print(f"    rejected as {k:<12s}        {v / n:8.2f}/frame")
    tot = max(st["sky_kept"], 1)
    print("  where the kept ones sit (stage-1 label):")
    for k, v in st["sky_by_label"].items():
        print(f"    {k:<10s} {v / n:7.2f}/frame   {v / tot * 100:5.1f}%")
    print(f"  sigma_ring hit the floor in {st['sky_floored'] / tot * 100:.1f}% of kept "
          f"candidates" + ("  <-- c is reporting the floor, not the sky"
                           if st["sky_floored"] / tot > 0.25 else ""))

    lab = max(st["sky_drone_frames"], 1)
    print(f"\n  DRONE: detected in {st['sky_drone_hit']} of {st['sky_drone_frames']} "
          f"labelled frames ({st['sky_drone_hit'] / lab * 100:.0f}%)")
    if st["sky_drone_c"]:
        d = np.asarray(st["sky_drone_c"])
        print(f"    c on the drone   median {np.median(d):6.1f}  p10 {np.percentile(d, 10):6.1f}"
              f"  min {d.min():6.1f}")
        print(f"    measured size    median {np.median(st['sky_drone_scale']):.1f} px diameter")
        print(f"    sigma_ring there median {np.median(st['sky_drone_sigma']):.2f} grey levels")
    if st["sky_clutter_c"]:
        c = np.asarray(st["sky_clutter_c"])
        print(f"    c on everything else   median {np.median(c):6.1f}  "
              f"p99 {np.percentile(c, 99):6.1f}  max {c.max():6.1f}   ({len(c)} candidates)")
        if st["sky_drone_c"]:
            d = np.asarray(st["sky_drone_c"])
            sep_frac = float((c >= np.percentile(d, 10)).mean())
            print(f"    SEPARATION: {sep_frac * 100:.1f}% of clutter reaches the drone's "
                  f"p10 contrast")
            print(f"      -- if this is near zero, c alone separates target from clutter;")
            print(f"         if it is large, c is a ranking score and needs a second test.")
        thr = np.array([4.0, 6.0, 8.0, 12.0, 20.0])
        far = silhouette.false_alarm_curve(c, thr, n)
        print("    measured false alarms/frame vs the Gaussian rate the same threshold "
              "would nominally buy:")
        n_ind = (W * H) / (np.pi * (2.5 * silhouette.SIGMA_MIN) ** 2)
        for t, fa in zip(thr, far):
            print(f"      c >= {t:5.1f}   measured {fa:8.2f}/frame   "
                  f"Gaussian {silhouette.gaussian_rate(t, n_ind):9.2e}/frame")
        print("      The gap between the two columns IS the finding: where they diverge,")
        print("      `c` is a ranking score and must not be quoted as a false-alarm rate.")

    print("\nSTAGE 2b -- THE SCENE BRANCH (layered homography)")
    print(f"  two distinct planes found in {st['two_layers']} of {n} frames "
          f"({st['two_layers'] / n * 100:.0f}%)")
    if st["sep"]:
        s = np.asarray(st["sep"])
        print(f"  plane separation   median {np.median(s):.2f} px   p90 {np.percentile(s, 90):.2f}"
              f"   (threshold {layers.MIN_SEPARATION_PX:.2f})")
    lt = max(sum(st["lay_share"].values()), 1)
    for k, name in ((0, "dominant"), (1, "second"), (-1, "ambiguous")):
        extra = ""
        if k in (0, 1) and st["lay_y"][k]:
            extra = f"   mean y {np.median(st['lay_y'][k]):.0f}"
        print(f"    {name:<10s} {st['lay_share'][k] / lt * 100:5.1f}% of grid points{extra}")

    print("\n  IS THE RESIDUAL FIELD ACTUALLY PARALLAX? (the question the layers were for)")
    print(f"    {'':<28s}{'two planes':>12s}{'ONE plane':>12s}   chance / expected")
    print(f"    {'epipole reliable':<28s}{st['epi_ok_lay'] / n * 100:11.0f}%"
          f"{st['epi_ok_one'] / n * 100:11.0f}%")
    print(f"    {'agreement with background':<28s}{_pct(st['agree_lay']):>12s}"
          f"{_pct(st['agree_one']):>12s}   {CHANCE:.3f}  (chance)")
    print(f"    {'corr(|mu|, dist to epipole)':<28s}{_pct(st['corr_lay']):>12s}"
          f"{_pct(st['corr_one']):>12s}   strongly +ve if parallax")
    print("    Parallax means mu = gamma * (e - x), so |mu| MUST grow with distance from")
    print("    the epipole. Near-zero correlation means the field is a uniform shift and")
    print("    the epipole is at infinity, where fitting it as a point is meaningless.")

    c = st["scene_counts"]
    tot = max(st["scene_seeds"], 1)
    print(f"\n  candidates {tot / n:.1f}/frame")
    for k in ("flash", "rejected", "unjudged", "survives"):
        print(f"    {k:<9s} {c[k] / n:7.1f}/frame  {c[k] / tot * 100:5.1f}%")
    if a.direction:
        judged = c["rejected"] + c["survives"]
        if judged:
            got = c["rejected"] / judged
            print(f"    rejection among judged {got * 100:.1f}%  vs chance {CHANCE * 100:.1f}%"
                  f"  -> lift x{got / CHANCE:.2f}")
    else:
        print("    (direction test OFF -- `survives` here means `persisted`)")
    sl_tot = max(sum(st["surv_label"].values()), 1)
    print("  where the survivors sit:  " + "  ".join(
        f"{k} {v / sl_tot * 100:.1f}%" for k, v in st["surv_label"].items()))

    print("\nBOTH BRANCHES ON THE SAME TARGET")
    lb = max(st["labelled"], 1)
    print(f"  labelled frames                  {st['labelled']}")
    print(f"  found by the sky branch          {st['sky_drone_hit']:5d}  "
          f"{st['sky_drone_hit'] / lb * 100:5.1f}%")
    print(f"  found by the scene branch        {st['scene_drone_kept']:5d}  "
          f"{st['scene_drone_kept'] / lb * 100:5.1f}%")
    print(f"  found by BOTH                    {st['both']:5d}  {st['both'] / lb * 100:5.1f}%")
    print(f"  found by EITHER                  {st['either']:5d}  {st['either'] / lb * 100:5.1f}%")
    print(f"  MISSED by both                   {st['neither']:5d}  "
          f"{st['neither'] / lb * 100:5.1f}%")
    print("  `either` is the number that matters: two branches in different regimes are")
    print("  worth having only if they fail on different frames.")


if __name__ == "__main__":
    main()
