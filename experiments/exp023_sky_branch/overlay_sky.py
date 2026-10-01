"""The sky branch with depth-aware rings, rendered over a span and measured against itself.

    PYTHONPATH="experiments/exp023_sky_branch;experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
        py -3.13 -m overlay_sky --start 650 --end 964

    PYTHONPATH="experiments/exp023_sky_branch/analog_catch_2;experiments/exp023_sky_branch;experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
        py -3.13 -m overlay_sky --start 441 --end 800

`--no-video` prints the report without writing frames. EXP-022's `overlay_stage2.py` in
`exp017_motion_first/` stays frozen as the whole-ring baseline.

What this run answers
---------------------
EXP-022 implemented the depth-aware ring, self-checked it, and then never passed it -- the
whole run used an undifferentiated ring. This run turns it on, and measures the difference
**on the same candidates in the same frames** rather than against EXP-022's printed
numbers: `contrast` and `contrast_plain` are computed for every candidate, so the A/B is
internal and no execution-to-execution drift can be mistaken for an effect.

**Defaults reproduce EXP-022.** The ring restriction and the band refusal are opt-in, since
neither wins on both clips at matched false-alarm rate.

**Read the size diagnostic before any recall figure.** The scale ladder tops out below the
median labelled target on both clips, so the detector fires at 0.05x (O4) and 0.16x
(analog) the target's true size: it is finding a small dark sub-feature inside a grown box,
not the airframe.

It also reports `snr`, the area-aware statistic, beside `c`. EXP-022's `c` uses the scale
only to place the core and ring and never as evidence, so a 3 px and a 12 px blob at the
same contrast score identically -- and worse, `min(core)` is a biased order statistic whose
bias grows with core size, so size was already leaking into `c` backwards. `snr` uses the
core's mean and the standard error of that mean instead.

The scene branch is not re-run here. EXP-022 measured it and the layered homography is
already filed for deletion; repeating it would only add cost and invite the two changes to
be confused. This is the sky branch alone.

Colours
-------
  * RED circle, radius = detected scale -- kept. Thicker above twice the threshold.
  * DARK CYAN circle -- rejected, drawn ONLY with `--show-rejected`. There are ~190 of
                        these per frame against ~10 kept, so drawing them by default fills
                        the picture with circles that are not false alarms and makes the
                        render appear to contradict its own report.
  * MAGENTA circle   -- kept by the depth-aware ring but NOT by the whole ring: the
                        candidates this change ADDS.
  * ORANGE circle    -- the reverse: kept by the whole ring, dropped by the depth-aware
                        one. These are the false alarms the change REMOVES.

Every circle is drawn at least `MIN_DRAW_DIAMETER` (10 px) across, so a 3 px detection is
still visible on the video. Drawing only: the radius in the report and the dump is the
detected scale, unchanged.
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
from overlay_video import LAYER_COLOUR, TINT, load_boxes, on_drone, px, text_block, tint

W, H = CLIP["width"], CLIP["height"]
S = W / 1440.0

KEPT, REJ = (0, 0, 255), (110, 110, 40)
ADDED, REMOVED = (230, 80, 230), (40, 160, 240)
MIN_DRAW_DIAMETER = 10   # px in the output video; drawing only, never fed back into scoring


def draw_radius(diameter: float) -> int:
    """The radius a candidate is drawn at: its detected scale, floored at MIN_DRAW_DIAMETER."""
    return int(round(max(diameter, MIN_DRAW_DIAMETER) / 2.0))


def label_map(sl: skyline.Skyline) -> np.ndarray:
    """Stage 1 as an integer map, which is what the depth-aware ring indexes.

    `uncertain` is its own label rather than being folded into either side, and with
    `--refuse-uncertain` a candidate standing in it is dropped rather than scored. A ring built from the
    uncertain band samples exactly the pixels stage 1 declined to call -- a thin, fairly
    uniform ribbon -- which deflates `sigma_ring` and inflates `c`. Measured on a 17-frame
    probe before this was added: 85% of all kept candidates landed in a band covering 7%
    of the frame.

    Both this restriction and the refusal are opt-in (`--depth-ring`, `--refuse-uncertain`).
    EXP-023 measured them at matched false-alarm rate and neither wins on both clips, so
    the default stays EXP-022's whole ring.
    """
    m = np.full((H, W), silhouette.SCENE, np.int32)
    m[sl.sky] = silhouette.SKY
    m[sl.uncertain] = silhouette.UNCERTAIN
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=CLIP["n_frames"])
    ap.add_argument("--contrast", type=float, default=6.0,
                    help="threshold on c, in units of sigma_ring")
    ap.add_argument("--min-diameter", type=float, default=0.0,
                    help="reject blobs smaller than this, in px. OFF by default, and "
                         "note it cuts on the DETECTED SCALE, which on these clips is "
                         "0.05-0.16x the target's true size -- so a floor read off the "
                         "detected-scale distribution cuts an artefact of the ladder, "
                         "not small targets.")
    ap.add_argument("--depth-ring", action="store_true",
                    help="restrict each ring to the candidate's own stage-1 label. OFF by "
                         "default: EXP-023 measured it against the whole ring at matched "
                         "false-alarm rate and neither wins on both clips.")
    ap.add_argument("--refuse-uncertain", action="store_true",
                    help="drop candidates standing in the stage-1 uncertain band. Needs "
                         "--depth-ring. OFF by default: it cuts false alarms hard but "
                         "costs 17 (O4) and 22 (analog) drone frames.")
    ap.add_argument("--show-rejected", action="store_true",
                    help="also draw rejected candidates. OFF by default: there are ~190 "
                         "of them per frame against ~10 kept, so drawing them fills the "
                         "picture with circles that are NOT false alarms and makes the "
                         "render disagree with its own report.")
    ap.add_argument("--no-sky", action="store_true",
                    help="do not draw stage 1: no sky / uncertain tint, no horizon dots, "
                         "no stage-1 line in the caption. Stage-0 mask tints stay.")
    ap.add_argument("--no-truth", action="store_true",
                    help="do not draw anything derived from the labels: no green box, no "
                         "FOUND/missed text, no zoom inset (it is centred on the label). "
                         "Scoring and the report are unchanged.")
    ap.add_argument("--cap", type=int, default=4000)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--dump-floor", type=float, default=3.0,
                    help="record a candidate only at or above this c. Structural "
                         "rejections (cloud, bloom, size, band) and anything on the "
                         "target are recorded whatever their c, so nothing auditable is "
                         "lost. 0 records every candidate, which on analog is 530k rows "
                         "and 60 MB, 94% of it below the c=1.8 ceiling measured on pure "
                         "noise sky -- i.e. rows that carry no information.")
    ap.add_argument("--dump", default=None,
                    help="where to write the candidate CSV (see --dump-floor). Any "
                         "later cut is then a GROUP BY rather than another span.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = a.out or os.path.join(CLIP["out"], f"sky_{CLIP['name']}_{a.start}_{a.end}.mp4")
    dump_path = a.dump or os.path.join(CLIP["out"],
                                       f"candidates_{CLIP['name']}_{a.start}_{a.end}.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    stage0 = masks.Stage0()
    print("[stage 0] " + ", ".join(f"{k} {v:.2f}%" for k, v in stage0.coverage().items()))
    print(f"[sky] c >= {a.contrast}, ring = "
          f"{'DEPTH-AWARE (stage-1 label)' if a.depth_ring else 'WHOLE (EXP-022 baseline)'}"
          f", min diameter {a.min_diameter or 'off'}, cap {a.cap}, "
          # With a whole ring there are no labels, so the refusal cannot fire whatever the
          # flag says. Printing "REFUSED" there claims a condition the run did not apply.
          f"uncertain band "
          f"{'REFUSED' if (a.depth_ring and a.refuse_uncertain) else 'scored'}")
    boxes = load_boxes()

    cap = cv2.VideoCapture(CLIP["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.start - 2)
    ok, prev = cap.read()
    if not ok:
        raise SystemExit(f"cannot read frame {a.start - 1}")
    writer = (None if a.no_video else
              cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H)))

    st = _new_stats()
    rows: list[dict] = []

    for f in range(a.start, a.end + 1):
        ok, cur = cap.read()
        if not ok:
            break
        hmat = common.homography(common.prep(prev, 11), common.prep(cur, 11))
        valid = stage0.valid(hmat)
        prev = cur
        sl = skyline.split(cur, valid)
        lmap = label_map(sl) if a.depth_ring else None

        hits = silhouette.detect(cur, valid, threshold=a.contrast, label_map=lmap,
                                 cap=a.cap, min_diameter=a.min_diameter,
                                 refuse_uncertain=a.refuse_uncertain)
        box = boxes.get(f)
        img = np.zeros((H, W, 3), np.uint8) if a.no_video else cur.copy()
        if not a.no_video:
            _draw_background(img, sl, stage0, sky=not a.no_sky)
        _score(hits, box, f, a, st, rows, img)
        if not a.no_truth:
            _draw_target(img, cur, box, hits)
        _caption(img, f, a, sl, hits)
        st["n"] += 1
        if writer is not None:
            writer.write(img)

    if writer is not None:
        writer.release()
    cap.release()
    _write_dump(dump_path, rows)
    report(st, a, out, dump_path)


def _new_stats() -> dict:
    return dict(n=0, total=0, kept=0, rej={}, by_label={0: 0, 1: 0, 2: 0},
                drone_frames=0, drone_hit=0, drone_hit_plain=0,
                drone_c=[], drone_c_plain=[], drone_snr=[], drone_size=[], drone_sigma=[],
                drone_true_size=[],
                clutter_c=[], clutter_c_plain=[], clutter_snr=[], clutter_size=[],
                added=0, removed=0, ring_kept=[], floored=0, refused=0, cap_bound=0)


def _draw_background(img, sl, stage0, sky: bool = True) -> None:
    if sky:
        tint(img, sl.sky, TINT["sky"], 0.22)
        tint(img, sl.uncertain, TINT["uncertain"], 0.30)
    for name, m in stage0.layers.items():
        tint(img, m, LAYER_COLOUR[name], 0.35)
    if not sky:
        return
    xs = np.nonzero(sl.horizon >= 0)[0]
    for x in xs[::4]:
        cv2.circle(img, (int(x), int(sl.horizon[x])), px(2), (40, 220, 230), -1)


def _row(s, frame: int, hit: bool) -> dict:
    """One candidate as a dump row. Written for KEPT and REJECTED alike.

    EXP-023's first dump held only kept candidates, which meant a threshold could be
    raised from the file but never lowered, and no rejection could be audited without
    re-running the span. `reason` is empty for a kept candidate and names the veto
    otherwise, so a later cut can put any of them back.
    """
    return dict(frame=frame, x=round(s.x, 1), y=round(s.y, 1),
                diameter=round(s.diameter, 2), sigma=round(s.sigma, 3),
                response=round(s.response, 3), c=round(s.contrast, 3),
                c_plain=round(s.contrast_plain, 3), snr=round(s.snr, 3),
                ring_sigma=round(s.ring_sigma, 3), ring_median=round(s.ring_median, 1),
                core_min=round(s.core_min, 1), core_mean=round(s.core_mean, 2),
                n_core=s.n_core, ring_kept=round(s.ring_kept, 3),
                edge_grad=round(s.edge_grad, 2), label=int(s.label),
                floored=int(s.floored), reason=s.reason, kept=int(s.kept),
                on_target=int(hit))


def _score(hits, box, frame, a, st, rows, img) -> None:
    """Count, draw and record every candidate, including what the ring change did to it."""
    st["total"] += len(hits)
    on_target_kept = False
    for s in hits:
        if s.reason and s.reason != "below threshold":
            st["rej"][s.reason] = st["rej"].get(s.reason, 0) + 1
        keeps_plain = s.contrast_plain >= a.contrast
        if s.kept and not keeps_plain:
            st["added"] += 1
        if keeps_plain and not s.kept and s.reason == "below threshold":
            st["removed"] += 1
        hit_any = box is not None and on_drone(s.x, s.y, box)
        # Structural rejections and target rows are always recorded: those are exactly the
        # ones a later question is about. The floor only drops the noise bulk.
        structural = bool(s.reason) and s.reason != "below threshold"
        if s.contrast >= a.dump_floor or structural or hit_any:
            rows.append(_row(s, frame, hit_any))
        if not s.kept:
            if a.show_rejected and s.reason != "below threshold":
                cv2.circle(img, (int(s.x), int(s.y)), draw_radius(s.diameter),
                           REMOVED if keeps_plain else REJ, max(1, px(1)))
            continue

        st["kept"] += 1
        st["by_label"][int(s.label)] = st["by_label"].get(int(s.label), 0) + 1
        st["ring_kept"].append(s.ring_kept)
        st["floored"] += 1 if s.floored else 0
        if hit_any:
            on_target_kept = True
            st["drone_c"].append(s.contrast)
            st["drone_c_plain"].append(s.contrast_plain)
            st["drone_snr"].append(s.snr)
            st["drone_size"].append(s.diameter)
            st["drone_sigma"].append(s.ring_sigma)
            # The target's TRUE size beside the detected scale. These differ by 20x
            # on O4, which is the single most important number this run prints.
            st["drone_true_size"].append(float(max(box[2], box[3])))
        else:
            st["clutter_c"].append(s.contrast)
            st["clutter_c_plain"].append(s.contrast_plain)
            st["clutter_snr"].append(s.snr)
            st["clutter_size"].append(s.diameter)
        colour = ADDED if not keeps_plain else KEPT
        cv2.circle(img, (int(s.x), int(s.y)), draw_radius(s.diameter),
                   colour, max(1, px(1) + int(s.contrast >= 2 * a.contrast)))

    if box is not None:
        st["drone_frames"] += 1
        st["drone_hit"] += 1 if on_target_kept else 0
        st["drone_hit_plain"] += 1 if any(
            s.contrast_plain >= a.contrast and not (s.reason and s.reason != "below threshold")
            and on_drone(s.x, s.y, box) for s in hits) else 0


def _draw_target(img, cur, box, hits) -> None:
    if box is None:
        return
    bx, by, bw, bh = [int(v) for v in box]
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (60, 230, 60), max(1, px(2)))
    mine = [s for s in hits if s.kept and on_drone(s.x, s.y, box)]
    half = int(np.clip(max(bw, bh) * 1.6, 24 * S, 140 * S))
    panel = int(W / 3.2)
    scale = max(2, int(panel / (2 * half)))
    x0 = int(np.clip(bx + bw / 2.0 - half, 0, W - 2 * half))
    y0 = int(np.clip(by + bh / 2.0 - half, 0, H - 2 * half))
    crop = cur[y0:y0 + 2 * half, x0:x0 + 2 * half]
    if crop.size:
        big = cv2.resize(crop, (2 * half * scale, 2 * half * scale),
                         interpolation=cv2.INTER_NEAREST)
        bhh, bww = big.shape[:2]
        p0, q0 = W - bww - int(16 * S), int(16 * S)
        if p0 >= 0 and q0 + bhh <= H:
            img[q0:q0 + bhh, p0:p0 + bww] = big
            cv2.rectangle(img, (p0, q0), (p0 + bww, q0 + bhh), (255, 255, 255), max(1, px(2)))
    if mine:
        b = max(mine, key=lambda s: s.contrast)
        txt = f"drone: FOUND  c={b.contrast:.1f}s (plain {b.contrast_plain:.1f})  snr={b.snr:.0f}  {b.diameter:.1f}px"
    else:
        txt = "drone: missed"
    text_block(img, [txt], x0=bx, y0=max(int(20 * S), by - int(10 * S)), scale=0.55)


def _caption(img, f, a, sl, hits) -> None:
    kept = [s for s in hits if s.kept]
    lines = [f"frame {f}    c >= {a.contrast} sigma    "
             f"ring {'DEPTH-AWARE' if a.depth_ring else 'WHOLE'}"]
    if not a.no_sky:
        lines.append(f"stage 1: sky {sl.sky_fraction * 100:.1f}%  "
                     f"uncertain {sl.uncertain.mean() * 100:.1f}%"
                     + ("  [NO SKY]" if not sl.has_sky else ""))
    lines.append(f"candidates {len(hits)}   kept {len(kept)}")
    text_block(img, lines)


def _write_dump(path: str, rows: list[dict]) -> None:
    """One row per kept candidate, so any later cut is a GROUP BY and not a re-run.

    The same discipline `src/eval/records.py` applies to the main evaluation path: every
    quantity a later question needs is written once, at the point it was computed.
    EXP-022 had none of this, so every threshold question cost a 14-minute span.
    """
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[dump] {len(rows)} candidates -> {path}")


def _q(v, p):
    return float(np.percentile(v, p)) if len(v) else float("nan")


def report(st, a, out, dump_path) -> None:
    n = st["n"]
    if not n:
        raise SystemExit("no frames rendered")
    print(f"\n[sky] {n} frames -> {out}\n")

    print("LOAD")
    print(f"  candidates before threshold   {st['total'] / n:8.1f}/frame")
    print(f"  kept at c >= {a.contrast:<4g}              {st['kept'] / n:8.1f}/frame")
    for k, v in sorted(st["rej"].items(), key=lambda kv: -kv[1]):
        print(f"    rejected as {k:<14s}  {v / n:8.2f}/frame")
    tot = max(st["kept"], 1)
    if st["by_label"].get(-1, 0):
        # A whole ring computes no label, so a per-label breakdown would be three zeros
        # pretending to be a measurement.
        print("    (no stage-1 breakdown: --depth-ring not set, so no labels were read)")
    else:
        for code, name in ((0, "scene"), (1, "sky"), (2, "uncertain")):
            c = st["by_label"].get(code, 0)
            print(f"    kept in {name:<10s}        {c / n:8.2f}/frame  {c / tot * 100:5.1f}%")
    if a.depth_ring and st["ring_kept"]:
        print(f"  ring surviving the label restriction: median "
              f"{np.median(st['ring_kept']) * 100:.0f}%, p10 {_q(st['ring_kept'], 10) * 100:.0f}%")

    # Only meaningful when the two rings actually differ. With a whole ring this section is
    # three zeros and an identity, which reads like a measured null result rather than a
    # comparison that was never made.
    if a.depth_ring:
        print("\nWHAT THE DEPTH-AWARE RING CHANGED (same candidates, same frames)")
        print(f"  candidates it ADDED (kept now, below threshold with a whole ring)   "
              f"{st['added'] / n:7.2f}/frame")
        print(f"  candidates it REMOVED (kept with a whole ring, below threshold now) "
              f"{st['removed'] / n:7.2f}/frame")
        print(f"  drone frames found  depth-aware {st['drone_hit']}   "
              f"whole ring {st['drone_hit_plain']}   of {st['drone_frames']}")
    else:
        print(f"\n  drone frames found  {st['drone_hit']} of {st['drone_frames']}"
              f"   (whole ring; --depth-ring not set)")

    d, c = np.asarray(st["drone_c"]), np.asarray(st["clutter_c"])
    print("\nSEPARATION: can the statistic tell the target from the clutter?")
    for name, dv, cv_ in (("c  (contrast)", d, c),
                          ("snr (area-aware)", np.asarray(st["drone_snr"]),
                           np.asarray(st["clutter_snr"]))):
        if not len(dv) or not len(cv_):
            print(f"  {name:<18s} no data")
            continue
        p10 = _q(dv, 10)
        frac = float((cv_ >= p10).mean())
        print(f"  {name:<18s} target median {np.median(dv):8.1f}  p10 {p10:8.1f}   "
              f"clutter median {np.median(cv_):8.1f}   "
              f"**{frac * 100:5.1f}% of clutter reaches the target's p10**")
    print("  Lower is better: it is the share of false alarms a threshold set to keep 90%")
    print("  of the target detections would still let through.")

    ts, dz = np.asarray(st["drone_true_size"]), np.asarray(st["drone_size"])
    if len(ts) and len(dz):
        ratio = float(np.median(dz / np.maximum(ts, 1e-6)))
        print("\nIS THE DETECTOR ACTUALLY FINDING THE AIRFRAME?")
        # Per DETECTION, not per labelled frame: a frame holding several hits contributes
        # its box size several times, so this runs larger than the clip's median box (112
        # vs 71.6 px on O4). Both are true of different populations, and labelling it
        # "target size" without saying which is how the error this diagnostic exists to
        # catch got into the ledger in the first place.
        print(f"  labelled box of the frames hit (max of w,h)  median {np.median(ts):6.1f} px")
        print(f"  DETECTED scale on target           median {np.median(dz):6.1f} px")
        print(f"  ratio detected / true              median {ratio:6.3f}")
        print(f"  scale ladder spans {2 * np.sqrt(2) * silhouette.SIGMA_MIN:4.1f} - "
              f"{2 * np.sqrt(2) * silhouette.SIGMA_MAX:.0f} px diameter")
        if ratio < 0.5:
            print("  ** The detector is firing far below the target's size: it is finding a")
            print("     small dark sub-feature inside a generously grown box, not the")
            print("     airframe. Read every recall figure here as `a candidate landed in")
            print("     the box region`, NOT as `the drone was detected`. **")

    print("\nSIZE -- the distribution a minimum-diameter floor would cut")
    ds, cs = np.asarray(st["drone_size"]), np.asarray(st["clutter_size"])
    if len(ds) and len(cs):
        print(f"  target  diameter  p10 {_q(ds, 10):5.1f}  median {np.median(ds):5.1f}  "
              f"p90 {_q(ds, 90):5.1f} px")
        print(f"  clutter diameter  p10 {_q(cs, 10):5.1f}  median {np.median(cs):5.1f}  "
              f"p90 {_q(cs, 90):5.1f} px")
        print("  a floor at D removes this share of each:")
        for thr in (3.0, 4.0, 6.0, 8.0):
            print(f"    D = {thr:4.1f} px   clutter {-100 * (cs < thr).mean():6.1f}%   "
                  f"TARGET {-100 * (ds < thr).mean():6.1f}%")
        print("  A floor only helps where it cuts far more clutter than target.")

    if st["drone_c"]:
        print(f"\n  c on the drone      median {np.median(d):6.1f}  "
              f"(whole ring {np.median(st['drone_c_plain']):6.1f})")
        print(f"  sigma_ring there    median {np.median(st['drone_sigma']):6.2f} grey levels")
    if len(c):
        thr = np.array([4.0, 6.0, 8.0, 12.0, 20.0])
        far = silhouette.false_alarm_curve(c, thr, n)
        print("\n  measured false alarms/frame:  " +
              "   ".join(f"c>={t:.0f}: {v:.2f}" for t, v in zip(thr, far)))
    print(f"\n  re-cut any of this from {dump_path} without re-running the span.")


if __name__ == "__main__":
    main()
