"""EXP-025d: the frame split into sky and ground, a different detector on each, top 3 by c.

    PYTHONPATH="experiments/exp023_sky_branch/analog_catch_2;experiments/exp025_top3;experiments/exp023_sky_branch;experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
        py -3.13 -m overlay_split

Stage 1 (`skyline.split`, EXP-017) is recomputed on every frame and its `sky` mask is the
split: a blob whose centre is on a sky pixel belongs to the sky section, anything else to
the ground section. The uncertain band is not a third section; it falls on whichever side
stage 1 called.

  * **sky section** -- EXP-023's sky branch as EXP-025 runs it: kept candidates, c >= 6.
  * **ground section** -- EXP-024's 2-of-4 motion window as EXP-025c uses it: a sky-branch
    blob (any c, cloud vetoes dropped) confirmed by a window survivor within 9 px.

Both sections are pooled and ranked by the sky branch's contrast `c`, and the top `--top`
are drawn: **one ranking per frame, not one per section.** `--merge R` first folds every blob
within R px of a stronger one into it, before either section's test (the strongest blob
anchors, the rest of its 2R-wide circle joins it), so one object is one candidate. A merged
blob keeps the anchor's centre, c and section, grows to cover its members, and is on the
drone if any member is; it passes the sky test if any member has c >= 6, the ground test
if a window survivor is within 9 px of any member. Drawn from EXP-023's candidate
dump and EXP-024's `seeds_k4_` dump; only stage 1 is computed here, no detector re-run.

The clean look: stage-0 tints, red circles at least 10 px across tagged `#rank c sky|gnd`,
`#1` thicker, a two-line caption. The split is drawn as stage 1's horizon (lowest sky row
per column), a thin line; `--no-split-line` drops it. Nothing is drawn from the labels.

Writes a CSV of every shown blob and one row per frame for the drone (`--hist` input to
`split_rank_hist.py`): its best rank in the pooled ranking, the section that ranked it, or
why it was not ranked.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import cv2
import numpy as np

import common
import masks
import skyline
from clipcfg import CLIP
from overlay_sky import KEPT, _draw_background, draw_radius
from overlay_video import load_boxes, px, text_block
from overlay_window_skyc import confirmed, load_survivors

RUNS = "runs/sofa_analog/"
W, H = CLIP["width"], CLIP["height"]
SECTIONS = ("sky", "ground")
TAG = {"sky": "sky", "ground": "gnd"}
SPLIT_LINE = (60, 200, 230)
SHOWN_FIELDS = ["frame", "rank", "section", "x", "y", "c", "diameter", "members",
                "on_target"]
DRONE_FIELDS = ["frame", "outcome", "rank", "section", "c"]


def load_dump(path: str) -> dict[int, list[dict]]:
    """EXP-023's candidates by frame, cloud vetoes left out (a shape rule applied at any c)."""
    by_frame = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["reason"] == "cloud":
                continue
            by_frame[int(r["frame"])].append(dict(
                x=float(r["x"]), y=float(r["y"]), c=float(r["c"]),
                diameter=float(r["diameter"]), kept=r["kept"] == "1",
                on_target=int(r["on_target"])))
    return by_frame


def section_of(sky: np.ndarray, x: float, y: float) -> str:
    xi, yi = int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1))
    return "sky" if sky[yi, xi] else "ground"


def merge(blobs: list[dict], radius: float) -> list[dict]:
    """One frame's blobs as objects: strongest first, each anchor takes every remaining blob
    within `radius` px. The anchor keeps its centre, c and section; the diameter grows to
    cover the members, and the object is on the drone if any member is. `parts` keeps the
    members for the sections' tests. `radius` 0 leaves every blob its own object."""
    left = sorted(blobs, key=lambda b: (-b["c"], b["on_target"]))
    out = []
    while left:
        a, rest = left[0], left[1:]
        dist = [np.hypot(b["x"] - a["x"], b["y"] - a["y"]) for b in rest]
        near = [b for b, d in zip(rest, dist) if d <= radius] if radius > 0 else []
        left = [b for b, d in zip(rest, dist) if not (radius > 0 and d <= radius)]
        reach = max([a["diameter"] / 2] + [np.hypot(b["x"] - a["x"], b["y"] - a["y"])
                                           + b["diameter"] / 2 for b in near])
        out.append(dict(a, diameter=2 * reach, members=1 + len(near), parts=[a, *near],
                        kept=any(b["kept"] for b in (a, *near)),
                        on_target=max(b["on_target"] for b in (a, *near))))
    return out


def pool(objs: list[dict], seeds: np.ndarray | None, radius: float) -> list[dict]:
    """Both sections' objects, highest c first. Ties put clutter ahead of the drone. A sky
    object passes when any member is kept (c >= 6), a ground object when a window survivor
    is within `radius` of any member."""
    upper = [o for o in objs if o["section"] == "sky" and o["kept"]]
    ground = [o for o in objs if o["section"] == "ground"]
    parts = [dict(p, obj=k) for k, o in enumerate(ground) for p in o["parts"]]
    hit = {p["obj"] for p in confirmed(parts, seeds, radius)}
    lower = [o for k, o in enumerate(ground) if k in hit]
    return sorted(upper + lower, key=lambda b: (-b["c"], b["on_target"]))


def drone_row(f: int, ranked: list[dict], blobs: list[dict], box, sky) -> dict:
    """Where the drone landed this frame: ranked (best rank, its section), `dropped` (a blob
    on it existed but its section's detector did not pass it), or `no blob`."""
    for i, b in enumerate(ranked, 1):
        if b["on_target"]:
            return dict(frame=f, outcome="ranked", rank=i, section=b["section"], c=b["c"])
    on = [b for b in blobs if b["on_target"]]
    if on:
        best = max(on, key=lambda b: b["c"])
        return dict(frame=f, outcome="dropped", rank="", section=best["section"], c=best["c"])
    bx, by, bw, bh = box
    return dict(frame=f, outcome="no blob", rank="",
                section=section_of(sky, bx + bw / 2, by + bh / 2), c="")


def draw(img, shown: list[dict], sl: skyline.Skyline, line: bool) -> None:
    if line:
        xs = np.nonzero(sl.horizon >= 0)[0]
        if len(xs):
            pts = np.stack([xs, sl.horizon[xs]], axis=1).astype(np.int32)
            cv2.polylines(img, [pts], False, SPLIT_LINE, max(1, px(1)), cv2.LINE_AA)
    for i, b in enumerate(shown, 1):
        p, r = (int(b["x"]), int(b["y"])), draw_radius(b["diameter"])
        cv2.circle(img, p, r, KEPT, max(1, px(1)) + (1 if i == 1 else 0))
        tag = f"#{i} {b['c']:.1f} {TAG[b['section']]}"
        org = (p[0] + r + 3, p[1] - r - 2)
        for colour, thick in (((0, 0, 0), 3), ((255, 255, 255), 1)):
            cv2.putText(img, tag, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, thick,
                        cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=441)
    ap.add_argument("--end", type=int, default=800)
    ap.add_argument("--k", type=int, default=4, help="window length of the seed dump")
    ap.add_argument("--min-appear", type=int, default=2)
    ap.add_argument("--top", type=int, default=3, help="at most this many per frame, pooled")
    ap.add_argument("--radius", type=float, default=9.0,
                    help="seed-to-blob distance that confirms a ground blob, px")
    ap.add_argument("--merge", type=float, default=0.0,
                    help="fold blobs within this many px of a stronger one into it, before "
                         "the sections' tests "
                         "(0: off, EXP-025d)")
    ap.add_argument("--window-dump", default=None,
                    help="EXP-024 seed dump (default: its seeds_k{k}_ CSV for this span)")
    ap.add_argument("--sky-dump",
                    default=RUNS + "exp023_sky_branch/candidates_catch_2_441_800.csv")
    ap.add_argument("--no-split-line", action="store_true",
                    help="do not draw stage 1's horizon")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    span = f"{CLIP['name']}_{a.start}_{a.end}"
    wdump = a.window_dump or RUNS + f"exp024_window_length/seeds_k{a.k}_{span}.csv"
    out = a.out or RUNS + (f"exp025_top3/split_sky_window{a.min_appear}of{a.k}"
                           f"_top{a.top}{f'_merge{a.merge:g}' if a.merge else ''}"
                           f"_{span}.mp4")
    stem = os.path.splitext(out)[0]
    os.makedirs(os.path.dirname(out), exist_ok=True)

    survivors = load_survivors(wdump, a.min_appear)
    blobs = load_dump(a.sky_dump)
    boxes = load_boxes()
    stage0 = masks.Stage0()
    print("[stage 0] " + ", ".join(f"{k} {v:.2f}%" for k, v in stage0.coverage().items()))
    print(f"[sky section] {a.sky_dump}, c >= 6\n[ground section] {a.min_appear} of {a.k} "
          f"from {wdump}, blob within {a.radius:g} px, any c\n[rank] pooled by c, top {a.top}"
          + (f", blobs within {a.merge:g} px merged" if a.merge else ""))

    cap = cv2.VideoCapture(CLIP["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.start - 2)
    ok, prev = cap.read()
    if not ok:
        raise SystemExit(f"cannot read frame {a.start - 1}")
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))
    shown_rows, drone_rows = [], []
    load = dict.fromkeys(SECTIONS, 0)
    shown_n = dict.fromkeys(SECTIONS, 0)
    sky_frac, folded = [], 0
    for f in range(a.start, a.end + 1):
        ok, cur = cap.read()
        if not ok:
            break
        hmat = common.homography(common.prep(prev, 11), common.prep(cur, 11))
        prev = cur
        sl = skyline.split(cur, stage0.valid(hmat))
        sky_frac.append(sl.sky_fraction)
        fb = [dict(b) for b in blobs.get(f, [])]
        for b in fb:
            b["section"] = section_of(sl.sky, b["x"], b["y"])
        objs = merge(fb, a.merge)
        folded += len(fb) - len(objs)
        ranked = pool(objs, survivors.get(f), a.radius)
        shown = ranked[:a.top]
        for b in ranked:
            load[b["section"]] += 1
        for b in shown:
            shown_n[b["section"]] += 1
        shown_rows += [dict(frame=f, rank=i, section=b["section"], x=b["x"], y=b["y"],
                            c=b["c"], diameter=b["diameter"], members=b["members"],
                            on_target=b["on_target"])
                       for i, b in enumerate(shown, 1)]
        if f in boxes:
            drone_rows.append(drone_row(f, ranked, objs, boxes[f], sl.sky))

        img = cur.copy()
        _draw_background(img, None, stage0, sky=False)
        draw(img, shown, sl, not a.no_split_line)
        text_block(img, [f"frame {f}    sky: sky branch c>=6   ground: {a.min_appear} of "
                         f"{a.k} window    top {a.top} by sky c",
                         f"ranked  sky {sum(b['section'] == 'sky' for b in ranked)}"
                         f"  ground {sum(b['section'] == 'ground' for b in ranked)}"
                         f"    shown {len(shown)}"])
        writer.write(img)
    writer.release()
    cap.release()

    for path, fields, rows in ((stem + ".csv", SHOWN_FIELDS, shown_rows),
                               (stem + "_drone.csv", DRONE_FIELDS, drone_rows)):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    report(a, out, stem, load, shown_n, sky_frac, drone_rows, folded)


def report(a, out, stem, load, shown_n, sky_frac, drone_rows, folded) -> None:
    n = len(sky_frac)
    print(f"\n[video] {n} frames -> {out}\n[dump] every shown blob -> {stem}.csv"
          f"\n[dump] the drone, per labelled frame -> {stem}_drone.csv")
    print(f"\n  stage 1 sky fraction: median {np.median(sky_frac):.0%}, "
          f"no sky in {sum(s == 0 for s in sky_frac)} frames")
    print(f"  ranked/frame  sky {load['sky'] / n:5.2f}  ground {load['ground'] / n:5.2f}"
          f"  total {sum(load.values()) / n:5.2f}")
    print(f"  shown/frame   sky {shown_n['sky'] / n:5.2f}  ground {shown_n['ground'] / n:5.2f}"
          f"  total {sum(shown_n.values()) / n:5.2f}")
    if a.merge:
        print(f"  merge {a.merge:g} px folded {folded} candidates ({folded / n:.2f}/frame) "
              f"into stronger ones, before the sections' tests")
    print(f"\n  drone, of {len(drone_rows)} labelled frames      sky  ground  total")
    for name, keep in (("ranked", lambda r: r["outcome"] == "ranked"),
                       (f"in the top {a.top}", lambda r: r["outcome"] == "ranked"
                        and r["rank"] <= a.top),
                       *((f"  #{i}", (lambda i: lambda r: r["rank"] == i)(i))
                         for i in range(1, a.top + 1)),
                       ("dropped by its section", lambda r: r["outcome"] == "dropped"),
                       ("no blob on it", lambda r: r["outcome"] == "no blob")):
        c = {s: sum(keep(r) and r["section"] == s for r in drone_rows) for s in SECTIONS}
        print(f"    {name:28s} {c['sky']:5d} {c['ground']:7d} {sum(c.values()):6d}")
    print("\n  'drone' = a blob's on_target from EXP-023's dump: inside the label box grown by"
          "\n  max(10 px, 25%). Section of a dropped drone = its best blob's; of no blob = the"
          "\n  label box centre's.")


if __name__ == "__main__":
    main()
