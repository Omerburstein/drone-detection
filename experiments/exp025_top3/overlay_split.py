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
if a window survivor is within 9 px of any member. `--merge-score sum` (EXP-030) scores
the object by the sum of its members' c instead of the anchor's; the tests are unchanged.
Drawn from EXP-023's candidate
dump and EXP-024's `seeds_k4_` dump; only stage 1 is computed here, no detector re-run.

The clean look: stage-0 tints, red circles at least `--min-draw` px across (default 10) tagged `#rank c sky|gnd`,
`#1` thicker, a two-line caption. The split is drawn as stage 1's horizon (lowest sky row
per column), a thin line; `--no-split-line` drops it. Nothing is drawn from the labels.

`--osd-grid` (EXP-027) first drops every blob that is a character in a row on the analog
OSD's grid -- the artificial horizon's white dashes, which slide with pitch and roll so no
static mask holds them. `src.algo.masking.on_osd_grid` on the current frame: copies of the
blob at two of the positions +-1 and +-2 columns away. Off by default.

`--kinematic` (EXP-028) then overrules every answer that moves faster than a drone can
(`src.algo.kinematics`). The pool's objects are *strong* and may start a track; merged
objects outside it at c >= `--c-keep` are *weak* and may only continue a confirmed one. Only
objects on a confirmed track are ranked, by c or (`--rank-by track`) by the track's decayed
sum of c, so a circle cannot appear somewhere new without a plausible history, and cannot
jump further than the speed limit allows. Off by
default; without it every output is byte-identical to EXP-027.

`--rank-by drone` (EXP-032) ranks the gate's survivors by `src.algo.selection.DroneSelector`
instead of by evidence alone. The cues are evidence, colour against the previous #1s,
closeness to recent #1s (decaying with age), the track's hit rate and age, how well its
path is predicted at constant velocity, and its speed against the scene. `--cues` weights
them.

**Every pixel flag is quoted at `--ref-width` px wide** (default 1440, as every other
constant in the project) and scaled to the clip's width: `--radius`, `--merge`,
`--min-move`, `--ceiling`, `--min-draw`, the OSD test's box and the selector's lengths.
A driver holding numbers tuned on the 960-wide analog clips passes `--ref-width 960`, and
its outputs are byte-identical to before. Without the scaling, the FIELD `.raw` at 4128 wide
read merge 20 and 30 identically (EXP-031).

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
from src.algo.kinematics import KinematicTracker, SpeedLimit
from src.algo.masking import on_osd_grid
from src.algo.scale import REF_WIDTH, PixelScale
from src.algo.selection import CUES, CueWeights, DroneSelector
from overlay_sky import KEPT, MIN_DRAW_DIAMETER, _draw_background
from overlay_video import load_boxes, px, text_block
from overlay_window_skyc import confirmed, load_survivors

RUNS = "runs/sofa_analog/"
RANKED_BY = {"c": "c", "track": "track evidence, decay {decay:g}",
             "drone": "the drone selector's score"}
W, H = CLIP["width"], CLIP["height"]
SECTIONS = ("sky", "ground")
TAG = {"sky": "sky", "ground": "gnd"}
SPLIT_LINE = (60, 200, 230)
SHOWN_FIELDS = ["frame", "rank", "section", "x", "y", "c", "diameter", "members",
                "on_target"]
DRONE_FIELDS = ["frame", "outcome", "rank", "section", "c"]
FRAME_FIELDS = ["frame", "sky_fraction", "ranked_sky", "ranked_ground", "shown_sky",
                "shown_ground", "folded", "osd_vetoed", "osd_vetoed_on_target"]
KINEMATIC_SHOWN = ["track", "reason", "evidence"]
KINEMATIC_FRAMES = ["born", "overruled", "continued_weak", "coasting", "too_still"]
SELECTOR_SHOWN = ["drone_score", *(f"cue_{c}" for c in CUES)]
# The grid test's box side, blob diameter clamped to a dash's size: 8-14 px at 960 wide,
# quoted at 1440 like every other length here.
OSD_BOX = (12.0, 21.0)
TAG_FONT = 0.675  # the circle tags' font scale at 1440 wide: 0.45 at 960, as before
# The pixel flags' defaults, quoted at REF_WIDTH whatever `--ref-width` a driver declares:
# a driver's --ref-width says where *its* numbers come from, not where these do.
FLAG_DEFAULTS = {"radius": 13.5, "ceiling": 37.5, "min_draw": MIN_DRAW_DIAMETER * 1.5}


def _blob(r: dict) -> dict:
    return dict(x=float(r["x"]), y=float(r["y"]), c=float(r["c"]),
                diameter=float(r["diameter"]), kept=r["kept"] == "1",
                on_target=int(r["on_target"]))


def load_dump(path: str) -> dict[int, list[dict]]:
    """EXP-023's candidates by frame, cloud vetoes left out (a shape rule applied at any c)."""
    by_frame = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["reason"] != "cloud":
                by_frame[int(r["frame"])].append(_blob(r))
    return by_frame


class DumpStream:
    """`load_dump` one frame at a time, for dumps too big to hold (FIELD: ~5M rows).

    `get(f)` must be called with increasing `f`, which is how `main` walks the span;
    `overlay_sky` writes its dump in frame order, so one forward pass serves the run."""

    def __init__(self, path: str) -> None:
        self._fh = open(path, encoding="utf-8")
        self._rows = csv.DictReader(self._fh)
        self._next = next(self._rows, None)

    def get(self, f: int, default=()) -> list[dict]:
        out = []
        while self._next is not None and int(self._next["frame"]) <= f:
            r, self._next = self._next, next(self._rows, None)
            if int(r["frame"]) == f and r["reason"] != "cloud":
                out.append(_blob(r))
        return out or list(default)


def section_of(sky: np.ndarray, x: float, y: float) -> str:
    xi, yi = int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1))
    return "sky" if sky[yi, xi] else "ground"


def osd_vetoed(gray: np.ndarray, blobs: list[dict], box: tuple[float, float]) -> list[dict]:
    """The blobs that sit in a row on the OSD grid, tested on a dash-sized square box whose
    side is clamped to `box`, in picture px."""
    out = []
    for b in blobs:
        side = float(np.clip(b["diameter"], *box))
        if on_osd_grid(gray, (b["x"] - side / 2, b["y"] - side / 2, side, side)):
            out.append(b)
    return out


def merge(blobs: list[dict], radius: float, score: str = "max") -> list[dict]:
    """One frame's blobs as objects: strongest first, each anchor takes every remaining blob
    within `radius` px. The anchor keeps its centre, c and section; the diameter grows to
    cover the members, and the object is on the drone if any member is. `parts` keeps the
    members for the sections' tests. `radius` 0 leaves every blob its own object.

    `score` "sum" (EXP-030) gives the object the sum of its members' c instead of the
    anchor's, keeping the anchor's own as `c_max`. Anchors are still chosen by their own c.

    Vectorised per anchor: FIELD frames carry ~1400 candidates, where a pairwise Python
    loop costs seconds a frame."""
    order = sorted(blobs, key=lambda b: (-b["c"], b["on_target"]))
    if not order:
        return []
    xy = np.array([[b["x"], b["y"]] for b in order])
    half = np.array([b["diameter"] / 2 for b in order])
    alive = np.ones(len(order), bool)
    out = []
    for i, a in enumerate(order):
        if not alive[i]:
            continue
        alive[i] = False
        d = np.hypot(xy[:, 0] - a["x"], xy[:, 1] - a["y"])
        idx = np.nonzero(alive & (d <= radius))[0] if radius > 0 else np.array([], int)
        alive[idx] = False
        near = [order[k] for k in idx]
        reach = max([a["diameter"] / 2, *(d[idx] + half[idx])])
        obj = dict(a, diameter=2 * reach, members=1 + len(near), parts=[a, *near],
                   kept=any(b["kept"] for b in (a, *near)),
                   on_target=max(b["on_target"] for b in (a, *near)))
        if score == "sum":
            obj.update(c=sum(b["c"] for b in (a, *near)), c_max=a["c"])
        out.append(obj)
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


def drone_row(f: int, ranked: list[dict], blobs: list[dict], box, sky,
              overruled: list[dict] = ()) -> dict:
    """Where the drone landed this frame: ranked (best rank, its section), `overruled` (the
    kinematic gate held a blob on it back), `dropped` (a blob on it existed but its section's
    detector did not pass it), or `no blob`."""
    for i, b in enumerate(ranked, 1):
        if b["on_target"]:
            return dict(frame=f, outcome="ranked", rank=i, section=b["section"], c=b["c"])
    on = [b for b in overruled if b["on_target"]]
    if on:
        best = max(on, key=lambda b: b["c"])
        return dict(frame=f, outcome="overruled", rank="", section=best["section"], c=best["c"])
    on = [b for b in blobs if b["on_target"]]
    if on:
        best = max(on, key=lambda b: b["c"])
        return dict(frame=f, outcome="dropped", rank="", section=best["section"], c=best["c"])
    bx, by, bw, bh = box
    return dict(frame=f, outcome="no blob", rank="",
                section=section_of(sky, bx + bw / 2, by + bh / 2), c="")


def speed_limit(a) -> SpeedLimit:
    """The `--kinematic` limit from the command line, at this clip's width and fps, with its
    pixels quoted at `--ref-width`. The slack is not a flag: its default is quoted at 1440
    whatever `--ref-width` says, as `FLAG_DEFAULTS`."""
    return SpeedLimit(width_px=W, fps=CLIP["fps"], v_max_ms=a.v_max, min_range_m=a.min_range,
                      hfov_deg=a.hfov, ceiling_px=a.ceiling, ref_width_px=a.ref_width,
                      slack_px=SpeedLimit.slack_px * a.ref_width / REF_WIDTH)


def selector(a) -> DroneSelector:
    """The `--rank-by drone` selector from the command line, at this clip's width; its own
    lengths are quoted at 1440. Colour is compared only where the clip config uses it: CVBS
    chroma is crawl, not colour."""
    return DroneSelector(PixelScale(W), CueWeights.parse(a.cues),
                         memory_decay=a.memory_decay, use_colour=CLIP.get("use_colour", True))


def gate(tracker: KinematicTracker, f: int, ranked: list[dict], objs: list[dict],
         c_keep: float, prev_to_cur: np.ndarray,
         rank_by: str = "c") -> tuple[list[dict], list[dict], object]:
    """One frame through the kinematic gate: the objects it passes, best first by `rank_by`
    (`c`, or the track's evidence for `track` and `drone`), and the ones it overruled.
    `ranked` is strong; the rest of `objs` at c >= `c_keep` is weak. Each object is tagged
    with its `track`, the gate's `reason`, the track's `evidence` and its `speed` against
    the scene in px/frame (NaN until the track has two sightings)."""
    pooled = {id(o) for o in ranked}
    cands = ranked + [o for o in objs if id(o) not in pooled and o["c"] >= c_keep]
    xy = np.array([[o["x"], o["y"]] for o in cands], float).reshape(-1, 2)
    res = tracker.step(f, xy, np.array([o["c"] for o in cands], float),
                       np.arange(len(cands)) < len(ranked), prev_to_cur)
    for o, tid, why, ev, moved, span in zip(cands, res.track_id, res.reason, res.evidence,
                                            res.moved, res.move_span):
        o["track"], o["reason"], o["evidence"] = int(tid), why, round(float(ev), 3)
        o["speed"] = float(moved) / span if span > 0 else float("nan")
    passed = [o for o, ok in zip(cands, res.shown) if ok]
    held = [o for o, ok in zip(cands, res.shown) if not ok]
    key = "c" if rank_by == "c" else "evidence"
    return sorted(passed, key=lambda b: (-b[key], b["on_target"])), held, res


def select(sel: DroneSelector, f: int, passed: list[dict], img: np.ndarray,
           prev_to_cur: np.ndarray) -> list[dict]:
    """The gate's survivors re-ranked by the drone selector, best first. Each is tagged with
    its `drone_score` and every cue. Called on every frame, even an empty one, so the
    selector's memory follows the camera without a gap.

    A merged object's colour is read at its anchor's own size: its merged diameter spans
    every member and would average the drone with the sky around it."""
    xy = np.array([[o["x"], o["y"]] for o in passed], float).reshape(-1, 2)
    looks = sel.look(img, xy, [o["parts"][0]["diameter"] for o in passed])
    res = sel.step(f, np.array([o["track"] for o in passed], int), xy,
                   np.array([o["evidence"] for o in passed], float),
                   np.array([o["speed"] for o in passed], float), looks, prev_to_cur)
    for k, o in enumerate(passed):
        o["drone_score"] = round(float(res.score[k]), 4)
        for c in CUES:
            o[f"cue_{c}"] = round(float(res.cues[c][k]), 4)
    return [passed[k] for k in res.order]


def top_jumps(rows: list[dict], reach: float) -> tuple[int, int]:
    """How often #1 moved further than `reach` between consecutive frames that both had
    one: (jumps, consecutive pairs). The user's complaint, counted."""
    first = {int(r["frame"]): (float(r["x"]), float(r["y"])) for r in rows
             if int(r["rank"]) == 1}
    pairs = [(first[f - 1], first[f]) for f in first if f - 1 in first]
    return sum(np.hypot(p[0] - q[0], p[1] - q[1]) > reach for p, q in pairs), len(pairs)


def draw(img, shown: list[dict], sl: skyline.Skyline, line: bool, min_draw: float) -> None:
    """Circles and tags for the shown objects, `min_draw` in picture px. Every stroke and
    the font scale with the clip's width, so a 4128-wide frame is as legible as a 960 one."""
    if line:
        xs = np.nonzero(sl.horizon >= 0)[0]
        if len(xs):
            pts = np.stack([xs, sl.horizon[xs]], axis=1).astype(np.int32)
            cv2.polylines(img, [pts], False, SPLIT_LINE, max(1, px(1)), cv2.LINE_AA)
    font = TAG_FONT * W / REF_WIDTH
    for i, b in enumerate(shown, 1):
        p, r = (int(b["x"]), int(b["y"])), int(round(max(b["diameter"], min_draw) / 2))
        cv2.circle(img, p, r, KEPT, max(1, px(1)) + (px(1.5) if i == 1 else 0))
        tag = f"#{i} {b['c']:.1f} {TAG[b['section']]}"
        if "drone_score" in b:
            tag += f" {b['drone_score']:.2f}"
        org = (p[0] + r + px(4.5), p[1] - r - px(3))
        for colour, thick in (((0, 0, 0), px(4.5)), ((255, 255, 255), px(1.5))):
            cv2.putText(img, tag, org, cv2.FONT_HERSHEY_SIMPLEX, font, colour, thick,
                        cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=441)
    ap.add_argument("--end", type=int, default=800)
    ap.add_argument("--k", type=int, default=4, help="window length of the seed dump")
    ap.add_argument("--min-appear", type=int, default=2)
    ap.add_argument("--top", type=int, default=3, help="at most this many per frame, pooled")
    ap.add_argument("--ref-width", type=float, default=REF_WIDTH,
                    help="every pixel flag below is quoted at this width and scaled to the "
                         "clip's (default 1440, the project's reference). Drivers holding "
                         "numbers tuned on the 960-wide analog clips pass 960")
    ap.add_argument("--radius", type=float, default=None,
                    help="seed-to-blob distance that confirms a ground blob, px at "
                         "--ref-width (default 13.5 at 1440 wide, 9 on analog)")
    ap.add_argument("--merge", type=float, default=0.0,
                    help="fold blobs within this many px (at --ref-width) of a stronger one "
                         "into it, before the sections' tests (0: off, EXP-025d)")
    ap.add_argument("--merge-score", choices=("max", "sum"), default="max",
                    help="a merged object's c: its anchor's (the strongest member's), or the "
                         "sum of every member's (EXP-030)")
    ap.add_argument("--osd-grid", action="store_true",
                    help="drop blobs in a row on the OSD character grid (the horizon "
                         "dashes) before merging (EXP-027)")
    ap.add_argument("--kinematic", action="store_true",
                    help="overrule answers that move faster than a drone can, and pops with "
                         "no plausible history (EXP-028)")
    ap.add_argument("--v-max", type=float, default=40.0,
                    help="--kinematic: the drone's top closing speed, m/s")
    ap.add_argument("--min-range", type=float, default=10.0,
                    help="--kinematic: the closest range the limit is computed at, m")
    ap.add_argument("--hfov", type=float, default=130.0,
                    help="--kinematic: the camera's horizontal field of view, degrees")
    ap.add_argument("--ceiling", type=float, default=None,
                    help="--kinematic: the limit never exceeds this, px/frame at --ref-width "
                         "(default 37.5 at 1440 wide, 25 on analog)")
    ap.add_argument("--c-keep", type=float, default=6.0,
                    help="--kinematic: the weakest c that may continue a confirmed track")
    ap.add_argument("--confirm", type=int, default=2,
                    help="--kinematic: hits a track needs before it is shown")
    ap.add_argument("--max-coast", type=int, default=5,
                    help="--kinematic: missed frames a confirmed track survives")
    ap.add_argument("--rank-by", choices=("c", "track", "drone"), default="track",
                    help="--kinematic: rank by this frame's c, by the track's decayed sum "
                         "of c (steadier #1), or by the drone selector's score over every "
                         "cue (EXP-032)")
    ap.add_argument("--cues", default="",
                    help="--rank-by drone: cue weights as <cue>=<w>,... over "
                         f"{', '.join(CUES)}; unnamed cues keep their defaults "
                         f"({CueWeights().label()})")
    ap.add_argument("--memory-decay", type=float, default=0.9,
                    help="--rank-by drone: per-frame decay of a past #1's support, and of "
                         "the learned drone colour's influence")
    ap.add_argument("--decay", type=float, default=0.8,
                    help="--kinematic: per-frame decay of a track's evidence")
    ap.add_argument("--min-move", type=float, default=0.0,
                    help="--kinematic: the MOVING FACTOR (EXP-029), the limit from below. "
                         "A confirmed track must have travelled this many px (at "
                         "--ref-width) against the static scene over --move-window frames "
                         "or it is overruled as too-still. 0 disables it, and then nothing "
                         "here changes. 20 at 960 (30 at 1440) is EXP-029's measured "
                         "setting on catch_2")
    ap.add_argument("--move-window", type=int, default=5,
                    help="--min-move: frames the travel is measured across")
    ap.add_argument("--window-dump", default=None,
                    help="EXP-024 seed dump (default: its seeds_k{k}_ CSV for this span)")
    ap.add_argument("--sky-dump",
                    default=RUNS + "exp023_sky_branch/candidates_catch_2_441_800.csv")
    ap.add_argument("--no-split-line", action="store_true",
                    help="do not draw stage 1's horizon")
    ap.add_argument("--min-draw", type=float, default=None,
                    help="smallest circle drawn, px across at --ref-width (default 15 at 1440 "
                         "wide, 10 on analog); appearance only, never filters")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--report-only", action="store_true",
                    help="no run: print the report from --out's existing _frames.csv and "
                         "_drone.csv (how run_clips.py reports chunks it has joined)")
    a = ap.parse_args()
    for key, quoted in FLAG_DEFAULTS.items():
        if getattr(a, key) is None:
            setattr(a, key, quoted * a.ref_width / REF_WIDTH)
    if a.rank_by == "drone" and not a.kinematic:
        ap.error("--rank-by drone ranks the kinematic gate's tracks: add --kinematic")
    try:
        weights = CueWeights.parse(a.cues)  # refused here, not after hours of work
    except ValueError as e:
        ap.error(str(e))

    span = f"{CLIP['name']}_{a.start}_{a.end}"
    wdump = a.window_dump or RUNS + f"exp024_window_length/seeds_k{a.k}_{span}.csv"
    out = a.out or RUNS + (f"exp025_top3/split_sky_window{a.min_appear}of{a.k}"
                           f"_top{a.top}{f'_merge{a.merge:g}' if a.merge else ''}"
                           f"{'_drone' if a.rank_by == 'drone' else ''}_{span}.mp4")
    stem = os.path.splitext(out)[0]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if a.report_only:
        report(a, out, stem, *read_tables(stem))
        return

    limit = speed_limit(a)
    scale = PixelScale(W, a.ref_width)
    radius, merge_px, min_draw = scale.px(a.radius), scale.px(a.merge), scale.px(a.min_draw)
    own = PixelScale(W)   # this module's own constants are quoted at 1440
    osd_box = (own.px(OSD_BOX[0]), own.px(OSD_BOX[1]))
    survivors = load_survivors(wdump, a.min_appear)
    blobs = DumpStream(a.sky_dump)
    boxes = load_boxes()
    stage0 = masks.Stage0()
    print("[stage 0] " + ", ".join(f"{k} {v:.2f}%" for k, v in stage0.coverage().items()))
    print(f"[scale] {W} px wide, pixel flags quoted at {a.ref_width:g}: x{scale.factor:.3f}")
    print(f"[sky section] {a.sky_dump}, c >= 6\n[ground section] {a.min_appear} of {a.k} "
          f"from {wdump}, blob within {radius:.1f} px, any c\n[rank] pooled by c, top {a.top}"
          + (f", blobs within {merge_px:.1f} px merged" if a.merge else "")
          + ("\n[osd grid] blobs in a row on the OSD grid dropped first" if a.osd_grid else "")
          + (f"\n[kinematic] {limit.v_max_ms:g} m/s at {limit.min_range_m:g} m through "
             f"{limit.hfov_deg:g} deg = {limit.physical_px:.1f} px/frame, ceiling "
             f"{limit.ceiling:.1f} -> {limit.px_per_frame:.1f} px/frame + {limit.slack:.1f} "
             f"slack; confirm {a.confirm}, coast {a.max_coast}, weak c >= {a.c_keep:g}"
             + (f"; moving factor {scale.px(a.min_move):.1f} px" if a.min_move else "")
             + f"; ranked by {RANKED_BY[a.rank_by].format(decay=a.decay)}"
             if a.kinematic else "")
          + (f"\n[selector] {weights.label()}; memory decay {a.memory_decay:g}, colour "
             f"{'on' if CLIP.get('use_colour', True) else 'off'}"
             if a.rank_by == "drone" else ""))
    tracker = KinematicTracker(limit, a.confirm, a.max_coast, a.decay,
                               min_move=a.min_move, move_window=a.move_window)
    drone_sel = selector(a) if a.rank_by == "drone" else None
    shown_fields = (SHOWN_FIELDS + (KINEMATIC_SHOWN if a.kinematic else [])
                    + (["c_max"] if a.merge_score == "sum" else [])
                    + (SELECTOR_SHOWN if drone_sel else []))
    frame_fields = FRAME_FIELDS + (KINEMATIC_FRAMES if a.kinematic else [])

    cap = cv2.VideoCapture(CLIP["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.start - 2)
    ok, prev = cap.read()
    if not ok:
        raise SystemExit(f"cannot read frame {a.start - 1}")
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))
    shown_rows, drone_rows, frame_rows = [], [], []
    for f in range(a.start, a.end + 1):
        ok, cur = cap.read()
        if not ok:
            break
        hmat = common.homography(common.prep(prev, 11), common.prep(cur, 11))
        prev = cur
        sl = skyline.split(cur, stage0.valid(hmat))
        fb = [dict(b) for b in blobs.get(f, [])]
        gone = []
        if a.osd_grid:
            gone = osd_vetoed(cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY), fb, osd_box)
            fb = [b for b in fb if not any(b is g for g in gone)]
        for b in fb:
            b["section"] = section_of(sl.sky, b["x"], b["y"])
        objs = merge(fb, merge_px, a.merge_score)
        ranked = pool(objs, survivors.get(f), radius)
        held, extra = [], {}
        if a.kinematic:
            ranked, held, res = gate(tracker, f, ranked, objs, a.c_keep, np.linalg.inv(hmat),
                                     a.rank_by)
            if drone_sel:
                ranked = select(drone_sel, f, ranked, cur, np.linalg.inv(hmat))
            extra = dict(born=res.born, overruled=sum(o["reason"] != "weak-orphan"
                                                      for o in held),
                         too_still=res.too_still,
                         continued_weak=res.continued_weak, coasting=res.coasting)
        shown = ranked[:a.top]
        frame_rows.append(dict(
            frame=f, sky_fraction=sl.sky_fraction,
            **{f"{k}_{s}": sum(b["section"] == s for b in rows)
               for k, rows in (("ranked", ranked), ("shown", shown)) for s in SECTIONS},
            folded=len(fb) - len(objs), osd_vetoed=len(gone),
            osd_vetoed_on_target=sum(b["on_target"] for b in gone), **extra))
        shown_rows += [dict(frame=f, rank=i, section=b["section"], x=b["x"], y=b["y"],
                            c=b["c"], diameter=b["diameter"], members=b["members"],
                            on_target=b["on_target"],
                            **({k: b[k] for k in KINEMATIC_SHOWN} if a.kinematic else {}),
                            **({"c_max": b["c_max"]} if a.merge_score == "sum" else {}),
                            **({k: b[k] for k in SELECTOR_SHOWN} if drone_sel else {}))
                       for i, b in enumerate(shown, 1)]
        if f in boxes:
            drone_rows.append(drone_row(f, ranked, objs, boxes[f], sl.sky,
                                        [o for o in held if o["reason"] != "weak-orphan"]))

        img = cur.copy()
        _draw_background(img, None, stage0, sky=False)
        draw(img, shown, sl, not a.no_split_line, min_draw)
        text_block(img, [f"frame {f}    sky: sky branch c>=6   ground: {a.min_appear} of "
                         f"{a.k} window    top {a.top} by "
                         + ("drone score" if drone_sel else
                            f"{'summed ' if a.merge_score == 'sum' else ''}sky c")
                         + (f"    kinematic {limit.px_per_frame:.0f} px/frame"
                            if a.kinematic else ""),
                         f"ranked  sky {sum(b['section'] == 'sky' for b in ranked)}"
                         f"  ground {sum(b['section'] == 'ground' for b in ranked)}"
                         f"    shown {len(shown)}"])
        writer.write(img)
    writer.release()
    cap.release()

    for path, fields, rows in ((stem + ".csv", shown_fields, shown_rows),
                               (stem + "_drone.csv", DRONE_FIELDS, drone_rows),
                               (stem + "_frames.csv", frame_fields, frame_rows)):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    report(a, out, stem, frame_rows, drone_rows)


def read_tables(stem: str) -> tuple[list[dict], list[dict]]:
    """The per-frame and drone tables a run wrote, typed as `main` builds them."""
    with open(stem + "_frames.csv", encoding="utf-8") as fh:
        frames = [{k: (float(v) if k == "sky_fraction" else int(v)) for k, v in r.items()}
                  for r in csv.DictReader(fh)]
    with open(stem + "_drone.csv", encoding="utf-8") as fh:
        drone = [dict(r, frame=int(r["frame"]), rank=int(r["rank"]) if r["rank"] else "")
                 for r in csv.DictReader(fh)]
    return frames, drone


def report(a, out, stem, frame_rows, drone_rows) -> None:
    n = len(frame_rows)
    sky_frac = [r["sky_fraction"] for r in frame_rows]
    load = {s: sum(r[f"ranked_{s}"] for r in frame_rows) for s in SECTIONS}
    shown_n = {s: sum(r[f"shown_{s}"] for r in frame_rows) for s in SECTIONS}
    folded = sum(r["folded"] for r in frame_rows)
    vetoed = dict(blobs=sum(r["osd_vetoed"] for r in frame_rows),
                  on_target=sum(r["osd_vetoed_on_target"] for r in frame_rows))
    print(f"\n[video] {n} frames -> {out}\n[dump] every shown blob -> {stem}.csv"
          f"\n[dump] the drone, per labelled frame -> {stem}_drone.csv")
    print(f"\n  stage 1 sky fraction: median {np.median(sky_frac):.0%}, "
          f"no sky in {sum(s == 0 for s in sky_frac)} frames")
    print(f"  ranked/frame  sky {load['sky'] / n:5.2f}  ground {load['ground'] / n:5.2f}"
          f"  total {sum(load.values()) / n:5.2f}")
    print(f"  shown/frame   sky {shown_n['sky'] / n:5.2f}  ground {shown_n['ground'] / n:5.2f}"
          f"  total {sum(shown_n.values()) / n:5.2f}")
    if a.merge:
        print(f"  merge {a.merge:g} px at {a.ref_width:g} wide folded {folded} candidates "
              f"({folded / n:.2f}/frame) "
              f"into stronger ones, before the sections' tests")
    if a.osd_grid:
        print(f"  osd grid dropped {vetoed['blobs']} candidates ({vetoed['blobs'] / n:.2f}/frame), "
              f"{vetoed['on_target']} of them on the drone")
    if a.kinematic:
        k = {key: sum(r[key] for r in frame_rows) for key in KINEMATIC_FRAMES}
        print(f"  kinematic gate ({speed_limit(a).px_per_frame:.1f} px/frame): {k['born']} "
              f"tracks born, {k['overruled']} candidates overruled "
              f"({k['overruled'] / n:.2f}/frame), {k['continued_weak']} weak continuations, "
              f"{k['coasting'] / n:.2f} confirmed tracks coasting/frame. 'ranked' above is "
              f"after the gate")
        if a.min_move > 0:
            print(f"  moving factor ({speed_limit(a).scale.px(a.min_move):.1f} px over "
                  f"{a.move_window} frames): "
                  f"{k['too_still']} of those overrulings were too-still "
                  f"({k['too_still'] / n:.2f}/frame) -- tracks that moved plausibly but "
                  f"not enough to be flying")
    if os.path.exists(stem + ".csv"):
        reach = speed_limit(a).reach(1)
        with open(stem + ".csv", encoding="utf-8") as fh:
            jumps, pairs = top_jumps(list(csv.DictReader(fh)), reach)
        print(f"  #1 jumped more than {reach:.0f} px in {jumps} of {pairs} consecutive frame "
              f"pairs that both had a #1")
    print(f"\n  drone, of {len(drone_rows)} labelled frames      sky  ground  total")
    for name, keep in (("ranked", lambda r: r["outcome"] == "ranked"),
                       (f"in the top {a.top}", lambda r: r["outcome"] == "ranked"
                        and r["rank"] <= a.top),
                       *((f"  #{i}", (lambda i: lambda r: r["rank"] == i)(i))
                         for i in range(1, a.top + 1)),
                       *((("overruled by the kinematic gate",
                           lambda r: r["outcome"] == "overruled"),) if a.kinematic else ()),
                       ("dropped by its section", lambda r: r["outcome"] == "dropped"),
                       ("no blob on it", lambda r: r["outcome"] == "no blob")):
        c = {s: sum(keep(r) and r["section"] == s for r in drone_rows) for s in SECTIONS}
        print(f"    {name:28s} {c['sky']:5d} {c['ground']:7d} {sum(c.values()):6d}")
    print("\n  'drone' = a blob's on_target from EXP-023's dump: inside the label box grown by"
          "\n  max(10 px, 25%). Section of a dropped drone = its best blob's; of no blob = the"
          "\n  label box centre's.")


if __name__ == "__main__":
    main()
