"""EXP-025c: top 3 sky-branch blobs confirmed by EXP-024's 2-of-4 motion window, ranked by c.

    PYTHONPATH="experiments/exp023_sky_branch/analog_catch_2;experiments/exp025_top3;experiments/exp023_sky_branch;experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
        py -3.13 -m overlay_window_skyc

Drawn from two dumps, no detector re-run:

  * EXP-024's per-seed dump at k=4 (`--window-dump`) -- a seed survives at `appearances >=
    --min-appear` (2 of 4).
  * EXP-023's candidate dump (`--sky-dump`) -- every sky-branch blob down to c=-0.96,
    less those its shape rules veto as `cloud` (`--keep-cloud` keeps them).

A blob is **confirmed** when a surviving window seed lies within `--radius` (9 px, the
analog appearance radius). Confirmed blobs are ranked by the sky branch's contrast `c` and
the top `--top` drawn, each blob once however many seeds confirm it. That is EXP-024's
`rank_hist.py --rank-by sky` drawn as video, with one difference: a window survivor with no
blob nearby has no `c`, so it cannot be ranked and is not drawn rather than tying for last.

There is **no c threshold**: the window does the filtering, the sky branch only orders
(and vetoes clouds, as it does at any threshold).
EXP-025's clean look: stage-0 tints, red circles at least 10 px across tagged `#rank c`,
`#1` thicker, a two-line caption. Nothing is drawn from the labels.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import cv2
import numpy as np

import masks
from clipcfg import CLIP
from overlay_gate import _Shown
from overlay_sky import _draw_background
from overlay_top3 import draw_ranked
from overlay_video import load_boxes, text_block

RUNS = "runs/sofa_analog/"
W, H = CLIP["width"], CLIP["height"]
FIELDS = ["frame", "rank", "x", "y", "c", "diameter", "seeds", "on_target"]


def load_survivors(path: str, min_appear: int) -> dict[int, np.ndarray]:
    """Window seeds at or above `min_appear`, by frame, as (N, 2) arrays of (x, y)."""
    by_frame = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if int(r["appearances"]) >= min_appear:
                by_frame[int(r["frame"])].append((float(r["x"]), float(r["y"])))
    return {f: np.asarray(v, float) for f, v in by_frame.items()}


def load_blobs(path: str, keep_cloud: bool = False) -> dict[int, list[dict]]:
    """Every sky-branch candidate the branch would rank, by frame.

    Only the c threshold is dropped. A `cloud` veto is a shape rule (large and soft-edged)
    the branch applies at any c, so those blobs are not candidates and are left out unless
    `keep_cloud`; they are 60% of the dump and would otherwise confirm clutter.
    """
    by_frame = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["reason"] == "cloud" and not keep_cloud:
                continue
            by_frame[int(r["frame"])].append(dict(
                x=float(r["x"]), y=float(r["y"]), c=float(r["c"]),
                diameter=float(r["diameter"]), on_target=int(r["on_target"])))
    return by_frame


def confirmed(blobs: list[dict], seeds: np.ndarray | None, radius: float) -> list[dict]:
    """Blobs with a surviving window seed within `radius`, highest c first, each with its
    seed count. A seed nearest to no blob within reach confirms nothing."""
    if seeds is None or not len(seeds) or not blobs:
        return []
    xy = np.array([[b["x"], b["y"]] for b in blobs])
    d = np.hypot(seeds[:, None, 0] - xy[None, :, 0], seeds[:, None, 1] - xy[None, :, 1])
    nearest = d.argmin(axis=1)
    within = d[np.arange(len(seeds)), nearest] <= radius
    counts = np.bincount(nearest[within], minlength=len(blobs))
    out = [dict(b, seeds=int(n)) for b, n in zip(blobs, counts) if n]
    return sorted(out, key=lambda b: -b["c"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=441)
    ap.add_argument("--end", type=int, default=800)
    ap.add_argument("--k", type=int, default=4, help="window length of the seed dump")
    ap.add_argument("--min-appear", type=int, default=2)
    ap.add_argument("--top", type=int, default=3, help="at most this many per frame")
    ap.add_argument("--radius", type=float, default=9.0,
                    help="seed-to-blob distance that confirms a blob, px")
    ap.add_argument("--window-dump", default=None,
                    help="EXP-024 seed dump (default: its seeds_k{k}_ CSV for this span)")
    ap.add_argument("--sky-dump",
                    default=RUNS + "exp023_sky_branch/candidates_catch_2_441_800.csv")
    ap.add_argument("--keep-cloud", action="store_true",
                    help="also rank blobs the sky branch vetoes as cloud (default: drop them)")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    span = f"{CLIP['name']}_{a.start}_{a.end}"
    wdump = a.window_dump or RUNS + f"exp024_window_length/seeds_k{a.k}_{span}.csv"
    out = a.out or RUNS + f"exp025_top3/window{a.min_appear}of{a.k}_skyc_top{a.top}_{span}.mp4"
    dump = os.path.splitext(out)[0] + ".csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    survivors = load_survivors(wdump, a.min_appear)
    blobs = load_blobs(a.sky_dump, a.keep_cloud)
    boxes = load_boxes()
    stage0 = masks.Stage0()
    print(f"[window] {a.min_appear} of {a.k} from {wdump}\n[sky] {a.sky_dump}\n"
          f"[rank] blobs with a survivor within {a.radius:g} px, by c, top {a.top}")

    cap = cv2.VideoCapture(CLIP["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.start - 1)
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))
    rows, n_conf, drone_rank = [], 0, {}
    for f in range(a.start, a.end + 1):
        ok, img = cap.read()
        if not ok:
            break
        ranked = confirmed(blobs.get(f, []), survivors.get(f), a.radius)
        n_conf += len(ranked)
        shown = ranked[:a.top]
        for i, b in enumerate(ranked, 1):
            if b["on_target"] and f not in drone_rank:
                drone_rank[f] = i
        rows += [dict(frame=f, rank=i, x=b["x"], y=b["y"], c=b["c"], diameter=b["diameter"],
                      seeds=b["seeds"], on_target=b["on_target"])
                 for i, b in enumerate(shown, 1)]
        _draw_background(img, None, stage0, sky=False)
        draw_ranked(img, [_Shown(b) for b in shown])
        n_surv = 0 if survivors.get(f) is None else len(survivors[f])
        text_block(img, [f"frame {f}    {a.min_appear} of {a.k} window, ranked by sky c,"
                         f" top {a.top}",
                         f"window survivors {n_surv}   with a sky blob {len(ranked)}"
                         f"   shown {len(shown)}"])
        writer.write(img)
    writer.release()
    cap.release()
    with open(dump, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    n = a.end - a.start + 1
    labelled = {f for f in boxes if a.start <= f <= a.end}
    ranks = [r for f, r in drone_rank.items() if f in labelled]
    print(f"\n[video] {n} frames -> {out}\n[dump] every shown blob -> {dump}")
    print(f"\n  confirmed blobs   {n_conf / n:5.2f}/frame    shown {len(rows) / n:5.2f}/frame")
    print(f"  drone confirmed in {len(ranks)} of {len(labelled)} labelled frames")
    print(f"  drone in the top {a.top} in {sum(r <= a.top for r in ranks)}:  "
          + " / ".join(f"#{i} {sum(r == i for r in ranks)}" for i in range(1, a.top + 1)))
    print("\n  'drone' = the blob's on_target from EXP-023's dump: inside the label box grown by"
          "\n  max(10 px, 25%).")


if __name__ == "__main__":
    main()
