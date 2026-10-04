"""EXP-024: histograms of the drone's rank among each frame's survivors, read from dumps.

    py -3.13 experiments/exp024_window_length/rank_hist.py [--rank-by sky]

One panel per gate. By default a gate's survivors in a frame are ranked by that detector's
own score `c`, highest first -- the motion-map peak height for a window gate, appearance
contrast for the sky branch, so ranks compare across panels but `c` does not.

`--rank-by sky` ranks the window gates' survivors by the **sky branch's** contrast instead:
each survivor takes the `c` of the nearest EXP-023 candidate within `SKY_RADIUS` in the
same frame (that dump holds every blob to c=-0.96; cloud vetoes are dropped), and one with no blob that close ranks
last. The window still decides *what survives*; only the order changes.

The drone's rank is the best rank of any survivor on it (`on_target`), counted
pessimistically: a survivor tied with the drone ranks ahead of it. Every labelled frame of
the span lands in exactly one bar: a rank, "filtered out" (a candidate on the drone existed
but none survived), or "not a candidate" (the detector never put one on it).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = "runs/sofa_analog/"
LABELS = "data/processed/SOFA-ANALOG/annotations/catch_2.json"
SPAN = (441, 800)
OUT = RUNS + "exp024_window_length/rank_hist_{}catch_2_441_800.png"
SKY_DUMP = RUNS + "exp023_sky_branch/candidates_catch_2_441_800.csv"
SKY_RADIUS = 9.0   # px -- the analog appearance radius, the window gate's own notion of "same place"

# (title, dump, which rows survive). Window dumps hold every seed, so the gate is
# `appearances >= m`; the EXP-023 dump holds every candidate to its floor, `kept` is c >= 6.
GATES = [
    ("2 of 3 window", RUNS + "exp024_window_length/seeds_k3_catch_2_441_800.csv",
     lambda r: int(r["appearances"]) >= 2),
    ("2 of 4 window", RUNS + "exp024_window_length/seeds_k4_catch_2_441_800.csv",
     lambda r: int(r["appearances"]) >= 2),
    ("sky branch, c >= 6", SKY_DUMP, lambda r: r["kept"] == "1"),
]

RANK_BINS = [str(n) for n in range(1, 11)] + ["11+"]
MISS_BINS = ["filtered\nout", "not a\ncandi-\ndate"]

INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES, MISS = "#2a78d6", "#b5b4ae"


def labelled_frames() -> set[int]:
    """Frames of the span carrying a drone box."""
    frames = json.load(open(LABELS, encoding="utf-8"))["frames"]
    return {int(k) for k, v in frames.items()
            if v and v.get("box") and SPAN[0] <= int(k) <= SPAN[1]}


def sky_index() -> dict[int, np.ndarray]:
    """EXP-023's candidates by frame, as (N, 3) arrays of (x, y, c).

    Blobs vetoed as `cloud` are left out: that is a shape rule the sky branch applies at any
    c, so they are not candidates whatever the threshold.
    """
    by_frame = defaultdict(list)
    with open(SKY_DUMP, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["reason"] == "cloud":
                continue
            by_frame[int(r["frame"])].append((float(r["x"]), float(r["y"]), float(r["c"])))
    return {f: np.asarray(v) for f, v in by_frame.items()}


def sky_score(sky: dict[int, np.ndarray], r: dict) -> float:
    """The sky branch's `c` for the blob nearest this row, or -inf if none is within reach."""
    blobs = sky.get(int(r["frame"]))
    if blobs is None:
        return -np.inf
    d = np.hypot(blobs[:, 0] - float(r["x"]), blobs[:, 1] - float(r["y"]))
    i = int(np.argmin(d))
    return float(blobs[i, 2]) if d[i] <= SKY_RADIUS else -np.inf


def drone_ranks(path: str, survives, score=lambda r: float(r["c"])
                ) -> tuple[dict[int, int], set[int]]:
    """{frame: drone's best rank among survivors}, and the frames it was a candidate at all."""
    by_frame, seen = defaultdict(list), set()
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            f, hit = int(r["frame"]), r["on_target"] == "1"
            if hit:
                seen.add(f)
            if survives(r):
                by_frame[f].append((score(r), hit))
    ranks = {}
    for f, rows in by_frame.items():
        rows.sort(key=lambda t: (-t[0], t[1]))   # ties: clutter first, the drone after
        rank = next((i for i, (_, hit) in enumerate(rows, 1) if hit), None)
        if rank is not None:
            ranks[f] = rank
    return ranks, seen


def bin_of(rank: int) -> str:
    return str(rank) if rank <= 10 else "11+"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rank-by", choices=("own", "sky"), default="own",
                    help="rank window survivors by their own motion score, or by the sky "
                         "branch's contrast at the same spot")
    a = ap.parse_args()
    sky = sky_index() if a.rank_by == "sky" else None
    labelled = labelled_frames()
    n = len(labelled)
    panels = []
    for title, path, survives in GATES:
        if sky is not None and path != SKY_DUMP:
            score = lambda r: sky_score(sky, r)  # noqa: E731
            title += ", ranked by sky c"
            with open(path, encoding="utf-8") as fh:
                rows = [r for r in csv.DictReader(fh) if survives(r)]
            blind = [r for r in rows if score(r) == -np.inf]
            print(f"{title}: {len(blind)} of {len(rows)} survivors "
                  f"({len(blind) / max(len(rows), 1):.0%}) have no sky blob within "
                  f"{SKY_RADIUS:g} px, {sum(r['on_target'] == '1' for r in blind)} of them "
                  "on the drone")
        else:
            score = lambda r: float(r["c"])  # noqa: E731
        ranks, seen = drone_ranks(path, survives, score)
        ranks = {f: r for f, r in ranks.items() if f in labelled}
        counts = dict.fromkeys(RANK_BINS + MISS_BINS, 0)
        for r in ranks.values():
            counts[bin_of(r)] += 1
        counts[MISS_BINS[0]] = len((seen & labelled) - set(ranks))
        counts[MISS_BINS[1]] = n - len(seen & labelled)
        with open(path, encoding="utf-8") as fh:
            load = sum(survives(r) for r in csv.DictReader(fh)) / (SPAN[1] - SPAN[0] + 1)
        top3 = sum(r <= 3 for r in ranks.values())
        med = sorted(ranks.values())[len(ranks) // 2] if ranks else None
        panels.append((title, counts, load, ranks, top3, med))
        print(f"{title:36s} load {load:5.2f}/frame  kept {len(ranks):3d}/{n}  "
              f"#1 {counts['1']:3d}  top3 {top3:3d}  median rank {med}  "
              f"filtered out {counts[MISS_BINS[0]]:3d}  "
              f"not a candidate {counts[MISS_BINS[1]]:3d}")

    labels = RANK_BINS + MISS_BINS
    xs = list(range(len(RANK_BINS))) + [len(RANK_BINS) + 0.8 + 1.3 * i for i in range(2)]
    ymax = max(max(c.values()) for _, c, *_ in panels) * 1.12
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True, facecolor=SURFACE)
    for ax, (title, counts, load, ranks, top3, med) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        colours = [SERIES] * len(RANK_BINS) + [MISS] * 2
        bars = ax.bar(xs, [counts[b] for b in labels], width=0.78, color=colours,
                      edgecolor=SURFACE, linewidth=1)
        for b, x, bar in zip(labels, xs, bars):
            if b in ("1", *MISS_BINS) and counts[b]:
                ax.text(x, bar.get_height() + ymax * 0.015, str(counts[b]),
                        ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_xticks(xs, labels, fontsize=8, color=INK_2)
        ax.set_ylim(0, ymax)
        ax.set_title(f"{title}  ({load:.1f} survivors/frame)", fontsize=11, color=INK,
                     loc="left")
        ax.text(0.22, 0.95,
                f"kept in {len(ranks)} of {n} labelled frames\n"
                f"#1 in {counts['1']},  top 3 in {top3}\nmedian rank {med}",
                transform=ax.transAxes, ha="left", va="top", fontsize=9, color=INK_2)
        ax.set_xlabel("drone's rank among the frame's survivors", fontsize=9, color=INK_2)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="y", colors=INK_2, labelsize=8, length=0)
    axes[0].set_ylabel("labelled frames", fontsize=9, color=INK_2)
    fig.suptitle(f"catch_2 {SPAN[0]}-{SPAN[1]}: where the drone ranks, by score, among each "
                 f"frame's survivors ({n} labelled frames per panel)",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    out = OUT.format("" if a.rank_by == "own" else "skyc_")
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
