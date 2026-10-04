"""EXP-024: histograms of the drone's rank among each frame's survivors, read from dumps.

    py -3.13 experiments/exp024_window_length/rank_hist.py

One panel per gate. A gate's survivors in a frame are ranked by that detector's own score
`c`, highest first -- the motion-map peak height for a window gate, appearance contrast
for the sky branch, so ranks compare across panels but `c` does not. The drone's rank is
the best rank of any survivor on it (`on_target`). Every labelled frame of the span lands
in exactly one bar: a rank, "filtered out" (a candidate on the drone existed but none
survived), or "not a candidate" (the detector never put one on it).
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = "runs/sofa_analog/"
LABELS = "data/processed/SOFA-ANALOG/annotations/catch_2.json"
SPAN = (441, 800)
OUT = RUNS + "exp024_window_length/rank_hist_catch_2_441_800.png"

# (title, dump, which rows survive). Window dumps hold every seed, so the gate is
# `appearances >= m`; the EXP-023 dump holds every candidate to its floor, `kept` is c >= 6.
GATES = [
    ("2 of 3 window", RUNS + "exp024_window_length/seeds_k3_catch_2_441_800.csv",
     lambda r: int(r["appearances"]) >= 2),
    ("2 of 4 window", RUNS + "exp024_window_length/seeds_k4_catch_2_441_800.csv",
     lambda r: int(r["appearances"]) >= 2),
    ("sky branch, c >= 6", RUNS + "exp023_sky_branch/candidates_catch_2_441_800.csv",
     lambda r: r["kept"] == "1"),
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


def drone_ranks(path: str, survives) -> tuple[dict[int, int], set[int]]:
    """{frame: drone's best rank among survivors}, and the frames it was a candidate at all."""
    by_frame, seen = defaultdict(list), set()
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            f, hit = int(r["frame"]), r["on_target"] == "1"
            if hit:
                seen.add(f)
            if survives(r):
                by_frame[f].append((float(r["c"]), hit))
    ranks = {}
    for f, rows in by_frame.items():
        rows.sort(key=lambda t: -t[0])
        rank = next((i for i, (_, hit) in enumerate(rows, 1) if hit), None)
        if rank is not None:
            ranks[f] = rank
    return ranks, seen


def bin_of(rank: int) -> str:
    return str(rank) if rank <= 10 else "11+"


def main() -> None:
    labelled = labelled_frames()
    n = len(labelled)
    panels = []
    for title, path, survives in GATES:
        ranks, seen = drone_ranks(path, survives)
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
        print(f"{title:20s} load {load:5.2f}/frame  kept {len(ranks):3d}/{n}  "
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
    fig.savefig(OUT, dpi=130, facecolor=SURFACE)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
