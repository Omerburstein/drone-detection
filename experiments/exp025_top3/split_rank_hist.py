"""EXP-025d: histogram of the drone's rank in the sky/ground split, stacked by section.

    py -3.13 experiments/exp025_top3/split_rank_hist.py [drone_csv]

Reads `overlay_split.py`'s `_drone.csv` (one row per labelled frame), so no re-render. Every
labelled frame lands in exactly one bar: its best rank in the pooled ranking, `dropped`
(a blob on the drone existed but its section's detector did not pass it: c < 6 in the sky,
no 2-of-4 survivor within 9 px on the ground), or `no blob` (no sky-branch candidate on it
at all). Each bar is split by the section, and so the detector, the frame is credited to:
the ranked blob's, a dropped drone's best blob's, or the label box centre's.
"""
from __future__ import annotations

import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT = "runs/sofa_analog/exp025_top3/split_sky_window2of4_top3_catch_2_441_800_drone.csv"
SECTIONS = (("sky", "sky section: sky branch, c >= 6", "#2a78d6"),
            ("ground", "ground section: 2-of-4 window, ranked by sky c", "#eb6834"))
RANK_BINS = [str(n) for n in range(1, 11)] + ["11+"]
MISS_BINS = ["dropped", "no\nblob"]

INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def bin_of(r: dict) -> str:
    if r["outcome"] == "dropped":
        return MISS_BINS[0]
    if r["outcome"] == "no blob":
        return MISS_BINS[1]
    rank = int(r["rank"])
    return str(rank) if rank <= 10 else "11+"


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    n = len(rows)
    labels = RANK_BINS + MISS_BINS
    counts = {s: dict.fromkeys(labels, 0) for s, *_ in SECTIONS}
    for r in rows:
        counts[r["section"]][bin_of(r)] += 1

    top3 = {s: sum(counts[s][b] for b in ("1", "2", "3")) for s, *_ in SECTIONS}
    for s, *_ in SECTIONS:
        print(f"{s:7s} " + "  ".join(f"{b.replace(chr(10), ' ')} {counts[s][b]}"
                                    for b in labels) + f"   top3 {top3[s]}")

    xs = list(range(len(RANK_BINS))) + [len(RANK_BINS) + 0.8 + 1.3 * i for i in range(2)]
    totals = [sum(counts[s][b] for s, *_ in SECTIONS) for b in labels]
    ymax = max(totals) * 1.12
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    base = [0] * len(labels)
    for s, name, colour in SECTIONS:
        h = [counts[s][b] for b in labels]
        ax.bar(xs, h, bottom=base, width=0.78, color=colour, edgecolor=SURFACE,
               linewidth=2, label=f"{name}  (top 3 in {top3[s]})")
        base = [b + v for b, v in zip(base, h)]
    for b, x, t in zip(labels, xs, totals):
        if t and b in ("1", "2", "3", *MISS_BINS):
            ax.text(x, t + ymax * 0.015, str(t), ha="center", va="bottom", fontsize=9,
                    color=INK)
    ax.set_xticks(xs, labels, fontsize=8, color=INK_2)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("drone's rank among the frame's pooled candidates (by sky c)", fontsize=9,
                  color=INK_2)
    ax.set_ylabel("labelled frames", fontsize=9, color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", colors=INK_2, labelsize=8, length=0)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper center")
    ranked = sum(r["outcome"] == "ranked" for r in rows)
    ax.set_title(f"catch_2 441-800, sky/ground split: drone ranked in {ranked} of {n} "
                 f"labelled frames, top 3 in {sum(top3.values())}, #1 in "
                 f"{totals[0]}", fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    out = path.replace("_drone.csv", "_rank_hist.png")
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
