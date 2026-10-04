"""EXP-025b: every frame the drone is in a top-N, and its rank there -- read from the dump.

    py -3.13 experiments/exp025_top3/drone_ranks.py runs/sofa_analog/exp025_top3/gate4of5_top3_catch_2_441_800.csv

Reads `overlay_gate.py`'s dump (every kept candidate with `appearances` and `rank_all`),
so any threshold can be tabled without a re-render. One row per frame where a candidate on
the drone is in the top N either **after the gate** or **before it** (EXP-025, where the
rank is `rank_all`); a dash means it was not in that top N. Writes a Markdown table.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict


def load(path: str) -> dict[int, list[dict]]:
    """The dump, grouped by frame, each frame's rows highest c first."""
    frames = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            frames[int(r["frame"])].append(dict(
                x=float(r["x"]), y=float(r["y"]), c=float(r["c"]),
                appearances=int(r["appearances"]), rank_all=int(r["rank_all"]),
                on_target=r["on_target"] == "1"))
    for rows in frames.values():
        rows.sort(key=lambda r: r["rank_all"])
    return frames


def drone_row(frame: int, rows: list[dict], min_appear: int, top: int) -> dict | None:
    """The drone's best rank in this frame with and without the gate, or None if neither."""
    gated = [r for r in rows if r["appearances"] >= min_appear][:top]
    g = next(((i, r) for i, r in enumerate(gated, 1) if r["on_target"]), None)
    u = next((r for r in rows[:top] if r["on_target"]), None)
    if g is None and u is None:
        return None
    best = g[1] if g else u
    return dict(frame=frame, gated=g[0] if g else None, ungated=u["rank_all"] if u else None,
                c=best["c"], appearances=best["appearances"], x=best["x"], y=best["y"])


def table(found: list[dict], min_appear: int, k: int, top: int) -> str:
    """The Markdown table, then a one-line count per rank."""
    head = (f"| frame | rank, {min_appear} of {k} gate | rank, no gate (EXP-025) | c "
            f"| appearances of {k} | x, y |\n| ---: | :---: | :---: | ---: | :---: | :--- |")
    dash = lambda v: "—" if v is None else f"#{v}"
    lines = [f"| {r['frame']} | {dash(r['gated'])} | {dash(r['ungated'])} | {r['c']:.1f} "
             f"| {r['appearances']} | {r['x']:.0f}, {r['y']:.0f} |" for r in found]
    count = lambda key, n: sum(r[key] == n for r in found)
    summary = [f"- **gated:** in top {top} in {sum(r['gated'] is not None for r in found)} frames — "
               + ", ".join(f"#{n} in {count('gated', n)}" for n in range(1, top + 1)),
               f"- **no gate:** in top {top} in {sum(r['ungated'] is not None for r in found)} "
               "frames — " + ", ".join(f"#{n} in {count('ungated', n)}"
                                        for n in range(1, top + 1))]
    return "\n".join([head, *lines, "", *summary]) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump")
    ap.add_argument("--min-appear", type=int, default=4)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--out", default=None, help="Markdown file; default beside the dump")
    a = ap.parse_args()

    frames = load(a.dump)
    found = [r for f in sorted(frames) if (r := drone_row(f, frames[f], a.min_appear, a.top))]
    md = table(found, a.min_appear, a.k, a.top)
    out = a.out or os.path.splitext(a.dump)[0] + "_drone_ranks.md"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(md)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
