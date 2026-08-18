"""Plot precision against target size in pixels, from one or more scoring dumps.

The one question the metric block cannot answer. `precision 0.99` is an average
over every box the detector emitted, and on this project box sizes span 8 px to
90 px -- a range over which the detector's reliability is the thing actually
under study. This draws precision as a function of the size of the box being
claimed, so "how small can a target get before a claim stops meaning anything"
becomes a line rather than an inference.

Bins on the **predicted** box size, not the target's: a false alarm has no
target, so its own size is the only one it has. See `src.eval.curves`.

The figure is redrawn from the dump CSVs, never from a live scoring, so it costs
nothing and cannot disagree with the numbers in the ledger. The binned values are
written out beside it for the same reason.

Example
-------
    py -3.13 -m src.plot_eval \
        --dump "centre@1x=runs/exp004_glad/matches_center.csv" \
        --dump "IoU@0.50=runs/exp004_glad/matches_iou50.csv" \
        --out runs/exp004_glad/precision_by_size.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # no display on this machine, and none wanted in a script
import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend choice)

from .eval.curves import (Curve, SIZE_EDGES, curve_rows, load_dump,  # noqa: E402
                          precision_by_size, recall_by_size)

# Validated categorical slots 1 and 2 (adjacent-pair CVD dE 24.7, normal 33.6).
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e3e2df"

MARKER_SIZE = 7.5  # points; ~9 px on the 200-dpi output
LINE_WIDTH = 2.0


def _style_axes(ax) -> None:
    """Recessive grid and axes: the marks carry the chart, not the furniture."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)  # y only; x is categorical
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)


def _plot_curve(ax, curve: Curve, color: str, label: str) -> None:
    """One precision line, with under-sampled bins drawn hollow.

    A bin of 12 predictions and a bin of 12,000 produce the same size dot
    otherwise, and the eye reads the noisy one as evidence.
    """
    x = np.arange(len(curve.values))
    ax.plot(x, curve.values, color=color, linewidth=LINE_WIDTH, zorder=3,
            label=label, solid_capstyle="round")
    reliable, sparse = curve.reliable, ~curve.reliable
    ax.plot(x[reliable], curve.values[reliable], linestyle="none", marker="o",
            markersize=MARKER_SIZE, color=color, markeredgecolor=SURFACE,
            markeredgewidth=2, zorder=4)
    ax.plot(x[sparse], curve.values[sparse], linestyle="none", marker="o",
            markersize=MARKER_SIZE, markerfacecolor=SURFACE, markeredgecolor=color,
            markeredgewidth=2, zorder=4)


def _plot_counts(ax, curve: Curve, labels: list[str]) -> None:
    """How many predictions each bin was computed from, on its own panel.

    A second y-scale on the precision axes would be the classic dual-axis
    mistake; a shared x and a separate panel says the same thing without
    inviting the two to be read against each other.
    """
    x = np.arange(len(curve.total))
    # Linear, not log: these bars are a sample-size cue, and a log height would
    # make 357 and 6,560 predictions look like the same amount of evidence.
    ax.bar(x, curve.total, width=0.62, color=INK_MUTED, zorder=2)
    ax.set_ylabel("predictions\nin bin", color=INK_SECONDARY, fontsize=9)
    ax.set_xticks(x, labels, color=INK_SECONDARY)
    ax.set_xlabel("predicted box size, $\\sqrt{w \\times h}$  (pixels)",
                  color=INK, fontsize=10.5)
    for i, total in enumerate(curve.total):
        ax.annotate(f"{int(total):,}", (i, total), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=7.5, color=INK_SECONDARY)


def render(curves: dict[str, Curve], counts: Curve, out: Path, title: str,
           subtitle: str) -> None:
    """Draw the figure and write it to `out`."""
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(9.2, 6.4), dpi=200, sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.12})
    fig.patch.set_facecolor(SURFACE)

    for ax in (top, bottom):
        _style_axes(ax)

    for (label, curve), color in zip(curves.items(), SERIES_COLORS):
        _plot_curve(top, curve, color, label)

    top.set_ylim(-0.03, 1.05)
    top.set_yticks(np.arange(0, 1.01, 0.2))
    top.set_ylabel("precision", color=INK, fontsize=10.5)
    top.legend(frameon=False, loc="lower right", fontsize=9.5,
               labelcolor=INK_SECONDARY)
    top.set_title(title, color=INK, fontsize=13, loc="left", pad=18, weight="bold")
    top.annotate(subtitle, xy=(0, 1.015), xycoords="axes fraction", fontsize=9.5,
                 color=INK_SECONDARY, ha="left", va="bottom")

    _plot_counts(bottom, counts, next(iter(curves.values())).labels)
    bottom.set_ylim(0, max(counts.total.max() * 1.28, 10))
    bottom.set_yticks([])
    bottom.spines["left"].set_visible(False)

    fig.text(0.008, 0.012,
             "Hollow markers: fewer than 30 predictions in the bin. Precision bins "
             "on the predicted box size — a false alarm has no target to bin on.",
             fontsize=8, color=INK_MUTED)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def write_curve_data(path: Path, curves: dict[str, Curve]) -> int:
    """Write the plotted numbers as CSV; return the row count."""
    rows = curve_rows(curves)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def parse_dump_arg(spec: str) -> tuple[str, Path]:
    """`LABEL=path/to/matches.csv` -> (label, path).

    The label is what the legend says, so it has to come from the caller: only
    they know whether a given dump was the centre-matched scoring or the strict
    one, and mislabelling those two is the specific error this project keeps
    warning about.
    """
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"expected LABEL=PATH, got {spec!r} (e.g. 'centre@1x=runs/x/matches.csv')")
    label, path = spec.split("=", 1)
    return label.strip(), Path(path.strip())


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for the size curves."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True, action="append", type=parse_dump_arg,
                    metavar="LABEL=CSV",
                    help="A dump written by `src.evaluate --dump`, and the name it "
                         "gets in the legend. Repeat to overlay scorings -- the "
                         "same run under two matching criteria is the useful case.")
    ap.add_argument("--out", type=Path, default=Path("precision_by_size.png"),
                    help="Where to write the figure (default precision_by_size.png).")
    ap.add_argument("--data-out", type=Path, default=None, metavar="CSV",
                    help="Also write the binned numbers behind the figure. Defaults "
                         "to the figure's path with a .csv suffix; pass a path to "
                         "override.")
    ap.add_argument("--title", default="Precision against target size",
                    help="Figure title.")
    ap.add_argument("--subtitle", default="",
                    help="Line under the title: the run, split and frame count.")
    return ap


def main() -> None:
    """Load each dump, bin it, and render the figure."""
    args = build_parser().parse_args()

    precision, recall, counts = {}, {}, None
    for label, path in args.dump:
        rows = load_dump(path)
        precision[label] = precision_by_size(rows, SIZE_EDGES)
        recall[f"{label} (recall)"] = recall_by_size(rows, SIZE_EDGES)
        counts = counts or precision[label]
        print(f"{label}: {len(rows)} rows from {path}")

    render(precision, counts, args.out, args.title, args.subtitle)
    print(f"Wrote {args.out}")

    data_out = args.data_out or args.out.with_suffix(".csv")
    written = write_curve_data(data_out, {**precision, **recall})
    print(f"Wrote {data_out} ({written} binned rows)")


if __name__ == "__main__":
    main()
