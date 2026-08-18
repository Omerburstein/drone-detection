"""Precision and recall as functions of target size, from the per-object dump.

The metric block reports recall in four coarse buckets and precision not by size
at all, because precision *cannot* be bucketed by target size: a false alarm has
no target, so it has no true size to bin on. The only size a false alarm has is
its own.

So the two curves bin on different quantities, and saying which is not a
footnote:

- **precision** bins every prediction on `pred_size` — the size of the box the
  detector emitted. Reads as "when this detector claims a drone *this big*, how
  often is it right".
- **recall** bins every target on `gt_size` — its true size. Reads as "of the
  drones that were *this big*, how many were found".

They are not two views of one axis and must not be read as a single trade-off
curve at a shared x. Plotted together they still belong on one x axis, because
the question both answer is the same one: what does size do to this detector.

Everything here consumes the CSV `records.py` writes, so any curve is a re-cut of
a scoring that already happened rather than a re-run of it.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Bin edges in pixels, on sqrt(w*h). Fine where this project lives -- ARD-MAV is
# 64.8% under 16 px -- and coarse above 32 px, where there is little data and
# nothing interesting happens. The open top bin catches oversized false alarms,
# which is where an appearance detector's failures pile up.
SIZE_EDGES = (0.0, 8.0, 12.0, 16.0, 20.0, 24.0, 32.0, 48.0, float("inf"))

# A bin holding fewer than this many objects is computed but flagged: a ratio
# over 20 samples is noise, and a chart that plots it at the same weight as a
# bin of 8,000 invites reading the noise as a trend.
MIN_RELIABLE = 30


@dataclass(frozen=True)
class Curve:
    """One metric binned by size, with the counts it was computed from.

    Counts travel with the values because a bare 1.0 precision means nothing
    until you know whether it came from 2 predictions or 2,000 -- and the
    per-bin sample count is exactly what a reader needs to judge the tail.
    """

    metric: str            # "precision" or "recall"
    binned_on: str         # the column the bins were taken over
    edges: tuple[float, ...]
    values: np.ndarray     # (n_bins,) the metric, NaN where the bin is empty
    hits: np.ndarray       # (n_bins,) true positives
    total: np.ndarray      # (n_bins,) denominator: preds for P, targets for R

    @property
    def labels(self) -> list[str]:
        """Human bin labels, `<8`, `8-12`, ... `>=48`."""
        names = []
        for lo, hi in zip(self.edges[:-1], self.edges[1:]):
            if lo == 0:
                names.append(f"<{hi:g}")
            elif hi == float("inf"):
                names.append(f">={lo:g}")
            else:
                names.append(f"{lo:g}-{hi:g}")
        return names

    @property
    def centres(self) -> np.ndarray:
        """A plotting x for each bin: its midpoint, the open top bin extrapolated."""
        edges = np.array(self.edges, dtype=float)
        top = edges[-2] + (edges[-2] - edges[-3])
        edges = np.concatenate([edges[:-1], [top]])
        return (edges[:-1] + edges[1:]) / 2

    @property
    def reliable(self) -> np.ndarray:
        """Which bins hold enough objects to be read as a measurement."""
        return self.total >= MIN_RELIABLE


def load_dump(path: Path) -> list[dict[str, str]]:
    """Read a dump CSV written by `records.write_dump`."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _floats(rows: list[dict[str, str]], column: str) -> np.ndarray:
    """One numeric column, with blank cells as NaN."""
    return np.array([float(row[column]) if row[column] not in ("", None) else np.nan
                     for row in rows], dtype=float)


def _binned(sizes: np.ndarray, hit: np.ndarray,
            edges: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin hits, totals and their ratio; NaN where a bin is empty."""
    index = np.digitize(sizes, np.array(edges[1:-1], dtype=float), right=False)
    n_bins = len(edges) - 1
    hits = np.array([hit[index == b].sum() for b in range(n_bins)], dtype=float)
    total = np.array([(index == b).sum() for b in range(n_bins)], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        values = np.where(total > 0, hits / np.maximum(total, 1), np.nan)
    return values, hits, total


def precision_by_size(rows: list[dict[str, str]],
                      edges: tuple[float, ...] = SIZE_EDGES) -> Curve:
    """Precision against the size of the box the detector emitted.

    Only prediction rows take part -- a missed target is not a wrong claim, so
    `fn` rows are excluded rather than counted as failures here.
    """
    preds = [r for r in rows if r["outcome"] in ("tp", "fp")]
    values, hits, total = _binned(_floats(preds, "pred_size"),
                                  np.array([r["outcome"] == "tp" for r in preds]),
                                  edges)
    return Curve("precision", "pred_size", edges, values, hits, total)


def recall_by_size(rows: list[dict[str, str]],
                   edges: tuple[float, ...] = SIZE_EDGES) -> Curve:
    """Recall against the target's true size.

    Only rows that carry a target take part: every `tp` (found) and every `fn`
    (missed). `fp` rows have no target and are excluded.
    """
    targets = [r for r in rows if r["outcome"] in ("tp", "fn")]
    values, hits, total = _binned(_floats(targets, "gt_size"),
                                  np.array([r["outcome"] == "tp" for r in targets]),
                                  edges)
    return Curve("recall", "gt_size", edges, values, hits, total)


def curve_rows(curves: dict[str, Curve]) -> list[dict[str, object]]:
    """The plotted numbers as flat rows, keyed by series name.

    Written out beside the figure so the chart can be redrawn, checked or
    re-styled without re-deriving anything from the dump.
    """
    rows = []
    for series, curve in curves.items():
        for i, label in enumerate(curve.labels):
            rows.append({
                "series": series,
                "metric": curve.metric,
                "binned_on": curve.binned_on,
                "bin": label,
                "bin_lo": curve.edges[i],
                "bin_hi": curve.edges[i + 1],
                "value": None if np.isnan(curve.values[i]) else round(
                    float(curve.values[i]), 4),
                "hits": int(curve.hits[i]),
                "total": int(curve.total[i]),
                "reliable": bool(curve.reliable[i]),
            })
    return rows
