"""Two-way cuts of the per-object dump: a size band crossed with a condition.

The metric block cuts a run one way at a time. `by target size` says how the
detector fares on tiny targets, pooled over every background; `by scene_category`
says how it fares on complex backgrounds, pooled over every size. Neither
answers the question this project keeps asking -- *what happens when both go
wrong at once* -- and the two marginals cannot be multiplied to find out, because
size and background are correlated: on ARD-MAV the `small_mav` videos are almost
entirely tiny targets, so a pooled "tiny" number is largely a statement about one
scene category.

So this groups the dump by (size band x condition label) and reports the same
quantities the metric block does, per cell.

Cells are built **frame by frame, not row by row**, and that is the load-bearing
decision here. A false alarm has no target, so it has no target size to bin on --
the point `curves.py` makes about precision. But "how often does this detector
cry wolf while hunting an 8 px drone" is a well-posed question anyway, because
the *frame* has a size even when the false alarm does not: it is the size of the
drone that was actually there. So a frame is assigned a band from its own
targets, every outcome in it inherits that band, and `far` -- false alarms over
frames -- means the same thing per cell as it does in the metric block.

A frame whose targets straddle two bands gets `mixed sizes` rather than being
forced into one, and a frame with no target at all gets `no target`. Both are
reported rather than dropped, so the cells still sum to the whole run. On ARD-MAV
every frame holds exactly one target, so neither appears in practice -- they are
there so this stays honest on a dataset where it matters.

Everything here consumes the CSV `records.py` writes, so a cross-cut is a re-cut
of a scoring that already happened rather than a second, subtly different one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .curves import MIN_RELIABLE, SIZE_EDGES, bin_index, bin_labels
from .metrics import f1_score

NO_TARGET = "no target"     # frame the detector fired on with nothing to find
MIXED = "mixed sizes"       # frame whose targets straddle a band edge
UNKNOWN = "uncategorised"   # frame the axis does not label

TP, FP, FN = "tp", "fp", "fn"


@dataclass(frozen=True)
class Cell:
    """One (size band, condition) cell: the counts, and the ratios over them.

    Counts are stored and ratios derived, never the other way round. A cell of
    3 targets and a cell of 3,000 produce the same `pd` and must not read the
    same, so every consumer gets the denominator alongside the number.
    """

    band: str
    condition: str
    n_frames: int
    n_gt: int
    tp: int
    fp: int
    fn: int
    mean_iou: float      # over matched pairs only; NaN when nothing matched
    loc_err: float       # median centre offset in target sizes, matched only
    loc_err_p90: float

    @property
    def pd(self) -> float:
        """Probability of detection: matched targets over targets present."""
        return self.tp / self.n_gt if self.n_gt else float("nan")

    @property
    def precision(self) -> float:
        """Of the alarms raised in this cell, the fraction that were real."""
        claims = self.tp + self.fp
        return self.tp / claims if claims else float("nan")

    @property
    def f1(self) -> float:
        """Harmonic mean of the two, NaN when either is undefined."""
        return f1_score(self.precision, self.pd)

    @property
    def far(self) -> float:
        """False alarms per frame in this cell -- the M5 rate, cell-local."""
        return self.fp / self.n_frames if self.n_frames else float("nan")

    @property
    def reliable(self) -> bool:
        """Whether the cell holds enough targets to be read as a measurement."""
        return self.n_gt >= MIN_RELIABLE


class _Tally:
    """Mutable accumulator for one cell while the dump is being walked."""

    def __init__(self, band: str, condition: str) -> None:
        self.band, self.condition = band, condition
        self.n_frames = self.tp = self.fp = self.fn = self.n_gt = 0
        self.ious: list[float] = []
        self.offsets: list[float] = []

    def add_frame(self, rows: list[dict[str, str]]) -> None:
        """Fold one frame's outcome rows in."""
        self.n_frames += 1
        for row in rows:
            outcome = row["outcome"]
            if outcome == FP:
                self.fp += 1
                continue
            self.n_gt += 1
            if outcome == FN:
                self.fn += 1
                continue
            self.tp += 1
            self.ious.append(number(row.get("iou")))
            self.offsets.append(number(row.get("center_dist_rel")))

    def freeze(self) -> Cell:
        """The immutable cell, with the matched-pair statistics computed."""
        ious = _finite(self.ious)
        offsets = _finite(self.offsets)
        return Cell(
            band=self.band, condition=self.condition, n_frames=self.n_frames,
            n_gt=self.n_gt, tp=self.tp, fp=self.fp, fn=self.fn,
            mean_iou=float(ious.mean()) if len(ious) else float("nan"),
            loc_err=float(np.median(offsets)) if len(offsets) else float("nan"),
            loc_err_p90=(float(np.percentile(offsets, 90)) if len(offsets)
                         else float("nan")),
        )


def number(cell: str | None) -> float:
    """One CSV cell as a float, blank as NaN."""
    return float(cell) if cell not in ("", None) else float("nan")


def _finite(values: list[float]) -> np.ndarray:
    """The values that exist, as an array -- blanks dropped, not zero-filled."""
    array = np.asarray(values, dtype=float)
    return array[~np.isnan(array)]


def group_frames(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    """The dump's rows bucketed by frame key, in the order they were written."""
    frames: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        frames.setdefault(row["key"], []).append(row)
    return frames


def frame_band(rows: list[dict[str, str]],
               edges: tuple[float, ...] = SIZE_EDGES) -> str:
    """Which size band a frame belongs to, from the targets it contains.

    `no target` when the frame holds none, `mixed sizes` when its targets do not
    all land in one band. Both are real answers, not failures: a frame whose
    only content is a false alarm genuinely has no target size, and forcing it
    into a band would put false alarms in a cell whose Pd they had no part in.
    """
    sizes = _finite([number(row.get("gt_size")) for row in rows
                     if row["outcome"] in (TP, FN)])
    if not len(sizes):
        return NO_TARGET
    labels = bin_labels(edges)
    bands = {labels[i] for i in bin_index(sizes, edges)}
    return bands.pop() if len(bands) == 1 else MIXED


def frame_condition(rows: list[dict[str, str]], axis: str) -> str:
    """The axis label a frame carries, `uncategorised` when the column is absent.

    Read from the frame's first row: `records.write_dump` writes axis labels as
    frame context, so every row of a frame carries the same value.
    """
    return rows[0].get(axis) or UNKNOWN


def cross_cut(rows: list[dict[str, str]], axis: str,
              edges: tuple[float, ...] = SIZE_EDGES) -> list[Cell]:
    """Every (size band, condition) cell present in the dump.

    Ordered by band -- declared size order first, then `mixed sizes` and
    `no target` -- and by condition label within a band, so two runs tabled side
    by side line their rows up.
    """
    tallies: dict[tuple[str, str], _Tally] = {}
    for frame_rows in group_frames(rows).values():
        key = (frame_band(frame_rows, edges), frame_condition(frame_rows, axis))
        tallies.setdefault(key, _Tally(*key)).add_frame(frame_rows)

    order = {label: i for i, label in enumerate(bin_labels(edges))}
    tail = len(order)
    return sorted((tally.freeze() for tally in tallies.values()),
                  key=lambda cell: (order.get(cell.band, tail), cell.band,
                                    cell.condition))


def select(cells: list[Cell], bands: tuple[str, ...] = (),
           conditions: tuple[str, ...] = ()) -> list[Cell]:
    """The cells matching the requested bands and condition labels.

    Empty filters mean "everything", so the two are independent: asking for one
    band across every background is as natural as asking for one background
    across every band.
    """
    return [cell for cell in cells
            if (not bands or cell.band in bands)
            and (not conditions or cell.condition in conditions)]


def pooled(cells: list[Cell]) -> Cell:
    """One cell summing a selection, for the total row under a table.

    Counts add; ratios are recomputed from the summed counts rather than
    averaged, since averaging per-cell ratios would weight a 3-target cell like a
    3,000-target one. The matched-pair statistics cannot be recovered from cells
    alone and are left NaN -- a median is not summable, and inventing one here
    would be the same mistake in a subtler place.
    """
    if not cells:
        return Cell("", "", 0, 0, 0, 0, 0, float("nan"), float("nan"), float("nan"))
    bands = {cell.band for cell in cells}
    conditions = {cell.condition for cell in cells}
    return Cell(
        band=bands.pop() if len(bands) == 1 else "all",
        condition=conditions.pop() if len(conditions) == 1 else "all",
        n_frames=sum(cell.n_frames for cell in cells),
        n_gt=sum(cell.n_gt for cell in cells),
        tp=sum(cell.tp for cell in cells),
        fp=sum(cell.fp for cell in cells),
        fn=sum(cell.fn for cell in cells),
        mean_iou=float("nan"), loc_err=float("nan"), loc_err_p90=float("nan"),
    )


def cell_rows(cells: dict[str, list[Cell]], axis: str) -> list[dict[str, object]]:
    """The table as flat rows, keyed by series name, ready for a CSV.

    Written out beside any printed table for the same reason `curves.py` writes
    its sidecar: so the numbers can be re-read, re-sorted or pasted into a
    document without re-deriving them from the dump.
    """
    return [
        {
            "series": series,
            "axis": axis,
            "band": cell.band,
            "condition": cell.condition,
            "frames": cell.n_frames,
            "targets": cell.n_gt,
            "tp": cell.tp,
            "fp": cell.fp,
            "fn": cell.fn,
            "pd": _rounded(cell.pd),
            "precision": _rounded(cell.precision),
            "f1": _rounded(cell.f1),
            "far": _rounded(cell.far),
            "mean_iou": _rounded(cell.mean_iou),
            "loc_err": _rounded(cell.loc_err),
            "loc_err_p90": _rounded(cell.loc_err_p90),
            "reliable": cell.reliable,
        }
        for series, series_cells in cells.items()
        for cell in series_cells
    ]


def _rounded(value: float) -> float | None:
    """A number for the CSV, or None where the cell had nothing to compute from."""
    return None if value != value else round(float(value), 4)
