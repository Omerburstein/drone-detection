"""False alarms binned by how far they landed from the nearest real drone.

A false-alarm count is one number standing over two completely different
failures. A box two pixels off a drone the detector plainly found, and a box on
an air-conditioning unit four hundred pixels away, are both `fp` — and the fix
for the first is a box regressor or a looser matching rule, while the fix for the
second is training data. Precision cannot tell them apart, and neither can
`far`.

Distance can. Every false alarm has a nearest target even though it has no
matched one, and where that distance sits says which failure it is:

- **under ~1 target size** — the alarm is *on* the drone. Under IoU matching
  these are the localisation misses the criterion charges twice, once as a false
  positive and once as a miss. Under centre matching at tolerance *t* a distance
  below *t* can only be a **second** box on an already-claimed target, since a
  first one would have matched: a duplicate, not a hallucination.
- **a few target sizes out** — near-misses on the drone's surroundings: its
  shadow, the contrail, the branch it is flying past.
- **tens of target sizes out** — genuine clutter. This is the population that
  says the detector does not know what a drone looks like.

Bins are in **multiples of the nearest target's own size** by default, not
pixels, for the reason the whole project bins that way: 20 px from a 60 px drone
is a graze and 20 px from a 6 px drone is a different object. Pixels are
available as `PX_EDGES` because an operator budgeting a rejection gate thinks in
pixels.

Reads the per-object dump, so this is a re-cut of a scoring that already
happened. Dumps written before `nearest_gt_dist` existed are handled by
re-deriving it from the frame's own rows -- every target in a frame appears
there exactly once, as a `tp` or an `fn`, so the frame's full ground truth is
recoverable. Both paths call `metrics.nearest_target`, so they cannot disagree
beyond the dump's own 4-decimal rounding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .crosscut import group_frames, number
from .curves import bin_index, bin_labels
from .metrics import box_size, nearest_target

FP = "fp"
TARGET_OUTCOMES = ("tp", "fn")

REL = "rel"
PX = "px"

# In multiples of the nearest target's size. The first edge is the centre
# criterion's own tolerance, so the first bin answers "would a looser matching
# rule have absorbed this alarm" directly.
REL_EDGES = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, float("inf"))

# In pixels, for the same distribution read as a rejection gate would see it.
PX_EDGES = (0.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, float("inf"))

DEFAULT_EDGES = {REL: REL_EDGES, PX: PX_EDGES}

NO_TARGET = "no target in frame"


@dataclass(frozen=True)
class Alarm:
    """One false alarm and its distance from the nearest real target.

    `distance` and `distance_rel` are NaN together, and only when the frame held
    no target at all. That is a distinct population from "far away" and is
    counted separately rather than dropped into the top bin.
    """

    key: str
    video: str
    group: str
    distance: float       # px to the nearest target in the frame
    distance_rel: float   # the same, in multiples of that target's size
    target_size: float    # that target's size, px

    @property
    def orphan(self) -> bool:
        """Whether this alarm fired in a frame with nothing to find."""
        return self.distance != self.distance

    def value(self, unit: str) -> float:
        """The distance in the requested unit."""
        return self.distance if unit == PX else self.distance_rel


@dataclass(frozen=True)
class AlarmTable:
    """Counts per distance bin, with the orphans kept out of the ladder.

    Shares are over **every** alarm including the orphans, so the column sums to
    one and no alarm is silently excluded from its own distribution.
    """

    unit: str
    edges: tuple[float, ...]
    group: str
    counts: np.ndarray    # (n_bins,) alarms per bin, orphans excluded
    orphans: int          # alarms in frames that held no target

    @property
    def labels(self) -> list[str]:
        """Bin names on the same convention the size ladder uses."""
        return bin_labels(self.edges)

    @property
    def total(self) -> int:
        """Every alarm this table was built from, orphans included."""
        return int(self.counts.sum()) + self.orphans

    @property
    def share(self) -> np.ndarray:
        """Each bin's fraction of all alarms; NaN when there are none at all."""
        if not self.total:
            return np.full(len(self.counts), np.nan)
        return self.counts / self.total

    @property
    def cumulative(self) -> np.ndarray:
        """Fraction of alarms at or nearer than the top of each bin.

        The column the table is actually read down: "how much of this
        detector's false-alarm count is within 2 target sizes of a real drone".
        """
        return np.cumsum(self.share)


def _boxes(rows: list[dict[str, str]], prefix: str) -> np.ndarray:
    """The xyxy corners of one prefix's boxes, as an (n, 4) array."""
    if not rows:
        return np.zeros((0, 4))
    return np.array([[number(row[f"{prefix}_{c}"]) for c in ("x0", "y0", "x1", "y1")]
                     for row in rows], dtype=float)


def _recorded(row: dict[str, str]) -> tuple[float, float, float] | None:
    """The dump's own nearest-target columns, or None if it predates them.

    A dump written before those columns existed has no key at all; one written
    after has blank cells only where the frame held no target. The two are
    different states and must not be conflated, which is why absence of the
    column is checked rather than emptiness of the cell.
    """
    if "nearest_gt_dist" not in row:
        return None
    return (number(row["nearest_gt_dist"]), number(row["nearest_gt_dist_rel"]),
            number(row.get("nearest_gt_size")))


def frame_alarms(rows: list[dict[str, str]], group: str = "") -> list[Alarm]:
    """Every false alarm in one frame, with its distance from the nearest target.

    Uses the dump's own columns when it has them and re-derives them from the
    frame's target rows when it does not, so a dump written before the columns
    landed still answers the question without being regenerated -- which matters,
    since regenerating one costs a full pass over the label tree.
    """
    alarms = [row for row in rows if row["outcome"] == FP]
    if not alarms:
        return []

    targets = [row for row in rows if row["outcome"] in TARGET_OUTCOMES]
    context = (rows[0]["key"], rows[0].get("video", ""),
               rows[0].get(group, "") if group else "")

    if _recorded(alarms[0]) is not None:
        return [Alarm(*context, *_recorded(row)) for row in alarms]

    if not targets:
        return [Alarm(*context, np.nan, np.nan, np.nan) for _ in alarms]

    gt_boxes = _boxes(targets, "gt")
    index, distance = nearest_target(_boxes(alarms, "pred"), gt_boxes)
    sizes = box_size(gt_boxes)
    return [
        Alarm(*context, float(distance[i]),
              float(distance[i] / sizes[index[i]]) if sizes[index[i]] > 0 else np.nan,
              float(sizes[index[i]]))
        for i in range(len(alarms))
    ]


def alarms(rows: list[dict[str, str]], group: str = "") -> list[Alarm]:
    """Every false alarm in a dump, in the order the file records them."""
    return [alarm for frame_rows in group_frames(rows).values()
            for alarm in frame_alarms(frame_rows, group)]


def bin_alarms(found: list[Alarm], unit: str = REL,
               edges: tuple[float, ...] | None = None,
               group: str = "") -> AlarmTable:
    """Count alarms into distance bins, orphans held aside.

    Half-open at the top of every bin, the same rule `curves.bin_index` uses, so
    an edge of 1.0 means "strictly inside one target size" and matches what a
    tolerance of 1.0 would have accepted.
    """
    edges = edges or DEFAULT_EDGES[unit]
    values = np.array([alarm.value(unit) for alarm in found], dtype=float)
    orphans = int(np.isnan(values).sum())
    placed = values[~np.isnan(values)]

    counts = np.zeros(len(edges) - 1, dtype=float)
    if len(placed):
        index = bin_index(placed, edges)
        for b in range(len(counts)):
            counts[b] = float((index == b).sum())
    return AlarmTable(unit=unit, edges=edges, group=group, counts=counts,
                      orphans=orphans)


def by_group(found: list[Alarm], unit: str = REL,
             edges: tuple[float, ...] | None = None) -> dict[str, AlarmTable]:
    """One table per distinct group label, in descending alarm count.

    Ordered by size because the question a grouped table answers is almost always
    "which sequence produces these", and on this project the answer has twice
    been "one of them" -- phantom63 contributed 129 of EXP-004's 188 alarms.
    """
    buckets: dict[str, list[Alarm]] = {}
    for alarm in found:
        buckets.setdefault(alarm.group, []).append(alarm)
    tables = {label: bin_alarms(group, unit, edges, label)
              for label, group in buckets.items()}
    return dict(sorted(tables.items(), key=lambda kv: -kv[1].total))


def table_rows(tables: dict[str, AlarmTable]) -> list[dict[str, object]]:
    """The binned counts as flat rows, keyed by series name, ready for a CSV.

    The orphan population gets its own row per series rather than a column, so
    every alarm in the file is on exactly one row and the counts sum to the
    total the metric block reports.
    """
    rows: list[dict[str, object]] = []
    for series, table in tables.items():
        for i, label in enumerate(table.labels):
            rows.append({
                "series": series,
                "group": table.group,
                "unit": table.unit,
                "bin": label,
                "bin_lo": table.edges[i],
                "bin_hi": table.edges[i + 1],
                "alarms": int(table.counts[i]),
                "share": _rounded(table.share[i]),
                "cumulative": _rounded(table.cumulative[i]),
            })
        rows.append({
            "series": series, "group": table.group, "unit": table.unit,
            "bin": NO_TARGET, "bin_lo": None, "bin_hi": None,
            "alarms": table.orphans,
            "share": _rounded(table.orphans / table.total) if table.total else None,
            "cumulative": 1.0 if table.total else None,
        })
    return rows


def _rounded(value: float) -> float | None:
    """A number for the CSV, or None where the bin had nothing to compute from."""
    return None if value != value else round(float(value), 4)
