"""Grouping evaluation frames by the conditions they were captured under.

Aggregate metrics hide the thing that matters most here: a detector can look
respectable overall while failing completely on complex backgrounds, on poorly
lit targets, or at long range. GLAD publishes its results split by scene
category, so scoring the same way is what makes our numbers comparable to
theirs at all -- and the other two axes are what separate failures the published
grouping conflates.

Axes come from `conditions.json`, written by `src.data.prepare_ardmav` and
`src.data.scene_stats`. An axis is either *video*-level (one label per video,
like the published `scene_category`) or *frame*-level (one label per frame, like
the measured `lighting` and `relative_range`, both of which vary within a
sequence).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from .labels import EvalFrame

UNKNOWN = "uncategorised"
VIDEO = "video"
FRAME = "frame"
LEGACY_KEY = "scene_category"


@dataclass(frozen=True)
class Axis:
    """One way of splitting a run, and the labels that do the splitting.

    `order` fixes how the buckets are reported. Contrast and range buckets read
    as a trend, and sorting them alphabetically would scramble that; an axis
    without a declared order falls back to sorted labels.
    """

    name: str
    level: str
    labels: dict[str, str]
    order: tuple[str, ...] = ()
    description: str = ""

    def label_for(self, key: str) -> str:
        """Bucket a frame key falls in, `UNKNOWN` when the axis does not cover it."""
        return self.labels.get(key if self.level == FRAME else video_of(key), UNKNOWN)

    def sort_key(self, label: str):
        """Position of a label in the report, declared order first."""
        return (self.order.index(label), "") if label in self.order else (len(self.order),
                                                                          label)


def video_of(key: str) -> str:
    """Video name for a frame key.

    Frame keys are stems like `phantom05_0001`, so the video is everything
    before the final underscore. Video-keyed runs use a bare frame index and
    have no video to recover, which is why conditions need an image-keyed run.
    """
    return key.rsplit("_", 1)[0] if "_" in key else key


def _axis_from_spec(name: str, spec: dict) -> Axis:
    """Build one axis from its `conditions.json` entry."""
    labels = spec.get("labels")
    if not isinstance(labels, dict):
        sys.exit(f"conditions: axis {name!r} has no 'labels' object")
    level = spec.get("level", VIDEO)
    if level not in (VIDEO, FRAME):
        sys.exit(f"conditions: axis {name!r} has level {level!r}, expected "
                 f"{VIDEO!r} or {FRAME!r}")
    return Axis(name=name, level=level, labels=labels,
                order=tuple(spec.get("order", ())),
                description=spec.get("description", ""))


def load_conditions(path: Path) -> list[Axis]:
    """Read `conditions.json` into the axes it declares.

    Accepts both shapes. The current one carries an `axes` object; the original
    carried a bare `scene_category` map and nothing else, and files in that
    shape are still read as a single video-level axis rather than rejected --
    an older processed tree stays scoreable without being regenerated.
    """
    if not path.exists():
        sys.exit(f"No conditions file at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))

    axes = data.get("axes")
    if isinstance(axes, dict) and axes:
        return [_axis_from_spec(name, spec) for name, spec in axes.items()]

    legacy = data.get(LEGACY_KEY)
    if isinstance(legacy, dict):
        return [Axis(name=LEGACY_KEY, level=VIDEO, labels=legacy)]

    sys.exit(f"{path}: expected an 'axes' object, or a '{LEGACY_KEY}' map "
             f"of video -> category")


def group_by_axis(frames: list[EvalFrame], axis: Axis) -> dict[str, list[EvalFrame]]:
    """Bucket frames along one axis, in that axis's reporting order.

    Frames the axis does not cover land in `uncategorised` rather than being
    dropped -- silently discarding them would change the denominator of every
    per-category number without saying so.
    """
    grouped: dict[str, list[EvalFrame]] = {}
    for frame in frames:
        grouped.setdefault(axis.label_for(frame.key), []).append(frame)
    return {label: grouped[label] for label in sorted(grouped, key=axis.sort_key)}


def group_by_condition(frames: list[EvalFrame],
                       conditions: dict[str, str]) -> dict[str, list[EvalFrame]]:
    """Bucket frames by a bare video -> category map.

    The one-axis convenience form, kept because a plain mapping is the natural
    thing to pass from a test or a notebook.
    """
    return group_by_axis(frames, Axis(LEGACY_KEY, VIDEO, conditions))


def as_axes(conditions: dict[str, str] | list[Axis] | None) -> list[Axis]:
    """Coerce whatever a caller passed as `conditions` into a list of axes."""
    if not conditions:
        return []
    if isinstance(conditions, dict):
        return [Axis(LEGACY_KEY, VIDEO, conditions)]
    return list(conditions)
