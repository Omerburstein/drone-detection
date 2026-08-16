"""Grouping evaluation frames by the condition they were captured under.

Aggregate metrics hide the thing that matters most here: a detector can look
respectable overall while failing completely on complex backgrounds or on the
smallest targets. GLAD publishes its results split this way, so scoring the same
way is what makes our numbers comparable to theirs at all.

Categories come from `conditions.json`, written by `src.data.prepare_ardmav`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .labels import EvalFrame

UNKNOWN = "uncategorised"


def load_conditions(path: Path) -> dict[str, str]:
    """Read `conditions.json` into a {video: category} mapping."""
    if not path.exists():
        sys.exit(f"No conditions file at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    mapping = data.get("scene_category")
    if not isinstance(mapping, dict):
        sys.exit(f"{path}: expected a 'scene_category' object mapping video -> category")
    return mapping


def video_of(key: str) -> str:
    """Video name for a frame key.

    Frame keys are stems like `phantom05_0001`, so the video is everything
    before the final underscore. Video-keyed runs use a bare frame index and
    have no video to recover, which is why conditions need an image-keyed run.
    """
    return key.rsplit("_", 1)[0] if "_" in key else key


def group_by_condition(frames: list[EvalFrame],
                       conditions: dict[str, str]) -> dict[str, list[EvalFrame]]:
    """Bucket frames by scene category, preserving the order categories appear.

    Frames whose video is absent from the mapping land in `uncategorised` rather
    than being dropped -- silently discarding them would change the denominator
    of every per-category number without saying so.
    """
    grouped: dict[str, list[EvalFrame]] = {}
    for frame in frames:
        grouped.setdefault(conditions.get(video_of(frame.key), UNKNOWN), []).append(frame)
    return grouped
