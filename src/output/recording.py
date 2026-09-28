"""The JSONL detection record and the run's headline counters."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Self

from ..algo.detections import Detections

BBOX_DECIMALS = 1  # sub-pixel precision is noise at these target sizes
CONF_DECIMALS = 4
MIN_ELAPSED = 1e-9  # seconds; floor for rate division on a run too short to time


def print_branches(branches: Counter) -> None:
    """Report which branch produced each processed frame.

    This is the paper's ablation table measured on our own run: how much of the
    recall is appearance and how much is motion. Here rather than in a CLI
    because `src.glad_detect` and `src.live_detect` both print it, and they used
    to hold byte-identical copies -- a drift in rounding or column width would
    have desynced the ledger's tables from what the tools actually print.
    """
    total = max(sum(branches.values()), 1)
    print("\nBranch that handled each frame:")
    for branch, count in branches.most_common():
        print(f"  {branch:<14} {count:>7}  {100 * count / total:5.1f}%")


class RunRecorder:
    """Owns `detections.jsonl` and the statistics computed from it.

    Every processed frame funnels through here regardless of source type, so the
    record schema and the run statistics are defined in exactly one place. Rows
    are appended as they are produced, so an interrupted run still leaves a
    readable file.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = path.open("w", encoding="utf-8")
        self.n_frames = 0
        self.n_dets = 0
        self.n_empty = 0
        self._t0 = time.perf_counter()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc) -> None:
        self._file.close()

    def record(self, key: dict[str, Any], dets: Detections) -> None:
        """Append one frame's detections and fold them into the counters.

        Frames with nothing found are written too, not skipped — on a baseline
        run the misses are the measurement.
        """
        row = dict(key)
        row["detections"] = dets.to_records(BBOX_DECIMALS, CONF_DECIMALS)
        self._file.write(json.dumps(row) + "\n")
        self.n_frames += 1
        self.n_dets += len(dets)
        self.n_empty += len(dets) == 0

    @property
    def elapsed(self) -> float:
        """Seconds since the recorder was created."""
        return time.perf_counter() - self._t0

    def print_progress(self, label: str) -> None:
        """One in-flight progress line, positioned by a caller-supplied label."""
        rate = self.n_frames / max(self.elapsed, MIN_ELAPSED)
        print(f"  {label}  |  {self.n_dets} dets  |  {rate:.2f} fps")

    def print_summary(self) -> None:
        """Final throughput and detection-rate report."""
        elapsed = self.elapsed
        frames = max(self.n_frames, 1)
        print(f"\n{self.n_frames} frames in {elapsed:.1f}s "
              f"({self.n_frames / max(elapsed, MIN_ELAPSED):.2f} fps)")
        print(f"{self.n_dets} detections, {self.n_dets / frames:.2f} per frame")
        # Most air-to-air frames contain exactly one drone, so on those datasets the
        # empty-frame rate is a rough miss rate -- the headline baseline number.
        print(f"Frames with no detection: {self.n_empty}/{self.n_frames} "
              f"({100 * self.n_empty / frames:.1f}%)")
        print(f"Wrote {self.path}")
