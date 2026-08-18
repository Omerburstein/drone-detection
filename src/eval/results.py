"""The append-only results log: one scored result per line, with its settings.

A metric block is uninterpretable without the settings that produced it. P/R/F1
under centre matching and under IoU matching are different measurements, and
`--iou 0.40` against `--iou 0.50` moved `small_mav` precision 23 points in
EXP-004 -- larger than any architectural difference this project has measured.
So the point of this file is not that it stores metrics; it is that it stores
them *attached to* the settings, in one place, so scoring the same run several
ways leaves a record that can be read back and compared instead of a series of
overwritten JSON files that all look alike.

Append-only and one JSON object per line, for the same reason `detections.jsonl`
is: re-scoring never destroys an earlier answer, and the file stays readable
while it is being written to.

`--json-out` is unchanged and still writes the bare `Metrics` schema the ledger
cites. This is the log; that is the snapshot.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .metrics import IOU, MatchCriterion, Metrics

# Bumped only if a field is removed or repurposed. Readers can then reject what
# they cannot interpret rather than silently mis-reading an older line.
SCHEMA = 1


@dataclass(frozen=True)
class EvalSettings:
    """Every input that changes the numbers, recorded beside them.

    Paths are stored as written on the command line rather than resolved: what
    matters when reading the log back is which run and which split, and an
    absolute path from another machine answers that no better.
    """

    pred: str
    labels: str
    match: str
    # Threshold for `iou`, tolerance in target sizes for `center`. One field,
    # because it is one knob -- `match` says which meaning applies.
    match_value: float
    conditions: str | None = None
    frame_size: list[int] | None = None

    @classmethod
    def from_args(cls, args, criterion: MatchCriterion) -> EvalSettings:
        """The settings of one `src.evaluate` invocation.

        The criterion is passed in rather than re-derived from `args`, so the
        value recorded is the one that actually scored the run.
        """
        return cls(
            pred=str(args.pred),
            labels=str(args.labels),
            match=criterion.kind,
            match_value=criterion.value,
            conditions=str(args.conditions) if args.conditions else None,
            frame_size=list(args.frame_size) if args.frame_size else None,
        )

    @property
    def is_comparable_to_published_map(self) -> bool:
        """Whether mAP@0.50:0.95 is even defined for this setting."""
        return self.match == IOU


def append_result(path: Path, settings: EvalSettings, metrics: Metrics,
                  when: datetime | None = None) -> int:
    """Append one scored result to `path`; return how many it now holds.

    The count is what the CLI prints, so a second scoring of the same run at
    different settings visibly grows the log rather than looking like a no-op.
    """
    record = {
        "schema": SCHEMA,
        "time": (when or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        "criterion": metrics.criterion,
        "settings": asdict(settings),
        "metrics": asdict(metrics),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return len(load_results(path))


def load_results(path: Path) -> list[dict]:
    """Every result recorded in `path`, oldest first.

    Blank lines are skipped; a malformed line is an error, because a log that
    silently drops what it cannot parse is worse than one that says so.
    """
    if not path.exists():
        return []
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number} is not valid JSON: {exc}") from exc
    return records
