"""Table false alarms by their distance from the nearest real drone.

`precision 0.86` and `0.13 false alarms per frame` both treat every wrong box as
the same wrong box. They are not. A box two pixels off a drone the detector
found, and a box on a rooftop four hundred pixels away, need opposite fixes —
the first a box regressor or a looser matching rule, the second training data —
and no headline number separates them.

This bins every `fp` row of a scoring dump by how far it landed from the nearest
ground-truth box in its own frame, in multiples of that target's size by default
and in pixels on request. Reads the persisted dumps, so it costs seconds and
cannot disagree with the ledger.

Example
-------
    py -3.13 -m src.alarm_eval \
        --dump "centre@1x=runs/exp004_glad/matches_center.csv" \
        --dump "IoU@0.50=runs/exp004_glad/matches_iou50.csv" \
        --csv runs/exp004_glad/alarm_distance.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .eval.alarms import (DEFAULT_EDGES, NO_TARGET, PX, REL, AlarmTable, alarms,
                          bin_alarms, by_group, table_rows)
from .eval.curves import load_dump

UNITS = {REL: "target sizes from the nearest drone",
         PX: "pixels from the nearest drone"}


def _bar(share: float, width: int = 24) -> str:
    """A proportional bar, so the shape of the distribution reads at a glance.

    Blocks rather than a plot: this table is usually read in a terminal beside
    the command that produced it, and the shape is the whole point — a mass at
    the near end and a mass at the far end are opposite diagnoses.
    """
    if share != share:
        return ""
    filled = int(round(share * width))
    return "#" * filled if filled else ("." if share > 0 else "")


def _print_table(series: str, table: AlarmTable) -> None:
    """One dump's alarm-distance ladder, with the orphan row under it."""
    print(f"\n  {series}  --  {table.total} false alarms by distance "
          f"({UNITS[table.unit]}):")
    print(f"    {'distance':<20}{'alarms':>8}{'share':>9}{'cum':>8}   shape")
    for i, label in enumerate(table.labels):
        print(f"    {label:<20}{int(table.counts[i]):>8}"
              f"{table.share[i]:>8.1%}{table.cumulative[i]:>8.1%}   "
              f"{_bar(table.share[i])}")
    if table.orphans:
        share = table.orphans / table.total
        print(f"    {NO_TARGET:<20}{table.orphans:>8}{share:>8.1%}"
              f"{'':>8}   {_bar(share)}")


def _parse_dump(spec: str) -> tuple[str, Path]:
    """`name=path`, or a bare path named after its parent run directory."""
    name, _, path = spec.partition("=")
    return (name, Path(path)) if path else (Path(name).parent.name or name,
                                            Path(name))


def _parse_edges(spec: str) -> tuple[float, ...]:
    """A comma-separated edge list; `inf` closes the top bin."""
    try:
        edges = tuple(float(part) for part in spec.split(",") if part.strip())
    except ValueError:
        sys.exit(f"--edges: expected comma-separated numbers, got {spec!r}")
    if len(edges) < 2 or list(edges) != sorted(edges):
        sys.exit(f"--edges: need at least two edges in increasing order, got {spec!r}")
    return edges


def build_parser() -> argparse.ArgumentParser:
    """The command line for the alarm-distance table."""
    parser = argparse.ArgumentParser(
        description="False alarms binned by distance from the nearest real target.")
    parser.add_argument("--dump", action="append", required=True, metavar="[NAME=]PATH",
                        help="scoring dump CSV; repeatable, one series each")
    parser.add_argument("--unit", choices=(REL, PX), default=REL,
                        help="bin in multiples of the target's own size (default) "
                             "or in pixels")
    parser.add_argument("--edges", type=_parse_edges,
                        help="bin edges, comma-separated; defaults to the ladder "
                             "for the chosen --unit")
    parser.add_argument("--group", metavar="COLUMN",
                        help="split each series by a dump column, e.g. video or "
                             "scene_category")
    parser.add_argument("--csv", type=Path,
                        help="write the table as well as printing it")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Print the alarm-distance table for every dump given."""
    args = build_parser().parse_args(argv)
    edges = args.edges or DEFAULT_EDGES[args.unit]

    tables: dict[str, AlarmTable] = {}
    for spec in args.dump:
        name, path = _parse_dump(spec)
        if not path.is_file():
            sys.exit(f"No dump at {path}")
        rows = load_dump(path)
        if rows and args.group and args.group not in rows[0]:
            sys.exit(f"{path}: no column {args.group!r}. "
                     f"Columns: {', '.join(rows[0])}")

        found = alarms(rows, args.group or "")
        if not found:
            sys.exit(f"{path}: no false alarms to bin -- nothing to table")
        if args.group:
            tables.update({f"{name} / {label}": table
                           for label, table in by_group(found, args.unit,
                                                        edges).items()})
        else:
            tables[name] = bin_alarms(found, args.unit, edges)

    for series, table in tables.items():
        _print_table(series, table)
    print(f"\n  Distance is to the nearest ground-truth box in the same frame,\n"
          f"  matched or not. `{NO_TARGET}` is counted separately: an alarm on an\n"
          f"  empty frame has no distance, and the top bin is not where it belongs.\n")

    if args.csv:
        rows = table_rows(tables)
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  wrote {len(rows)} rows to {args.csv}\n")


if __name__ == "__main__":
    main()
