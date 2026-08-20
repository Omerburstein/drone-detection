"""Cross-cut a scoring dump: detection rate and false alarms by size *and* condition.

The metric block answers "how does this detector do on tiny targets" and "how
does it do on complex backgrounds" separately. The question that actually decides
whether a system flies is the conjunction -- a 6 px drone against treeline -- and
neither marginal answers it, because the two axes are correlated on every
air-to-air dataset we have.

This reads the per-object dumps `src.evaluate --dump` wrote and prints the
(size band x condition) table, with Pd, precision, F1, false alarms per frame and
localisation error per cell. Several dumps can be tabled at once, which is how
one run is read under two matching criteria -- the comparison CLAUDE.md warns
about, where centre-matching and IoU@0.50 disagree by more on tiny targets than
any architecture change this project has measured.

Costs nothing and cannot disagree with the ledger: every number is a re-cut of a
scoring that already happened, never a second scoring.

Example
-------
    py -3.13 -m src.cross_eval \
        --dump "centre@1x=runs/exp004_glad/matches_center.csv" \
        --dump "IoU@0.50=runs/exp004_glad/matches_iou50.csv" \
        --axis scene_category --band "<8" --condition complex \
        --csv runs/exp004_glad/small_complex.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .eval.crosscut import (Cell, SIZE_EDGES, cell_rows, cross_cut, pooled,
                            select)
from .eval.curves import load_dump

ABSENT = "-"  # column filler where a cell has nothing to compute the number from
SPARSE = "*"  # marker on a cell holding too few targets to read as a measurement


def _ratio(value: float) -> str:
    """A ratio for the table, or a dash where it does not exist.

    Dashes rather than 0.0000, for the reason `report.py` gives: a cell with no
    targets has no detection rate, and a zero there reads as total failure --
    the opposite of "we did not measure this".
    """
    return ABSENT if value != value else f"{value:.4f}"


def _print_table(series: str, cells: list[Cell], axis: str) -> None:
    """One dump's cross-cut, plus the pooled row over the cells shown.

    The pooled row is recomputed from summed counts, not averaged, so it matches
    what the metric block would say about the same selection.
    """
    print(f"\n  {series}  --  by size band x {axis}:")
    print(f"    {'band':<11}{'condition':<14}{'frames':>8}{'targets':>8}"
          f"{'TP':>7}{'FN':>6}{'FP':>7}{'Pd':>9}{'P':>9}{'F1':>9}"
          f"{'FA/frm':>9}{'IoU':>8}{'offset':>8}")
    for cell in cells:
        flag = "" if cell.reliable else SPARSE
        print(f"    {cell.band:<11}{cell.condition + flag:<14}"
              f"{cell.n_frames:>8}{cell.n_gt:>8}{cell.tp:>7}{cell.fn:>6}"
              f"{cell.fp:>7}{_ratio(cell.pd):>9}{_ratio(cell.precision):>9}"
              f"{_ratio(cell.f1):>9}{_ratio(cell.far):>9}"
              f"{_ratio(cell.mean_iou):>8}{_ratio(cell.loc_err):>8}")

    if len(cells) > 1:
        total = pooled(cells)
        print(f"    {'-' * 99}")
        print(f"    {total.band:<11}{total.condition:<14}"
              f"{total.n_frames:>8}{total.n_gt:>8}{total.tp:>7}{total.fn:>6}"
              f"{total.fp:>7}{_ratio(total.pd):>9}{_ratio(total.precision):>9}"
              f"{_ratio(total.f1):>9}{_ratio(total.far):>9}"
              f"{ABSENT:>8}{ABSENT:>8}")


def _parse_dump(spec: str) -> tuple[str, Path]:
    """`name=path`, or a bare path named after its parent run directory."""
    name, _, path = spec.partition("=")
    return (name, Path(path)) if path else (Path(name).parent.name or name,
                                            Path(name))


def _parse_edges(spec: str) -> tuple[float, ...]:
    """A comma-separated edge list in pixels; `inf` closes the top bin.

    Edges are the band boundaries, so `0,8,inf` is the two-band cut this CLI
    exists for and the default is the same ladder `curves.py` plots on.
    """
    try:
        edges = tuple(float(part) for part in spec.split(",") if part.strip())
    except ValueError:
        sys.exit(f"--edges: expected comma-separated numbers, got {spec!r}")
    if len(edges) < 2 or list(edges) != sorted(edges):
        sys.exit(f"--edges: need at least two edges in increasing order, got {spec!r}")
    return edges


def build_parser() -> argparse.ArgumentParser:
    """The command line for the cross-cut table."""
    parser = argparse.ArgumentParser(
        description="Pd and false alarms per (target size x capture condition).")
    parser.add_argument("--dump", action="append", required=True, metavar="[NAME=]PATH",
                        help="scoring dump CSV; repeatable, one series each")
    parser.add_argument("--axis", default="scene_category",
                        help="condition column to cross with size "
                             "(default: scene_category)")
    parser.add_argument("--band", action="append", default=[], metavar="LABEL",
                        help="keep only these size bands, e.g. '<8'; repeatable")
    parser.add_argument("--condition", action="append", default=[], metavar="LABEL",
                        help="keep only these condition labels, e.g. complex")
    parser.add_argument("--edges", type=_parse_edges, default=SIZE_EDGES,
                        help="band edges in px, comma-separated (default: the "
                             "curves.py ladder)")
    parser.add_argument("--csv", type=Path,
                        help="write the table to this path as well as printing it")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Print the cross-cut for every dump given, and optionally write it out."""
    args = build_parser().parse_args(argv)

    tables: dict[str, list[Cell]] = {}
    for spec in args.dump:
        name, path = _parse_dump(spec)
        if not path.is_file():
            sys.exit(f"No dump at {path}")
        rows = load_dump(path)
        if rows and args.axis not in rows[0]:
            sys.exit(f"{path}: no column {args.axis!r} -- the run was scored "
                     f"without that axis. Columns: {', '.join(rows[0])}")
        cells = select(cross_cut(rows, args.axis, args.edges),
                       tuple(args.band), tuple(args.condition))
        if not cells:
            sys.exit(f"{path}: no cells match those --band/--condition filters")
        tables[name] = cells

    for name, cells in tables.items():
        _print_table(name, cells, args.axis)
    print(f"\n  {SPARSE} fewer than 30 targets in the cell -- a ratio, not a "
          f"measurement.\n")

    if args.csv:
        rows = cell_rows(tables, args.axis)
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  wrote {len(rows)} rows to {args.csv}\n")


if __name__ == "__main__":
    main()
