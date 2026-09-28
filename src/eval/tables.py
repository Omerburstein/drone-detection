"""Shared plumbing for the CLIs that read a scoring dump and write a table.

`cross_eval`, `alarm_eval` and `plot_eval` each load the same dump CSV, bin it
their own way, and write the result back out as CSV. The binning is genuinely
different in all three; everything around it was not, and was maintained as
three copies -- two of them byte-identical.

The risk that justifies collapsing them is not tidiness. `fieldnames=list(rows[0])`
makes the output schema follow the first row's key order, which is subtle and
load-bearing: it is why a table's columns match the order the row dicts were
built in. Three copies of that is three chances for one CLI's CSV to acquire a
different column order than its siblings for no stated reason.

Imports only stdlib, so `src/eval/` stays importable without torch, matplotlib
or a checkpoint present.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ..errors import UsageError


def rounded(value: float) -> float | None:
    """A number for the CSV, or None where the bin had nothing to compute from.

    `None` writes an empty cell. That is the right answer for a *summary* table,
    where a bin with no objects has no ratio -- a 0.0 there would read as total
    failure, the opposite of "we did not measure this".

    Note `records._round` deliberately does **not** call this: the per-object
    dump writes `""` rather than `None` and rounds to its own `DECIMALS`. That is
    a different schema with a different reader, not a fourth copy of this.
    """
    return None if value != value else round(float(value), 4)


def write_rows(path: Path, rows: list[dict]) -> int:
    """Write `rows` as CSV, creating the directory; return the row count.

    Columns come from the first row's key order, so the caller controls the
    layout by building its dicts in the order it wants read.
    """
    if not rows:
        raise ValueError(f"no rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def parse_dump_spec(spec: str, *, require_label: bool = False) -> tuple[str, Path]:
    """`LABEL=path` -> (label, path), for the `--dump` flag.

    `require_label` is what separates the two conventions this project already
    documents, and both are deliberate:

    * `cross_eval` and `alarm_eval` accept a bare path and name the series after
      its run directory. Their output is a terminal table read beside the command
      that produced it, so the path is on screen anyway.
    * `plot_eval` requires the label, because it goes into a **figure legend**
      that outlives the command. Only the caller knows whether a dump was the
      centre-matched scoring or the strict one, and confusing those two is the
      error this project keeps warning about.
    """
    label, sep, path = spec.partition("=")
    if not sep:
        if require_label:
            raise ValueError(
                f"expected LABEL=PATH, got {spec!r} "
                f"(e.g. 'centre@1x=runs/x/matches.csv')")
        # Bare path: the run directory is the most useful name available.
        return Path(spec).parent.name or spec, Path(spec)
    return label.strip(), Path(path.strip())


def parse_edges(spec: str) -> tuple[float, ...]:
    """A comma-separated edge list; `inf` closes the top bin.

    Edges are bin boundaries, so `0,8,inf` is a two-band cut. Sorted order is
    required rather than sorted for the caller: `0,16,8` is a typo, and silently
    reordering it would produce a table whose bands are not the ones asked for.
    """
    try:
        edges = tuple(float(part) for part in spec.split(",") if part.strip())
    except ValueError:
        raise UsageError(f"--edges: expected comma-separated numbers, got {spec!r}")
    if len(edges) < 2 or list(edges) != sorted(edges):
        raise UsageError(f"--edges: need at least two edges in increasing order, got {spec!r}")
    return edges
