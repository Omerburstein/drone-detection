# `src.cross_eval` — reference

Cuts a scoring dump by **target size and capture condition at once**, and prints Pd,
precision, F1, false alarms per frame and localisation error for each cell.

```
py -3.13 -m src.cross_eval --dump [LABEL=]CSV [--dump ...] [options]
```

## Why this table exists

The metric block cuts a run one way at a time. `by target size` says how the detector
does on tiny targets, pooled over every background; `by scene_category` says how it does
on complex backgrounds, pooled over every size. The question that decides whether a
system flies is the **conjunction** — a 6 px drone against treeline — and neither
marginal answers it.

They also cannot be multiplied together to get it, because the two axes are correlated.
On ARD-MAV the `small_mav` videos are almost entirely tiny targets: 5,032 of the 5,677
sub-8 px targets in the test split are in that category, so the pooled "tiny" recall is
mostly a statement about one scene category, not about size.

Every number is a re-cut of a scoring that already happened, so the table costs nothing
to produce and cannot disagree with the ledger. Pooling every cell of a run reproduces
that run's metric block exactly — the check `tests/unit/test_crosscut.py` pins and
EXP-004 confirms (pooled `all/all` → P 0.8559, Pd 0.7713, F1 0.8114, the recorded
headline).

## Parameters

| Parameter | Default | What it does |
| --- | --- | --- |
| `--dump` | required | `LABEL=path/to/matches.csv`, or a bare path named after its run directory. **Repeatable** — one table per series, which is how a run is read under two matching criteria at once. |
| `--axis` | `scene_category` | Which condition column to cross with size. Any axis column the dump carries works: `lighting` and `relative_range` once [scene_stats](scene_stats.md) has run. |
| `--band` | all | Keep only these size bands, e.g. `--band "<8"`. Repeatable. Labels are the ones in the table — `<8`, `8-12`, … `>=48`. |
| `--condition` | all | Keep only these condition labels, e.g. `--condition complex`. Repeatable. |
| `--edges` | `0,8,12,16,20,24,32,48,inf` | Band edges in px on `sqrt(w*h)`, comma-separated. `--edges 0,8,inf` collapses the ladder to the two-band cut. Must increase. |
| `--csv` | none | Write the table beside printing it, so the numbers can be re-read without re-deriving them. |

The default edges are the same ladder [`plot_eval`](plot_eval.md) bins on, deliberately:
a band in this table and a bin in that figure must mean the same thing, or the two
descriptions of one run silently disagree.

## Where a false alarm goes

This is the load-bearing decision, and it is the same problem
[plot_eval](plot_eval.md#what-is-on-each-axis) solves differently. **A false alarm has no
target, so it has no target size.** Precision-against-size sidesteps that by binning on
the alarm's own box. A cross-cut cannot: "how often does this detector cry wolf while
hunting an 8 px drone" is a question about the *hunt*, not about the alarm's appearance.

So cells are built **frame by frame, not row by row**. A frame gets a band from the
targets it contains, and every outcome in that frame — alarms included — inherits it. The
frame has a size even when the alarm does not: it is the size of the drone that was
actually there. That is what makes `FA/frm` mean the same thing per cell as it does in
the metric block: this cell's false alarms over this cell's frames.

Two bands exist for the cases where a frame genuinely has no single size:

- **`no target`** — a frame the detector fired on with nothing to find. It has real false
  alarms and no Pd at all, which the table prints as `-` rather than `0`.
- **`mixed sizes`** — a frame whose targets straddle a band edge. Not forced into one.

Both are reported rather than dropped, so cells still account for every row of the dump
exactly once. On ARD-MAV every frame holds exactly one target, so `mixed sizes` never
appears and `no target` holds only the 18 frames with no ground truth.

## Reading the table

```
  centre@1x  --  by size band x scene_category:
    band       condition       frames targets     TP    FN     FP       Pd        P       F1   FA/frm     IoU  offset
    <8         complex            640     640    552    88      2   0.8625   0.9964   0.9246   0.0031  0.5537  0.2236
```

- **`Pd`** is recall: matched targets over targets present. The project's primary number.
- **`FA/frm`** is the M5 false-alarm rate, cell-local. Read it *with* `Pd` — a detector
  that never fires scores 0 false alarms.
- **`P`** is precision over the alarms raised in this cell. With a large `frames` and a
  tiny `FP` it saturates near 1.0 and stops being informative; `FA/frm` does not.
- **`IoU`** and **`offset`** are over matched pairs only — "when it was found, how well
  was the box placed". A cell whose targets were all missed shows `-`, never `0`: a zero
  there would read as perfect placement.
- **`*`** on a condition label means fewer than 30 targets in the cell. A ratio, not a
  measurement.
- The **pooled row** under a multi-cell table recomputes ratios from summed counts, never
  by averaging per-cell ratios. `IoU` and `offset` are absent there because a median of
  medians is not a median.

## Two criteria, one table

Pass the same run's two dumps and the size-threshold interaction CLAUDE.md warns about
becomes visible directly:

```
py -3.13 -m src.cross_eval \
    --dump "centre@1x=runs/exp004_glad/matches_center.csv" \
    --dump "IoU@0.50=runs/exp004_glad/matches_iou50.csv" \
    --axis scene_category --band "<8" --condition complex \
    --csv runs/exp004_glad/small_complex.csv
```

On EXP-004 that one cell moves from Pd 0.8625 / 0.003 FA per frame under centre matching
to Pd 0.5844 / 0.281 FA per frame under IoU@0.50 — 28 points of Pd and a 90× false-alarm
rate, from the matching threshold alone, on an unchanged set of predictions. The gap
closes monotonically as targets grow and is gone by 24 px. **Never table the two
criteria's numbers together unlabelled**, and state the criterion whenever quoting one.

## Refusals

The CLI stops rather than printing a table that would be read wrong:

- A dump with no such axis column — every cell would read `uncategorised`.
- A `--band`/`--condition` filter matching nothing — an empty table is
  indistinguishable from a run that found nothing.
- Edges that are not increasing.

## See also

- [evaluate.md](evaluate.md) — producing the dump this reads, and the matching criteria.
- [plot_eval.md](plot_eval.md) — the same size axis as a figure, one criterion per line.
- [experiments.md](experiments.md) — EXP-004, whose cross-cut is quoted above.
