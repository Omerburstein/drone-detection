# `src.plot_eval` — reference

Draws precision as a function of target size in pixels, from one or more dumps written by
`src.evaluate --dump`.

```
py -3.13 -m src.plot_eval --dump LABEL=CSV [--dump LABEL=CSV ...] [options]
```

## Why this chart exists

The metric block gives precision as a single number and recall in four coarse buckets.
Neither answers the question this project actually turns on: **how small can a target get
before a claim about it stops meaning anything.** On ARD-MAV the boxes span roughly 8 px
to 90 px, and 64.8% of them are under 16 px — a range over which the detector's
reliability is the thing under study, not a nuisance to average over.

## Parameters

| Parameter | Default | What it does |
| --- | --- | --- |
| `--dump` | required | `LABEL=path/to/matches.csv`. The label is what the legend says. **Repeatable** — overlaying the same run under two matching criteria is the useful case. |
| `--out` | `precision_by_size.png` | Where to write the figure. |
| `--data-out` | `--out` with a `.csv` suffix | The binned numbers behind the figure, so it can be redrawn or checked without re-deriving anything. Carries three series per dump — precision, recall and the size-normalised centre offset — of which only precision is drawn. |
| `--title` | `Precision against target size` | Figure title. |
| `--subtitle` | none | The line under it. Put the run, split and frame count here. |

The label has to come from you because only you know which scoring a dump was: mixing up
the centre-matched file and the IoU-matched one is precisely the error
[evaluate.md](evaluate.md#matching-criteria) keeps warning about, and the figure cannot
detect it.

## What is on each axis

**Precision bins on `pred_size` — the size of the box the detector emitted, not the
target's true size.** This is forced: a false alarm has no target, so its own box is the
only size it has. The line reads *"when this detector claims a drone this big, how often
is it right"*.

Recall, which the same command writes into the data CSV but does **not** plot, bins on
`gt_size` instead — *"of the drones that were this big, how many were found"*. The two
are not two views of one axis, and a single chart showing both at a shared x would invite
reading them as a trade-off curve at one size. They are in the same file, labelled by
`binned_on`, and kept off the same panel.

The **centre offset** (`metric = loc_error`) is written to the CSV as well, also binned
on `gt_size`, over `tp` rows only — a false alarm has no target to be offset from and a
miss has no box. It carries `median` and `p90` where the ratio series carry `hits`, since
a distribution has no numerator. Read it against the recall series over the same bins:
recall says how often a drone of that size was found, the offset says how well the box
sat on it once it was. Rising offset with falling size, at steady recall, is a matching
threshold story rather than a detection one.

Bin edges are 8 / 12 / 16 / 20 / 24 / 32 / 48 px with an open bin above — fine where the
data lives, coarse where it thins out. The open top bin is deliberate: an appearance-only
detector's false alarms pile up above any fixed edge (EXP-001–003 boxed window recesses
and air-conditioning units at ~1.7× the true target size).

## Reading it

- **Hollow markers mean fewer than 30 predictions in that bin.** A ratio over a handful
  of samples is noise, and at the same dot size the eye reads noise as trend.
- **The lower panel is the sample count per bin**, on its own axes. It is a second panel
  and not a second y-scale on purpose — a dual-axis chart is the one form this project
  will not draw.
- Two lines from the same run under two criteria are **not** two detectors. Their gap is
  the ruler, not the model: it is localisation slack being charged as a false alarm at
  small sizes, which is the whole subject of EXP-004.

## Example

```bash
py -3.13 -m src.plot_eval \
    --dump "centre@1x=runs/exp004_glad/matches_center.csv" \
    --dump "IoU@0.50=runs/exp004_glad/matches_iou50.csv" \
    --out runs/exp004_glad/precision_by_size.png \
    --subtitle "GLAD released weights · ARD-MAV official 15-video test split · 28,337 frames"
```

Writes the figure and `precision_by_size.csv` beside it:

| Column | What it is |
| --- | --- |
| `series` | The `--dump` label the row came from. |
| `metric` | `precision` or `recall`. |
| `binned_on` | `pred_size` or `gt_size` — **read this before comparing two rows.** |
| `bin`, `bin_lo`, `bin_hi` | The bucket, and its edges in pixels. |
| `value` | The metric. Empty where the bin held nothing — not `0`, which would claim the detector was tried at that size and failed. |
| `hits`, `total` | Numerator and denominator. A value without them is unreadable in the tail. |
| `reliable` | Whether `total` cleared the 30-object floor. |

## Cost

Seconds. It reads CSVs, never the detector or the images, so a re-style or a new bin
edge costs nothing. Re-binning wants `--dump` regenerated only if you change the
*matching*, not the plotting.

## See also

[`src.cross_eval`](cross_eval.md) cuts the same dumps on the same bin edges, but crosses
size with a condition axis and prints a table instead of drawing a line. Use it when the
question is a conjunction — tiny **and** cluttered — and this when the question is size
alone across the whole run. The edges are shared deliberately: a band there and a bin
here must mean the same thing.
