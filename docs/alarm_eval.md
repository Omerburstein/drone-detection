# `src.alarm_eval` — reference

Bins every false alarm in a scoring dump by **how far it landed from the nearest real
drone**, in multiples of that target's own size or in pixels.

```
py -3.13 -m src.alarm_eval --dump [LABEL=]CSV [--dump ...] [options]
```

## Why this table exists

`precision 0.86` and `0.13 false alarms per frame` both treat every wrong box as the same
wrong box. They are not the same, and the difference decides what to do next:

| Where the alarm landed | What it is | What fixes it |
| --- | --- | --- |
| under ~1 target size | a box **on** the drone, scored wrong | a box regressor, or the matching rule |
| a few target sizes out | the drone's surroundings — shadow, contrail, the branch it passed | context, or NMS |
| tens of target sizes out | genuine clutter | training data |

No headline number separates those. A detector at 0.86 precision whose alarms are all in
the first row is a **localisation** problem and nearly solved; the same 0.86 with alarms
in the third row does not know what a drone looks like. On EXP-004 this is not a
hypothetical — the two criteria's alarm counts differ by a factor of 19, and the table
shows the entire difference sitting in the first bin.

## Parameters

| Parameter | Default | What it does |
| --- | --- | --- |
| `--dump` | required | `LABEL=path/to/matches.csv`, or a bare path named after its run directory. **Repeatable** — one ladder per series. |
| `--unit` | `rel` | `rel` bins on multiples of the nearest target's own size; `px` bins on raw pixels. |
| `--edges` | the ladder for `--unit` | Bin edges, comma-separated, increasing. `inf` closes the top bin. |
| `--group` | none | Split each series by a dump column — `video`, `scene_category`, `branch`. Tables come back largest-first. |
| `--csv` | none | Write the table beside printing it. |

Default ladders: `rel` is `0,1,2,4,8,16,32,inf`; `px` is `0,5,10,25,50,100,250,500,inf`.

**Relative is the default deliberately.** 20 px from a 60 px drone is a graze; 20 px from
a 6 px drone is a different object. Pixels are there because an operator budgeting a
rejection gate thinks in pixels, not in target widths.

## What "distance" means here

**Distance to the nearest ground-truth box in the same frame, matched or not.** A false
alarm has no *matched* target — that is what makes it false — but it always has a nearest
one, and that is the quantity being binned. The matched target is explicitly not
privileged: a duplicate box sitting on the drone the detector already found is measured
against that drone, which is exactly the reading wanted.

The first bin means different things under the two criteria, and this is worth stating:

- Under **`--match iou`**, a sub-1 distance is a localisation miss — the box is on the
  drone but overlaps too little, so the criterion charges it twice, once as a false
  positive and once as a miss.
- Under **`--match center` at tolerance *t***, a distance below *t* can only be a
  **second** box on an already-claimed target, since a first one would have matched. It
  is a duplicate, not a hallucination.

**`no target in frame` is a separate row, not the top bin.** An alarm on a frame that
held nothing has no distance at all, and sweeping it into `>=32` would manufacture
evidence of long-range clutter the run never produced. It is counted, shown, and included
in the share denominator so the column still sums to 100%.

## Reading the table

```
  centre@1x  --  188 false alarms by distance (target sizes from the nearest drone):
    distance              alarms    share     cum   shape
    <1                         0    0.0%    0.0%
    1-2                        8    4.3%    4.3%   #
    ...
```

- **`alarms`** is the count. **`share`** is over every alarm including the orphan row.
- **`cum`** reads from the near end: "how much of this detector's false-alarm count is
  within *n* target sizes of a real drone". It is the column the table is actually read
  down.
- **`shape`** is a proportional bar. A mass at the near end and a mass at the far end are
  opposite diagnoses, and the shape makes which one obvious before the numbers are read.

## Where the numbers come from

The dump carries `nearest_gt_dist`, `nearest_gt_dist_rel` and `nearest_gt_size` on every
prediction row — see [evaluate.md](evaluate.md#the-per-object-dump). Dumps written before
2026-08-20 predate those columns, and this CLI **re-derives the distance from the frame's
own rows** in that case: every target appears exactly once, as a `tp` or an `fn`, so a
frame's full ground truth is recoverable from the dump alone. Both paths call
`metrics.nearest_target`, so they agree to the dump's 4-decimal rounding —
`tests/unit/test_records.py` pins that.

That fallback is why every run already in the ledger could be tabled without re-scoring.
Regenerating a dump costs a full pass over the label tree (~25 min for ARD-MAV); reading
one costs seconds.

## Cost

Seconds. It reads CSVs, never the detector, the labels or the images.

## See also

- [evaluate.md](evaluate.md) — producing the dump, and the matching criteria whose choice
  this table makes visible.
- [cross_eval.md](cross_eval.md) — the same dumps cut by target size and capture
  condition instead.
- [experiments.md](experiments.md) — EXP-004's alarm-distance ladder and what it settled.
