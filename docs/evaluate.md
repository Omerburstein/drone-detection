# `src.evaluate` — reference

Scores a `detections.jsonl` against ground-truth labels. Matching defaults to the centre
rule (`--match center`); `--match iou` is COCO's protocol exactly.

```
py -3.13 -m src.evaluate --pred <jsonl> --labels <dir> [options]
```

## Parameters

| Parameter | Default | What it does |
| --- | --- | --- |
| `--pred` | required | `detections.jsonl` written by `baseline_detect.py`. |
| `--labels` | required | Directory of YOLO-format `.txt` label files. Matched to predictions by **filename stem** — `img042.jpg` pairs with `img042.txt`. Video runs pair frame index to `<idx>.txt`. |
| `--iou` | `0.5` | IoU threshold for the precision/recall/size breakdowns, when `--match iou`. AP@0.50 and mAP@0.50:0.95 always use the standard sweep regardless of this. Ignored under the default criterion. |
| `--match` | `center` | How a prediction claims a target: `center` (centre-to-centre distance, the default) or `iou` (COCO overlap). See **Matching criteria** below. |
| `--match-tol` | `1.0` | For `--match center`: the centre-distance tolerance in multiples of the target's own size. `1.0` means "more than one drone-width off centre is a miss". |
| `--frame-size W H` | none | Frame dimensions in pixels. **Required for video-keyed predictions** — the JSONL stores frame indices, not sizes. For image runs it is read from the image file, and this flag overrides it. |
| `--conditions` | none | `conditions.json` from `src.data.prepare_ardmav`. Adds one breakdown table per axis the file declares — `scene_category` (required to compare against papers that report by category), plus the measured `lighting` and `relative_range`. See [scene_stats.md](scene_stats.md). |
| `--json-out` | none | Also write metrics as JSON — one snapshot, overwritten each run. Pass it for anything you intend to cite in the ledger. |
| `--save` | none | Append this result **and the settings that produced it** to a results log (JSONL). See **The results log** below. |
| `--dump` | none | Write one CSV row per true positive, false positive and missed target — the table the metric block is a `GROUP BY` over. See **The per-object dump** below. |

A missing label file is read as "no ground-truth boxes in this frame," not an error —
that is how legitimate negative frames are represented. A missing *image* when the size
cannot be resolved is a hard error.

## Metrics glossary

**AP@0.50** — Average Precision at IoU 0.50. Sort every prediction by confidence, walk
down the list accumulating true and false positives, and integrate precision over
recall (101-point interpolation). The single headline number. A prediction counts as
correct if it overlaps a ground-truth box by ≥50% IoU and that box is not already
claimed.

**mAP@0.50:0.95** — the same, averaged over IoU thresholds 0.50, 0.55 … 0.95. Stricter,
because it rewards tight boxes. If this is much lower than AP@0.50, targets are being
*found* but boxed *loosely*.

**Precision** — TP / (TP + FP). Of the boxes emitted, how many were real. Low precision
means false alarms on clutter, birds, or sky texture.

**Recall** — TP / (TP + FN). Of the drones present, how many were found. Low recall
means misses. **On this project, recall is usually the binding constraint.**

**F1** — harmonic mean of precision and recall. One number when you need one, but it
hides which side is failing, so never read it alone.

**Mean IoU** — average overlap across *matched* boxes only. Pure localisation quality,
scale-invariantly expressed. This is the well-posed version of "how far off are the box
coordinates" — but because it ignores every miss and every false alarm by construction,
it is only meaningful next to precision and recall.

**Frames with a miss** — frames containing at least one unfound ground-truth drone.
Directly comparable to the empty-frame rate printed by `baseline_detect.py`, but honest:
it uses labels, so a box on a cloud no longer counts as a hit.

**Recall by target size** — recall bucketed by ground-truth box area: tiny (<16 px),
small (16–32), medium (32–96), large (>96). Buckets are finer at the small end than
COCO's, because on air-to-air data nearly everything would otherwise land in one bucket.

> **This is the most diagnostic block in the report.** Strong recall on medium/large
> collapsing to near-zero on tiny means the problem is input resolution, not
> architecture. Raise `--imgsz` or turn on `--tile` before reaching for a bigger model.

**Scores by condition** — precision, recall, F1 and AP@0.50 within each bucket, printed
as **one table per axis**, enabled by `--conditions`. ARD-MAV declares three:

| Axis | Level | Buckets |
| --- | --- | --- |
| `scene_category` | per video | `ordinary` / `complex` / `small_mav` — GLAD's published grouping |
| `lighting` | per frame | target-vs-background separation: `invisible (<5)` → `strong (>=30)`, plus `backlit` |
| `relative_range` | per frame | apparent size as range: `near (<2x)` → `very far (>5x)` |

> Aggregates hide the failure that matters. A detector can look uniformly mediocre
> overall while actually being adequate on sky and useless against urban clutter — two
> situations needing completely different fixes. `scene_category` is also the only form
> in which results can be compared against GLAD, which publishes per category and not in
> aggregate.
>
> The other two axes exist because `scene_category` **conflates things that need
> different fixes**: ARD-MAV's `small_mav` videos are both the longest-range *and* the
> worst-lit in the split, and those two handicaps compound while being nearly
> uncorrelated (r = 0.071). See [scene_stats.md](scene_stats.md) for how they are
> measured and [experiments.md](experiments.md) for what they showed.

**Never sum across axes.** Every axis covers every frame, so counts partition *within* an
axis and double-count across them. That is why the tables are printed separately.

A bucket with no ground truth reports **NaN**, not 0.0: NaN says "not measured", whereas
0.0 would claim the detector was tried there and failed. Frames an axis does not cover
are bucketed as `uncategorised` rather than dropped — discarding them would silently
change every per-bucket denominator.

Bucket order follows the axis's declared order where it has one, so lighting and range
read worst-to-best rather than alphabetically.

## Matching rules

- **Greedy, in descending confidence order.** The most confident prediction picks first.
- **Each ground-truth box can be claimed once.** A second box on an already-matched
  drone becomes a false positive — this is what penalises duplicate detections.
- **Class-aware.** A `bird` prediction cannot satisfy a `drone` ground truth.

These hold under either criterion below.

## Matching criteria

One number cannot do both jobs at these target sizes, so the criterion is a choice.

### `--match iou`

A prediction claims a target when their IoU is at least `--iou`. COCO's rule, and **the
only setting comparable to a published mAP** — pass it, and state the threshold, whenever
that is the comparison being made.

Its weakness is exactly this project's regime. IoU falls off with *relative* offset, so a
fixed threshold gets stricter as targets shrink. Measured on EXP-004: moving the threshold
from 0.50 to 0.40 moved `small_mav` precision 23 points and recall 19, while barely
touching `ordinary`. A 3 px error on a 12 px drone is scored a miss *and* a false alarm;
the same 3 px on a 100 px target is a clean hit.

### `--match center` (default)

A prediction claims a target when their **centres are within `--match-tol` target sizes**
of each other, where size is `sqrt(w*h)` of the ground-truth box — the same notion of size
the recall-by-size buckets use.

This answers *did the detector find the drone*, which is the question a false-alarm rate is
actually asking. It is size-relative by construction, so it is equally strict on a 10 px
and a 100 px target. Box dimensions are ignored entirely: a correctly centred but badly
sized box is a detection, and its sizing error shows up in **mean IoU** instead.

**mAP@0.50:0.95 is reported as `n/a` under this criterion.** That metric is defined by
averaging over IoU thresholds; sweeping the distance tolerance instead would produce a
number that looks like mAP, is not, and would eventually be compared to one.

### Which to use

| Question | Criterion |
| --- | --- |
| How well are the boxes placed? (regression quality) | `iou`, plus **mean IoU** |
| Did we find the drone? Is this a false alarm? | `center` |
| Comparing to a published mAP | `iou`, and state the threshold |

`center` is the default because it is the right ruler for this project's own question —
is the detector finding 10-30 px drones, and are its false alarms real. `iou` is one flag
away, and required the moment the comparison is against someone else's published number.

**P/R/F1 from the two criteria are different measurements and must never be compared.**
The criterion is recorded in the printed report, in the `criterion` field of `--json-out`,
and on every line of the `--save` log, for exactly that reason.

The criterion does not simply flatter a detector. Re-scoring the four runs in the ledger,
false alarms fell 95% for GLAD (3,658 → 188) — its "false alarms" were mostly its own boxes
sitting just off a real drone — and by 0.2% for the off-the-shelf baselines, whose false
alarms are genuinely on air-conditioning units and window recesses.

## The results log

Re-scoring is free — `detections.jsonl` is persisted, so the same run can be scored
under any number of settings without touching the detector. `--save` is what makes that
worth doing: it **appends** one JSON object per scoring, so the settings stay attached to
the numbers instead of the previous answer being overwritten.

```bash
# Score under the default centre rule.
py -3.13 -m src.evaluate --pred runs/exp004/detections.jsonl     --labels data/processed/ARD-MAV/labels/test --frame-size 1920 1080     --save runs/exp004/results.jsonl

# Then again under COCO's rule, and again at a looser threshold.
py -3.13 -m src.evaluate ... --match iou --iou 0.50 --save runs/exp004/results.jsonl
py -3.13 -m src.evaluate ... --match iou --iou 0.40 --save runs/exp004/results.jsonl
```

Each line holds:

| Field | What it is |
| --- | --- |
| `schema` | Log format version. Bumped only if a field is removed or repurposed. |
| `time` | UTC timestamp, to the second. |
| `criterion` | The criterion label as printed — `centre@1x target size`, `IoU@0.40`. |
| `settings` | `pred`, `labels`, `match`, `match_value`, `conditions`, `frame_size`. |
| `metrics` | The complete `--json-out` schema, `by_size` and `by_condition` included. |

`match_value` is one field because it is one knob: the threshold under `iou`, the
tolerance in target sizes under `center`. `match` says which meaning applies.

Read it back with `src.eval.results.load_results`, or from the shell:

```bash
py -3.13 -c "import json,sys; [print(r['criterion'], '%.4f %.4f' % (r['metrics']['precision'], r['metrics']['recall'])) for r in map(json.loads, open('runs/exp004/results.jsonl'))]"
```

A malformed line is an error rather than a silent skip — a history that quietly shortens
itself is worse than one that says it is broken.

`--json-out` is unchanged: one file, overwritten, holding the bare `Metrics` schema the
ledger cites. Use it for the number being cited and `--save` for the history behind it;
they are independent and can be passed together.

## The per-object dump

`--json-out` and `--save` store *summaries*. A summary answers one question and discards
what the next one needs: `precision 0.9926` cannot be re-cut by target size, `recall by
size` cannot be re-cut by video, and neither survives a change of criterion without
re-running the scorer.

`--dump` writes the table those summaries are computed from — **one row per prediction
and one row per ground-truth box, each appearing exactly once**:

```bash
py -3.13 -m src.evaluate     --pred runs/exp004_glad/detections.jsonl     --labels data/processed/ARD-MAV/labels/test     --conditions data/processed/ARD-MAV/conditions.json     --frame-size 1920 1080     --dump runs/exp004_glad/matches_center.csv
```

| Column | What it is |
| --- | --- |
| `key`, `video` | Frame stem and the sequence it belongs to. Grouping by video needs no lookup. |
| `outcome` | `tp` (both boxes), `fp` (prediction only), `fn` (target only). |
| `score` | Prediction confidence. Blank on `fn`. Constant 1.0 for GLAD, which emits none. |
| `pred_x0…pred_y1`, `pred_size`, `pred_area` | The emitted box. `pred_size` is `sqrt(w*h)`. |
| `gt_x0…gt_y1`, `gt_size`, `gt_area` | The target. `gt_size` is what the recall buckets bin on. |
| `gt_class`, `pred_class` | Class ids, so a class-confusion cut is possible. |
| `iou` | **Real IoU of the pair, whatever criterion matched them.** Under centre matching this is the only column still measuring how well the box was placed. |
| `center_dx`, `center_dy`, `center_dist` | Offset between the two centres, px. |
| `center_dist_rel` | The same in multiples of `gt_size` — *the quantity `--match center` thresholds on*. A tolerance of *t* accepts exactly the rows below *t*, so the effect of a different `--match-tol` can be read off the file without re-scoring. |
| one column per axis | `scene_category`, and `lighting` / `relative_range` when `--conditions` declares them. |
| one column per recorded field | Whatever the run wrote per frame. GLAD writes `branch` — the state-machine path that produced the box. |

CSV, not JSONL, unlike the rest of the pipeline: this file exists to be loaded by
something else, and its rows are genuinely flat. It is **overwritten**, not appended —
two criteria's rows in one file would be silently double-counted by anything grouping it,
so give each criterion its own path.

Because every object appears exactly once, counting rows reproduces the headline numbers
rather than approximating them:

```bash
# precision, from the file alone
py -3.13 -c "import csv,collections; c=collections.Counter(r['outcome'] for r in csv.DictReader(open('runs/exp004_glad/matches_center.csv'))); print(c['tp']/(c['tp']+c['fp']))"
```

```python
# every other cut, in pandas
import pandas as pd
d = pd.read_csv("runs/exp004_glad/matches_center.csv")

d.groupby("video").outcome.value_counts().unstack()        # false alarms per sequence
d.groupby("branch").outcome.value_counts().unstack()       # which GLAD branch is wrong
d[d.outcome == "tp"].groupby("scene_category").iou.mean()  # localisation by background
d[d.outcome == "tp"].center_dist_rel.quantile([.5, .9])    # how tight the centres are
```

`tests/unit/test_records.py` pins the claim that matters: the `tp`/`fp`/`fn` row counts
equal what `evaluate` reports, under either criterion.

## Comparability

Two runs are comparable only if they share the validation split, `--imgsz`, `--conf`,
`--iou`, and tiling setting. Small targets make mAP unusually sensitive to input
resolution — a resolution change alone can move AP@0.50 by tens of points and look like
an architectural win.

Also confirm the split was made **by sequence, not by frame**. Frame-level splits put
near-identical adjacent frames in both train and val and inflate every metric here.
