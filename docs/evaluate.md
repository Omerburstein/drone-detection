# `scripts/evaluate.py` — reference

Scores a `detections.jsonl` against ground-truth labels using the COCO protocol.

```
py -3.13 scripts/evaluate.py --pred <jsonl> --labels <dir> [options]
```

## Parameters

| Parameter | Default | What it does |
| --- | --- | --- |
| `--pred` | required | `detections.jsonl` written by `baseline_detect.py`. |
| `--labels` | required | Directory of YOLO-format `.txt` label files. Matched to predictions by **filename stem** — `img042.jpg` pairs with `img042.txt`. Video runs pair frame index to `<idx>.txt`. |
| `--iou` | `0.5` | IoU threshold for the precision/recall/size breakdowns. AP@0.50 and mAP@0.50:0.95 always use the standard sweep regardless of this. |
| `--frame-size W H` | none | Frame dimensions in pixels. **Required for video-keyed predictions** — the JSONL stores frame indices, not sizes. For image runs it is read from the image file, and this flag overrides it. |
| `--json-out` | none | Also write metrics as JSON. Pass it for anything you intend to cite in the ledger. |

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

## Matching rules

- **Greedy, in descending confidence order.** The most confident prediction picks first.
- **Each ground-truth box can be claimed once.** A second box on an already-matched
  drone becomes a false positive — this is what penalises duplicate detections.
- **Class-aware.** A `bird` prediction cannot satisfy a `drone` ground truth.

## Comparability

Two runs are comparable only if they share the validation split, `--imgsz`, `--conf`,
`--iou`, and tiling setting. Small targets make mAP unusually sensitive to input
resolution — a resolution change alone can move AP@0.50 by tens of points and look like
an architectural win.

Also confirm the split was made **by sequence, not by frame**. Frame-level splits put
near-identical adjacent frames in both train and val and inflate every metric here.
