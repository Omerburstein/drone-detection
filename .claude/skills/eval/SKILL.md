---
name: eval
description: Score a detection run against ground truth and interpret the result — AP@0.50, mAP@0.50:0.95, precision/recall, localisation quality, and recall broken down by target size. Records the run in the experiment ledger. Use when the user asks to "evaluate", "score", "measure", "what's the mAP", "how did that run do", or wants two runs compared.
---

# eval

Turn a `detections.jsonl` into trustworthy numbers, then say what they mean.

Wraps `src.evaluate`. Do not hand-roll metrics — if the script lacks something,
extend it so every run is scored identically. Metrics computed two different ways are
not comparable, and that defeats the point of measuring.

## Why not MSE on box coordinates

This comes up often enough to state plainly. Detection is a **matching** problem before
it is a regression problem, and a coordinate error like MSE cannot express it:

- **Nothing says which prediction pairs with which ground-truth drone.** MSE needs a
  correspondence that does not exist until you have matched by IoU.
- **It cannot represent a miss.** A frame where the drone was never found has no
  predicted box to subtract from — the error is undefined, and the worst possible
  failure silently drops out of the average.
- **It cannot represent a false positive.** A model boxing every cloud has no
  coordinate error to speak of on the frames it happens to get right.
- **It is not scale-invariant.** A 10 px error on a 20 px drone is a total miss; the
  same error on a 300 px drone is a rounding detail. MSE scores them identically —
  fatal on this project, where target size varies by an order of magnitude.
- **Confidence is ignored,** so you cannot trade precision against recall.

The right pipeline is *match by IoU → count TP/FP/FN → integrate precision over
recall*. That is AP, and it is what `evaluate.py` computes.

Localisation error is still worth knowing — it is reported as **mean IoU over matched
boxes**, which is the scale-invariant form of the question MSE was reaching for. Read
it *alongside* recall, never instead of it.

## Running

```
py -3.13 -m src.evaluate --pred runs/<exp>/detections.jsonl \
    --labels data/processed/<dataset>/labels/val \
    [--iou 0.5] [--frame-size W H] [--json-out runs/<exp>/metrics.json]
```

`--frame-size` is required for video-keyed predictions, since the JSONL carries frame
indices rather than dimensions. Always pass `--json-out` — the ledger entry should cite
a file, not a screenshot of a terminal.

## Before trusting any number

Check these first. A confident wrong number is the expensive failure here.

1. **Is the split sequence-level?** Check the dataset's `MANIFEST.md`. If validation
   was split frame-wise, near-identical adjacent frames sit on both sides and every
   metric is inflated — often by tens of points. Stop and report this rather than
   quoting the number.
2. **Do labels and predictions actually align?** If ground-truth count looks wrong, or
   recall is implausibly near 0, suspect a key mismatch (label stems vs image stems)
   before concluding the model is bad.
3. **Was this the test split?** Test is spent once, at the end. Development iterates on
   val.
4. **Same `imgsz`, `--conf`, `--iou`, and tiling setting as the run being compared
   against?** If not, the comparison is invalid — say so.

## Interpreting

Report precision and recall **together**, always. A single mAP hides which of two
opposite problems you have:

- **Low recall, high precision** → drones are being missed. Look at resolution first
  (`--imgsz`, `--tile`), then at whether small targets dominate the size breakdown.
  On this project this is the usual failure.
- **High recall, low precision** → clutter, birds, and sky texture are firing. Look at
  the confidence threshold and at whether bird negatives are in the training data.
- **Good AP@0.50 but poor mAP@0.50:0.95** → the drone is being found but boxed
  loosely. Localisation, not detection. Confirm with mean IoU.

**The size breakdown is the most informative block for this project.** If recall is
strong on `medium`/`large` and collapses on `tiny`, the problem is input resolution,
not architecture — and a bigger model will not fix it. Recommend a resolution or tiling
experiment before anything more expensive.

## After scoring

Hand the result to `algo-agent` to write the ledger entry in `docs/experiments.md`, or
write it directly using that file's format. It must cite weights, split rule, exact
command, and every metric. **An unrecorded run will be re-run at full cost later.**

Then state the next experiment: what to run, what it costs, and what outcome would
change the plan.
