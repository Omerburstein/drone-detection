# Experiment ledger

Every run gets an entry — **including failed and abandoned ones.** Negative results are
what stop the project re-treading dead ends.

An entry missing weights, split rule, or the exact command is incomplete. Mark unknown
fields `UNKNOWN` explicitly rather than omitting them.

Maintained by `algo-agent`. Metrics come from `src.evaluate` (see
[evaluate.md](evaluate.md)) — cite the `--json-out` file, not a terminal screenshot.

---

## Template

```markdown
## EXP-000 — <one-line title>
- **Date:** YYYY-MM-DD
- **Question:** what this run is meant to answer
- **Model / weights:** architecture + exact checkpoint path
- **Data:** dataset(s), split rule, train/val/test proportions
- **Hyperparameters:** imgsz, batch, lr, epochs, conf, iou, tiling
- **Hardware:** where it ran, wall-clock, cost
- **Command:** the exact invocation
- **Metrics:** AP@0.5 | mAP@0.5:0.95 | precision | recall | mean IoU | recall-by-size
- **Result:** what it means in one or two sentences
- **Caveats:** what this does *not* establish
- **Next:** what it implies, and the EXP id that follows
```

---

## Runs

### On comparing EXP-001 / 002 / 003

These three share weights, threshold, split and frame set. They differ **only** in the
size the target has shrunk to when the network sees it — a 20 px drone in a 1920×1080
frame arrives as 6.7 px, 13.3 px and 20 px respectively.

Normally a differing `imgsz` makes runs incomparable, and that rule stands. It exists to
stop an accidental resolution confound being sold as an architectural win. Here
resolution **is** the independent variable and the model is held constant, which is the
opposite situation. Do not cite this trio as precedent for comparing runs that differ in
`imgsz` for any other reason.

The **medium bucket is the control.** It is already large enough to survive a 1280
letterbox, so tiling should not help it. It does not (0.3216 → 0.3266), which is what
licenses attributing the tiny/small gains to resolution rather than to tiling adding
some general benefit.

---

## EXP-001 — off-the-shelf YOLOv8s, whole-frame @640
- **Date:** 2026-08-16
- **Question:** What does a single-class drone detector give us on air-to-air footage with no fine-tuning, at the default input size?
- **Model / weights:** YOLOv8s, `weights/yolov8s_eo_drone.pt` (HF `IRIS-Computer-Vision/YOLOv8s_EO_Drone_Detection`, ~1,000 annotations derived from Anti-UAV, single class `drone`)
- **Data:** ARD-MAV, official GLAD 15-video test split, every 10th frame = 2,834 frames / 2,816 boxes
- **Hyperparameters:** imgsz 640, conf 0.15, iou 0.5, no tiling
- **Hardware:** i7-1255U CPU, 875 s, 3.24 fps
- **Command:** `py -3.13 -m src.baseline_detect --weights weights/yolov8s_eo_drone.pt --source data/processed/ARD-MAV/images/test --stride 10 --max-frames 3000 --conf 0.15 --imgsz 640 --no-save-frames --out runs/exp001_whole_640`
- **Metrics:** `runs/exp001_whole_640/metrics.json` — AP@0.5 **0.0058** | mAP@0.5:0.95 0.0034 | P 0.0528 | R 0.0121 | mean IoU 0.7635 | TP 34 / FP 610 | recall by size: tiny **0.0000**, small 0.0090, medium 0.1357
- **Result:** Effectively total failure. 34 correct detections against 2,816 targets.
- **Caveats:** Off-the-shelf weights on out-of-domain data; says nothing about a fine-tuned model.
- **Next:** EXP-002 — same everything, double the input size.

## EXP-002 — same weights, whole-frame @1280
- **Date:** 2026-08-16
- **Question:** Is the failure primarily resolution loss in the letterbox?
- **Model / weights:** identical to EXP-001
- **Data:** identical to EXP-001
- **Hyperparameters:** imgsz **1280**, conf 0.15, iou 0.5, no tiling
- **Hardware:** i7-1255U CPU, 2,138 s, 1.33 fps
- **Command:** as EXP-001 with `--imgsz 1280 --out runs/exp002_whole_1280`
- **Metrics:** `runs/exp002_whole_1280/metrics.json` — AP@0.5 **0.0219** | mAP@0.5:0.95 0.0150 | P 0.0450 | R 0.0550 | mean IoU 0.7926 | TP 155 / FP 3,293 | recall by size: tiny **0.0000**, small 0.1164, medium 0.3216
- **Result:** Doubling input size raised recall 4.5× and AP 3.8×, at 2.4× the compute. Tiny targets stayed at exactly zero.
- **Caveats:** Precision fell (0.053 → 0.045) — the extra recall came bundled with 5× the false positives.
- **Next:** EXP-003 — remove the resize entirely.

## EXP-003 — same weights, tiled at native resolution
- **Date:** 2026-08-16
- **Question:** With no downscaling at all, does the detector find the tiny targets?
- **Model / weights:** identical to EXP-001
- **Data:** identical to EXP-001
- **Hyperparameters:** imgsz 640, **tiled** 640 px crops, overlap 0.2 (8 tiles/frame), conf 0.15, iou 0.5
- **Hardware:** i7-1255U CPU, ~4,400 s, 0.64 fps
- **Command:** as EXP-001 with `--tile --tile-size 640 --tile-overlap 0.2 --out runs/exp003_tiled_640`
- **Metrics:** `runs/exp003_tiled_640/metrics.json` — AP@0.5 **0.0251** | mAP@0.5:0.95 0.0145 | P 0.0300 | R 0.1246 | mean IoU 0.7570 | TP 351 / FP **11,353** | recall by size: tiny 0.0289, small 0.2980, medium 0.3266
- **Result:** Recall 10× EXP-001, and tiny finally moves off zero — to 2.9%. But AP barely improves over EXP-002 (0.0251 vs 0.0219) because precision collapses to 3%: 32 false positives for every true positive. mAP@0.5:0.95 actually *falls* slightly, tiled boxes being looser (IoU 0.757 vs 0.793).
- **Caveats:** Every tile is an independent opportunity to false-positive, so FP count scales with tile count. 0.64 fps is far outside any edge budget.
- **Next:** M4a — GLAD's released weights, as a check on our pipeline rather than on GLAD.

### What EXP-001–003 establish

1. **Resolution is the binding constraint on recall.** 0.012 → 0.055 → 0.125 as the target
   survives at 6.7 → 13.3 → 20 px. Monotonic, large, and the medium-bucket control
   confirms the mechanism.
2. **Resolution is not sufficient.** Even with *no* downscaling, 97% of tiny targets are
   still missed and precision is 3%. The ceiling is the model, not the pipeline.
3. **Off-the-shelf appearance-only detection does not transfer to air-to-air.** Peak
   AP@0.5 of 0.025 against a fine-tuned YOLOv5's published 0.53 on ARD100. Inspection of
   false positives shows the model firing on window recesses, AC units and scooters —
   dark compact rectangles, which is what a drone looks like in its ground-based,
   sky-background training set. Its predicted boxes are ~1.7× too large (median 36 px vs
   ground truth 21 px; 15.8% exceed the largest ground-truth box in the split).
4. **Localisation is not the problem.** Mean IoU 0.76–0.79 whenever a target is found.
5. **This is empirical support for motion-based methods.** At 10–30 px there is not
   enough appearance information to separate a drone from a window recess; the
   discriminating signal is that one moves relative to the background and the other does
   not. Independently re-derives the conclusion Rozantsev et al. reached in 2015 and that
   GLAD and YOLOMG are built on.

**Do not compare these to the published 0.53.** That figure is YOLOv5 *trained on*
ARD100 — a fine-tuned baseline, not an off-the-shelf model. The plan originally cited it
as the expected value for EXP-001, which was an error.
