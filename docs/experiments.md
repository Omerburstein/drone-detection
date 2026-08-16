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

### Scores by scene condition

Added by M2a and re-scored from the persisted JSONL, so no inference was re-run.

| Category | gt | EXP-001 P/R | EXP-002 P/R | EXP-003 P/R | GLAD measured (EXP-004) | GLAD published |
| --- | --- | --- | --- | --- | --- | --- |
| ordinary | 924 | .082 / .021 | **.126 / .125** | .072 / **.266** | .987 / .965 | 0.99 / 0.96 |
| complex | 957 | .076 / .016 | .026 / .042 | .022 / .108 | .907 / .828 | 0.94 / 0.86 |
| small_mav | 935 | **.000 / .000** | **.000 / .000** | .0005 / .0021 | .642 / .522 | 0.82 / 0.67 |

Two things the aggregate was hiding:

- **The domain-shift signature is visible directly.** At 1280, precision on *ordinary*
  backgrounds is 5x that on *complex* (0.126 vs 0.026). A model trained on drones against
  sky does relatively better against sky and collapses against urban clutter — which is
  exactly what inspecting the false positives showed (window recesses, AC units).
- **`small_mav` is a total wipeout**, not merely weak: zero detections at both whole-frame
  resolutions, and 2 correct out of 935 when tiled. GLAD reports F1 0.73 on this same
  category. That gap is the single clearest statement of what an appearance-only detector
  cannot do at this scale.

These are not like-for-like with GLAD's column: ours are stride-10 stills scored on
2,834 frames, GLAD's are full-rate video with motion. The gap is large enough to be
meaningful anyway, but it is not a measured head-to-head — EXP-004 is.

**Do not compare these to the published 0.53.** That figure is YOLOv5 *trained on*
ARD100 — a fine-tuned baseline, not an off-the-shelf model. The plan originally cited it
as the expected value for EXP-001, which was an error.

---

## EXP-004 — GLAD's released weights on ARD-MAV, as a harness check
- **Date:** 2026-08-16
- **Question:** Do M1's evaluation math and M2's VOC→YOLO conversion reproduce a published
  number? **This is a test of our pipeline, not of GLAD.** These weights were trained on
  ARD-MAV's other 45 videos and the architecture was tuned against this very split, so the
  result is optimistic by construction and must never be cited as "GLAD scores X for us".
- **Model / weights:** GLAD (Guo et al., T-ITS 2024), released checkpoints —
  `third_party/GLAD/weights/{yolov5s_GLAD.pt, yolov5s_GLAD-crop.pt, Net_best.pth}`.
  CPU port at `src/algo/glad/`; see [glad_detect.md](glad_detect.md).
- **Data:** ARD-MAV, official GLAD 15-video test split, **every frame** of the original
  `.mp4`s — 28,337 frames / 28,160 boxes. Contiguous by necessity: both motion branches
  difference against the previous frame.
- **Hyperparameters:** all fixed at the released values — GAD conf 0.5, LAD-tracking conf
  0.1 within 50 px, LAD-acquire conf 0.5 within 10 px, NMS IoU 0.4, 320×320 search region,
  30-miss fallback. Letterbox fill **black**, reproducing the upstream padding defect.
- **Hardware:** i7-1255U CPU, 10,602 s (2.9 h), 2.67 fps
- **Command:** `py -3.13 -m src.glad_detect --pad released --out runs/exp004_glad`
  then `py -3.13 -m src.evaluate --pred runs/exp004_glad/detections.jsonl --labels
  data/processed/ARD-MAV/labels/test --conditions data/processed/ARD-MAV/conditions.json
  --frame-size 1920 1080 --json-out runs/exp004_glad/metrics.json`
- **Metrics:** `runs/exp004_glad/metrics.json` — P **0.8559** | R **0.7713** | F1 **0.8114**
  | mean IoU 0.7255 | TP 21,721 / FP 3,658 | recall by size: tiny **0.6624**, small 0.9783,
  medium 0.9487. AP@0.5 0.7046 — **ignore it**, see caveats.

  | Category | Ours P/R/F1 | Published P/R/F1 |
  | --- | --- | --- |
  | ordinary | 0.987 / 0.965 / 0.976 | 0.99 / 0.96 / 0.97 |
  | complex | 0.907 / 0.828 / 0.866 | 0.94 / 0.86 / 0.90 |
  | small_mav | 0.642 / 0.522 / 0.576 | 0.82 / 0.67 / 0.73 |
  | total | 0.856 / 0.771 / 0.811 | 0.91 / 0.81 / 0.86 † |

  † the released code is the paper's `GAD+GMD+LAD+LMD` ablation row, not full GLAD — there
  is no Kalman filter and the search region is a fixed 320×320, not `L = 300 + 4·T_lost`.
  That row, not the 0.92/0.82/0.87 headline, is the honest target.

- **Result:** **The harness is validated.** `ordinary` reproduces the published row almost
  exactly (0.987/0.965 against 0.99/0.96), and the aggregate lands within 0.054 precision
  and 0.039 recall of the released-code ablation. A conversion or scoring bug could not
  produce a near-exact `ordinary` row — coordinate, normalisation and frame-numbering
  errors are all scale-free and would damage every category alike.
- **The residual gap is a scoring threshold, not a detector.** It is concentrated entirely
  in the smallest targets, and re-scoring the same JSONL at looser IoU shows why:

  | Category | @0.50 | @0.40 | @0.30 | Published |
  | --- | --- | --- | --- | --- |
  | ordinary | .987/.965 | .995/.973 | .997/.975 | 0.99/0.96 |
  | complex | .907/.828 | .975/.890 | .993/.907 | 0.94/0.86 |
  | small_mav | .642/.522 | .869/.707 | .955/.777 | 0.82/0.67 |

  At **IoU 0.40 our numbers bracket the published ones in all three categories**, slightly
  above rather than below. The shortfall at 0.50 scales inversely with target size, which
  is the signature of a matching-criterion difference: at 12 px a 2 px centre offset drops
  IoU under 0.5 while the detection is unambiguously correct. Mean IoU of 0.726 says the
  matched boxes are well placed, so these are marginal misses, not bad ones.

  **We do not know GLAD's threshold** — the paper does not state one, and this is inference
  from the shape of the discrepancy, not a fact. Treat it as the leading explanation.
- **Branch attribution** (the paper's ablation measured on our own run): `local yolo`
  88.4%, `local miss` 7.4%, `global miss` 2.9%, `local mod` 1.0%, `global mod` 0.1%,
  `global yolo` 0.1%. Acquisition happens **42 times in 28,337 frames**; LAD inside the
  search region does essentially all the work thereafter. This is the clearest possible
  statement of why the local regime is the architecture's whole idea.
- **Supporting measurement — the letterbox fill is worth nothing.** Porting exposed an
  upstream defect: `copyMakeBorder`'s seventh positional parameter is `dst`, not `value`,
  so GLAD's requested 128 grey is discarded and the bars are black — over 44% of a 1080p
  frame at 640×640, against weights yolov5 trained with 114. GAD alone, every 60th frame
  (473 frames / 468 targets, IoU 0.50): black **0.717/0.152**, 114 **0.726/0.147**, 128
  **0.719/0.147**. Two detections separate them. The intuitive "44% out-of-distribution
  input must cost recall" argument is wrong. `--pad` now defaults to the correct 114
  because it is free, not because it helps. That same run reproduces the paper's `GAD only`
  ablation (0.76/0.17) at 0.72/0.15 — a second, independent harness check.
- **Caveats:**
  - **Not a measurement of GLAD's quality.** Trained on the same capture campaign and
    tuned against this split. It says nothing about generalisation — that is M4b.
  - **AP is meaningless here and is not comparable to EXP-001–003.** GLAD emits no
    confidence, so every box is recorded at 1.0; with constant scores AP degenerates to
    roughly P×R (0.856 × 0.771 = 0.660 against the reported 0.705). Likewise
    mAP@0.50:0.95 of 0.296. Score this run on P/R/F1 only.
  - **Single-target by construction.** `MOD2_global` breaks on its first accepted
    candidate and the state machine tracks one box, so recall is capped at one drone per
    frame. Harmless on ARD-MAV, fatal for a multi-intruder use case.
  - fp32 on CPU against the engines' build precision, and one NMS pass where upstream runs
    the plugin's and then `cv2.dnn.NMSBoxes` again. Neither is expected to move a
    detection, but neither has been isolated.
- **Next:** M4b — GLAD on video it has never seen, where the number finally means
  something about the model rather than about us.

### What EXP-004 changes

1. **The harness is trustworthy.** Every number EXP-001–003 produced can now be read as a
   statement about the detector rather than a possible artefact of our conversion. This is
   the whole reason M4a existed.
2. **Published P/R cannot be compared without knowing the IoU threshold.** On tiny targets
   the choice between 0.40 and 0.50 moves precision by 23 points and recall by 19 in the
   `small_mav` category — larger than most architectural differences this project will ever
   measure. Every future comparison against a paper must state the threshold or be marked
   uncertain.
3. **Motion + local search is worth ~30x over off-the-shelf appearance.** Against EXP-003's
   best tiled run: recall 0.771 vs 0.125, precision 0.856 vs 0.030, and tiny-target recall
   0.662 vs 0.029. EXP-001–003 concluded from failure that the discriminating signal at
   10–30 px is motion, not appearance; this measures the same conclusion from success.
4. **The real ceiling is acquisition, not tracking.** 2,958 frames end with no detection
   and 3,080 targets are never found even at IoU 0.25, while the local regime holds lock
   88% of the time. Effort belongs on the branch that finds a target from cold — which is
   also the branch the authors' own hovering-target failure mode attacks.
