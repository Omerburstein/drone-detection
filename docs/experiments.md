# Experiment ledger

Every run gets an entry — **including failed and abandoned ones.** Negative results are
what stop the project re-treading dead ends.

An entry missing weights, split rule, or the exact command is incomplete. Mark unknown
fields `UNKNOWN` explicitly rather than omitting them.

Maintained by `algo-agent`. Metrics come from `src.evaluate` (see
[evaluate.md](evaluate.md)) — cite the `--json-out` file, not a terminal screenshot.

> **Reproducing a pre-2026-08-18 command:** `src.baseline_detect` tiled only when
> given `--tile` until 2026-08-18; it now tiles by default. Whole-frame entries below
> (EXP-001, EXP-002) record commands with no tiling flag at all — **add `--no-tile`**
> to re-run them as they were run. Every entry from EXP-004 on states its flag
> explicitly.
>
> The same date, `src.evaluate` switched its matching default from IoU to centre
> distance. Every `src.evaluate` command recorded below was run under the old default,
> so **add `--match iou`** to reproduce the numbers in that entry; without it the run is
> scored by the centre rule and the P/R will not match. Pass `--save` when re-scoring —
> the log keeps each criterion's answer instead of overwriting the last one.

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

#### `small_mav` is two handicaps, not one

The category name says scale, and the earlier reading above attributed the wipeout to
scale alone. Measuring the imagery directly (M2b, `src.data.scene_stats`) shows that is
only half of it — those videos are also **the worst lit in the split**. Share of each
video's targets below 5 grey levels of separation from their own background:

| Video | Category | Targets under 5 grey levels |
| --- | --- | --- |
| phantom19 | small_mav | **18.8%** |
| phantom46 | small_mav | **13.2%** |
| phantom30 | ordinary | 10.6% |
| phantom47 | ordinary | 10.3% |
| … | | |
| phantom10 | ordinary | 1.7% |
| phantom09 | ordinary | 0.7% |

A **27× spread** between the best and worst video, and it does not follow the published
category boundaries — `phantom30` is nominally *ordinary* and ranks third worst.

The two effects are **independent and compounding**: contrast and apparent size correlate
at only r = 0.071 across all 28,160 targets, so they are separate axes that happen to
land together on `small_mav`. Scored on EXP-004 (GLAD, full-rate, all 28,160 targets),
holding apparent size fixed:

| Recall | <5 contrast | 5–10 | 10–20 | 20–40 | >40 | drop |
| --- | --- | --- | --- | --- | --- | --- |
| w < 12 px | 47.1% | 42.6% | 53.0% | 53.9% | 66.1% | **−19 pt** |
| w 12–20 px | 70.5% | 77.4% | 83.3% | 88.4% | 85.9% | −15 pt |
| w > 20 px | 91.3% | 91.6% | 97.3% | 97.8% | 97.2% | −6 pt |

**Poor lighting costs three times more at long range than at short range.** GLAD absorbs
it almost entirely on large targets and loses a third of its remaining recall on small
ones; worst cell 47.1% against best cell 97.8%. The baselines carry no readable signal
here — EXP-001 to EXP-003 sit at 0–40% recall in every cell with no monotone trend, and
fail for reasons that swamp lighting.

Consequence for future work: **`small_mav` results are not attributable to scale**. Use
the `lighting` and `relative_range` axes (`--conditions`) to separate them, and do not
read a `small_mav` number as a pure long-range measurement.

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
- **Watchable version (2026-08-20):** `runs/exp004_glad/examples/phantom19_overlay.mp4`,
  all 2,158 scored frames of phantom19 with the ground truth and GLAD's box drawn
  together and coloured by outcome — `src.render_video`, see
  [render_video.md](render_video.md). phantom19 is `small_mav` and the worst-lit video in
  the split, so this is the 0.642/0.522 row as footage. Not committed (`runs/` is
  gitignored, and it is 251 MB); re-render from the persisted JSONL in ~4 minutes.

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

---

### EXP-004 re-scored on centre distance

Added after EXP-004 landed, on the observation that IoU is the wrong ruler for a
false-alarm rate at 10–30 px. `--match center --match-tol 1.0`: a prediction claims a
target when their centres are within one target size (`sqrt(w*h)`), whatever the box
dimensions. Free — the JSONL is persisted, so no inference was re-run. See
[evaluate.md](evaluate.md#matching-criteria).

| | IoU@0.50 | centre@1× |
| --- | --- | --- |
| precision | 0.8559 | **0.9926** |
| recall | 0.7713 | **0.8946** |
| F1 | 0.8114 | **0.9410** |
| TP / FP | 21,721 / 3,658 | 25,191 / **188** |
| mean IoU | 0.7255 | 0.6829 |
| recall, tiny | 0.6624 | **0.8493** |
| ordinary P/R | .987 / .965 | .998 / .976 |
| complex P/R | .907 / .828 | .998 / .912 |
| small_mav P/R | .642 / .522 | **.980 / .797** |

**GLAD produces 188 false alarms in 28,337 frames, not 3,658.** 95% of what IoU@0.50
charged as false positives were GLAD's own boxes sitting slightly off a real drone —
counted twice over, once as a false alarm and once as a miss.

**The criterion does not simply inflate everything.** Re-scored the same way, the
off-the-shelf baselines barely move:

| Run | FP @ IoU 0.50 | FP @ centre | Change |
| --- | --- | --- | --- |
| EXP-004 GLAD | 3,658 | 188 | **−95%** |
| EXP-001 whole @640 | 610 | 609 | −0.2% |
| EXP-002 whole @1280 | 3,293 | 3,287 | −0.2% |
| EXP-003 tiled | 11,353 | 11,325 | −0.2% |

That asymmetry is the criterion working as intended: it distinguishes "the box is a few
pixels off the drone" from "the box is on an air-conditioning unit", and forgives only the
first. EXP-001–003's conclusions are unchanged under either rule.

**mean IoU is the check that keeps this honest.** It falls slightly (0.726 → 0.683)
because centre matching admits pairs IoU would have rejected, and it is still computed as
real IoU under both criteria. Localisation quality is measured, not assumed — the boxes are
simply no longer required to be well-sized in order to count as detections.

**Not comparable to the published table.** GLAD's figures are IoU-based (whatever their
threshold), so the centre-matched column belongs only in comparisons against our own runs.
`mAP@0.50:0.95` is reported as `n/a`: it is an IoU sweep by definition.

---

### EXP-004 broken down by feature

Added 2026-08-18. No inference re-run and no re-scoring beyond what `src.evaluate`
already does — both scorings now also write a **per-object dump**
(`--dump`, see [evaluate.md](evaluate.md#the-per-object-dump)), one CSV row per true
positive, false positive and missed target. Every table below is a `GROUP BY` over
`runs/exp004_glad/matches_center.csv` and `matches_iou50.csv`; the headline numbers they
sum to are byte-identical to the ones already in this entry, which is the check that the
dump is a re-cut and not a re-measurement.

**Distance.** ARD-MAV ships no telemetry range, but every target is the same airframe, so
apparent size *is* range up to one scale factor (`d ∝ 1/size`; see
[scene_stats.md](scene_stats.md)). Buckets are multiples of the split's own closest
approach, p95 apparent size = **34.2 px**. Frame-level, so a false alarm is charged to
the conditions it happened under.

| Range | frames | targets | P centre | R centre | P IoU@0.50 | R IoU@0.50 |
| --- | --- | --- | --- | --- | --- | --- |
| near (<2×) | 9,190 | 9,190 | 0.9997 | 0.9812 | 0.9943 | 0.9760 |
| mid (2–3×) | 5,867 | 5,867 | 0.9993 | 0.9204 | 0.9548 | 0.8795 |
| far (3–5×) | 10,345 | 10,345 | 0.9958 | 0.8395 | 0.7713 | 0.6503 |
| **very far (>5×)** | 2,758 | 2,758 | **0.9431** | **0.7574** | **0.3905** | **0.3136** |

**Background** (the published `scene_category` axis, video-level):

| Category | frames | targets | P centre | R centre | P IoU@0.50 | R IoU@0.50 |
| --- | --- | --- | --- | --- | --- | --- |
| ordinary | 9,244 | 9,230 | 0.9981 | 0.9758 | 0.9866 | 0.9646 |
| complex | 9,582 | 9,578 | 0.9981 | 0.9116 | 0.9070 | 0.8284 |
| small_mav | 9,352 | 9,352 | 0.9798 | 0.7969 | 0.6420 | 0.5222 |

**Target size**, the same axis the recall buckets use:

| Size | targets | P centre | R centre | P IoU@0.50 | R IoU@0.50 |
| --- | --- | --- | --- | --- | --- |
| tiny (<16 px) | 18,265 | 0.9899 | 0.8493 | 0.7755 | 0.6624 |
| small (16–32) | 7,927 | 0.9971 | 0.9835 | 0.9823 | 0.9783 |
| medium (32–96) | 1,968 | 0.9968 | 0.9563 | 0.9915 | 0.9487 |

**Branch** — the state-machine path that produced each box, recorded per frame by
`src.glad_detect` and carried into the dump:

| Branch | frames | P centre | P IoU@0.50 | mean IoU of its hits |
| --- | --- | --- | --- | --- |
| local yolo | 25,041 | 0.9935 | 0.8654 | 0.687 |
| **local mod** | 296 | **0.9358** | **0.0709** | **0.321** |
| global mod | 26 | 1.0000 | 0.8846 | 0.682 |
| global yolo | 16 | 0.6250 | 0.4375 | 0.624 |
| global miss / local miss / first frame | 2,799 | — (no box emitted) | — | — |

#### What the breakdown adds

1. **Distance is the dominant feature, and it is not the same statement as "small
   targets".** Precision holds at ≥0.994 out to 5× closest approach and only breaks in the
   final bucket (0.943); recall decays monotonically from 0.981 to 0.757 across the four.
   Under IoU@0.50 the same axis collapses to 0.39/0.31 in the far bucket — the ruler, not
   the detector, as established earlier in this entry.
2. **The motion branch is correctly located and badly sized.** `local mod` — LMD's
   contour-derived boxes — scores precision **0.94 under centre matching and 0.07 under
   IoU@0.50**, with mean IoU 0.321 against `local yolo`'s 0.687. A motion blob marks
   *where* the drone is, not *how big* it is. This is the single clearest instance in the
   project of a branch that any IoU-based score would call broken and that is in fact
   doing its job. It is 296 frames, so it changes no headline — but it is the branch to
   fix a box regressor onto, not to discard.
3. **`global yolo` is the weakest branch by a wide margin** (P 0.625 centre, 16 frames).
   Consistent with GAD's published 0.17 recall and with the acquisition ceiling noted
   above; too few frames to conclude more.
4. **Per-video spread is larger than any per-category number.** Under IoU@0.50 the videos
   run from phantom30 at P/R 0.995/0.989 to phantom43 at 0.420/0.384 — a spread the
   three-category aggregate hides completely. phantom43, phantom46 and phantom63 carry
   most of the loss; all three are `small_mav`.
5. **Recall dips again at the top of the size range** — 0.83 centre in the ≥48 px bucket,
   against 0.99 at 32–48. Only 450 targets, and the likely cause is a target close enough
   to leave the 320×320 search region between frames rather than an appearance failure.
   Flagged, not concluded.

The figure `runs/exp004_glad/precision_by_size.png` plots the first row of this story:
precision against the size of the box being claimed, under both criteria, with the
per-bin sample counts under it. Regenerate with
[`src.plot_eval`](plot_eval.md); the binned numbers are in `precision_by_size.csv`.

---

### EXP-004 with false-alarm rate and localisation error (M5)

Added 2026-08-20. **No inference re-run** — `detections.jsonl` was persisted on
2026-08-16, so this is `src.evaluate` over the same file under both criteria, now
reporting three quantities the metric block did not previously carry: `far` (false alarms
per frame), `loc_err` (centre offset in multiples of the target's own size) and
`loc_by_size`. Every headline number reproduces the entries above **exactly** under both
criteria — P/R/AP/mean IoU and all TP/FP/FN counts — so nothing earlier in this entry is
disturbed. Snapshots: `metrics_center.json`, `metrics.json`; log `results.jsonl`.

**False alarms per frame.** Precision already said what fraction of alarms were wrong;
this says how often the alarm fires at all, which is the number an operator budgets
against.

| | centre@1× | IoU@0.50 | ratio |
| --- | --- | --- | --- |
| **overall** | **0.0066** | **0.1291** | 19.5× |
| `ordinary` | 0.0018 | 0.0130 | 7.2× |
| `complex` | 0.0018 | 0.0843 | 47.9× |
| `small_mav` | **0.0165** | **0.2912** | 17.7× |

**Size-normalised localisation error**, centre@1× matching, over matched pairs only. The
fine bins are the `loc_error` series in `precision_by_size.csv`; the four coarse buckets
are what the metric block prints.

| gt size | pairs | mean | median | p90 |
| --- | --- | --- | --- | --- |
| <8 px | 4,462 | **0.255** | 0.250 | 0.368 |
| 8–12 | 7,313 | 0.176 | 0.171 | 0.260 |
| 12–16 | 3,738 | 0.121 | 0.116 | 0.180 |
| 16–20 | 2,333 | 0.093 | 0.088 | 0.142 |
| 20–24 | 1,863 | 0.071 | 0.067 | 0.104 |
| 24–32 | 3,600 | 0.063 | 0.059 | 0.096 |
| 32–48 | 1,510 | **0.051** | 0.047 | 0.079 |
| ≥48 | 372 | 0.071 | 0.060 | 0.112 |

#### What this adds

1. **The IoU/centre gap now has a mechanism, measured.** Relative localisation error
   degrades **5× monotonically** as targets shrink — 0.051 target-widths at 32–48 px to
   0.255 below 8 px. At 0.255 a pair cannot clear IoU 0.50 however correct the detection
   is, which is why the same run posts recall 0.89 under one ruler and 0.77 under the
   other. The earlier sections of this entry asserted this from the P/R gap; this is the
   quantity itself, and it is the metric to watch when a future model claims a small-target
   win.
2. **FAR and precision genuinely diverge.** `small_mav` alarms **9× more often** than
   `ordinary` (0.0165 vs 0.0018) while scoring 0.98 precision against 0.998 — a two-point
   precision difference standing in for an order-of-magnitude difference in how often the
   thing fires. Either number alone misleads about the other. This is why `far` is now on
   `ConditionScore` and not only on the aggregate.
3. **False alarms are a sequence-level failure, not a rate.** Under centre matching
   **phantom63 alone contributes 129 of the 188** false alarms — 69% from one of 15
   videos, with phantom43 next at 19. Quoting 0.0066/frame as a property of the detector
   is therefore wrong: on twelve of these videos it is near zero, and on one it is not.
   Consistent with item 4 of the breakdown above, where phantom43/46/63 carried most of
   the per-video loss.
4. **Do not compare offsets across criteria.** The IoU@0.50 column of the same table reads
   *lower* (0.199 in the <8 px bin against centre's 0.255) purely because IoU matching
   admits only pairs that were already well placed — the offsets are censored at the
   tolerance, not better. Compare `loc_err` between runs only under the same criterion.
5. **Above 48 px the error rises again** (0.071 from 0.051), on 372 pairs. Same shape as
   the recall dip noted in item 5 of the breakdown above, and the same likely cause — a
   target close enough to leave the 320×320 search region. Two independent metrics now
   point at it; still 372 samples, so still flagged rather than concluded.

**Cost: zero GPU, ~25 min wall-clock, and none of it was the detector.** Re-scoring reads
28,337 per-frame label files; the scoring itself is seconds. The `matches_*.csv` dumps
already carried `gt_size` and `center_dist_rel`, so every number in the two tables above
was derivable from them without touching a label at all — the re-run existed only to write
the new fields into the persisted JSON this ledger cites. Noted because it is the trigger
condition in [todo.md](todo.md) for storing labels per video instead of per frame.

---

### EXP-001–003 re-scored with dumps — resize vs tiled, by feature

Added 2026-08-18. No inference re-run: all three `detections.jsonl` were persisted, so
this is `src.evaluate --dump` over each, under both criteria. Headline numbers reproduce
the entries above exactly. Figure: `runs/compare_resize_vs_tile/precision_by_size.png`.

**These three are the resize-vs-tile axis, and they are the *baseline* model, not GLAD.**
GLAD has no such switch — see the note below.

Distance (same buckets as EXP-004; p95 closest approach 34.2 px), centre@1× matching:

| Range | targets | resize @640 P/R | resize @1280 P/R | tiled, 8 crops P/R |
| --- | --- | --- | --- | --- |
| near (<2×) | 924 | .128 / .038 | .125 / .173 | **.075 / .315** |
| mid (2–3×) | 583 | .000 / .000 | .001 / .002 | .027 / .129 |
| far (3–5×) | 1,028 | .000 / .000 | .000 / .000 | .003 / .012 |
| very far (>5×) | 281 | .000 / .000 | .000 / .000 | .001 / .004 |

Background:

| Category | targets | resize @640 P/R | resize @1280 P/R | tiled P/R |
| --- | --- | --- | --- | --- |
| ordinary | 924 | .087 / .022 | .124 / .132 | **.077 / .283** |
| complex | 957 | .076 / .016 | .026 / .043 | .025 / .120 |
| small_mav | 935 | .000 / .000 | .000 / .000 | .001 / .003 |

Target size:

| Size | targets | resize @640 R | resize @1280 R | tiled R |
| --- | --- | --- | --- | --- |
| tiny (<16 px) | 1,835 | 0.0000 | 0.0000 | **0.0387** |
| small (16–32) | 782 | 0.0090 | 0.1240 | **0.3095** |
| medium (32–96) | 199 | 0.1407 | 0.3216 | 0.3317 |

#### What this adds over the original entries

1. **Tiling buys recall everywhere and costs precision everywhere.** Under centre
   matching — which forgives loose boxes and therefore cannot be blamed for the drop —
   tiled recall beats resize@640 in every distance bucket and every category, while
   precision falls in all but the nearest. 8 crops per frame is 8 independent chances to
   false-alarm, and against a model this far out of domain that is what dominates.
2. **Neither mode works past ~3× closest approach.** Every configuration is at or below
   0.02 recall in the two far buckets. The resize/tile choice moves the near-range
   numbers; it does not extend the range at which this detector functions at all.
3. **Resize@640 has no tiny-target detections to bin.** Its `tiny` row is `nan` precision
   over zero predictions — it never emits a box that small. Its false alarms are large:
   478 in the `medium` bucket and 120 above 96 px, against 199 medium targets and no
   large ones in the split at all. That is the 1.7×-too-large box distribution from the
   original entry, now visible per bucket.
4. **The comparison is not confounded by the matching rule.** Re-scored under centre
   matching the baselines' false alarms fall by 0.2% (see the EXP-004 centre-distance
   section), so these are genuine misplacements, not localisation slack.

#### Note: GLAD has no resize/tile switch

Worth recording because it has come up twice. The "8 crops per frame" path is
`src.baseline_detect --tile`, a property of **our baseline harness**. GLAD's two regimes
— global (whole frame letterboxed to 640) and local (one 320×320 crop around the previous
box, upscaled to 640) — are *states of one state machine*, alternating automatically on
detection success, not modes a user selects. There is no configuration of `src.glad_detect`
that makes GLAD tiled, and no flag that disables its resize. Running GAD alone
(`--pad`-style measurements in [glad_detect.md](glad_detect.md)) is the closest thing to
"GLAD in resize mode", and running GAD over tiles is item 7 in
[glad-model.md §6](glad-model.md) — an unbuilt experiment, not an option.

---

### EXP-004 cross-cut — size and background at the same time

Added 2026-08-20. **No inference and no re-scoring.** This is a `GROUP BY` over the same
two persisted dumps every section above is cut from, `runs/exp004_glad/matches_center.csv`
and `matches_iou50.csv`, along **two axes at once** rather than one. New CLI
[`src.cross_eval`](cross_eval.md) over `src/eval/crosscut.py`.

- **Command:**
  ```
  py -3.13 -m src.cross_eval \
      --dump "centre@1x=runs/exp004_glad/matches_center.csv" \
      --dump "IoU@0.50=runs/exp004_glad/matches_iou50.csv" \
      --axis scene_category --band "<8" --condition complex \
      --csv runs/exp004_glad/small_complex.csv
  ```
  The full ladder is the same command without `--band`/`--condition`, written to
  `runs/exp004_glad/size_by_scene.csv`.
- **Validity check:** pooling every cell reproduces this entry's headline **exactly** —
  28,178 frames / 28,160 targets, P 0.8559, Pd 0.7713, F1 0.8114, FA 0.1298 per frame
  under IoU@0.50. A cross-cut that did not sum back to the metric block would be a second
  measurement pretending to be a re-cut.

**Why one-way cuts could not answer this.** The `Background` and `Target size` tables in
the breakdown section above are marginals, and they cannot be multiplied together because
the axes are correlated: **5,032 of the split's 5,677 sub-8 px targets are `small_mav`**,
so the published "tiny" row is mostly a statement about one scene category.

**The asked-for cell — targets under 8 px on `complex` background.** 640 frames, 640
targets (ARD-MAV is one target per frame).

| | TP | FN | FP | Pd | P | F1 | FA/frame | mean IoU | median offset |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| centre@1× | 552 | 88 | **2** | **0.8625** | 0.9964 | 0.9246 | **0.0031** | 0.5537 | 0.2236 |
| IoU@0.50 | 374 | 266 | **180** | **0.5844** | 0.6751 | 0.6265 | **0.2812** | 0.6112 | 0.1891 |

**The full ladder within `complex`**, both criteria, Pd and FA per frame:

| Band | frames | targets | Pd centre | FA/frm centre | Pd IoU@0.50 | FA/frm IoU@0.50 |
| --- | --- | --- | --- | --- | --- | --- |
| **<8 px** | 640 | 640 | **0.8625** | **0.0031** | **0.5844** | **0.2812** |
| 8–12 | 3,671 | 3,671 | 0.8891 | 0.0022 | 0.7559 | 0.1354 |
| 12–16 | 2,535 | 2,535 | 0.8982 | 0.0008 | 0.8556 | 0.0434 |
| 16–20 | 1,080 | 1,080 | 0.9380 | 0.0009 | 0.9204 | 0.0185 |
| 20–24 | 668 | 668 | 0.9775 | 0.0000 | 0.9731 | 0.0045 |
| 24–32 | 601 | 601 | 0.9834 | 0.0000 | 0.9834 | 0.0000 |
| 32–48 | 373 | 373 | 0.9946 | 0.0000 | 0.9946 | 0.0000 |
| ≥48 † | 10 | 10 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| no target † | 4 | 0 | — | 1.0000 | — | 1.0000 |
| **pooled** | 9,582 | 9,578 | 0.9116 | 0.0018 | 0.8284 | 0.0850 |

† under 30 targets — a ratio, not a measurement.

**The sub-8 px band across all three backgrounds**, centre@1×:

| Background | frames | targets | TP | FN | FP | Pd | FA/frm | mean IoU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| complex | 640 | 640 | 552 | 88 | 2 | 0.8625 | 0.0031 | 0.5537 |
| ordinary † | 5 | 5 | 5 | 0 | 0 | 1.0000 | 0.0000 | 0.6261 |
| small_mav | 5,032 | 5,032 | 3,905 | 1,127 | 145 | 0.7760 | 0.0288 | 0.5071 |
| **pooled** | 5,677 | 5,677 | 4,462 | 1,215 | 147 | 0.7860 | 0.0259 | — |

Under IoU@0.50 the same band reads complex 0.5844 Pd / 0.2812 FA, small_mav 0.4108 /
0.3941, pooled 0.4309 / 0.3810.

**The same cell in the three baseline runs**, centre@1×. Those runs sampled every 10th
frame, so the cell holds 68 targets against EXP-004's 640 — same split, same labels, same
criterion, **different frame sampling**. Pd and FA/frame are both densities, so the
comparison is legitimate in kind; the 68-target denominator is what limits it, not the
sampling itself. Read the Pd column as exact and the FA column as ±1 alarm.

| Run | targets | TP | FN | FP | Pd | FA/frame |
| --- | --- | --- | --- | --- | --- | --- |
| EXP-001 whole @640 | 68 | 0 | 68 | 1 | **0.0000** | 0.0147 |
| EXP-002 whole @1280 | 68 | 0 | 68 | 8 | **0.0000** | 0.1176 |
| EXP-003 tiled @640 | 68 | 0 | 68 | 63 | **0.0000** | 0.9265 |
| EXP-004 GLAD | 640 | 552 | 88 | 2 | **0.8625** | 0.0031 |

#### What the cross-cut adds

1. **The hardest cell is not the one the marginals point at.** Sub-8 px on `complex`
   scores Pd 0.8625 — *higher* than the pooled sub-8 px 0.7860, and only 5 points below
   the `complex` category's own aggregate 0.9116. The pooled tiny number is dragged down
   by `small_mav` (0.7760 on 5,032 targets), not by background complexity. **Background
   clutter and target size are not additive failures here.**
2. **Within `complex`, size costs 13 points of Pd and clutter costs almost nothing.**
   Pd runs 0.8625 → 0.9946 monotonically from the <8 px band to 32–48 px, while false
   alarms run 0.0031 → 0.0000. Whatever `complex` means visually, GLAD's local search
   regime is largely immune to it once the target is above ~12 px.
3. **The criterion gap is concentrated exactly here and nowhere else.** In this one cell,
   moving from centre@1× to IoU@0.50 costs **28 points of Pd (0.8625 → 0.5844) and
   multiplies the false-alarm rate 90× (0.0031 → 0.2812)** — on an unchanged set of
   predictions. The gap closes monotonically with size and is **zero at and above 24 px**.
   That is the localisation-error mechanism from the M5 section stated as one number: at
   a median offset of 0.22 target-widths on a 6 px target, IoU 0.50 is unreachable while
   the detection is plainly correct. **Any comparison of this cell against a published
   figure is meaningless without the paper's IoU threshold.**
4. **The complex/small_mav difference at <8 px is video identity, not background —
   settled, not suspected.** Re-cutting the same band with `--axis video` (free, the
   dumps already carry the column) shows that **562 of the 640 sub-8 px `complex` targets
   are one video, phantom58**, at Pd 0.9128. The remaining four `complex` videos
   contribute 78 targets between them, three of them below the 30-target reliability
   floor:

   | Video | category | targets | Pd | FA/frm | mean IoU |
   | --- | --- | --- | --- | --- | --- |
   | phantom58 | complex | 562 | 0.9128 | 0.0018 | 0.5502 |
   | phantom86 | complex | 45 | 0.7556 | 0.0000 | 0.6310 |
   | phantom65 † | complex | 13 | 0.1538 | 0.0769 | 0.2774 |
   | phantom08 † | complex | 12 | 0.2500 | 0.0000 | 0.4736 |
   | phantom05 † | complex | 8 | 0.0000 | 0.0000 | — |
   | phantom19 | small_mav | 765 | 0.8967 | 0.0013 | 0.5397 |
   | phantom41 | small_mav | 315 | 0.8032 | 0.0032 | 0.6017 |
   | phantom43 | small_mav | 1,320 | 0.8917 | 0.0144 | 0.4730 |
   | phantom46 | small_mav | 899 | 0.7264 | 0.0011 | 0.5138 |
   | phantom63 | small_mav | 1,733 | 0.6555 | 0.0710 | 0.4978 |
   | phantom47 † | ordinary | 5 | 1.0000 | 0.0000 | 0.6261 |

   † under 30 targets. Among the seven reliable cells, Pd spans **0.6555 to 0.9128 — a
   26-point range, three times the 8.7-point complex-vs-small_mav gap it is supposed to
   explain.** phantom43 is `small_mav` and scores 0.8917, above every `complex` video but
   phantom58. **`scene_category` does not predict sub-8 px performance; the video does.**
   Quote the 0.8625 figure as "phantom58, plus 78 targets of noise", never as "GLAD on
   tiny targets against complex backgrounds".
5. **Mean IoU falls with size even among the targets that were found** — 0.5537 in the
   sub-8 px complex cell against 0.8379 at 32–48 px. The boxes that do land are landing
   loosely, which is the same story as (3) from the localisation side and confirms these
   are marginal misses rather than a detector defeated by clutter.
6. **The baseline runs are a clean floor.** All three score **exactly zero** in this cell
   while their false-alarm rate spans 0.015 to 0.93 per frame — tiling buys 63 alarms and
   no detections at all below 8 px on complex background. The ~30× motion-over-appearance
   result from the original entry is, in this cell, unbounded.

#### Next

- **Already run (item 4 above):** `--axis video --band "<8"`. Its result is the reason
  the recommendations below changed. **`scene_category` has stopped being a useful
  predictor at small sizes**, so per-video variance is now the thing to model, and M4b's
  ARD100 run should be scored per video from the start rather than per published
  category. Budget for it: none — it is a flag on a CLI that reads a persisted CSV.
- **Also free:** the same cross-cut against `relative_range` instead of
  `scene_category`, once the lighting/range axes land for the ARD-MAV test split (open
  item in [todo.md](todo.md)). Range is the dominant marginal per the breakdown section;
  crossing it with size separates "far" from "small", which apparent size alone conflates
  by construction.
- **Not yet worth running:** anything that costs a GPU. The sub-8 px complex cell already
  scores 0.86 Pd at 0.003 false alarms per frame under the project's own criterion. The
  measured deficit is in `small_mav` and at range, and M4b (GLAD on unseen video) is
  still the experiment that decides whether any of these numbers survive contact with
  data GLAD was not tuned on.

---

### EXP-001–004 — where the false alarms actually land

Added 2026-08-20. **No inference and no re-scoring.** A `GROUP BY` over the same persisted
dumps, binning every `fp` row by its distance from the nearest ground-truth box in its own
frame. New CLI [`src.alarm_eval`](alarm_eval.md) over `src/eval/alarms.py`.

- **Command:**
  ```
  py -3.13 -m src.alarm_eval \
      --dump "centre@1x=runs/exp004_glad/matches_center.csv" \
      --dump "IoU@0.50=runs/exp004_glad/matches_iou50.csv" \
      --csv runs/exp004_glad/alarm_distance.csv
  ```
  Baselines: `runs/compare_resize_vs_tile/alarm_distance.csv`.
- **Why the count alone was not enough.** `far` says how often the detector cries wolf.
  It cannot say whether the wolf was two pixels from a real drone or four hundred, and
  those need opposite fixes — a box regressor versus training data.
- **Distance is to the nearest target, matched or not.** A false alarm has no *matched*
  target by definition, but it always has a nearest one. Frames holding no target at all
  give an alarm no distance; those are counted on their own row rather than swept into
  the far bin, which would manufacture clutter the run never produced.

**EXP-004 GLAD, both criteria.** Distance in multiples of the nearest target's own size.

| Distance | centre@1× | share | IoU@0.50 | share |
| --- | --- | --- | --- | --- |
| **<1** | **0** | 0.0% | **3,470** | **94.9%** |
| 1–2 | 8 | 4.3% | 8 | 0.2% |
| 2–4 | 9 | 4.8% | 9 | 0.2% |
| 4–8 | 38 | 20.2% | 38 | 1.0% |
| 8–16 | 32 | 17.0% | 32 | 0.9% |
| 16–32 | 30 | 16.0% | 30 | 0.8% |
| ≥32 | 53 | 28.2% | 53 | 1.4% |
| no target in frame | 18 | 9.6% | 18 | 0.5% |
| **total** | **188** | | **3,658** | |

**The same run in pixels** (`--unit px`, `runs/exp004_glad/alarm_distance_px.csv`).
Median alarm distance is **100.2 px** under centre matching and **2.2 px** under
IoU@0.50.

| Distance | centre@1× | share | cum | IoU@0.50 | share | cum |
| --- | --- | --- | --- | --- | --- | --- |
| **<5 px** | **0** | 0.0% | 0.0% | **3,427** | **93.7%** | 93.7% |
| 5–10 | 5 | 2.7% | 2.7% | 37 | 1.0% | 94.7% |
| 10–25 | 11 | 5.9% | 8.5% | 19 | 0.5% | 95.2% |
| 25–50 | 46 | 24.5% | 33.0% | 49 | 1.3% | 96.6% |
| 50–100 | 23 | 12.2% | 45.2% | 23 | 0.6% | 97.2% |
| **100–250** | **53** | **28.2%** | 73.4% | 53 | 1.4% | 98.6% |
| 250–500 | 28 | 14.9% | 88.3% | 28 | 0.8% | 99.4% |
| ≥500 | 4 | 2.1% | 90.4% | 4 | 0.1% | 99.5% |
| no target in frame | 18 | 9.6% | | 18 | 0.5% | |
| **total** | **188** | | | **3,658** | | |

**Read the two unit views together — they disagree in an informative way.** In target
sizes the two criteria are identical from the 1–2 bin outward. In pixels they are not:
IoU@0.50 adds 32 alarms in the 5–10 px bin, 8 in 10–25 and 3 in 25–50, on top of the
3,427 under 5 px. Those 43 extras (3,427 + 32 + 8 + 3 = **3,470**, exactly the criterion
gap) are still all inside one target size — the largest sits at **0.97** — they simply sit
on *large* targets, up to 69 px, so a sub-1 offset is tens of pixels wide.

That is the whole argument for binning relative by default. **The matching boundary is
size-relative, so only the relative ladder shows it as a clean cut**; the pixel ladder
smears it across three bins and would invite reading 43 localisation misses as clutter.
Pixels are still the right view for an operator sizing a rejection gate, which is why
both are available — but the relative one is what a criterion argument should be made on.

**The three baselines**, centre@1× (every 10th frame, so counts are a tenth-scale sample):

| Distance | EXP-001 whole@640 | EXP-002 whole@1280 | EXP-003 tiled@640 |
| --- | --- | --- | --- |
| <1 | 0 | 0 | 9 |
| 1–2 | 0 | 5 | 14 |
| 2–4 | 0 | 10 | 121 |
| 4–8 | 8 | 41 | 389 |
| 8–16 | 25 | 252 | 973 |
| 16–32 | 94 | 813 | 2,395 |
| **≥32** | **476 (78.2%)** | **2,127 (64.7%)** | **7,332 (64.7%)** |
| no target in frame | 6 | 39 | 92 |
| **total** | **609** | **3,287** | **11,325** |
| median distance, px | 868 | 683 | 622 |
| median distance, target sizes | 55.9 | 45.6 | 45.5 |

#### What this settles

1. **Every one of the 3,470 alarms IoU@0.50 adds over centre matching is inside one
   target size.** The two columns of the EXP-004 table are *identical* from the 1–2 bin
   outward — 8, 9, 38, 32, 30, 53, 18 under both criteria. The criterion does not find
   different clutter; it reclassifies boxes that are sitting **on** the drone. The
   centre-distance section above asserted "95% of what IoU@0.50 charged as false
   positives were GLAD's own boxes slightly off a real drone" from the count; this is the
   distance itself, and it is 94.9% with a median of 2.2 px.
2. **GLAD's real false-alarm population is 188 boxes, and it is not near-misses.** Under
   centre matching **zero** alarms fall inside one target size — as they must, since a
   first box that close would have matched, so a sub-1 alarm could only be a duplicate and
   GLAD emits one box per frame. Of the 170 that have a distance at all, **153 (90%) are
   4 or more target sizes out** and 53 are beyond 32; median 100 px. These are genuine
   wrong-object detections, not sloppy boxing.
3. **This is the strongest separation yet between the baselines and GLAD, and it is not
   about counts.** EXP-003 emits 60× GLAD's alarms, but the *shape* is the finding: 64.7%
   of tiled alarms are ≥32 target sizes from any drone, median 622 px — the detector is
   boxing objects that have nothing to do with the target. Doubling input resolution
   (EXP-001 → 002) moves the median from 868 px to 683 px and tiling to 622 px, so
   more resolution buys alarms that are *nearer* the drone but nowhere near enough to
   matter. **The failure was never localisation.**
4. **EXP-003's 9 sub-1 alarms are the only duplicates in the project.** Tiling gives
   overlapping crops an independent chance at the same target, and class-aware NMS at the
   merge does not always collapse them. Nine boxes out of 11,325 — real, negligible, and
   the only place the tile-merge path is visibly imperfect.
5. **The 18 no-target frames are worth their own line.** GLAD fires on 18 of the split's
   frames that hold no drone at all — 9.6% of its alarm budget, from 18 of 28,178 frames.
   Small, but it is the one population no amount of box regression or matching-rule change
   will touch.

#### Next

- **Free:** `--group video` on EXP-004 confirms the concentration already recorded above —
  phantom63 supplies 129 of the 188, and its shape is the far-clutter shape (38.8% beyond
  32 target sizes). phantom43's 19 alarms are the opposite: 84% within 8 target sizes,
  i.e. the drone's surroundings rather than unrelated objects. **Two different failures
  in one aggregate, in the two videos that already dominate the loss.**
- **What this changes about M7.** A box-regression head or a tighter matching rule would
  recover almost the whole IoU@0.50 penalty and **none** of the 188 real alarms. If the
  goal is a deployable false-alarm rate, the work is hard-negative mining on the far
  population — clutter phantom63 flies past — not localisation. That is a data question
  for `dataset-agent`, not an architecture question.
- **Not worth running:** anything on the near population. It is already zero under the
  criterion this project scores on.
