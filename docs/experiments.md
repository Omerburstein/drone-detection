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
>
> **Run artifacts for EXP-001–009 were deleted on 2026-09-22.** `runs/` now keeps only
> the FIELD and SOFA runs (EXP-010 on). Every `runs/exp00*` and `runs/compare_*` path
> cited in EXP-001–009 — detections, match dumps, metric JSONs, figures, example crops —
> no longer exists on disk. The numbers in those entries are the record; to regenerate a
> file, re-run the entry's recorded command (with the flag notes above).
>
> **`runs/` is grouped by footage** (2026-09-22): `runs/field/` (EXP-010–012),
> `runs/sofa_o4/` (EXP-011 O4, 012a, 012b), `runs/sofa_analog/` (EXP-013, 014). Each
> experiment is a subfolder, with its log beside it. Paths below use this layout, and
> that includes the `--out` in recorded commands.

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

---

## EXP-005 — GLAD on ARD100, video it has never seen (M4b)
- **Date:** 2026-08-23
- **Question:** How much of EXP-004's performance does GLAD keep on video it was not
  trained on? EXP-004 is optimistic by construction; this is the run where the number
  means something. **The one variable is the video content** — same weights, same code
  path, same `--pad released`, same 1920×1080, same full-rate contiguous decode.
- **Model / weights:** identical to EXP-004 — `third_party/GLAD/weights/{yolov5s_GLAD.pt,
  yolov5s_GLAD-crop.pt, Net_best.pth}`, CPU port at `src/algo/glad/`. Nothing retrained,
  nothing re-tuned.
- **Data:** ARD100 test split ∩ "not in our local ARD-MAV 60" = **15 videos**, matching
  EXP-004's count: `phantom03, 92, 93, 94, 95, 97, 102, 110, 113, 119, 133, 135, 136, 141,
  144`. **34,287 frames / 33,517 boxes**, every frame of the original `.mp4`s. 777 frames
  are genuine negatives.
- **Hyperparameters:** all fixed at the released values, byte-identical to EXP-004.
  Letterbox fill **black** (`--pad released`).
- **Hardware:** i7-1255U CPU, 9,531 s (2.65 h), **3.60 fps**.
- **Command:**
  `py -3.13 -m src.glad_detect --dataset ARD100 --pad released --out runs/exp005_glad_ard100`
  then `py -3.13 -m src.evaluate --pred runs/exp005_glad_ard100/detections.jsonl --labels
  data/processed/ARD100/labels/test --conditions data/processed/ARD100/conditions.json
  --frame-size 1920 1080 --match center --match-tol 1.0 --dump
  runs/exp005_glad_ard100/matches_center.csv --json-out
  runs/exp005_glad_ard100/metrics_center.json`
- **Metrics:** `runs/exp005_glad_ard100/metrics_center.json` (centre@1×) and
  `metrics_iou50.json` (IoU@0.50).

  | | EXP-004 (ARD-MAV) | EXP-005 (ARD100) | Δ |
  | --- | --- | --- | --- |
  | **centre@1× P** | 0.9926 | **0.9460** | −0.047 |
  | **centre@1× R** | 0.8946 | **0.6880** | **−0.207** |
  | **centre@1× F1** | 0.9410 | **0.7966** | −0.144 |
  | false alarms (n) | 188 | **1,316** | 7.0× |
  | false alarms / frame | 0.0066 | **0.0384** | 5.8× |
  | mean IoU (matched) | 0.6829 | 0.6805 | −0.002 |
  | IoU@0.50 P / R / F1 | 0.856 / 0.771 / 0.811 | **0.834 / 0.606 / 0.702** | −0.165 R |

  Recall by target size, centre@1× — **the drop is in every bucket, including the easiest**:

  | bucket | EXP-004 n / R | EXP-005 n / R |
  | --- | --- | --- |
  | tiny (<16 px) | 18,265 / 0.8493 | 20,617 / **0.5997** |
  | small (16–32) | 7,927 / 0.9835 | 12,157 / **0.8401** |
  | medium (32–96) | 1,968 / 0.9563 | 730 / **0.6575** |
  | large (>96 px) | 0 / n/a | 13 / 0.1538 |

- **Result: GLAD retains roughly three-quarters of its recall on unseen video from the
  same campaign — 0.895 → 0.688 — and its false-alarm rate rises almost sixfold.**
  Precision holds up well (0.993 → 0.946); the loss is overwhelmingly missed detections,
  not spurious ones.

- **The backlit control, and it does not explain the gap.** Exposure was the confound that
  had to be cleared before reading anything as generalisation. **Corrected and made
  symmetric 2026-08-23** by deriving ARD-MAV's own `lighting` axis (`src.data.scene_stats`,
  28,337 frames, ~1 h) and re-scoring EXP-004 through it —
  `runs/exp004_glad/metrics_center_lit.json`:

  | subset | EXP-004 R | EXP-005 R | Δ |
  | --- | --- | --- | --- |
  | all frames | 0.8946 | 0.6880 | −0.207 |
  | backlit only | 0.9165 | 0.6277 | −0.289 |
  | **non-backlit only** | **0.8896** | **0.7126** | **−0.177** |

  **A 17.7-point gap survives a like-for-like non-backlit comparison**, so backlighting
  accounts for 3.0 of the 20.7 points and the rest is the video content itself. This is the
  entry's load-bearing number: without it the headline could have been dismissed as an
  exposure artefact, and it cannot be.

  **Two corrections to what this entry first claimed**, both from that measurement:

  1. **ARD-MAV test is 18.6% backlit, not "~0%".** The 0.0–0.2% figure quoted from
     [prepare_ardmav.md](prepare_ardmav.md) is a **per-video mean** `pct_blown`; the
     `backlit` bucket is a **per-frame** test at 2% blown, and the two are different
     quantities. Measured per frame the split is ARD-MAV 18.6% against ARD100 29.7% — a
     real exposure gap, but nothing like the total absence first assumed. The original
     asymmetric control (ARD100 non-backlit against EXP-004 aggregate) happened to land
     within half a point of the symmetric one, so the conclusion never moved; the reasoning
     behind it was wrong all the same.
  2. **`backlit` does not predict difficulty — it is the easiest bucket on ARD-MAV and the
     second-hardest on ARD100.** EXP-004 scores **0.9165** backlit, *above* its own 0.8946
     aggregate and second only to `strong`; EXP-005 scores 0.6277, below everything but
     `invisible`. So the label is not measuring an intrinsic difficulty of blown highlights.
     Whatever makes ARD100's backlit frames hard is specific to those frames, and the
     backlit gap (−28.9) being *wider* than the aggregate gap (−20.7) is the one place
     exposure and content plausibly interact.

- **Nor is it size composition.** The obvious second explanation — ARD100's targets are
  smaller — fails on its own evidence: recall falls in the **medium (32–96 px)** bucket
  from 0.956 to 0.658. That is the easy control bucket, targets large enough that neither
  tininess nor exposure is plausible, and it lost 30 points. A composition shift moves the
  aggregate by reweighting buckets; it cannot move the buckets themselves.

- **The mechanism is visible in the branch summary.** Same pipeline, same thresholds:

  | branch | EXP-004 | EXP-005 |
  | --- | --- | --- |
  | `local yolo` | 88.4% | **66.6%** |
  | `local miss` | 7.4% | **16.6%** |
  | `global miss` | 2.9% | **12.3%** |
  | `local mod` | 1.0% | 4.3% |

  GLAD is a lock-and-track state machine, and on this data it **loses lock four times as
  often** (2.9% → 12.3% unlocked-and-finding-nothing). Detections fall to 0.71 per frame
  from EXP-004's 0.90, and 28.9% of frames come back empty. The 3.60 fps versus EXP-004's
  2.67 fps is consistent: the expensive motion path runs on demand, and a pipeline that has
  given up looking is a pipeline doing less work.

- **Localisation is untouched.** Mean IoU on matched boxes is 0.6805 against 0.6829, and
  the centre offset is 0.082 target sizes. **When GLAD finds a drone here it places the box
  exactly as well as it does on its training campaign.** The failure is acquisition, not
  regression — the same conclusion EXP-004's alarm analysis reached, now on unseen data.

- **Range degrades hard.** near (<2×) 0.784, mid (2–3×) 0.561, far (3–5×) **0.164**. The
  far bucket is only 1,046 frames, but losing 84% of targets there is the sharpest
  single-axis collapse this project has measured.

- **Caveats — what may and may not be said:**
  - **"GLAD retains 77% of its recall on unseen video from the same campaign"** is the
    honest sentence. *Not* "GLAD generalises" — ARD100 is the same lab, same rigs, likely
    the same capture campaign, so this is the optimistic end of any generalisation estimate.
  - **No per-category rows.** ARD100 publishes no `ordinary`/`complex`/`small_mav`
    grouping, so EXP-004's headline cut has no counterpart. The comparison above is
    aggregate plus size in pixels, which is why size is quoted in pixels and **not** via
    `relative_range` — that axis is scaled to each split's own closest approach and the two
    are not comparable across datasets.
  - **Ignore the AP** (0.5226 @ IoU 0.50, mAP 0.2031). GLAD emits no confidence; every box
    is recorded at 1.0, so a ranking metric over a constant score is degenerate. P, R and F1
    are the only honest columns, exactly as in EXP-004.
  - **Not comparable to any published ARD100 number.** YOLOMG reports on all 100 videos;
    this is 15, chosen for non-overlap with our local 60.

- **Watchable version (2026-08-23):** `runs/exp005_glad_ard100/examples/phantom119_overlay.mp4`
  (3,297 scored frames, 110 s), `phantom97_overlay.mp4` (1,798, 60 s) and
  `phantom144_overlay.mp4` (2,698, 90 s), ground truth and
  GLAD's box drawn together and coloured by outcome — `src.render_video`, see
  [render_video.md](render_video.md). Rendered with `--zoom-span 320 --zoom 2`: the inset
  is **LAD's own 320×320 search region** (`REGION_HALF = 160`,
  `src/algo/glad/pipeline.py:44`) drawn at 2×, i.e. 640×640 on the 1080p frame. That span
  is deliberate — the panel shows the crop the local detector actually receives, so a miss
  can be read as *the drone was not in the region* or *it was there and LAD did not fire*,
  which the branch column alone cannot separate. 2× because at a 13 px median target 1×
  leaves the drone an unreadable speck, and 3× (the `fit_zoom` ceiling here) would take 89%
  of the frame height. The pair was chosen as the two ends of this split: phantom119 is
  P 1.000 / R 0.768 on 2,932 targets with **one** false alarm in the whole video,
  phantom97 is P 0.854 / R 0.378 — the worst recall of the fifteen **despite larger
  targets** (20.5 px median against 119's 13.4). Size does not explain phantom97, which is
  what makes it the video to watch before M7. phantom144 was added afterwards and is the
  middle case, P 0.961 / R 0.514 at 13.9 px. Not committed (`/runs/` is gitignored);
  re-render from the persisted JSONL in ~2 minutes each.

  **What the search-region panel showed, and what checking it across the split found.**
  The panel splits a miss into *the drone was not in the region* and *it was there and LAD
  did not fire*. phantom97 turned out to be overwhelmingly the second — 680 `local miss`
  against 294 `global miss` — and phantom144 repeats it at 834 against 387. phantom119 is
  the one that inverts, 405 global against 275 local. Cutting **all fifteen videos** the
  same way (from `matches_center.csv`, `branch` on `outcome == fn`):

  | | share of all 10,458 misses |
  | --- | --- |
  | `local miss` | **52.1%** (5,453) |
  | `global miss` | 35.2% (3,679) |
  | `local yolo` — fired, matched nothing | 11.6% (1,213) |

  **`local miss` is the dominant failure in 13 of the 15 videos**, and the two exceptions
  are phantom119 (ratio 0.68) and phantom102 (0.37). That matters because **phantom102
  alone supplies 45% of every `global miss` in the split** (1,647 of 3,679); it is also the
  worst video at R 0.231. Drop it and the local:global ratio goes from 1.48 to **2.38**.

  **This qualifies the *Next* section below.** Targeting `global miss` is defensible on
  *change*: it grew 2.9% → 12.3% of frames from EXP-004, a 4.2× rise against `local miss`'s
  2.2× (7.4% → 16.6%), so GAD re-acquisition is what generalises worst. But it is not the
  larger pool — `local miss` is bigger in the branch table already (16.6% vs 12.3%), bigger
  as a share of misses (52.1% vs 35.2%), and dominant per-video in 13 of 15 — and the
  aggregate that makes `global miss` look central is 45% one video. A fine-tune aimed only
  at re-acquisition leaves the larger half of the misses untouched.

  Note the `lighting` axis does not separate any of this: phantom119 and phantom97 sit at
  ~50% `backlit` and fail by opposite mechanisms, and phantom144 is 67% backlit. No cut
  taken before this one would have surfaced it.

- **Derived cut, 2026-08-23 — fine-binned recall vs target size, and what it says about
  camera choice.** Re-binned from `matches_center.csv` at 2 px granularity below 16 px,
  because the `tiny (<16 px)` bucket hides the shape of the collapse:

  | gt size (px) | 4–6 | 6–8 | 8–10 | 10–12 | 12–14 | 14–16 | 16–20 | 20–24 | 24–32 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | n | 4 | 300 | 1,850 | 5,270 | 6,975 | 6,218 | 6,712 | 3,244 | 2,201 |
  | recall | 0.000 | 0.160 | 0.348 | 0.527 | 0.635 | 0.719 | 0.798 | 0.881 | 0.908 |

  **GLAD's practical floor on this data is ~8 px** — below it recall is under a third, and
  below 6 px there is not one true positive in 33,517 targets. Recall then rises roughly
  linearly to 24–32 px and *falls* again past 32 (0.65 / 0.70 / 0.15), which is the
  reacquisition failure, not a size effect — those buckets are 653 / 77 / 13 targets.
  This curve is the input to
  [edge-budget.md §4.3](edge-budget.md#43-the-opposite-direction--an-analog-fpv-camera-iflight-racecam-r1-mini),
  which projects it onto a candidate airframe camera; **an analog FPV camera at 720 px /
  130–165° shrinks every target here to 1–5 px and projected recall to ≈0.005.** Kept in the
  ledger because it is a cut of *this* run, taken with no re-inference.

  **Reading this curve for a target that is not ARD100's.** The cut is in *pixels*, so
  transferring it to another airframe needs the physical-size ratio. ARD100 flies
  Phantom/Mavic-class targets (~0.4–0.5 m); this project's is a **10-inch quad, ~0.59 m**
  (user, 2026-08-23), so ours is ~1.2–1.4× more pixels at equal range. That credit moves the
  R1 Mini projection from 0.005 to **0.013–0.020** — real, and far too small to matter
  against 5.3–6.7× of lost angular resolution. **Never compare `gt_size` across datasets
  without this ratio**, the same trap `relative_range` carries.

- **New observation: the motion module's 50-candidate cap fired 35 times.**
  `third_party/GLAD/MOD2.py:60,183` returns an **empty** candidate list when a frame yields
  more than 50 motion rects — it gives up rather than degrading. 35 frames of 34,287 is
  negligible for these metrics, and it had never been triggered on ARD-MAV, so it is
  recorded as observed rather than as an effect. It would matter on genuinely cluttered
  data, where it is a silent recall floor.

#### Next

- **M4b is answered; the open question is now M7's target.** The deficit is acquisition at
  range and after lock-loss, not localisation and not box quality. Fine-tuning from these
  weights is the obvious lever. **Which failure to aim it at is no longer obvious, and the
  per-video branch cut in the watchable-version note above is why.** `global miss` is the
  branch that generalises worst — 2.9% → 12.3% of frames, a 4.2× rise — but `local miss` is
  the larger pool on every measure that is not a delta: 16.6% of frames against 12.3%,
  52.1% of all misses against 35.2%, and the dominant failure in 13 of the 15 videos, while
  45% of the split's `global miss` comes from phantom102 alone. **Decide between LAD and
  GAD on that evidence before renting a GPU**, and if the answer is "both", say so in the
  training proposal rather than discovering it after the run.
- **Already run:** the per-video cut. EXP-004's alarms concentrated in two videos out of
  fifteen, and EXP-005's do the same — see the branch table in the watchable-version note.
  Per-video variance is confirmed as the thing to model, not the aggregate. Still open on
  this dump: `--group video` against the **false alarms** specifically, which is a
  different question from the misses cut above and equally free.
- **The backlit slice is closed, not open.** ARD-MAV's `lighting` axis was derived on
  2026-08-23 and the control is now symmetric: it costs 3.0 points of 20.7 and does not
  change this entry's conclusion. `relative_range` is the opposite case and is now
  *demonstrably* not comparable — ARD-MAV carries a `very far (>5x)` bucket of 2,758 frames
  that ARD100 has no counterpart for, and 36.5% of ARD-MAV's frames are `far` against
  ARD100's 3.0%. The axis is scaled to each split's own closest approach, exactly as
  warned. Compare `gt_size` in pixels, never range labels.

---

## EXP-006 — half rate on ARD-MAV: does GLAD survive 15 fps?
- **Date:** 2026-08-24
- **Question:** A camera delivers 30 fps and GLAD sustains 2.67. If half the frames
  are dropped, what does it cost? **Frame *striding* is ruled out** for GLAD
  ([edge-budget.md](edge-budget.md) §3 item 7) because both motion branches difference
  against the previous frame — but that objection is about the *gap*, not about
  processing fewer frames. Halving the rate keeps every processed frame adjacent to
  the one before it; only the interval doubles, from 33 ms to 66 ms.
- **Model / weights:** identical to EXP-004 — released GLAD checkpoints, CPU port at
  `src/algo/glad/`. Nothing retrained.
- **Data:** ARD-MAV official 15-video test split. **14,172 of 28,337 frames processed
  (50.01%)**; every frame still decoded, because decode cost is not what a duty cycle
  saves.
- **Hyperparameters:** all released values, `--pad released`, byte-identical to
  EXP-004. The one variable is the schedule.
- **Hardware:** i7-1255U CPU, 3,911 s (1.09 h), **3.62 fps**.
- **Command:**
  `py -3.13 -m src.glad_detect --pad released --sample nth --sample-n 2 --out runs/exp006_half_ardmav`
  then `py -3.13 -m src.evaluate --pred runs/exp006_half_ardmav/detections.jsonl --labels
  data/processed/ARD-MAV/labels/test --conditions data/processed/ARD-MAV/conditions.json
  --frame-size 1920 1080 --match center --match-tol 1.0 --dump
  runs/exp006_half_ardmav/matches_center.csv --json-out
  runs/exp006_half_ardmav/metrics_center.json`
- **The control, and why it is the only legal comparison:** a duty-cycled run records
  only the frames it processed, and `src.eval.labels` reads prediction rows rather than
  the label directory, so it is scored on exactly those frames. Comparing that to a
  full-rate run over *all* frames would compare two frame populations. `--keys-from`
  restricts EXP-004's persisted JSONL to this run's 14,172 keys — no inference, same
  weights, same criterion:
  `py -3.13 -m src.evaluate --pred runs/exp004_glad/detections.jsonl --keys-from
  runs/exp006_half_ardmav/detections.jsonl --labels data/processed/ARD-MAV/labels/test
  --frame-size 1920 1080 --match center --match-tol 1.0 --json-out
  runs/exp006_half_ardmav/control_metrics_center.json`
- **Metrics:** `runs/exp006_half_ardmav/metrics_center.json` against
  `control_metrics_center.json`, both centre@1× over the same 14,172 frames.

  | | EXP-004 full rate | EXP-006 half rate | Δ |
  | --- | --- | --- | --- |
  | **recall** | 0.8943 | **0.8182** | −0.076 |
  | precision | 0.9927 | 0.9777 | −0.015 |
  | F1 | 0.9409 | 0.8908 | −0.050 |
  | false alarms / frame | 0.0066 | 0.0186 | 2.8× |
  | mean IoU (matched) | 0.6826 | 0.6875 | +0.005 |
  | TP / FP | 12,596 / 93 | 11,523 / 263 | |

  **Recall retained: 91.5%.**

- **Result: half the frames costs 7.6 points of recall, and the tracking lock survives.**
  The predicted failure — the local regime losing lock once target displacement doubled
  against `TrackingDetector.MAX_DISTANCE = 50` px — **did not happen**. The branch mix
  barely moves: `local yolo` 88.4% → 81.3%, `global miss` 2.9% → 2.3%. What rises is
  `local miss`, 7.4% → 14.4%: the tracker keeps its search region and more often finds
  nothing inside it. The 50 px gate bites, but gently at n=2, so GLAD's local regime has
  margin at 15 fps.
- **The compute saving is larger than the frame saving here**, which is the opposite of
  the ARD100 result and worth stating: 30 fps ÷ 2.67 = 11.24 s of compute per second of
  video at full rate, against 15 ÷ 3.62 = 4.14 at half — **2.71×**. Per *processed* frame
  this run is 36% faster than EXP-004, because the expensive global regime fires less
  (`global miss` + `global mod` 2.5% against 3.0%). Compare EXP-008, where the same
  policy on different content was 23% *slower* per frame. **Duty-cycle compute savings
  are content-dependent and must not be quoted as a fixed ratio.**

---

## EXP-007 — two-frame bursts on ARD-MAV
- **Date:** 2026-08-24
- **Question:** If the pipeline is far slower than the feed, can it process two
  *adjacent* frames and then sleep? The pair keeps differencing coherent — the frames
  are a true 33 ms apart — while the duty cycle collapses. What does a cold start cost?
- **Model / weights / hyperparameters:** identical to EXP-004, `--pad released`.
- **Data:** ARD-MAV test 15. **949 of 28,337 frames processed (3.35%)**, as 2-frame
  bursts every 60 frames.
- **Hardware:** i7-1255U CPU, 531 s, **1.79 fps**.
- **Command:**
  `py -3.13 -m src.glad_detect --pad released --sample burst --burst-length 2 --burst-period 60 --out runs/exp007_burst_ardmav`
- **Why period 60 and not 900 (the 30-second scheme actually proposed):** the period sets
  detection *latency*, not per-burst detection probability — each burst is an independent
  acquisition attempt on an arbitrary frame. A 900-frame period yields ~63 bursts across
  the split, too thin to measure. 60 yields ~474 attempts at the identical quantity, and
  **the measured per-burst probability applies to any period.** Measure dense, deploy
  sparse.
- **Metrics:** `runs/exp007_burst_ardmav/metrics_center.json`. **Read the
  second-of-burst row, not the aggregate.** The first frame of every burst follows a
  pipeline reset and has nothing to difference against, so it is a guaranteed miss by
  construction — confirmed empirically: **0 TP from 471 targets**. Over the 474 usable
  frames, against EXP-004 on those same frames:

  | | EXP-004 full rate | EXP-007 burst | Δ |
  | --- | --- | --- | --- |
  | **recall** | 0.8702 | **0.4191** | −0.451 |
  | precision | 0.9903 | 0.9517 | −0.039 |
  | TP / FP | 409 / 4 | 197 / 10 | |

  **Recall retained: 48.2%.**

- **Result: a burst keeps under half the recall, and every burst runs in the global
  regime.** Branch mix is 50.1% `first frame`, 28.1% `global miss`, 11.4% `global yolo`,
  10.4% `global mod` — and **zero `local yolo`**, because a lock cannot survive the
  sleep. That zero is also the harness's correctness check: a non-zero `local yolo`, or a
  `first frame` share away from 1/K, would mean the per-burst reset had misfired and the
  motion branches were differencing across the sleep.
- **Appearance and motion contribute equally here**: of 197 detections, 98 came from
  `global yolo` and 99 from `global mod`. Contrast EXP-009, where on unseen video
  appearance collapses and motion carries 77%.
- **Cost:** 1 frame per second of video ÷ 1.79 fps = 0.56 s of compute per second of
  video, against 11.24 at full rate — **20× cheaper**. Per *processed* frame it is 33%
  more expensive than EXP-004 (1.79 against 2.67 fps), because the expensive global path
  runs every time it runs at all. **The saving is duty cycle alone, never per-frame cost.**

---

## EXP-008 — half rate on ARD100, video GLAD has never seen
- **Date:** 2026-08-24
- **Question:** EXP-006 measured the half-rate penalty where GLAD is strong. Does the
  penalty grow where the model is already weak? EXP-005 established that ARD100 costs
  20.7 points of recall on content alone; if a duty cycle compounds that, the deployment
  answer changes.
- **Model / weights / hyperparameters:** identical to EXP-005, `--pad released`.
- **Data:** ARD100 test 15 (the same split as EXP-005). **17,148 of 34,287 frames
  processed (50.01%)**.
- **Hardware:** i7-1255U CPU, 6,154 s (1.71 h), **2.79 fps**.
- **Command:**
  `py -3.13 -m src.glad_detect --dataset ARD100 --pad released --sample nth --sample-n 2 --out runs/exp008_half_ard100`
  with the control `--keys-from runs/exp008_half_ard100/detections.jsonl` against
  `runs/exp005_glad_ard100/detections.jsonl`.
- **Metrics:** centre@1× over the same 17,148 frames.

  | | EXP-005 full rate | EXP-008 half rate | Δ |
  | --- | --- | --- | --- |
  | **recall** | 0.6876 | **0.6290** | −0.059 |
  | precision | 0.9459 | 0.9321 | −0.014 |
  | F1 | 0.7963 | 0.7511 | −0.045 |
  | false alarms / frame | 0.0385 | 0.0448 | +16% |
  | mean IoU (matched) | 0.6805 | 0.6867 | +0.006 |

  **Recall retained: 91.5%** — identical to EXP-006's 91.5% to three significant figures.

  Recall by size, against the control on the same frames:

  | bucket | n | full rate | half rate | Δ |
  | --- | --- | --- | --- | --- |
  | tiny (<16 px) | 10,315 | 0.5987 | 0.5344 | −0.064 |
  | small (16–32) | 6,079 | 0.8411 | 0.7954 | −0.046 |
  | medium (32–96) | 366 | 0.6557 | 0.5410 | **−0.115** |

- **Result: the half-rate penalty is a property of the policy, not of the content.**
  91.5% retention on both ARD-MAV and ARD100 is the entry's load-bearing number: it means
  the cost of running at 15 fps can be quoted as a single figure and does **not** compound
  with the generalisation gap. Contrast the burst policy, which does compound (EXP-007
  48.2% → EXP-009 36.0%).
- **The loss is worst on medium targets**, which is counter-intuitive and thinly
  sampled (n=366, so do not lean hard on it). Medium targets are nearer and traverse more
  pixels per frame, so doubling the interval hurts them most — consistent with the 50 px
  gate being the mechanism.
- **Cost: 1.55×, not 2×.** 30 ÷ 3.60 = 8.33 s of compute per second of video at full
  rate, against 15 ÷ 2.79 = 5.38 at half. Per processed frame this run is **23% slower**
  than EXP-005, because losing lock more often means paying the motion path more often
  (`local miss` 16.6% → 21.6%, `global miss` unchanged at 12.3%). Halving the frames does
  not halve the work.

---

## EXP-009 — two-frame bursts on ARD100
- **Date:** 2026-08-24
- **Question:** The burst policy where the model is weakest. This is the run that decides
  whether a 30-second duty cycle is deployable at all.
- **Model / weights / hyperparameters:** identical to EXP-005, `--pad released`.
- **Data:** ARD100 test 15. **1,146 of 34,287 frames processed (3.34%)**, 2-frame bursts
  every 60 frames.
- **Hardware:** i7-1255U CPU, 908 s, **1.26 fps**.
- **Command:**
  `py -3.13 -m src.glad_detect --dataset ARD100 --pad released --sample burst --burst-length 2 --burst-period 60 --out runs/exp009_burst_ard100`
- **The sampling is representative, and this was checked rather than assumed.** EXP-005
  restricted to these 1,146 frames scores **recall 0.6878** against its full-run 0.6880 —
  1,146 frames out of 34,287 reproduce the whole run to within 0.0002. Nothing below is a
  sampling artefact.
- **Metrics:** over the 561 second-of-burst frames (the first of each pair yielded
  **0 TP from 560 targets**, as designed):

  | | EXP-005 full rate | EXP-009 burst | Δ |
  | --- | --- | --- | --- |
  | **recall** | 0.6934 | **0.2496** | −0.444 |
  | precision | 0.9534 | 0.9150 | −0.038 |
  | TP / FP | 389 / 19 | 140 / 13 | |

  **Recall retained: 36.0%.**

- **Result: burst pairs keeps about a third of the recall on unseen video, and the
  penalty compounds where half rate's does not.**

  | policy | ARD-MAV | ARD100 | compounds? |
  | --- | --- | --- | --- |
  | half rate | 91.5% | 91.5% | **no** |
  | burst pairs | 48.2% | 36.0% | **yes** |

- **The mechanism is visible in which branch found each target.** On ARD-MAV appearance
  and motion split the work 98/99; here it is **32 `global yolo` against 108 `global
  mod`** — GAD collapses on unseen video and motion carries 77% of what remains. That is
  independently consistent with EXP-005's finding that the deficit is acquisition rather
  than localisation, and it means burst mode on unseen video rides almost entirely on a
  single frame-pair difference.
- **Latency is what disqualifies the scheme, not recall.** Each burst is an independent
  attempt at 25.0%, so the expected number of attempts before a first detection is 4.0:

  | period | mean delay to first detection | closing at 40 m/s |
  | --- | --- | --- |
  | 2 s (measured) | 8.0 s | **320 m** |
  | **30 s (as proposed)** | **120 s** | **4,806 m** |
  | full rate | ~0.05 s | ~2 m |

  At a 30-second period a target closes nearly 5 km before it is expected to be seen.
  **No recall figure rescues that**, and it is the reason the live path
  ([live_detect.md](live_detect.md)) treats burst pairs as a fallback for hardware that
  cannot sustain half rate, never as a power-saving choice.
- **Cost:** 0.79 s of compute per second of video against 8.33 at full rate — **10.5×
  cheaper**, and ~157× at a 30-second period. The saving is real; the latency is what
  cannot be paid for.

#### Next

- **Half rate is the deployable policy and burst pairs is the fallback.** Both are wired
  into `src.live_detect --policy auto`, which measures sustained throughput against the
  feed's own rate and takes the most accurate policy the machine can actually hold.
- **Open: 10 fps (`--sample-n 3`).** Half rate cost 8.5% of recall with the lock intact
  and `global miss` unmoved, so the gate has margin left. One run, same code, and it
  would establish whether the penalty is linear in interval or has a knee.
- **Open: `--burst-length 3`.** Every burst currently spends half its frames on a
  guaranteed miss. A 3-frame burst gives two usable frames for 1.5× the cost and might
  let the local regime engage once, which no 2-frame burst can.
- **Not open: the per-stage profile** ([todo.md](todo.md)) still gates the edge budget,
  and these runs do not substitute for it — they measure whole-pipeline throughput, not
  where the time goes.

---

## EXP-010 — GLAD on our own footage, which nobody has labelled

- **Date:** 2026-09-17
- **Question:** The first video in this project that came off **our** airframe rather than
  someone else's dataset. Does GLAD fire on it at all, does it fire on the right thing,
  and what does it cost per frame at this resolution?
- **Model / weights:** GLAD released pipeline, `third_party/GLAD/weights/` — `yolov5s_GLAD.pt`, `yolov5s_GLAD-crop.pt`, `Net_best.pth`. `--pad released`, matching EXP-004–009.
- **Data:** `data/raw/FIELD/videos/captured_raw_20260616_040253_004.mp4` — 1032×752, 30 fps, **3,600 frames (120.0 s)**, arid hillside, hard sky/ridge horizon. **No labels.** Provenance in [datasets.md](datasets.md).
- **Hyperparameters:** every fixed threshold of the released source; no stride, no duty cycle (`--sample every`, 100% duty)
- **Hardware:** i7-1255U CPU, 1,179.4 s wall-clock, **3.05 fps**
- **Command:** `py -3.13 -m src.glad_detect --videos data/raw/FIELD/videos --video-names captured_raw_20260616_040253_004 --record-all --images data/processed/FIELD/images/test --pad released --out runs/field/exp010_field_glad`
- **Metrics:** **none, and none are possible.** No ground truth exists for this video, so
  there is no AP, mAP, precision, recall or `far`. What the run has is 866 detections over
  3,600 frames (0.24/frame), 2,734 frames empty (75.9%), and this branch split:

| Branch | Frames | Share |
| --- | ---: | ---: |
| `global miss` | 2,388 | 66.3% |
| `local yolo` | 850 | 23.6% |
| `local miss` | 345 | 9.6% |
| `local mod` | 10 | 0.3% |
| `global yolo` | 6 | 0.2% |
| `first frame` | 1 | 0.0% |

- **Result:** GLAD works on our footage. Detections are not scattered — **96% of them fall
  inside three sustained lock-ons** (frames 2–17 with a scattered tail to 176, 1101–1553,
  and 3140–3600 essentially unbroken), which is the local regime holding a track, and the
  gaps between them are `global miss`. A **seeded random sample of 24 of the 866
  detections, inspected as zoomed crops, contained 23 drones and 1 patch of ground
  clutter.** Read as precision conditional on firing that is **23/24 ≈ 0.96** (95% CI
  roughly 0.79–0.999 on n=24) — far above anything this project has measured on a
  prepared dataset, because almost every detection here comes from a held track against
  clean sky rather than from per-frame acquisition.
- **Caveats:** Four, and the first two are load-bearing.
  - **This is not a score and cannot be compared to EXP-001–009.** The 0.96 is a sampled
    estimate of one direction only. **Recall is unmeasured**, and it is demonstrably not
    1: at frames 19, 101 and 176 the drone is plainly visible in the sky while the tracker
    holds a box on a bush — a false alarm and a miss in the same frame.
  - **Resolution.** GLAD's motion constants are absolute pixels tuned for 1920×1080 (blob
    area 30–3000, `a=160`, `dist_ref=200`, blur 11, `MAX_DISTANCE=50`). This capture's
    diagonal is 1276 against 2203 — **0.579× linear, 0.335× in area** — so the run
    measures our failure to rescale alongside the detector. The same confound
    [todo.md](todo.md) raises for FL-Drones.
  - **One video, one flight, one target.** 120 s.
  - **Throughput was measured on a machine that was not idle** — a few short commands ran
    against it. Treat 3.05 fps as approximate.
- **What it does establish, that no prepared dataset could:**
  - **3.05 fps at 1032×752**, against 2.67 on 1080p ARD-MAV. Still 10× short of a 30 fps
    feed, so `src.live_detect --policy auto` would pick **burst pairs** on this footage —
    the 36% retention fallback, not half rate.
  - **Two thirds of the run is `global miss`** — the expensive branch, GAD then GMD then
    LAD, spent finding nothing. On ARD-MAV that branch is rare. Whatever the edge budget
    ends up being, this is the regime it has to survive, and it is the *slowest* one.
  - **Ground clutter takes the lock.** The drift onto a bush at frames 19–176, while the
    real target is in frame, is the failure mode this footage adds that ARD-MAV's cleaner
    backgrounds do not exercise.
- **Watch it:** `runs/field/exp010_field_glad/overlay.mp4` — all 3,600 frames with GLAD's boxes
  drawn back on, rendered by `src.render_video --no-labels --zoom 3 --zoom-span 100`.
  **Every hit on one sheet:** `runs/field/exp010_field_glad/all_hits.png` — all 866 detections
  as crops ordered by branch and bordered in its colour (`global yolo` 6, `local yolo` 850,
  `local mod` 10). The three clutter episodes are visible as blocks of hillside among
  otherwise clean sky: frames ~12–100 (the known bush lock), ~1520–1540, and ~3141–3156.
  **Unscored on purpose:** the normal renderer colours every box by match outcome, so on
  unlabelled footage it would paint all 866 detections red and caption each a false alarm
  — a precision claim of zero against a run that sampled 23/24 correct. The unscored mode
  draws one neutral colour and says on the strip that nothing here is known to be right.
- **Next: labels** — see [todo.md](todo.md). The run is keyed at
  `data/processed/FIELD/images/test/`, so annotating even the three target episodes turns
  this JSONL into a real score with **no second inference pass**.
- **No appearance-only reference exists for this footage.** A tiled `yolov8s_eo_drone`
  run (the EXP-001–003 model, conf 0.15) was started on 2026-09-17 and **cancelled at
  1,786 of 3,600 frames**; its partial output was deleted rather than kept, because a
  half-finished run that looks like a run is worse than none. It sustained ~0.5 fps — a
  full pass is ~2 h — so if the comparison is ever wanted, budget for that or stride it.

---

## EXP-011 — the same footage at the right scale, which made it worse

- **Date:** 2026-09-17
- **Question:** EXP-010 looked far worse than the prepared-dataset runs, and three causes
  were on the table: the background, the target's payload, and **resolution**. GLAD's motion
  constants are absolute pixels tuned at 1920×1080 and this capture is 1032×752, so the
  ledger ranked resolution first: EXP-010 measured our failure to rescale alongside the
  detector. Remove that confound and see what is left.
- **Model / weights:** identical to EXP-010 — GLAD released pipeline,
  `third_party/GLAD/weights/`, `--pad released`. Nothing about the detector changed.
- **Data:** identical to EXP-010 — `captured_raw_20260616_040253_004.mp4`, 1032×752, 3,600
  frames. **Still no labels.**
- **The one change:** `--scale auto` (new; `src.algo.glad.scaling`). Each frame is resized to
  the 1920×1080 **diagonal** preserving aspect — **1032×752 → 1780×1297, 1.7248× linear,
  2.975× in pixels** — and boxes are mapped back, so the record stays in original
  coordinates and is directly comparable to EXP-010's.
- **Hardware:** i7-1255U CPU, 1,351.7 s wall-clock, **2.66 fps**. The machine was **not
  idle** — a second session ran against it throughout. Treat as approximate.
- **Command:** `py -3.13 -m src.glad_detect --videos data/raw/FIELD/videos --video-names captured_raw_20260616_040253_004 --record-all --images data/processed/FIELD/images/test --pad released --scale auto --out runs/field/exp011_field_glad_scaled`
- **Metrics:** **none, and none are possible** — same as EXP-010, and for the same reason.
  What follows is what the detector *did*, not how well it did it.

| | EXP-010 native | EXP-011 scaled |
| --- | ---: | ---: |
| detections | 866 (0.241/frame) | **763 (0.212/frame)** |
| empty frames | 2,734 (75.9%) | **2,837 (78.8%)** |
| `global miss` | 2,388 (66.3%) | **2,376 (66.0%)** |
| `local yolo` | 850 (23.6%) | 736 (20.4%) |
| `local miss` | 345 (9.6%) | 460 (12.8%) |
| `local mod` | 10 (0.3%) | 16 (0.4%) |
| `global yolo` | 6 (0.2%) | 10 (0.3%) |
| median box, √area | 12.4 px | **9.6 px** |

- **Result: scaling does not fix it, and on the balance of the evidence it makes it worse.**
  The headline number the whole exercise targeted — `global miss`, 66.3% — moved to **66.0%**.
  That is noise. But the *composition* changed a great deal, and inspecting it is what
  settles the question:
  - **The three known drift frames are fixed.** At frames 19, 101 and 176 EXP-010 held a box
    on a bush while the drone was in clear sky. EXP-011 emits **no box** at any of them, and
    the whole 2–176 episode collapses from 41 detections to 2.
  - **But it invented two larger clutter locks.** 113 detections appear in regions EXP-010
    never fired in at all — sustained runs at **frames ~1832–1885 and ~2374–2464**. A seeded
    sample of 24 of those 113, inspected as zoomed crops, was **24 out of 24 ground
    clutter**: the box on a pale rock or bush against dark hillside, every time, held by
    `local yolo`. Not one contained a drone.
  - **And it lost real targets.** Inside the two genuine episodes it fired 301 times against
    365 (frames 1101–1553) and 347 against 458 (3140–3600). There are **186 in-episode
    frames where EXP-010 fired and EXP-011 did not**; a seeded sample of 24 of those showed
    **large, sharp, unmistakable multirotor silhouettes against blue sky** in all but two.
    These were not marginal detections. They were the easiest ones in the video.
  - The median box shrank from 12.4 px to 9.6 px √area, which is what a population shifting
    from drones to clutter blobs looks like.
- **Why upscaling would hurt — a mechanism, offered as hypothesis:** interpolation adds no
  information, but it does move everything through the **blob-area gate**, which is
  `30 < area < 3000` px² in both `MOD2_global` and `MOD2_local`. At 2.975× in area, clutter
  that sat *below* the 30 px² floor at native resolution — small pale rocks, bush crowns —
  is lifted into the admissible window, while the drone was already comfortably inside it
  and gains nothing. **The gate's lower bound was doing real clutter rejection, and
  upscaling defeated it.** Cubic interpolation also softens the target's edges, which is
  consistent with `local yolo` falling 850 → 736. If that reading is right, the
  constant-side fix — scaling the thresholds *down* rather than the frame *up* — is not
  equivalent to this run and could still help; it is [glad-model.md](glad-model.md)
  improvement #4's remaining half, and it is expensive for the reasons recorded there.
- **What this settles about EXP-010's three candidate causes:**
  - **Resolution: largely exonerated** as the explanation for how EXP-010 looked. The
    confound is real and worth knowing about, but correcting it does not recover
    performance — it trades one failure for a worse one.
  - **Background: strongly implicated.** Every false alarm inspected in this run, and the
    one in EXP-010's sample, is ground clutter on an arid hillside. The failure mode is the
    tracker taking a lock on terrain and holding it for tens of seconds, which is exactly
    what ARD-MAV's cleaner backgrounds never exercise.
  - **Payload: still untestable and still unsupported.** There is no payload-free control
    flight of this airframe. The indirect evidence continues to point away from it: payload
    is an appearance-channel hypothesis, and `global yolo` fires 0.2–0.3% here against 0.1%
    on ARD-MAV — GAD is near-useless from cold on *both*, so it cannot explain the gap.
- **Caveats:** Three, and the first is structural.
  - **"Inside an episode" is defined by where EXP-010 fired**, so the episode arithmetic is
    circular and cannot on its own prove EXP-011 lost true positives. **The visual
    inspection is not circular** — the crops show drones or they show hillside — and that is
    what the claim rests on. Both sheets are in the run directory.
  - **Recall is unmeasured in both runs.** EXP-011 is worse than EXP-010 by this evidence;
    neither absolute number is known, and neither will be until the footage is labelled.
  - **Sampling error.** Two samples of 24. The 24/24-clutter result is strong; the
    "overwhelmingly real drones" result rests on eyeballing 24 crops.
- **A throughput result worth keeping:** 2.66 fps at **2.975× the pixels**, against 3.05 fps
  native — only **13% slower for triple the pixel count**. The motion branch is therefore
  *not* what this pipeline spends its time on; the YOLO forward passes are, and GAD
  letterboxes to 640 regardless of source resolution. That matters for the edge budget:
  optimising MOD2 would buy almost nothing.
- **Evidence in the run directory** (`runs/field/exp011_field_glad_scaled/`, gitignored):
  `exp011_new24.png` (the 24 out-of-episode detections, all clutter), `exp010_lost24.png`
  (24 of the 186 in-episode detections EXP-011 dropped, nearly all real drones),
  `exp011_sample24.png` (a seeded 24 of all 763, comparable to EXP-010's 23/24), plus the
  two throwaway scripts that produced them. `all_hits.png` puts **all 763** on one sheet,
  ordered by branch and bordered in its colour, where the invented clutter locks read as an
  unbroken block of dark hillside running from frame ~1833 to ~2464 — roughly a fifth of
  the sheet, against clean sky either side. **No overlay video was rendered.** EXP-010's is
  388 MB and this is a negative result nothing will be built on; the contact sheets carry
  the finding at a fraction of the size.
- **Next:** this run exhausts what unlabelled footage can answer. Both surviving questions —
  how bad the clutter false-alarm rate actually is, and what recall on our own camera is —
  are measurements, and both need **labels**. See [todo.md](todo.md).

---

## EXP-011 — GLAD on our O4 intercept trials, and what a burned-in HUD does to it

- **Date:** 2026-09-17
- **Question:** Six clips off our own DJI O4 downlink, five `catch` and one `miss`. Does GLAD find the target on the footage the airframe actually produces?
- **Model / weights:** identical to EXP-004/005/010 — `third_party/GLAD/weights/{yolov5s_GLAD.pt, yolov5s_GLAD-crop.pt, Net_best.pth}`, `--pad released`, every threshold at the released value. Nothing retrained.
- **Data:** `data/processed/SOFA-O4/videos/` — six clips, **7,386 frames, 4.1 min, 1440×1080**. Losslessly cropped from 2520×1080 goggles screen recordings; **7,386/7,386 frames verified bit-identical** to the raw crop. [MANIFEST](../data/processed/SOFA-O4/MANIFEST.md). **No labels.**
- **Hyperparameters:** full rate, contiguous, 100% duty cycle.
- **Hardware:** i7-1255U CPU, 2,375.9 s, **3.11 fps**.
- **Command:** `py -3.13 -m src.glad_detect --videos data/processed/SOFA-O4/videos --video-names first_catch second_catch third_catch forth_catch catch_5 miss_1 --record-all --images data/processed/SOFA-O4/images/test --pad released --out runs/sofa_o4/exp011_sofa_o4_glad`
- **Metrics:** none possible — no ground truth. 1,347 detections over 7,386 frames (0.18/frame), 6,039 frames empty (81.8%).

| Branch | Frames | Share |
| --- | ---: | ---: |
| `global miss` | 3,456 | 46.8% |
| `local miss` | 2,577 | 34.9% |
| `local yolo` | 1,091 | 14.8% |
| `local mod` | 233 | 3.2% |
| `global yolo` | 15 | 0.2% |
| `global mod` | 8 | 0.1% |

### Result: the run measures the overlay, not the detector

**A seeded random sample of 24 of the 1,347 detections, inspected as zoomed crops:**

| What the box was on | Count |
| --- | ---: |
| HUD glyph — battery digits `3`/`4`/`8`, the `v`, the `A` of `AIR`, the centre reticle | **21** |
| Bare ground clutter | 3 |
| **A drone** | **0** |

Corroborated by geometry: **49.7% of all detections fall in the bottom telemetry strip**
(y ≥ 900). The two sustained lock-ons — `second_catch` 709–1162 and `miss_1` 1425–1724 —
are the tracker holding station on the **battery voltage digit**, visible in
`examples/miss_1_overlay.mp4` at frame 1490.

And it costs real detections: `first_catch` frame **962** contains an unmistakable
quadcopter against clean sky, and GLAD fired on **nothing** in frames 950–964.

| Clip | Frames | Fired | Fired % |
| --- | ---: | ---: | ---: |
| `first_catch` | 964 | 166 | 17.2% |
| `second_catch` | 1,551 | 509 | 32.8% |
| `third_catch` | 1,155 | 33 | 2.9% |
| `forth_catch` | 1,060 | 48 | 4.5% |
| `catch_5` | 932 | 73 | 7.8% |
| `miss_1` | 1,724 | 518 | 30.0% |

- **Caveats:** **Do not cite any number here as GLAD's performance on our footage.** The
  detection count is dominated by an artifact of the recording method. No labels exist, so
  recall is unmeasured — but it is visibly poor, and the reason is legible: the tracker is
  captured by a static glyph and stops looking. Resolution is a second-order confound
  (1440×1080 is 0.817× the 1080p diagonal GLAD's absolute-pixel constants assume).
- **Watch it:** `runs/sofa_o4/exp011_sofa_o4_glad/examples/<clip>_overlay.mp4`, six files, all
  7,386 frames, unscored (`src.render_video --no-labels --zoom 3 --zoom-span 100`).
- **What this does establish:**
  - **3.11 fps at 1440×1080**, against 2.67 on 1080p ARD-MAV and 3.05 on FIELD.
  - **A burned-in OSD is a blocker, not a nuisance.** The appearance branch fires on
    high-contrast glyphs and the tracker then holds them for hundreds of frames. Any
    deployment that taps a goggles feed rather than a clean camera inherits this.
  - **The trial `raw/` folder is not a way out** — it is 320×240 analog DVR, far worse
    than the goggles capture.
- **Next:** crop the telemetry strip (zero risk, removes half the detections) and decide
  whether to inpaint the centre reticle, which cannot be cropped because it sits where
  targets appear. Then re-run. See [todo.md](todo.md).

---

## EXP-012 — inverting the image, which found a drone the other two runs never saw

- **Date:** 2026-09-17
- **Question:** [glad-model.md §5b](glad-model.md) measured that ARD-MAV's targets are 84.4%
  **brighter** than their background (white DJI Phantoms over roads and concrete), while our
  field target is 98.5% **darker** and our ground false alarms are 99.1% brighter — our
  clutter sits in the training distribution more comfortably than our own drone does. That
  is a correlation. Inverting the image (`255 - pixel`) swaps both polarities at once and
  tests it directly.
- **Model / weights / data:** identical to EXP-010 in every respect. `--pad released`, native
  1032×752, no scaling. **The only change is `--invert`.** Still no labels.
- **Hardware:** i7-1255U, 5,159.1 s, **0.70 fps** — against 3.05 fps for the same footage
  un-inverted. The machine was **not idle**, but a 4.4× gap is too large for contention
  alone; the likely cause is more motion candidates surviving the blob-area gate and each
  paying a LeNet call. Not investigated.
- **Command:** `py -3.13 -m src.glad_detect --videos data/raw/FIELD/videos --video-names captured_raw_20260616_040253_004 --record-all --images data/processed/FIELD/images/test --pad released --invert --out runs/field/exp012_field_glad_inverted`
- **Metrics:** none possible, as with EXP-010 and EXP-011.

**The prediction was wrong, and the hypothesis survived anyway.** The stated test was that
cold appearance acquisition would rise sharply. `global yolo` went **6 → 11** frames out of
3,600. That is noise, and GAD remains useless on this footage whichever way up it is. The
branch mix barely moved either (`global miss` 66.3% → 62.2%).

What changed is *what got detected*, and it changed exactly the way the polarity argument
predicts:

| Window | EXP-010 native | EXP-011 scaled | EXP-012 inverted |
| --- | ---: | ---: | ---: |
| 1101–1553 sky episode | 365 | 301 | 360 |
| 3140–3600 sky episode | 458 | 347 | 394 |
| 2–176 bush lock (false) | 41 | 2 | 14 |
| 2350–2480 white structures (false) | 1 | **75** | **0** |
| **1990–2120 drone over terrain** | **0** | **0** | **40** |
| 1540–1700 undetermined | 0 | 0 | 13 |

- **It found a real drone episode both other runs missed entirely.** Frames ~1990–2120, 40
  boxes, **median 22 px**, and at that size the target is unambiguous in the crops:
  a multirotor with visible arms and rotors, **carrying a bright payload slung beneath the
  airframe** — and the box sits on the *payload* rather than the body.
  `zoom_drone_over_terrain.png` in the run directory. EXP-010 and EXP-011 recorded **zero**
  boxes anywhere in that window.
- **Acquisition came from `global mod`, not `global yolo`** — a motion candidate confirmed by
  LAD within 10 px at conf 0.5, at frame 2008. That branch fired **0 times in EXP-010** and
  once in EXP-011. Everything after it is `local yolo` (24) and `local mod` (15) holding the
  track. So inversion did not improve appearance acquisition; it made the **motion** path's
  candidate survive confirmation.
- **It eliminated EXP-011's clutter.** The 2350–2480 block that EXP-011 produced 75
  detections in — small white man-made structures on the hillside, 4–8 px, confirmed at high
  zoom in `runs/field/exp011_field_glad_scaled/zoom_white_structures.png`, with a second identical
  unboxed structure visible below-left in most frames — drops to **zero**. The bush lock
  falls 41 → 14.
- **The polarity preference itself is confirmed and stable.** Measured on the **original**
  frames, 98.6% of EXP-012's detections are *darker* than their background — which means
  that in the inverted image the detector actually saw, they were **brighter**. EXP-010's
  false alarms were 93.5% brighter as seen. So across both runs the detector consistently
  picks what is **brighter than its local background in whatever image it is given**.
  Inverting did not remove that preference; it changed which physical objects satisfy it.

- **Reading:** GLAD acquires targets that are brighter than their local background, which is
  the ARD-MAV white-Phantom signature. Against sky our drone is a dark silhouette — the wrong
  polarity — but high-contrast enough that once locked, `local yolo` at conf 0.1 holds it, so
  the sky episodes work in every run. **Against terrain the drone is darker than its
  surroundings and never gets acquired at all** — until inversion supplies the training
  polarity, at which point the motion branch's candidate passes confirmation and 40
  detections follow. This is the first direct evidence that the appearance prior, not the
  resolution (EXP-011) and not the payload as such, is what costs us recall on our own
  footage.
- **Caveats:**
  - **Not a deployment fix, and must not be read as one.** It is a probe. Inverting would
    invert the problem on any footage that does look like ARD-MAV, and the sky episodes did
    lose ground (458 → 394 on the second). The remedy for an appearance mismatch is
    fine-tuning on our own footage, which needs labels.
  - **Still unscored.** 40 "true" detections is a visual judgement on crops, not a
    measurement; 13 detections at 1540–1700 are genuinely ambiguous (`zoom_ambiguous_1540_1700.png`)
    and are counted as neither.
  - **0.70 fps** makes this the slowest configuration measured, on a machine that was not idle.
  - One video, one flight, one target.
- **An option this suggests, not yet tested:** running both polarities and taking the union
  would have caught every episode in the video — the sky ones from the normal pass and the
  terrain one from the inverted. It costs 2× compute, which the edge budget probably cannot
  afford, but it brackets what a polarity-robust detector would be worth.
- **Evidence** (`runs/field/exp012_field_glad_inverted/`, gitignored): `overlay.mp4` (all 3,600
  frames, `src.render_video --no-labels --zoom 3 --zoom-span 100`, matching EXP-010's so the
  two are watchable side by side), `all_hits.png` (all 831 by branch),
  `zoom_drone_over_terrain.png`, `zoom_ambiguous_1540_1700.png`, `new_regions24.png`.
  The sheets are reproducible with [`src.crops`](crops.md), which the throwaway scripts that
  first made them have since become; see the `/inspect` skill for how to read one.

---

## EXP-012a — the O4 fixes on one clip: the mask works, the motion fix does not

> **Corrected 2026-09-22 by [EXP-012b](#exp-012b--exp-012as-motion-claim-re-measured-on-the-labels).**
> The HUD-veto result stands. The explanation for the motion miss **does not**: it compared
> the target's *raw* image motion (4.5 px) with the *raw* background spread, when the
> test that matters is the target's motion *relative to the compensated background*
> (~22 px at 954–964) against the residual *left after* the homography (p90 7.4 px). The
> drone is not buried in compensation error. It is in the difference image. GLAD drops it
> later, in the crowding bail and the blob shape test. The "~955–964" passage was also
> wrong: the labels put the drone in view for frames 708–964.

- **Date:** 2026-09-17
- **Question:** EXP-011 measured the overlay instead of the detector. With the HUD vetoed and the motion branches no longer discarding their candidates, does GLAD find the drone?
- **Model / weights:** identical to EXP-004/005/010/011. `--pad released`.
- **Data:** `data/processed/SOFA-O4/videos/first_catch.avi` — 964 frames, 1440×1080. **No labels**, but one target passage confirmed by eye and by tracker: frames ~955–964, 80×34 px against clean sky at frame 962.
- **Hyperparameters:** `--hud-mask` (1.80% of frame) and `--motion-profile clutter` (ranked candidates, blob ceiling 3,000 → 12,000 px²).
- **Hardware:** i7-1255U CPU, 701.2 s, **1.37 fps**.
- **Command:** `py -3.13 -m src.glad_detect --videos data/processed/SOFA-O4/videos --video-names first_catch --record-all --pad released --hud-mask data/processed/SOFA-O4/hud_mask.png --motion-profile clutter --out runs/sofa_o4/exp012a_first_catch`

### Result

| | EXP-011 | EXP-012a |
| --- | ---: | ---: |
| Detections | 166 | **9** |
| `global miss` | 40.4% | **93.6%** |
| `local yolo` | 95 | **0** |
| Throughput | 3.11 fps | **1.37 fps** |
| Drone found at frame 962 | no | **no** |

**The HUD veto works.** 166 detections fall to 9, and every `local yolo` lock on a glyph is
gone. **The motion fix does not buy recall.** Frames 950–964 are `global miss` without
exception: the unmistakable 80×34 px quadcopter at 962 is still missed, and all 9 survivors
were inspected as crops — **HUD ladder dashes and ground, not one drone.**

Five of the nine sit on the **pitch ladder**, the one HUD element the mask does not cover
because it sweeps vertically with pitch. That gap is now a measured consequence rather than
a caveat.

### Why the motion branches cannot see this target

Measured over frames 954–964, target displacement against background displacement:

| | px per frame |
| --- | ---: |
| **Target** | **4.1 – 9.5** (4.5 at the moment of the catch) |
| Background, median | 18.9 – 24.3 |
| Background, p90 | 27.2 – 34.3 |

The background's own **spread is ~10 px between median and p90** — that is parallax, and a
single homography cannot represent it. The residual it leaves behind is **larger than the
target's entire differential motion of 4.5 px**. The drone is buried in compensation error,
and no threshold, blob ceiling or candidate ranking recovers something that is quieter than
the noise it sits in.

This is intercept geometry doing it. Closing on a target puts it near the focus of
expansion, where image motion is *least*, while the near ground streams past at 20+ px. The
premise GLAD's motion branches rest on — that the target moves differently from a
compensable background — is inverted here.

- **Correction to the EXP-011 write-up.** That entry blamed `motion_compensate`'s 50 px
  flow-rejection cap. **Measured, it is not binding on this segment**: background flow is
  18.9–24.3 px median, p90 34.3, all under 50. The cap would bite lower and faster; what
  bites *here* is parallax spread, which is a different problem and not fixed by moving
  that constant.
- **Caveats:** one clip, one passage, still no labels — so "missed" is established by eye
  on a target that is unmistakable, but precision and recall remain unquantified. The
  `clutter` profile also costs **2.3× the compute** for no recovered target.
- **Next:** the appearance branch has to carry this, which means **fine-tuning on our own
  labels** — Stage E of the plan and the existing M7. Before that, `--motion-profile
  clutter` should not be adopted: it is slower and, on this evidence, buys nothing.

## EXP-012b — EXP-012a's motion claim re-measured on the labels

- **Date:** 2026-09-22
- **Question:** EXP-012a concluded the drone was "quieter than the noise it sits in": target
  4.5 px/frame against a ~10 px background spread, so no motion threshold could recover it.
  That was measured over 11 frames placed by eye. `first_catch` now has 257 labelled
  frames (708–964, [datasets.md](datasets.md)). Does the claim survive them?
- **Model / weights:** none for the first part. The second part runs GLAD's own
  `motion_compensate` and `MotionPort` (both `UPSTREAM` and `CLUTTER` profiles, HUD mask
  on) with the released LeNet gate.
- **Data:** `data/processed/SOFA-O4/videos/first_catch.avi`, frames 708–964, boxes from
  `annotations/first_catch.json`. 67 boxes placed by hand, 190 by the follower.
- **Hardware:** i7-1255U CPU, a few minutes.
- **Scripts:** `runs/sofa_o4/exp012b_motion_check/motion_check.py` and `stage_check.py`, run from
  the repo root with `PYTHONPATH=.`. They are throwaway (gitignored with their outputs
  `motion.csv` and `stages.json`), not tested code.

### What was measured, and how

For each consecutive labelled pair: GLAD's grid-KLT homography, as `motion_compensate`
computes it, with background points taken outside the target box and the HUD mask.

- **Differential motion:** where the background model says the target's previous centre
  should now be, minus where it actually is. This is what a differencing detector sees.
- **Residual:** how far the background points themselves miss after compensation. This is
  the noise the target has to beat.

Follower boxes are biased: the follower moves the box *with* the image, which pulls
differential motion toward zero. So the headline uses only **hand-placed to hand-placed
spans** (66), chaining the per-frame homographies across the 3–4 frame gap.

| Frames | Box (median) | Target differential, hand spans | Residual median | Residual p90 | Spans above residual p90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 708–800 | 43 px | **4.3** px/f | 0.6 | 2.8 | 70% |
| 801–900 | 87 px | **10.1** | 1.3 | 4.1 | 97% |
| 901–953 | 104 px | **13.0** | 0.9 | 4.2 | 100% |
| 954–964 | 112 px | **22.2** | 1.2 | 7.4 | 100% |

At the pixel level (blurred, grey, compensated difference image, as `MOD2_global` builds
it), frames 954–964 have **target-box p90 difference 61** against a **background p99 of
54**. The drone is among the loudest things in the frame.

### Where EXP-012a went wrong

1. **The wrong subtraction.** 4.5 px was the target's raw image motion. 19–34 px was the
   background's raw motion. The ~10 px "median to p90 spread" is mostly rotation and
   perspective, which a homography models. What it cannot model, the residual, has a
   median of **~1 px**. The target's motion relative to the background is roughly
   background motion minus target motion, about 22 px. The claim that "intercept geometry
   inverts GLAD's premise" has it backwards: a target the camera is holding still, over
   streaming ground, is exactly the differential motion a compensator exposes.
2. **The wrong passage.** "~955–964" was a guess. The drone is in view from frame 708, at
   17–142 px.

### Where GLAD actually loses it

Frame by frame through `MOD2_global`, with the stage where the labelled drone drops out
(257 frames, each against the one before it):

| Stage | `upstream` | `clutter` |
| --- | ---: | ---: |
| No blob on the target at all | 58 (all in 708–800, small and slow) | 58 |
| Blob fails the area / aspect test | 73 (2 too big) | 71 |
| Crowding: >50 candidates, upstream **returns nothing** | **126** | — |
| Crowding: ranked below the top 50 | — | **90** |
| Not rigid (`_moving_coherently`) | — | 15 |
| LeNet gate rejects it | — | 19 |
| Another candidate accepted first | — | 4 |
| **Found** | **0** | **0** |

- **In 199 of 257 frames the drone is in the difference image.** The motion branch does
  not miss it for lack of signal.
- **Crowding is the largest loss.** Frames carry a median ~115 candidates, so the ground
  residual is real clutter. But the target is *one of them*, not below them. `clutter`'s
  shape-based ranking (`candidate_score`: fill × squareness) puts it outside the top 50 in
  90 frames.
- **The shape test is the second.** 79% of those blobs are within the area range, so the
  0.6–3.0 aspect ratio rejects them. The likely cause, not verified, is a moving target's
  difference blob being the target plus its ghost, stretched along the motion. At 962 the
  on-target blob is 2,841 px² and is rejected on shape.

### The ~100 other blobs: how big, how close

`runs/sofa_o4/exp012b_motion_check/blob_census.py` writes every blob in `MOD2_global`'s binary
image to `blobs.csv` (67k rows). A *candidate* is a blob that passes upstream's area
(30–3,000 px²) and aspect tests. Frames carry ~276 blobs and a median **103** candidates.

- **They are small.** Background candidates are 90 px² median (p10 38, p90 430, p99 2,017),
  longest side 16 px median. 55% are under 100 px². They are the same size throughout the
  approach, 87–106 px² median in every stretch.
- **They are mostly far from the drone.** Median 557 px from its centre, and only 1.6%
  within 100 px. Per frame: a median of **1** within 100 px, 11 within 200 px, and the
  nearest at 87 px. They come from the terrain, not from a halo around the target.
- **The drone's blob is larger and stronger than the typical one, but not the largest or
  strongest.** Its area grows with the approach: 48 → 634 → 1,084 → 2,825 px² median, against
  a background median of ~90. Its mean difference is 18–46 against 14–22 for the background,
  but the background's p90 is 33–52. In the 126 frames where it is a candidate, its median
  rank is 13th by area, 12th by area × strength, and 30th by strength alone. It is in the
  top 10 by area in 46% of frames and ranked first in 3%. The current shape score ranks it
  about 66th.

**Seen, not only counted:** `draw_blobs.py` in the same directory renders
`blobs_708_964.mp4` (every labelled frame, 10 fps) and stills with zoom sheets for frames
750 / 850 / 925 / 958 / 962. Each blob is coloured by fate: grey rejected, orange
candidate, red top-10 by area, green on the drone. Judged by eye in those stills, not
counted: the large blobs are **tree canopies, roof edges, the fisheye rim and near ground
at the frame bottom**. Many small ones are **HUD elements the mask misses**: pitch-ladder
dots, side-scale ticks and chevrons, and the ALT / voltage / bitrate digits. At 962 the
drone's blob is a clean quadcopter outline, widened along its motion past the 3:1 aspect
limit. **In most of the passage (750, 850, 925) the drone is against trees or terrain, not
sky.** Only the last ~10 frames put it against open sky.

**Why static objects make blobs.** `why_static.py` measures, on the same five frames,
how far each pixel is still misaligned after GLAD's warp (dense Farneback flow from the
compensated previous frame to the current one) and how textured it is. It excludes the
drone, the HUD mask and the warp border. It also writes `residual_<frame>.jpg` heatmaps.

| Frame | Threshold | Misalignment in blobs / elsewhere (px) | Gradient in / out (grey/px) | Gradient × misalignment in / out |
| --- | ---: | ---: | ---: | ---: |
| 750 | 10 | 3.00 / 0.13 | 3.4 / 0.6 | 11.3 / 0.1 |
| 850 | 14 | 4.34 / 0.33 | 4.5 / 0.6 | 20.6 / 0.3 |
| 925 | 12 | 4.01 / 0.58 | 3.9 / 1.4 | 17.8 / 1.1 |
| 958 | 14 | 5.55 / 0.56 | 2.8 / 1.1 | 18.0 / 0.8 |
| 962 | 14 | 5.26 / 0.38 | 3.0 / 1.1 | 16.2 / 0.6 |

A difference-image value is roughly gradient × misalignment. Inside blobs that product
is 11–21 grey levels, just over the 10–14 threshold. Outside, it is about 1. Blobs appear
where both things are true: the warp leaves 3–5 px of error **and** the surface is
textured. The heatmaps show where that is:
- the tree line and canopies, which sit at a different depth from the ground the warp
  mostly fits;
- the near bunker, where parallax is largest;
- the fisheye rim, which a homography cannot model;
- the HUD, which is fixed to the screen. The warp moves it along with the scene, so it
  is misaligned by the whole camera motion.

This refines the earlier "residual ~1 px median": that median is dominated by sky and
far ground. Objects at other depths keep 3–5 px. It is still below the drone's
differential motion late in the approach, but it is concentrated on high-contrast
edges, which is what makes it look like a moving target.

So ranking alone does not isolate it. Ranking by size or by size × strength would keep it
inside a budget of 50 in most frames where the shape score drops it. Picking it out of the
~10 blobs still ahead of it needs something else, such as persistence over several frames
or distance from the last known position.

### Reading

- **EXP-012a's mechanism is retracted.** Motion is a live cue on this footage, and in the
  late approach it is a strong one.
- **Its practical conclusion still holds:** `--motion-profile clutter` found the drone in
  **0** frames, so it should still not be adopted. The reason is ranking and shape
  filtering, not physics, and both are fixable.
- **Next candidates**, cheapest first: rank candidates by difference strength rather than
  shape; loosen or drop the aspect test in the global branch; then re-run EXP-012a.
- **Caveats:** one clip, one approach, a target larger than this project's 10–30 px brief
  (median 72 px). The early small-target stretch (708–800) is the weakest: differential
  4.3 px/f, 58 frames with no blob at all. That stretch, not the close approach, is the
  one a long-range detector has to win. Follower boxes were used only for the per-frame
  pixel statistics and the stage trace, not for the headline differential.

---

## EXP-013 — GLAD on analog goggles footage: every box is the OSD, no drone found

- **Date:** 2026-09-18
- **Question:** asked directly: detect the drone in `catch 2` of the Sofa Base analog
  recordings and render an annotated video.
- **Model / weights:** identical to EXP-012a (GLAD released pipeline, `--pad released`).
  **Default motion profile** (vendored `MOD2`). EXP-012a said not to adopt `clutter`, so
  this run does not use it. **No `--scale`**: 960×720 is 0.545× the 1080p diagonal, so
  every motion constant is off by that factor, and EXP-011 showed `--scale auto` did not
  help on the FIELD capture.
- **Data:** `data/raw/SOFA-ANALOG/videos/catch_2.mp4`, 890 frames, 960×720 @ 30 fps. This is
  an **analog FPV** goggles recording, a different link from the O4 clips. Staged from
  `Downloads/EXP Sofa Base 24-08-26/processed/analog/catch 2.mp4` (MD5 `e89ab834…`).
  **No labels.** The source folder is mixed: `catch 1` is O4-format 2520×1080 and
  `catch 9` is 1280×720. Only the nine 960×720 clips were staged.
- **HUD mask:** `data/processed/SOFA-ANALOG/hud_mask.png`, built from those nine clips.
  **The O4 default `--white-level 225` does not work here**: it caught only the date strip,
  because analog OSD glyphs are soft grey, not saturated. `--white-level 180` covers the
  telemetry blocks and the date strip, with no sky masked (25,905 px, **3.75%**). It cannot
  cover the scrolling compass tape or the artificial-horizon dashes, which both move.
- **Hardware:** i7-1255U CPU. It ran at **5.1–5.3 fps** unimpeded. The overall 4,237 s
  (0.21 fps) comes from a stall between frames 500 and 750 (0.18 fps), most likely host
  sleep or contention. **Do not quote it as throughput.**
- **Command:** `py -3.13 -m src.glad_detect --videos data/raw/SOFA-ANALOG/videos --video-names catch_2 --record-all --pad released --hud-mask data/processed/SOFA-ANALOG/hud_mask.png --images data/processed/SOFA-ANALOG/images/test --out runs/sofa_analog/exp013_analog_catch2`

### Result

82 detections on 82 of 890 frames: `global yolo` 3, `local yolo` 48, `local mod` 31.
**All 82 were inspected as crops, and none is a drone:**

| Frames | Branch | What it is |
| --- | --- | --- |
| 363–401 | `global yolo` → `local yolo` (38) | the compass tape's **"N"** glyph. Moving HUD, so the mask cannot reach it |
| 402–837 | `local mod` (31) | telemetry digits (`7:100`, RSSI) at the edges of the masked cells |
| 769–811 | `global yolo` → `local yolo` (15) | **one dash of the artificial-horizon bar.** It sits in a row of identical unboxed dashes and steps 596 → 563 → 530 px, which is the dash spacing |
| 242–244 | `global yolo` → `local yolo` (3) | a ~40×26 px dark-and-white object on the ground. Clutter |
| 540 | `local yolo` (1) | an 8×6 px blob. Undetermined |

The 769–811 span needed a full-resolution look to call: in a 230 px crop it resembles a
small quadcopter against sky. Seen at frame scale beside its neighbours, it is plainly the
horizon bar.

- **Correction (2026-09-18, EXP-014):** there *was* a drone to find. It is in open sky for
  roughly **frames 510–585**, closing from a speck to ~25 px, and this run emitted **no box
  on it**. During that span GLAD was locked onto telemetry digits: the `local yolo` at 540
  and the `local mod` hits at 542, 563 and 578. The 8-frame scan below skipped the whole
  passage. It stands as a record of why an eyeball negative is not evidence.
- **What this does *not* establish:** that there was no drone to find. The clip was
  scanned by eye at 8 frames (600–870), and no airframe was visible at that scale. By the
  standing warning in `datasets.md`, that is **not evidence of absence** at these sizes.
  `catch` is a trial outcome, not a detection label.
- **What it does establish:** on analog footage, the burned-in OSD dominates GLAD's output
  more thoroughly than it did on O4. The moving elements are the failure: the compass
  letters and the horizon dashes are high-contrast, drone-sized glyphs against sky. A
  static mask cannot fix that.
- **Artefacts (gitignored):** `runs/sofa_analog/exp013_analog_catch2/overlay.mp4` (unscored,
  `--zoom 3 --zoom-span 100`), `all_hits.png` (all 82), `sky_769_811.png`,
  `ground_242_244.png`, plus the mask preview `runs/sofa_analog/exp013_hud_preview.png`.
- **Next:** locate the target in `catch_2` by hand (`/annotate`) before any further run
  on this footage. Without a known passage, a miss and an empty clip look identical. If a
  target passage exists and GLAD misses it, this becomes the same story as EXP-012a, and
  fine-tuning (M7) is again the answer, not masking.

## EXP-014 — the same clip with the moving OSD vetoed: the drone found, once

- **Date:** 2026-09-18
- **Question:** EXP-013's 82 detections were all OSD. If the moving OSD is vetoed too, does
  GLAD find the target?
- **Model / weights / data:** identical to EXP-013. `catch_2`, 890 frames, 960×720,
  `--pad released`, default motion, no `--scale`.
- **Changes:**
  - **Block mask.** `data/processed/SOFA-ANALOG/hud_mask.png` was rebuilt with
    `--white-level 180 --block-fraction 0.08 --picture-rows 150:530`: OSD text blocks
    outside the picture rows, each filled to its rectangle, 13.15% of the frame.
  - **`--osd-twins`.** A box is vetoed if it has an identical copy 0.5–2.5 OSD columns
    away, at a score of at least 0.70.
  - Both are described in [hud_mask.md](hud_mask.md#analog-osd).
- **Hardware:** i7-1255U CPU, 255.8 s, **3.48 fps**. The pytest suite ran concurrently for
  part of the run.
- **Command:** `py -3.13 -m src.glad_detect --videos data/raw/SOFA-ANALOG/videos --video-names catch_2 --record-all --pad released --hud-mask data/processed/SOFA-ANALOG/hud_mask.png --osd-twins --images data/processed/SOFA-ANALOG/images/test --out runs/sofa_analog/exp014_analog_catch2_osd`

### Result

| | EXP-013 | EXP-014 |
| --- | ---: | ---: |
| Detections | 82 | **8** |
| on OSD | 81 | **2** |
| **on the drone** | **0** | **1** (frame 547) |
| `local yolo` frames | 48 | 2 |
| `global miss` | 45.6% | 83.6% |

All 8 were inspected at frame scale:

| Frame | Branch | What it is |
| --- | --- | --- |
| **547** | `global yolo` | **the target.** A dark multirotor silhouette in open sky, confirmed by stepping 540–552 |
| 242–244 | `global yolo` → `local yolo` | ground clutter, as in EXP-013 |
| 483 | `global yolo` | a horizon-height blob on an analog breakup frame. Undetermined |
| 555 | `local mod` | a building edge, the motion branch's fallback after 547 |
| 811 | `global yolo` | a horizon dash whose twins scored below 0.70 on this frame |
| 822 | `local mod` | the RSSI glyph just below the telemetry block's rectangle |

- **The veto worked on both sides.** Of the 81 OSD false alarms, 79 are gone. Freed from the
  telemetry locks EXP-013 sat in, the global detector reached the target, which **no
  earlier run on this footage had boxed**.
- **The veto did not reach the drone.** Its twin score was measured on its own positions:
  **0.42, 0.59, 0.63** at frames 560, 570 and 580, against a 0.70 veto. By 580 it is flying
  level with the horizon dashes, so **the margin there is 0.07**. The dashes themselves
  scored 0.58–0.95, so the two populations overlap, and 0.70 sits in the overlap.
- **Recall is still ~1 frame in ~75.** The target is visible for roughly frames 510–585,
  judged by eye. GLAD boxed it once and did not hold it: `local yolo` never locked on, and
  the motion fallback wandered to a building at 555. **Nothing here is a score:** the
  passage bounds are eyeballed and there are no labels.
- **Next:** label 510–585 with `/annotate`, so this clip scores against real ground truth.
  The appearance branch is the gap, as in EXP-012a on O4. A 25 px dark quad against sky is
  the opposite of GLAD's white-Phantom-over-ground training prior (glad-model.md §5b), so
  fine-tuning (M7) remains the remedy. `--invert` (EXP-012) is the cheap probe to try first
  on this passage.
- **Artefacts (gitignored):** `runs/sofa_analog/exp014_analog_catch2_osd/overlay.mp4`, `all_hits.png`
  and the mask preview `runs/sofa_analog/exp014_hud_preview.png`.

---

## EXP-015 — edge-normalised frame difference on `first_catch`

- **Date:** 2026-09-22
- **Question:** EXP-012b found that static objects make blobs because the difference
  image is roughly gradient × misalignment. The single homography leaves 3–5 px of
  misalignment on textured edges at other depths. Dividing the compensated difference by
  the local gradient should turn it into roughly "pixels of displacement". Static edges
  would then read their misalignment (3–5 px) and the drone its differential motion
  (4–22 px). Does that isolate the drone better than the plain difference?
  Is motion worth pursuing further before a learned model?
- **Model / weights:** none. GLAD's grid-KLT homography, reproduced in
  `common.homography` with H returned. It was checked **bit-for-bit identical** to
  `Functions.motion_compensate`'s warp on frame 849→850.
- **Data:** `data/processed/SOFA-O4/videos/first_catch.avi`, 1440×1080. Labels:
  `annotations/first_catch.json`, 257 drone frames (708–964, 67 hand-placed and 190
  follower). Empty frames 2–707, every second one (353 frames), for the false-alarm load.
- **Hardware / cost:** i7-1255U CPU. The run took ~9.5 min for all 20 maps plus the peaks.
  One normalised map costs well under 0.1 s/frame on top of the homography.
- **Scripts (gitignored, throwaway):** `runs/sofa_o4/exp015_normalised_motion/`.
  - `common.py`: homography, masks, maps and peaks.
  - `collect.py`: writes `peaks.pkl`.
  - `score.py`: `MASK=none|props|fixed|strict`, writes `score_<mask>.csv`.
  - `categorise.py`: where the false peaks sit.
  - `draw.py`: the stills.

  All run from the repo root with `PYTHONPATH=".;runs/sofa_o4/exp015_normalised_motion"`.

### Method

- **Frame pair** (n−1, n), warped with GLAD's homography (fitted on MOD2's 11×11-blurred
  grey frames). D = |current − warped previous|.
- **Maps.**
  - `plain_b{11,5}`: D, at MOD2's 11×11 blur and at a lighter 5×5.
  - `norm_b*_g{cur,max}_e{1,2,4}`: D / max(G, ε). G is the Sobel gradient in grey
    levels/px, taken either from the current frame or as the max of both frames.
    ε is the floor, 1, 2 or 4 grey/px.
  - `win_b*_e*`: sqrt(Σ D² / (Σ G_max² + ε²)) over a 9×9 window. This is the
    least-squares normal-flow magnitude, the cheap Lucas–Kanade-style version.
  - `dis_b5`: dense DIS optical-flow magnitude from the warped previous frame to the
    current one, as a flow residual.
- **Candidates are peaks,** not blobs: local maxima with a 15 px radius, strongest
  first, so ranks do not depend on a threshold.
- **Match criterion:** a peak is "on the drone" if it lies inside the label box grown by
  max(10 px, 25% of its longest side). That margin covers the ghost at the previous
  position. The drone's rank is 1 + the number of off-drone peaks stronger than its best
  on-drone peak.
- **Operating point:** the threshold τ10 gives **10 candidates per empty frame** (1–707)
  for each map. The table reports "found" (drone peak ≥ τ10) and candidates per drone
  frame at that τ, so every map carries the same empty-frame false-alarm load.
- **Masks, in order of strength:**
  - **Always on:** the warp border (eroded 8 px), a 24 px edge margin, and the HUD mask
    dilated 7 px.
  - `fixed`: also drops screen-fixed clutter. These are pixels where `plain_b11`'s top-50
    peaks land in ≥5% of the empty frames (2.7% of the frame before a 4 px dilation). It is
    calibrated only on frames without the drone.
  - `strict`: a **headroom estimate only.** It adds a 60 px edge band and everything
    within 30 px of an overlay, 42% of the frame. It removes half the drone box in
    901–964, and it was chosen after seeing this clip.

### First finding: the biggest clutter is not the scene

With only the HUD mask, `plain_b11`'s τ10 is **126 grey levels**, far above anything
terrain produces. The top peaks sit in two places, both **fixed to the screen**:
- **the host drone's own propeller blades**, visible at the left and right edges at
  y≈500–620;
- **the pitch ladder dashes and HUD digits** that the mask misses.

See `screen_fixed_800.jpg`. Every number below uses the `fixed` mask unless it says
otherwise. The drone touches that mask in 43 labelled frames (877–891, 926–930 and
others, near the ladder). The pitch ladder moves with pitch, so a fixed-position mask
cannot catch all of it.

### Results (`fixed` mask)

Each cell reads top-1 / top-5 / top-10, then the median rank.

| Frames | Box (median) | `plain_b11` (MOD2 difference) | `norm_b11_gmax_e2` | `win_b11_e4` (windowed) | `norm_b5_gmax_e4` |
| --- | ---: | --- | --- | --- | --- |
| 708–800 | 43 px | 0 / 0 / 0, 183 | 0 / 0 / 0, 213 | 0 / 0 / 0, 102 | 0 / 0 / 0, 98 |
| 801–900 | 87 px | 0 / .04 / .06, 41 | .01 / .11 / .22, 20 | 0 / .13 / .40, 14 | 0 / .09 / .23, 19 |
| 901–953 | 104 px | .09 / .32 / .42, 14 | .21 / .47 / .68, 6 | 0 / .06 / .32, 14 | .19 / .40 / .55, 8 |
| 954–964 | 112 px | 0 / 0 / .09, 18 | .09 / .55 / 1.00, 5 | 0 / 0 / .09, 17 | 0 / .09 / .82, 9 |
| **708–964** | 72 px | **.02 / .08 / .11, 56** | **.05 / .16 / .27, 29** | .00 / .06 / .23, 27 | .04 / .12 / .24, 25 |

At the same load of **10 candidates per empty frame**:

| Frames | `plain_b11` found / cands per drone frame | `norm_b11_gmax_e2` found / cands |
| --- | --- | --- |
| 708–800 | 0.00 / 15.3 | 0.00 / 7.6 |
| 801–900 | 0.21 / 29.8 | 0.23 / 13.0 |
| 901–953 | 0.91 / 33.9 | 0.87 / 16.8 |
| 954–964 | 1.00 / 46.1 | 1.00 / 53.6 |
| **all** | **0.31 / 26.1** | **0.31 / 13.6** |

- **Other variants.**
  - Gradient from the max of both frames beats the current frame alone in every setting.
    With the current frame only, the ghost next to the drone is amplified, and so is
    every canopy-against-sky edge.
  - Lighter blur (5×5) does not rescue the early stretch.
  - Without the `fixed` mask, every map is worse: top-10 is 0.09 plain vs 0.15 normalised.
- **Dense flow (`dis_b5`) is useless here:** top-10 is 0 in every stretch. DIS invents
  large flow on low-texture sky and on the warped border (top off-drone peak 137 px
  against the drone's 5.7). Farneback costs 1.2 s/frame at full resolution and was not
  run over the clip.
- **Headroom under `strict`:** top-10 overall rises to 0.49 normalised and 0.28 plain.
  The last two stretches reach median rank 1. **708–800 stays at 0–1%.**

### What the stills show

`still_<n>.jpg` has four panels. Top: the top-10 peaks, plain on the left and normalised on
the right, with green marking a peak on the drone. Bottom: the two heatmaps.
`zoom_<n>.jpg` crops the frame and both maps around the label. I looked at all of them.

- **Static textured clutter is suppressed.** At 850 and 925, the shed's corrugated roof,
  the tree canopies and the far tree line are the brightest things in the plain
  difference, and they are nearly black in the normalised map. That is the predicted
  effect. The windowed map puts a number on it: under `strict`, the strongest off-drone
  scene peak reads **4.8–6.0 px** from frame 801 on (8.1 px in 708–800), close to
  EXP-012b's 3–5.5 px misalignment.
- **What survives normalisation:**
  - the edges of the prop blades just outside the mask;
  - the left edge of the picture;
  - the **pitch-ladder dashes**, which at 925 and 962 sit a few tens of px from the drone;
  - HUD digits;
  - canopy tops against sky, where gradient is low and D is not.

  Of the top-10 off-drone normalised peaks on drone frames, **40% are within 60 px of
  the frame edge and 29% within 30 px of an overlay.** Only 32% are in open scene.
- **962:** normalised puts peaks on the drone as N1 and N6. Plain puts its top 10 on the
  ladder, the canopy and the props.
- **750, the long-range case:** the drone does not show in either map (`zoom_750.jpg`).
- **600, empty:** the top peaks are the prop edges and near ground at the bottom left.

**Video:** `overlay_exp015.mp4` (`draw_video.py`, same directory) covers frames 650–964
at 10 fps, starting 58 drone-free frames before the drone appears. Top row: the top-10
peaks and the drone's rank, plain vs `norm_b11_gmax_e2`, under the `fixed` mask. Bottom
row: the two heatmaps, plain scaled to 40 grey levels and normalised to 10 px.

### Displacement read-out (hand-placed boxes only)

The windowed map reads the drone at **1.4 / 4.4 / 4.6 / 8.1 px** over the four stretches.
EXP-012b measured 4.3 / 10.1 / 13.0 / 22.2 px/frame of true differential motion, so the
drone reads **2–3× low**. Normal flow under-reads once displacement exceeds the blurred
edge width, and a window averages over flat pixels. The map is a usable ranking score. It
is not a calibrated velocity.

### Reading

- **Normalisation works as designed, and it helps.** At an equal empty-frame load it
  **halves the candidates carried in drone frames** (26 → 14). It **roughly halves the
  drone's median rank** (56 → 29) and **doubles or better top-5 and top-10** (0.08 → 0.16,
  0.11 → 0.27). It does this for about the cost of a Sobel. The mid and late approach
  (801–964) is where it pays: top-10 is 0.68 at 901–953 and 1.00 at 954–964.
- **It does nothing for 708–800,** the long-range stretch this project cares most about:
  **0 of 93 frames in the top 10 for every variant.** The drone reads 20 grey levels plain
  and ~1.4 px windowed, against a top clutter peak of 109 grey levels / 8 px. With two
  frames and this blur, the small, slow drone is not in the motion signal. That agrees with
  EXP-012b's 58 frames without a blob, all in this stretch.
- **The remaining false alarms are mostly own-airframe and overlay, not scene.** This
  needs a mask fix (prop blades, and a pitch-ladder mask that moves with it), not a
  better motion cue.
- **Caveats:**
  - One clip and one close approach. The median drone is 72 px, above the 10–30 px brief.
  - The `fixed` mask is calibrated on this clip's own empty frames, so it is in-sample for
    the overlay layout, though not for the drone. `strict` is tuned on this clip and is a
    ceiling, not a result.
  - Ranks and top-k use follower boxes as well as hand-placed ones. Displacement claims
    use hand-placed boxes only.
  - No `detections.jsonl` was written. Peaks have no extent, so a P/R at an IoU threshold
    would mean nothing; the match criterion above stands in for it.
- **Next:**
  1. Rank the global branch's candidates by windowed or normalised difference instead of
     `candidate_score`. Add the prop and pitch-ladder masks. Re-score `first_catch`
     (CPU, about an hour, under the open todo).
  2. For long range, two-frame motion is not the answer on this evidence. Try multi-frame
     accumulation over 708–800 before paying for a learned model.
  3. Before choosing or training any learned model (YOLOMG, which is GPL-3.0, or a GLAD
     fine-tune), label more O4 clips with long-range spans. The whole evaluation set is
     one close approach.
