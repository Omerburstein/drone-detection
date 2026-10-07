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

> **Label note (2026-09-28).** The three spellings of this label -- `no target`,
> `no target in frame` and `no_target` -- were unified to **`no target`**. Rows above
> predate that and are left as the runs printed them; a re-run today prints
> `no target` wherever they say otherwise.

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

> **Label note (2026-09-28).** The three spellings of this label -- `no target`,
> `no target in frame` and `no_target` -- were unified to **`no target`**. Rows above
> predate that and are left as the runs printed them; a re-run today prints
> `no target` wherever they say otherwise.

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

> **Label note (2026-09-28).** The three spellings of this label -- `no target`,
> `no target in frame` and `no_target` -- were unified to **`no target`**. Rows above
> predate that and are left as the runs printed them; a re-run today prints
> `no target` wherever they say otherwise.

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

> **Label note (2026-09-28).** The three spellings of this label -- `no target`,
> `no target in frame` and `no_target` -- were unified to **`no target`**. Rows above
> predate that and are left as the runs printed them; a re-run today prints
> `no target` wherever they say otherwise.
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

---

## EXP-016 — multi-frame local-ring differential motion, stage 1: the bars are not met

- **Date:** 2026-09-23
- **Question:** EXP-015 ended with the long-range drone outside the top 10 of every
  two-frame motion map on `first_catch` 708–800. The user's observation was that over
  10–15 frames it becomes obvious which blob is the drone. Does verifying each motion peak
  over time — by its motion **relative to its own local background ring**, with a
  sequential confirmation test — find the target at an affordable false-alarm rate?
  Stage 1 is a falsification run in `runs/`, deliberately before any code lands in `src/`.
- **Model / weights:** none. No learned component anywhere. GLAD's grid-KLT homography
  (`runs/sofa_o4/exp015_normalised_motion/common.py::homography`, unchanged) plus OpenCV
  pyramidal Lucas–Kanade.
- **Data, both clips carrying equal weight:**
  - **O4** `data/processed/SOFA-O4/videos/first_catch.avi`, 1440×1080, 964 frames.
    Labels `annotations/first_catch.json`: 257 drone frames (708–964; 67 hand-placed, 190
    follower). Empty frames 2–707 = **23.5 s**.
  - **Analog** `data/raw/SOFA-ANALOG/videos/catch_2.mp4`, 960×720, 890 frames.
    Labels `annotations/catch_2.json`: 224 drone frames in five spans (491–591, 647–678,
    682–683, 687–735, 746–785), median box 37 px. Empty frames = 665 = **22.2 s**.
  - **No split rule applies** — nothing is trained. z\* is calibrated on empty frames only,
    two-fold, frozen before any drone frame is scored, separately per clip.
- **Hardware / cost:** i7-1255U CPU, no GPU. `collect` 406 s (O4) / 202 s (analog); one
  `verify` pass 4–20 min (O4) / 45–110 s (analog) depending on z\*; 7 ladder passes per
  clip plus the frozen pass, ~3 h wall clock in total, all backgrounded.
- **Scripts (gitignored, throwaway):** implementation in
  `runs/sofa_o4/exp016_multiframe/`, clip settings in
  `runs/sofa_analog/exp016_multiframe/clipcfg.py`. Every pixel constant is quoted at
  1440 px wide and scaled by 960/1440 for analog, so analog is the same rule read on a
  smaller picture rather than a re-tuned one.
  - `mf.py` — masks, seeds, background flow, the `Hypothesis`, the speed cap.
  - `collect.py` — per consecutive pair: H, 200 seeds, the grid flow. `win_b5_e4` is
    asserted **bit-identical** to EXP-015's `maps()` on a sample frame in every run.
  - `verify.py` — the tracker, one pass per z\*.
  - `calibrate.py` — two-fold z\* on empty frames. `sweep.py` — the operating curve.
  - `score.py`, `falsecheck.py`, `longbase.py`, `draw_video.py`, `stills.py`,
    `seedcheck.py`, `ringcheck.py`, `screenfixed.py`, `interlace_check.py`.
  - Run from the repo root, e.g.
    `PYTHONPATH="runs/sofa_analog/exp016_multiframe;runs/sofa_o4/exp016_multiframe;runs/sofa_o4/exp015_normalised_motion;." py -3.13 -m verify --zstar 20 --tag frozen`

### Method

1. **Seeds, blind.** The top 200 peaks per frame of EXP-015's `win_b5_e4`, anywhere in the
   valid frame. No centre prior — stage 1 is the acquisition question. Masks: the warp
   border, a 24 px edge margin, the clip's HUD mask dilated 7 px, and the screen-fixed map.
2. **Background flow.** GLAD's homography plus forward-backward LK on a 24 px grid. A grid
   point's **residual** is its tracked motion minus the homography's prediction for it.
3. **Hypothesis update.** Each live hypothesis is LK-tracked with a forward-backward check.
   Its **differential vector** is its own motion, minus the homography prediction, minus
   the median residual of grid points in its **25–120 px ring**. Accumulated as
   `z_k = |Σd| / sqrt(Σ_j (1.48·ring_MAD_j + 0.3 px)²)`, which equals the plan's
   `|Σd| / (√k·σ)` for a constant ring.
4. **Confirm** at the first `k ≥ 3` with `z ≥ z*` and `|Σd| ≥ 6 px`. **Kill** at `k = 15`
   (the user's 0.5 s ceiling), after two forward-backward failures, or on the speed cap.
   A confirmed track is held by LK with a template re-lock and dropped when its trailing
   15-frame z has sat below `z*/2` for 10 frames.
5. **Overlay veto.** Raw image motion < 0.5 px/f while the ring flows > 3 px/f ⇒
   screen-fixed. On analog, EXP-014's OSD-twin test is applied at confirmation as well.
6. **Speed cap, in physical units.** A 0.6 m airframe at ≤30 m/s, worst case 60 m/s
   closing, at 30 fps bounds image motion at **`2·v_max/(fps·size_m) = 3.33 × apparent
   size px` per frame**. The ratio is lens-, range- and resolution-independent, so the same
   number serves O4 and analog with no scaling — the code derives it from
   `target_size_m`, `v_max_ms`, `closing_factor` and the frame rate rather than storing a
   pixel threshold. It also sets the KLT re-lock gate and the seed association gate per
   hypothesis.
7. **Match criterion**, unchanged from EXP-015: a position is on the drone if it lies
   inside the label box grown by max(10 px, 25% of its longest side).

### Pre-flight on analog: not interlaced, but KLT does suffer

`interlace_check.py`. Median vertical/horizontal neighbour difference **1.33**, and
decimating vertically *raises* the neighbour difference (ratio 0.42) rather than lowering
it — the comb signature of interlace is absent. Row-band frame differences show no seam.
The 1.33 is analog's horizontal low-pass along the scanline, not two fields.

**Forward-backward failure rates**, reported separately from mask rejections because they
mean different things:

| | grid FB failure | grid FB error, median frame | hypothesis FB failure |
| --- | ---: | ---: | ---: |
| O4, gate 1.00 px | **46.1%** | 0.013 px | **17.5%** |
| analog, gate 0.67 px | **49.2%** | 0.218 px | **35.2%** |

Analog's median forward-backward error is **17× O4's**, which is the grain the plan warned
about. On the fixed grid the *failure rate* is nearly the same on both clips, because both
distributions are bimodal — a large population tracks to a hundredth of a pixel and a large
population does not track at all. Where the difference shows is on the **hypotheses**,
which sit on motion peaks rather than on an arbitrary grid: **35.2% of analog hypothesis
tracks fail the check against 17.5% on O4**, so analog grain costs a tracker twice as many
frames of evidence. Masked and wild tracks account for a further 18.2% / 10.0% (analog) and 9.2% /
4.2% (O4). About 39% (analog) and 49% (O4) of grid points survive, which leaves 20–65
usable ring points per hypothesis.

### The premise, re-derived rather than trusted

`ringcheck.py`, from the labels on `first_catch` 708–800:

| label source | raw motion | after H | after the ring | ring σ | z per frame |
| --- | ---: | ---: | ---: | ---: | ---: |
| hand-placed, ≤3 frames apart (n=10) | 4.10 | 4.31 | **4.23** | **0.63** | **7.7** |
| every consecutive label, follower included (n=92) | 3.57 | 0.84 | 0.87 | 0.47 | 1.1 |

The design pass's ~9σ is **confirmed on hand-placed labels** (4.2 px/frame against a
0.63 px ring scatter) and **not reproduced on follower labels**, where the drone reads
0.87 px/frame. The follower smooths and lags, so its frame-to-frame displacement
under-reads the target's true motion by ~5×. Any future differential-motion number on this
project must say which labels it used.

**Seed coverage, re-derived** (`seedcheck.py`, consecutive frames, top 200 of `win_b5_e4`):
O4 708–800 is **76%**, not the 92% the design pass quoted; 91% over 708–964. Analog is
**86%** on 491–591 and 77% overall. The 92% figure was the whole drone stretch, not the
long-range part of it. A seed usually exists; it is ranked 19–95 at the median.

### z\* calibration, empty frames only, two-fold

Empty frames split into two contiguous halves; z\* tuned on one and counted on the other,
both ways round; frozen at the more conservative of the two before any drone frame was
scored. Selection rule fixed in advance: the smallest ladder value at or under 5 confirmed
false tracks per minute.

**Confirmed false tracks per minute, by fold:**

| z\* | O4 fold A | O4 fold B | analog fold A | analog fold B |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 953.5 | 912.7 | 189.8 | 243.2 |
| 6 | 719.0 | 387.5 | 70.5 | 81.1 |
| 8 | 545.6 | 244.8 | 38.0 | 59.5 |
| 12 | 326.3 | 153.0 | 10.8 | 10.8 |
| 16 | 219.3 | 96.9 | 5.4 | 5.4 |
| 20 | 96.9 | 40.8 | 5.4 | 0.0 |
| 25 | 56.1 | 20.4 | — | — |
| 30 | 20.4 | 15.3 | — | — |
| 40 | 15.3 | 0.0 | — | — |
| 60 | **0.0** | **0.0** | — | — |

- **Analog freezes at z\* = 20**, held-out rate **5.4 false tracks/min** (one track in a
  fold). One track moves the rate by 5.4/min, so the calibration is coarse: 22 s of empty
  footage is what this clip has.
- **O4 freezes at z\* = 60**, the first ladder value at which fold A is clean; fold B is
  clean from z\* = 40. Held-out rate **0.0 false tracks/min**. But at z\* = 60 **the drone
  never confirms either** — its maximum z anywhere on the clip is 53, and on 708-800 it is
  **13.9**. The threshold the false-alarm budget demands is **four times the largest value
  the long-range target ever produces**. That single sentence is the result.
- The analog ladder was stopped at z\* = 20 once the budget was met; the O4 ladder ran to
  60.

**At the frozen z\* = 60, O4 confirms one track in the entire clip and none of it is the
drone:**

| stretch | n | coverage | fragments | top-1 | top-5 | **top-10** | median rank | seen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **708-800** | 93 | 0.00 | 0 | 0.00 | 0.12 | **0.24** | 12 | 0.66 |
| 801-900 | 100 | 0.00 | 0 | 0.05 | 0.15 | 0.28 | 14 | 0.93 |
| 901-953 | 53 | 0.00 | 0 | 0.57 | 0.91 | 0.92 | 1 | 1.00 |
| 954-964 | 11 | 0.00 | 0 | 0.00 | 1.00 | 1.00 | 2 | 1.00 |
| 708-964 | 257 | 0.00 | 0 | 0.14 | 0.33 | 0.43 | 10 | 0.85 |

### Analog `catch_2` at the frozen z\* = 20

47,147 hypotheses born, **4 confirmed tracks in the whole clip**, 3 vetoed by the OSD-twin
test, 1 confirmed false track = **2.7 per minute**.

| stretch | n | coverage | fragments | top-1 | top-5 | top-10 | median rank | seen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **491–591** | 101 | **0.15** | 2 | 0.33 | 0.42 | 0.43 | 2 | 0.75 |
| 647–678 | 32 | 0.00 | 0 | 0.00 | 0.16 | 0.22 | 35 | 0.56 |
| 687–735 | 49 | 0.00 | 0 | 0.02 | 0.12 | 0.27 | 34 | 0.61 |
| 746–785 | 40 | 0.00 | 0 | 0.00 | 0.02 | 0.02 | 36 | 0.70 |
| 491–785 | 224 | 0.07 | 2 | 0.15 | 0.24 | 0.29 | 22 | 0.69 |

First confirmed on-drone frame on the primary span: **567, i.e. +76 frames (2.53 s)**
after the target's first labelled frame.

**Verdict against the pre-set analog bars:**

| bar | result | |
| --- | --- | :-: |
| confirmed within 30 frames of 491 | +76 frames | **FAIL** |
| ≥ 40% coverage of a span | 15% on 491–591, 0% elsewhere | **FAIL** |
| false tracks per minute reported | 2.7/min | reported |

### The operating curve, which is what the single point cannot show

`sweep.py`, primary span 491–591. **z\* was frozen on empty frames; this table exists to
show the shape of the trade, not to pick a better point afterwards.**

| z\* | FA/min | acquisition | coverage | fragments | top-10 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 216.5 | +17 | 0.49 | 5 | 0.57 |
| 6 | 75.8 | +17 | 0.49 | 5 | 0.57 |
| 8 | 48.7 | +43 | 0.30 | 3 | 0.50 |
| 12 | 10.8 | +28 | 0.30 | 4 | 0.52 |
| 16 | 5.4 | +78 | 0.09 | 2 | 0.41 |
| **20 (frozen)** | **2.7** | **+76** | **0.15** | 2 | 0.43 |

Both analog bars *are* reachable — but not together, and not at the false-alarm budget.
Acquisition within 30 frames needs z\* ≈ 12, which costs **10.8 FA/min**; 40% coverage
needs z\* ≤ 6, which costs **76 FA/min**, 15× the budget.

### The comparison arm: a longer baseline makes it worse

`longbase.py`, frame n against n−k through chained homographies, ranked exactly as
EXP-015 ranks peaks. Analog 491–591, drone in the top 10 of the map:

| k | 1 | 3 | 6 |
| --- | ---: | ---: | ---: |
| top-10 | **0.19** | 0.02 | 0.00 |
| median rank | 29 | 49 | 138 |

Chaining homographies accumulates misalignment faster than the drone accumulates
displacement, so the cheap multi-frame arm is worse than the two-frame one it was meant to
improve. That is the negative result which justifies paying for per-hypothesis tracking
rather than a wider difference.

### O4 `first_catch`, at the threshold where it does acquire

The frozen z\* = 60 table is above and it is all zeros. The table below is at **z\* = 6**,
the point where the drone is acquired on 708–800 at all. It runs at **553 false tracks per
minute, 110× the budget**, and is reported so the failure has a shape — not as an operating
point, and not as a number to quote.

| stretch | n | coverage | fragments | top-1 | top-5 | top-10 | median rank | seen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **708–800** | 93 | **0.37** | 2 | 0.00 | 0.12 | **0.30** | 12 | 0.68 |
| 801–900 | 100 | 0.63 | 9 | 0.00 | 0.25 | 0.39 | 13 | 0.94 |
| 901–953 | 53 | 0.91 | 22 | 0.25 | 0.49 | 0.87 | 6 | 1.00 |
| 954–964 | 11 | 1.00 | 5 | 0.00 | 0.91 | 0.91 | 2 | 1.00 |
| 708–964 | 257 | 0.61 | 34 | 0.05 | 0.28 | 0.48 | 9 | 0.86 |

First confirmed on-drone frame: **742**, i.e. +34 frames (1.13 s) after the drone appears.

**The operating curve** (`sweep.py`, primary span 708–800):

| z\* | FA/min | acquisition | coverage 708–800 | top-10 708–800 |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 933.1 | frame 708 (+0) | **0.57** | 0.00 |
| 6 | 553.3 | frame 742 (+34) | 0.37 | 0.30 |
| 8 | 395.2 | frame 759 (+51) | 0.12 | 0.24 |
| 12 | 239.7 | never | 0.00 | 0.19 |
| 16 | 158.1 | never | 0.00 | 0.05 |
| 20 | 68.8 | never | 0.00 | 0.27 |
| 25 | 38.2 | never | 0.00 | — |
| 30 | 17.8 | never | 0.00 | 0.22 |
| 40 | 7.6 | never | 0.00 | — |

The drone's z on 708–800 peaks at **13.9**; the surviving false tracks reach **62**. There
is no threshold at which the long-range drone is above the clutter on this clip.

### What survives: the false tracks, named

`falsecheck.py`, classifying each confirmed false track by where it confirmed.

**Analog**, at z\* = 4 so there are enough to characterise (80 tracks):

| where | share |
| --- | ---: |
| upper half — sky and the horizon tree line | **57.5%** |
| frame edge | 16.2% |
| within 30 px of an overlay | 13.8% |
| lower half — ground | 12.5% |

Their median OSD-twin score is **0.42 and none reaches the 0.70 veto**, so the moving OSD
is *not* what survives — the twin test had already removed 54 candidates earlier in the
pass. What survives is the **horizon**: a depth discontinuity where the ring straddles sky
(no parallax) and near tree line (several px), so the ring median no longer describes the
hypothesis's own background and the residual it leaves looks like differential motion.
At the frozen z\* = 20 one false track remains, and it is at the horizon.

**O4**, at z\* = 20 (27 tracks): **59.3% confirm within 30 px of a burned-in overlay**,
18.5% sky/horizon, 14.8% frame edge, 7.4% ground. The pitch ladder moves with pitch, so
the static screen-fixed mask cannot hold it, and **81% of these score at or above the 0.70
OSD-twin veto** — which is not applied on O4.

**But that veto does not transfer.** Applying it post hoc (`twinveto.py`) across the whole
ladder:

| z\* | FA/min | FA/min with the veto | on-drone tracks | kept |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 553.3 | 247.3 | 34 | 10 |
| 20 | 68.8 | 12.7 | 9 | 6 |
| 25 | 38.2 | **5.1** | 7 | 5 |
| 30 | 17.8 | 5.1 | 2 | 1 |

It cuts the false rate 5–8× and would reach the budget at z\* ≈ 25 — but the **on-drone
confirmed tracks score a median 0.76 on the same test**, above the veto. **On analog the
same measurement comes back 0.25 median, 0.68 max**, comfortably under it, and the veto
costs no on-drone track at any z\*. The test is safe where there is a character grid and
unsafe where there is not, which is the distinction to carry forward. O4's HUD is a
digital overlay with no MAX7456 character grid, so "one character column away" is not a
meaningful distance there and the test fires on scene texture. The missing veto is real,
the available veto is not the one to use, and that is a prerequisite this experiment
cannot claim as solved.

O4, the same arm (`longbase.py`), drone in the top 10:

| k | 708–800 | 801–900 | 901–953 | 954–964 | seed present, 708–800 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | **0.00** | 0.22 | 0.08 | 0.00 | 0.76 |
| 3 | 0.00 | 0.11 | 0.00 | 0.91 | 0.71 |
| 6 | 0.00 | 0.03 | 0.02 | 0.45 | 0.40 |

Same conclusion on both clips: the long baseline is worse, and it is worse because chained
homographies drift. Only the final 11 frames, where the target is 112 px and moving 22
px/frame, benefit from k = 3.

### Verdict against the pre-set bars

Every bar was written into the plan before any of this ran.

| clip | bar | result | |
| --- | --- | --- | :-: |
| O4 | first confirmation by frame 760 | never confirms at z\* = 60; frame 742 at z\* = 6 (553 FA/min) | **FAIL** |
| O4 | ≥ 50% coverage of 708–800 | 0% at z\* = 60; 37% at z\* = 6, 57% at z\* = 4 | **FAIL** |
| O4 | ≤ 5 false tracks/min | 0.0/min at z\* = 60 — but only because nothing confirms at all | pass, vacuously |
| O4 | top-10 in 708–800 ≥ 30 pts above EXP-015's 0% | **0.24** at z\* = 60; 0.30 at z\* = 6 | **FAIL by 6 pts** |
| analog | confirmed within 30 frames of 491 | +76 frames at z\* = 20; +28 at z\* = 12 (10.8 FA/min) | **FAIL** |
| analog | ≥ 40% span coverage | 15% at z\* = 20; 49% at z\* = 6 (76 FA/min) | **FAIL** |
| analog | false tracks/min reported | 2.7/min at the frozen z\* | reported |

**Stage 1 does not pass. Stage 2 is not authorised by this result.**

### What did work, and it is not nothing

- **The drone's rank by z is transformed.** On `first_catch` 708–800, EXP-015 put the drone
  in the top 10 of **0 of 93 frames** for every two-frame map. Ranking live hypotheses by
  accumulated differential z puts it in the top 10 of **0.22–0.30** of those frames at every
  threshold from z\* = 6 upward, **0.24 at the frozen z\* = 60**, and in the top 5 of 0.12.
  (Two thresholds break the pattern — 0.05 at z\* = 16 and 0.00 at z\* = 4 — because at
  those settings the frame is crowded with *held* false tracks that outrank the drone.)
  The signal the design pass measured is real and the accumulation extracts it. What fails
  is the **absolute** threshold: the clutter produces the same statistic, only more of it.
- **The local ring does cancel parallax as predicted.** Re-derived above: 4.2 px/frame of
  drone motion against a 0.63 px ring scatter on hand-placed labels.
- **The analog target is acquired at all.** EXP-014 boxed it once in ~75 frames with GLAD;
  here a single confirmed track holds it from 567 through the rest of the span at rank #1,
  with one false track in the clip. That is a better hold than anything this project has had
  on analog — at 2.5× the acquisition bar, and only once the target is ~60 px.
- **The long-baseline arm is dead.** Measured on both clips, at k = 3 and 6. It does not
  need trying again.

### Why it fails, as far as this run can say

1. **The clutter has the same statistic.** A hypothesis confirms when it moves consistently
   against its ring. At a **depth discontinuity** the ring is not its background: on analog
   57% of false tracks confirm at the sky/tree-line boundary, where half the ring has no
   parallax and half has several px. The test's own premise — "the surroundings share the
   parallax" — is exactly what a horizon violates.
2. **The speed cap barely bites.** It is physically correct and it is free, but it scales
   with apparent size, and clutter seeds are *large*: **53% of O4 seeds and 40% of analog
   seeds sit at the 120 px size clamp**, giving a 400 px/frame cap that rejects nothing.
   The cap is a real constraint only for small candidates, which is where it is least
   needed because they move least. It killed a negligible share of hypotheses in every run.
3. **The latency ceiling and the threshold fight each other.** z grows as √k, so a 15-frame
   ceiling caps the achievable z at roughly 3.9× the single-frame z. On `first_catch`
   708–800 the drone's single-frame z is ~4–8, so its ceiling is ~14 — measured max 13.9 —
   while clutter reaches 62 because a clutter track that survives is one whose ring is
   *systematically* wrong, not one that is noisy.
4. **Analog's later spans are over ground.** 647–785 puts the drone against textured
   terrain rather than sky, and coverage there is 0% at every z\*. The one span that works
   is the one against sky.

### What the overlay videos show

Watched frame by frame, not summarised from the tables.

- **Analog `overlay_exp016.mp4`, frames 441–800**, and `still_500/520/545/570/585.png`.
  - At **500** the target is **19×20 px**, a faint dark smudge just above the tree line,
    barely separable by eye in the picture and invisible in the `win_b5_e4` panel. **No
    hypothesis is on it.** This is the acquisition failure, and it is a seeding-and-signal
    failure, not a verification failure.
  - At **520** (36×23 px) a hypothesis is on it and it is **rank #1 by z at 13.2** — below
    the frozen 20, so it is not confirmed. The margin at that moment is the whole result.
  - At **570** (61×33 px) the drone is an unmistakable dark quadrotor against sky, the
    brightest blob in the `win_b5_e4` crop, confirmed, rank #1, z 31.
  - At **574** the whole frame carries **169 live hypotheses and exactly one confirmation**,
    and it is the drone.
  - At **585** (47×26 px) the drone is plainly visible and **no hypothesis is on it at
    all** — the screen-fixed mask has a blob sitting on the tree line the drone has flown
    across. That mask is calibrated on this clip's own empty frames and it is costing
    recall here; see the caveats.
  - At **699** (span 687–735) the target is low in the frame against fast-moving ground,
    the seed map is uniformly speckled, and nothing tracks it. That is the 0% coverage on
    the later spans.

### Caveats

- **Both clips are one approach each**, and the bars are decided by a handful of events:
  the analog false-track rate at the frozen z\* is **one track in 22 seconds**, so ±1 track
  moves it by 5.4/min. Every false-alarm-per-minute figure here has Poisson error of the
  same order as the budget. Longer empty footage is the fix, and this project does not have
  it yet.
- **The screen-fixed mask is calibrated in-sample on each clip's own empty frames**, as it
  was in EXP-015. On analog it is worse than in-sample: it masks a patch of the *tree line*
  the drone later flies across (frame 585), because the camera pointed the same way during
  the empty frames. A deployed system cannot build this mask and would not have it.
- **The analog screen-fixed map is this experiment's own reconstruction** of EXP-015's
  rule (`screenfixed.py`); EXP-015 kept the `.npy` but not the script. Re-deriving it on O4
  gives an IoU of 0.23 against the original and covers 0.62% of the frame against 2.67%, so
  it is *more conservative*, not less. O4 keeps using EXP-015's original file so the two
  experiments stay comparable.
- **The ~9σ premise holds only on hand-placed labels** (n = 10 usable pairs in 708–800).
  On follower labels the same measurement gives 1.1σ. This is the single most fragile
  number in the chain.
- **O4's "long range" drone is 43 px**, well above the 10–30 px brief. Analog's 491–591
  span starts at 17–20 px, which is the honest small-target case, and it is the case that
  failed.
- **Top-k here is not EXP-015's top-k.** EXP-015 ranked peaks in a static map; this ranks
  live hypotheses by accumulated z, and the populations differ in size and composition.
  The comparison is the one the plan asked for, and it is directional, not exact.
- **No `detections.jsonl` was written and nothing was scored through `src.evaluate`.**
  Hypotheses are points, not boxes, so an IoU-thresholded P/R would mean nothing. The match
  criterion above stands in for it, exactly as in EXP-015.
- The confirmation-hold rule (drop a confirmed track when its trailing 15-frame z sits
  below z\*/2 for 10 frames) is a stage-1 invention, not from the plan, and it affects
  coverage and fragment counts. It does not affect acquisition latency or the false-track
  count, which are decided at confirmation.

### Next

1. **Do not build stage 2.** `src/algo/temporal/` is not authorised by this result. The
   plan's stage 2 was conditional on stage 1 passing and it did not. Promoting a detector
   that needs 110× its false-alarm budget to acquire would repeat EXP-012a's mistake at a
   larger scale.
2. **The evidence base is the binding constraint, not the algorithm.** Every bar here is
   decided by single-digit event counts over 22–23 s of empty footage and one approach per
   clip. **Label a second analog clip with long-range spans** (`/annotate`) before any
   further algorithm work on this branch — see the todo. It is the cheapest thing that
   could change the verdict, and it is also the only way the current verdict can be
   trusted.
3. **If the branch is picked up again, the two things worth trying are named by the
   failure**, not by the plan:
   - **A ring that respects depth.** Segment the ring by flow magnitude and take the mode
     rather than the median, or reject a hypothesis whose ring is bimodal. 57% of analog's
     false tracks are one geometry — the horizon — and it is detectable in the ring itself.
   - **A moving-overlay veto that suits a digital HUD.** O4's dominant surviving source is
     the pitch ladder; the MAX7456 twin test cuts it 5× but takes the drone with it
     (on-drone median twin score 0.76). A ladder-shaped mask that moves with pitch is the
     already-open todo and it is a prerequisite, not an optimisation.
4. **The speed-capped velocity bank is still open** but its trigger is not met the way the
   plan expected: the O4 miss on 708–800 is only **24%** a seeding failure (76% of frames
   do carry a seed on the drone), so a recall booster under the ring test addresses the
   smaller half of the problem.
5. **Separately, for deploy-agent:** the analog optics. 120° across 960 px is 8 px/degree,
   so a 0.6 m target is **275/R px** — **10 px at ~27 m**, 5 px at ~55 m.
   [edge-budget.md](edge-budget.md) §1 assumes 60° across 1920 px and puts 10 px at
   ~110 m. Analog is a **4× shorter detection range for the same pixel count**, on the
   footage the user cares about most, and this experiment's 0.5–2.5 s confirmation latency
   has to be bought out of a budget that is 4× smaller than the one in the document.
   Raised as its own todo.

- **O4 `overlay_exp016.mp4`, frames 641–964**, drawn at the over-budget **z\* = 6**, because
  at the frozen z\* = 60 nothing confirms and there would be nothing to watch. Stills
  `still_720/750/780/800/860/925/962.png`.
  - At **720** the label is **19×18 px** — the stretch starts far smaller than its 43 px
    median. The target is a speck **against the tree canopy**, not against sky: invisible
    in the picture at 4× magnification and black in the `win_b5_e4` panel. Nothing tracks
    it.
  - At **750** (50×35 px) it is a faint dark smudge still inside the canopy, still black in
    the seed map, and a confirmed track does sit on it at **rank #9, z 9.0**.
  - At **759** the frame carries **115 live hypotheses and 16 simultaneous confirmations** —
    on the shed, on the terrain, on the overlay glyph blocks — with the drone at **rank #4**.
    That picture is the 553 false tracks per minute made visible, and it is the reason the
    bar exists.
  - The plan asked whether 708–800 would show a confirmed track on the drone. **It does,
    from frame 742 — but only at a threshold that also confirms sixteen other things.**
  - **The background is the story on this clip.** EXP-015 reported 708–800 as "long range";
    what the stills show is that it is also the stretch where the drone is **superimposed
    on the tree line**. Its differential motion is measured against a ring that is half
    canopy at one depth and half ground at another.

### Artefacts (all gitignored)

`runs/sofa_o4/exp016_multiframe/` and `runs/sofa_analog/exp016_multiframe/`:
`collect.pkl`, `verify_z*.pkl` (the ladder), `verify_frozen.pkl`, `longbase.pkl`,
`overlay_exp016.mp4`, `still_*.png`, `calibrate.log`, `sweep.log`, `ladder.log`,
`twinveto.log`, `screenfixed.log`, `collect.log`, `finalise*.log`.

---

## EXP-017 — the EXP-016 motion test on our own field footage, unchanged

- **Date:** 2026-09-23
- **Question:** the user asked to run the motion algorithm on the FIELD capture. EXP-016
  built and rejected a multi-frame local-ring differential-motion test on two *goggles*
  clips — O4 and analog. FIELD is our own airframe's raw camera feed, the footage
  [research-notes.md](research-notes.md) names as the actual target domain. What does the
  same code, with no constant re-tuned, do on it?
- **Model / weights:** none. No learned component. EXP-016's `mf.py`/`collect.py`/
  `verify.py` verbatim, GLAD's grid-KLT homography plus OpenCV pyramidal Lucas–Kanade.
- **Data:** `data/raw/FIELD/videos/captured_raw_20260616_040253_004.mp4`, 1032×752,
  3600 frames, 30 fps, 120.0 s. **No labels** — see `data/raw/FIELD/PROVENANCE.md`.
- **Hardware / cost:** i7-1255U CPU, no GPU. `screenfixed` 148 s, `collect` 1211 s,
  `verify` 594 s (z\* = 20) and 2299 s (z\* = 6) run in parallel, overlay render ~25 min.
  ~1.5 h wall clock, all backgrounded.
- **Scripts (gitignored, throwaway):** `runs/field/exp017_multiframe/clipcfg.py`. Everything
  else is EXP-016's, run against it by putting the FIELD clip dir first on `PYTHONPATH`:

      PYTHONPATH="runs/field/exp017_multiframe;runs/sofa_o4/exp016_multiframe;runs/sofa_o4/exp015_normalised_motion;." py -3.13 -m verify --zstar 20

  Three small changes landed in EXP-016's shared modules, each defaulting to previous
  behaviour so both SOFA clips re-import byte-identical (verified: O4 706 empty frames,
  analog 665, matching EXP-016): `mf.py` accepts `labels=None` and `hud=None`, and takes
  `drone_ranges` to keep provenance-named episodes out of the screen-fixed calibration
  set; `draw_video.py` no longer captions a label box or a drone rank it does not have.

### What cannot be concluded, stated before the numbers

**FIELD has no ground truth, so there is no Pd, no precision and no recall here, and
nothing in this entry may be placed beside EXP-004–009.** Two consequences run through
everything below:

1. **z\* was not calibrated on this clip.** EXP-016 froze z\* on each clip's empty frames,
   two-fold, before scoring. That needs labels. The two values used here are *imported*
   from the SOFA clips (analog's frozen z\* = 20, and z\* = 6, the value O4's overlay was
   drawn at). **The threshold is borrowed, not earned.**
2. **"On the target" means "co-located with an EXP-010 GLAD box"**, which PROVENANCE
   describes as where that detector fired and was right. A track far from one is
   *unexplained*, not false. PROVENANCE is explicit that a gap may be a stretch the
   detector missed rather than a stretch with no drone, so every false-alarm rate below is
   an **upper bound**.

### Result: it acquires in all three episodes, at 38 false tracks/min

At the imported z\* = 20, over 3600 frames: **112,732 hypotheses born, 97 confirmed
tracks.** Requiring a track to sit within 30 px of a GLAD box on **at least half of the
frames the two overlap, over at least 10 frames** — sustained association, not one lucky
pass — gives **8 tracks on the target, in all three episodes**:

| Confirmed | Ended | Frames near a GLAD box | Seed size | z |
| --- | --- | --- | --- | --- |
| 8 | 45 | 11 / 16 | 16.0 | 28.0 |
| 1100 | 1325 | **183 / 185** | 39.0 | 19.8 |
| 1278 | 1322 | 45 / 45 | 33.0 | 28.4 |
| 1360 | 1622 | 137 / 146 | 86.0 | 111.8 |
| 3166 | 3176 | 11 / 11 | 26.0 | 23.1 |
| 3166 | 3203 | 38 / 38 | 21.0 | 10.9 |
| 3482 | 3567 | 86 / 86 | 22.0 | 42.7 |
| 3508 | 3567 | 60 / 60 | 86.0 | 31.2 |

This is the thing EXP-016 could not get on either goggles clip. On O4 the drone's maximum
z over 708–800 was 13.9 against a budget-mandated z\* of 60, a 4× gap; here a track on the
target reaches z = 111.8 and holds for 262 frames. **A caution against over-reading one
row:** a track confirmed at frame 10 that lived 1344 frames passes within 7.4 px of a GLAD
box once and is *not* in the table — it is clutter that happened to start near the target.
The sustained criterion is what separates the two, and a one-pass criterion would have
reported 12 tracks instead of 8.

Against the 5 false tracks/min budget EXP-016 set, counting every confirmation outside the
three episodes as false:

| z\* | Confirmed tracks | On target | False (83.5 s of believed-empty) | Rate |
| --- | --- | --- | --- | --- |
| 20 | 97 | 8 | 53 | **38.1/min** |
| 6 | 315 | 9 | 185 | 132.9/min |

**z\* = 6 buys nothing.** It triples the false rate and adds one target track. z\* = 20 is
the better of the two operating points and is what the overlay is drawn at.

### Why it does better here: the clip is easier for the flow stage

The background-flow stage, which everything downstream rests on, is markedly healthier on
FIELD than on either goggles clip:

| | O4 `first_catch` | analog `catch_2` | **FIELD** |
| --- | --- | --- | --- |
| Usable grid points | 0.489 | 0.389 | **0.661** |
| Forward–backward failure | 0.461 | 0.492 | **0.338** |
| Masked out | 0.092 | 0.182 | **0.006** |

No HUD to mask and no analog grain to break KLT. **A good part of what EXP-016 measured as
a failure of the algorithm was a property of goggles recordings**, not of air-to-air
geometry. That does not overturn EXP-016's verdict on those clips — it narrows what the
verdict was about.

The screen-fixed map bears this out: on FIELD it covers **0.5% of the frame** (O4 and
analog mask burned-in overlays), and it is two blobs in the top corners at y 12–97, warp
and lens-edge residue rather than scene content, peaking at frac 0.09 against the 0.05
threshold. **No EXP-010 GLAD detection anywhere in the clip falls inside it** (lowest
detection y = 95), so it is not hiding the target.

### The false alarms are concentrated in time, and the driver is host angular rate

This is the finding worth carrying forward. The 53 false tracks are spread evenly across
the frame — 21% top third, 42% middle, 38% bottom — so there is no spatial structure of the
kind EXP-016 found (analog: 57% at the sky/tree-line boundary; O4: 59% within 30 px of an
overlay). **In time they are not spread at all.** 50 of 53 fall in frames 1800–2699:

| Block | Median background image motion |
| --- | --- |
| 0–1799 | 1.1 – 3.3 px/frame |
| **1800–2699** | **5.4 – 7.8 px/frame** |
| 2700–3599 | 1.7 – 4.5 px/frame |

Frames 2000–2600 are a hard banking manoeuvre — horizon at ~45°, a near-vertical look-down
at 2300 with lens flare, rolling back by 2600. The *residual* after the homography stays
small throughout (0.16–0.25 px), so the homography is still fitting; what changes is raw
image motion against an 11 px KLT window, and the hypothesis tracker's differential vectors
become unreliable before the background model does.

Splitting there:

| | Duration | False tracks | Rate |
| --- | --- | --- | --- |
| Hard manoeuvre 1800–2699 | 30.0 s | 50 | 100.0/min |
| Rest of the believed-empty clip | 53.5 s | 3 | **3.4/min** |

**3.4/min is inside EXP-016's 5/min budget, while acquiring the target in all three
episodes.** That is the first time any configuration of this branch has been under budget
and acquiring at the same time.

**This number must not be quoted as a result.** The 1800–2699 boundary was chosen *after
looking at where the false tracks fell*, on this clip's own frames — in-sample, post-hoc,
and exactly the kind of split that manufactures a good number from noise. What makes it
worth recording rather than discarding is that it has an independent mechanism (background
image motion, measured separately and 3–5× higher in that window) and that the mechanism
corresponds to a signal **available on the aircraft from the IMU without looking at the
video at all**. It is a hypothesis with a proposed test, not a measurement.

### Reading

Three clips have now been read with the same unchanged code, and they fail differently:
O4 on its burned-in overlay, analog on depth discontinuity at the horizon, FIELD on host
angular rate. **Only the third is a property of the flight rather than of the recording
chain**, and it is the only one of the three that is both detectable without the video and
plausibly fixable by gating rather than by a better detector.

EXP-016's "do not build stage 2" stands — nothing here is a calibrated result, and the
only under-budget number in this entry is in-sample. But its stated reason for stopping was
that *the evidence base, not the algorithm, was the binding constraint*, and this entry is
evidence that the two goggles clips were not representative of the domain the project
actually targets.

### Next

1. **Label a FIELD episode.** This is now the cheapest high-value labelling in the project,
   ahead of a second analog clip: FIELD is the target domain, it carries **83.5 s** of
   believed-empty footage against 22–23 s on either SOFA clip, and every number above turns
   into a real one the moment labels exist — z\* calibrated on this clip's own empty frames,
   a true Pd per episode, and a false-alarm rate that is not an upper bound. Episode 3
   (3140–3600, 15.4 s, ~18 px falling to ~6 px) is the most informative.
2. **Test the angular-rate hypothesis out-of-sample**, which does not need labels: derive
   background image motion per frame from `collect.pkl` alone, gate confirmations above a
   threshold chosen on *one* clip, and read the false-alarm rate on another. If it holds,
   it is an IMU gate on the aircraft, not an algorithm change.
3. **Do not re-tune constants on FIELD** before either of the above. Every pixel constant
   here is EXP-016's, scaled by 1032/1440 = 0.717, and that is what makes the three clips
   comparable at all.

### Artefacts (all gitignored)

`runs/field/exp017_multiframe/`: `clipcfg.py`, `collect.pkl` (128 MB), `collect_fb.pkl`,
`screen_fixed_frac.npy`, `verify_z20.pkl`, `verify_z6.pkl`, `overlay_exp017.mp4`
(frames 1–3600 at z\* = 20), `screenfixed.log`, `collect.log`, `verify_z20.log`,
`verify_z6.log`, `draw.log`.

---

## EXP-018 — is there any fisheye to correct, and does stage 1 care?

- **Date:** 2026-09-29
- **Question:** the motion-first rebuild's stage 1 splits each frame into sky and scene as
  a two-level depth prior. Does correcting the lens first change that split? And, before
  that can be asked at all: **is there a radial distortion in this footage to correct?**
  `docs/todo.md`'s fisheye item has been open since 2026-09-17 with its premise withdrawn
  and never settled either way.
- **Model / weights:** none. No learned component.
- **Data:** `data/processed/SOFA-O4/videos/first_catch.avi` (1440×1080, 964 frames) and
  `data/raw/SOFA-ANALOG/videos/catch_2.mp4` (960×720). 40 frames sampled evenly from each;
  35 and 29 respectively had a horizon crossing ≥55% of the frame width.
- **Hardware / cost:** i7-1255U CPU. ~6 min per clip per lambda swept over images; the
  coordinate-only sweeps are seconds.
- **Scripts (gitignored):** `runs/sofa_o4/exp017_motion_first/undistort.py` (division
  model, both directions, plus `py -3.13 -m undistort` self-check) and `check_fisheye.py`.

      PYTHONPATH="runs/sofa_o4/exp017_motion_first;runs/sofa_o4/exp015_normalised_motion;." \
          py -3.13 -m check_fisheye --frames 40 --range 1.2 --lam=-0.20

  **Numbering clash, recorded rather than renamed:** `runs/sofa_o4/exp017_motion_first/`
  and `runs/sofa_analog/exp017_motion_first/` were created in a different session that did
  not know EXP-017 was already taken by the FIELD run above. The directory name stays as
  it is — paths are cited in that run's own README — but EXP-017 in this ledger means
  FIELD, and the motion-first build is **not yet in this ledger at all**; its README is
  currently its only record.

### Method, and the control that makes it readable

Stage 1 already emits a horizon (lowest sky row per column). A horizon is the only plumb
line this footage offers, but a tree line has real relief, so **a bow is not by itself
evidence of a lens.** The discriminator is that a lens bows a line by an amount set by the
line's distance from the optical axis — pinned to image coordinates, zero when the line
runs through the axis — while terrain is pinned to the scene.

Each frame's horizon gets a MAD-trimmed quadratic fit; `sag` is the bow at mid-span
(exactly `-a` with x normalised to [-1, 1]) and the fit residual RMS measures the tree
line's own roughness. A **positive control** bends a synthetic straight horizon at each
real frame's own height and span by a known lambda and requires the sweep to recover it —
without it, "found nothing" and "cannot find anything" are the same output.

    positive control, both clips: injected lambda -0.160, recovered -0.160, residual |sag| 0.000 px

### Result: the two airframes give opposite answers

| | O4 (digital link) | analog (CVBS) |
| --- | ---: | ---: |
| frames with a fittable horizon | 35 of 40 | 29 of 40 |
| median \|sag\| | 159.5 px | 38.7 px |
| median fit residual (tree relief) | 40.1 px | 13.7 px |
| sag vs horizon height, r | +0.547 | +0.621 |
| sag crosses zero at row | **325** (axis is 540) | **368** (axis is 360) |
| best lambda | none — runs to +1.2 and still falling | **-0.60**, a clean interior minimum |
| mean \|sag\| at best lambda | 149.7 → 72.9 px (absurd lambda) | 43.4 → **26.7** px |

**O4 has no radial signature.** The zero crossing misses the optical axis by 215 px, the
sweep never finds an interior minimum over ±1.2 and wants *pincushion* — the opposite of a
fisheye — and the decisive frames settle it directly: **11 frames whose horizon runs
within 30 px of the optical centre bow by a median 209 px**, where a radial model demands
zero. Rendering one confirms it by eye: tall near trees at both frame edges, distant tree
line low in the middle. The bow is scene structure. This is consistent with the O4 air
unit dewarping before the link, and it closes the fisheye item for O4.

**Analog does have one.** The zero crossing lands 8 px from the optical axis, and the
sweep has a real bowl bottoming at lambda = -0.60 (26.7 px against 43.4 px uncorrected).
Resampled, the horizon's median \|sag\| falls **38.7 → 14.2 px** against a tree-line
residual of 11.5 px — i.e. as straight as this plumb line can show. Caveat, stated: only
**one** analog frame has its horizon near the optical axis, so the direct refutation that
settles O4 has no power here and the evidence is the regression and the bowl.

### But stage 1 does not benefit, on either clip

| lambda | FOV retained | sky fraction (like for like) | horizon \|sag\| | split IoU vs old |
| --- | ---: | ---: | ---: | ---: |
| O4, -0.20 | 80.0% | 30.93% → 30.38% | 159.5 → 145.3 | **0.989** |
| analog, -0.60 | **59.3%** | 28.48% → 25.13% | 38.7 → 14.2 | **0.885** |

**Undistortion is a bijection on pixels, so it relocates the sky boundary and cannot
relabel it.** Stage 1 is a per-pixel appearance decision — texture, luminance, blue excess,
each against a percentile of the frame's own distribution — and a pixel that looked like
tree still looks like tree at its new coordinates. The IoU against the old split carried
through the same remap is 0.989 on O4; what little it moves is resampling at the boundary,
not a better decision. Sky fraction is flat or slightly **down**, and on analog the
correction costs **40.7% of the field of view**, which on a shallow-elevation target is
exactly the picture one cannot afford to throw away.

**One measurement was wrong before it was right, and the correction matters.** The first
pass reported "remap keeps 100.0% of the frame" and a sky fraction *rising* 28.0% → 30.6%.
Both were artefacts: `undistort_maps`'s `inside` flag answers which *destination* pixels
have a source, and for a barrel correction the answer is always all of them, because that
map zooms in. The discarded picture is at the *source* edges and only the forward map sees
it. So a 20% crop was being reported as "keeps 100%", and the sky fraction rose only
because the crop removed ground. Compared over the retained field of view it does not rise.

### What this settles, and what it does not

- **Fisheye undistortion is not a stage-1 lever on either clip.** Closed.
- **No undistorted copy was written to `data/processed/`**, though the user authorised one
  on 2026-09-22. On O4 there is nothing to undo; on analog the only honest lambda costs
  41% of the frame on a fixed canvas. If analog undistortion is ever wanted, it should
  expand the canvas rather than crop, trading centre resolution for field of view.
- **Untested: whether analog's lambda = -0.60 helps stage 2.** That is where a radial error
  actually bites — the homography, the epipole and the parallax residual all assume a
  pinhole camera, and `undistort_points` corrects coordinates with no resampling and no
  FOV loss at all. EXP-018 did not measure it. It is the open question this leaves.

---

## EXP-019 — stage 1 and stage 2 rendered together, and measured over two spans

- **Date:** 2026-09-29
- **Question:** the motion-first rebuild has a sky/scene split (stage 1) and an epipolar
  direction test (stage 2), each checked in isolation. Drawn on the same frame and
  measured over a whole span, do they hold up, and do the candidates stage 2 fails to
  reject sit on stage 1's boundary the way EXP-016's confirmed false tracks did?
- **Model / weights:** none. No learned component.
- **Data:** `data/processed/SOFA-O4/videos/first_catch.avi` frames 650–964 (315 pairs,
  257 labelled) and `data/raw/SOFA-ANALOG/videos/catch_2.mp4` frames 441–800 (360 pairs,
  224 labelled). `tau = 1.661`, EXP-017's frozen threshold budget, unchanged.
- **Hardware / cost:** i7-1255U CPU. ~7 min per span, both backgrounded.
- **Scripts (gitignored):** `runs/sofa_o4/exp017_motion_first/overlay_video.py`.
  `--no-video` prints the report without writing frames.

      PYTHONPATH="runs/sofa_o4/exp017_motion_first;runs/sofa_o4/exp015_normalised_motion;." \
          py -3.13 -m overlay_video --start 650 --end 964

  Artifacts: `stages_first_catch_650_964.mp4`, `stages_catch_2_441_800.mp4`.
  See EXP-018 for the EXP-017 directory-numbering clash.

### Stage 1

| | O4 650–964 | analog 441–800 |
| --- | ---: | ---: |
| sky fraction, median (p10 / p90) | 21.2% (17.4 / 31.0) | 21.5% (**0.0** / 24.0) |
| uncertain band | 7.0% of frame | 8.9% of frame |
| frames with no sky at all | 0 of 315 | **75 of 360** |
| frame-to-frame sky IoU, median | 0.960 | 0.960 |
| pairs below 0.90 IoU | 30 of 314 | 56 of 359 |
| **where the labelled drone is put** | **scene 257, sky 0** | **scene 158, sky 44, uncertain 22** |

**The split calls the airborne target "scene" in 257 of 257 O4 frames and 158 of 224
analog frames.** The user's correction — "the drone is in the sky, just shallow one (for me
tree line is sky as well)" — is confirmed with a number: the rule separates *blue sky* from
everything, while what the ring test needs is **far from near**, and a target at shallow
elevation against a distant tree line is far. This is a definition error, not a threshold.

**Analog loses the sky entirely in 21% of frames**, so a ring test gated on stage 1 would
have to refuse on a fifth of that clip. Flicker is secondary: 0.960 median IoU is healthy.

### Stage 2, against a chance baseline that had been missing

| | O4 | analog |
| --- | ---: | ---: |
| epipole reported reliable | 297 of 315 | 310 of 360 |
| epipole anisotropy, median | 0.38 (threshold 0.15) | 0.29 |
| epipole frame-to-frame jump, median | **126.8 px** (p90 293.4) | **98.9 px** (p90 232.5) |
| epipole agreement with background (angular) | **34.2%** | **29.4%** |
| candidates/frame | 87.4 | 47.2 |
| rejected as epipolar | 32.5% | 22.8% |
| unjudged (FOE / rotation) | 18.2% | 28.6% |
| rejection among **judged** candidates | **39.7%** | **31.9%** |
| the same for **random** directions | 22.8% | 22.8% |
| **lift over chance** | **×1.74** | **×1.40** |
| drone survives | 134 of 163 (82%) | 65 of 67 (97%) |

**A candidate whose direction is random is rejected with probability `2·asin(0.35)/π =
22.8%`,** because the test rejects when |sin| to the epipolar line is under 0.35. Every
rejection rate must be read against that floor, and EXP-017's step 2 did not do so. Read
correctly the direction test runs at **1.74× chance on O4 and 1.40× on analog**. It is
doing something real and it is doing far less than the several-fold predicted in the plan.
**That prediction is now twice contradicted and should not be restated.**

**The epipole hops 99–127 px between consecutive frames**, where a real focus of expansion
drifts smoothly with the manoeuvre. This is single-pair noise, not a bad estimator:
EXP-012b measured the post-homography residual at ~1 px median, and a 1 px residual's
direction is dominated by tracking noise. A RANSAC fit over the same points (200
hypotheses, 3 px tolerance, inlier refit) was **worse** — median jump 454.7 px against
least squares' 93.9 px on frames 700–758, the two answers 282 px apart — which is the
signature of noise-limited rather than outlier-limited data. The fix is to fit the epipole
**once per window** rather than per pair, which the tracker needs anyway.

### A metric of ours that was broken, and is now fixed

`geometry.estimate_epipole`'s `inlier_fraction` counted voting lines within a fixed 3 px of
the solution on `|n · (p − x)|`, the epipole's offset from the line through x along mu.
**That offset scales with `|p − x|` for a fixed angular error**, so the fixed pixel
tolerance was only meaningful next to the epipole and read ~0.008 everywhere else — it was
measuring distance, not agreement. It is now the angular form, the fraction of background
points within the same 0.35 sin threshold the direction test uses. It is reporting-only and
was never part of `reliable`, so no earlier verdict in EXP-017 changes.

Read that way the epipole explains **34.2%** of the O4 background and **29.4%** of
analog's, against the 22.8% a random direction would hit — ×1.50 and ×1.29. That is an
independent route to the same conclusion as the candidate lift (×1.74 / ×1.40): the
epipole is real, it is weak, and on analog it is barely locating anything.

### Where the survivors sit — the reason to draw both stages together

| | band's share of frame | survivors in the band |
| --- | ---: | ---: |
| O4 | 7.0% | **12.3%** of 13,582 |
| analog | 8.9% | **7.9%** of 8,264 |

**EXP-016's horizon failure does not appear at the candidate stage.** O4 is 1.76×
over-represented at the sky/scene boundary; analog is 0.89×, i.e. *under*-represented.
EXP-016's 57.5% figure was about confirmed **multi-frame tracks**, so these do not
contradict it — but they locate the horizon problem in the ring test and the accumulation
rather than in candidate generation, which is an argument for **deprioritising the
depth-aware ring** relative to the tracker. First evidence either way.

### Two defects the render shows that the tables do not

- **Candidates survive in the gaps between HUD glyph boxes.** The dilated HUD mask covers
  each glyph but not the space between them, and glyph edges leak difference energy there.
  Visible on O4 frame 946 around `4.04v` and `24.3V`.
- **The uncertain band is a 7–9% ribbon**, as `UNCERTAIN_PX = 24 px` dilate/erode
  specifies — a lot of frame to declare unjudgeable when the target spends its time near
  that boundary.

### Still standing from EXP-017

`[WARNING] stage-0 masks cover the labelled drone in 55 frames, first at 588` on analog.
Unchanged, and still the first thing to fix on that clip.

## EXP-020 — the analog HUD mask was eating a fifth of the ground truth

- **Date:** 2026-09-29
- **Question:** the analog mask covers 13.15% of the frame while the visible OSD is much
  smaller. Is the difference free, or is it deleting targets?
- **Data:** `data/raw/SOFA-ANALOG/videos` (9 clips, all 960×720), the 494 labelled boxes in
  `data/processed/SOFA-ANALOG/labels`, and EXP-013's 82 recorded detections.
- **Hardware:** i7-1255U CPU. Mask rebuild ~2 min; no inference re-run.

### Where the mask goes

The mask is six filled rectangles. Only 14,130 px of the 90,883 are pixels the loose
threshold ever called OSD — `osd_blocks` fills each joined block to its **bounding
rectangle**, so a line of telemetry with gaps between its fields becomes one solid slab.
Scored per rectangle against the labels and against EXP-013:

| Region | px | Labelled targets vetoed | EXP-013 detections vetoed |
| --- | ---: | ---: | ---: |
| top centre 285×41 | 11,685 | 26 | 34 |
| top right 183×74 | 13,542 | 0 | 30 |
| top left 186×30 | 5,580 | 0 | 0 |
| single glyph 18×24 | 298 | 0 | 0 |
| **bottom left 468×93** | **43,524** | **68** | **0** |
| **bottom right 378×43** | **16,254** | **6** | **0** |

The two bottom rectangles are **66% of the mask**, veto **74 of 494 real targets**, and
have never vetoed a single false alarm — no EXP-013 detection landed below row 274.

### The cause, and the fix

`--picture-rows 150:530` read as "the sky is the picture". Rows 530–720 were therefore
*outside* it and open to the fill, and the fill is deliberately generous. But 118 of the
494 labelled boxes are below row 530: the drone flies over the ground, not only the sky.

Declaring the band `150:720` is the whole fix — no code change.

| | `150:530` | `150:720` |
| --- | ---: | ---: |
| Mask | 90,883 px (13.15%) | **53,257 px (7.71%)** |
| Labelled targets vetoed | 100 / 494 (20.2%) | **26 / 494 (5.3%)** |
| EXP-013 detections vetoed | 64 / 82 | 64 / 82 |

The new mask is a strict **subset** of the old one and its veto set is **bit-for-bit
identical**, so EXP-014 (82 → 8 detections, drone found at frame 547) is untouched.

- **Command:** `py -3.13 -m src.data.hud_mask --videos data/raw/SOFA-ANALOG/videos --out data/processed/SOFA-ANALOG/hud_mask.png --white-level 180 --block-fraction 0.08 --picture-rows 150:720 --preview runs/sofa_analog/exp020_hud_preview.png`

### It also closes most of EXP-017's standing warning

EXP-017 ended on `[WARNING] stage-0 masks cover the labelled drone in 55 frames, first at
588`. **47 of those 55 are the HUD layer** — `catch_2` frames 689 onward — and the new
mask covers **0** labelled boxes in that clip. The remainder, including the first at 588,
come from another stage-0 layer (`ladder_mask` or `prop_mask`) and are still open.

### Two things that did not work, so they are not worth retrying

- **Glyph-tight masking instead of rectangles.** Dilating the loose glyph mask 15×15 gives
  8.77% and 31 targets lost — no better than the row fix, and it *loses* 8 vetoes in the
  top-right block. Filling to the rectangle is correct where the rectangle earns it.
- **A median OSD "plate" to tell a drone-on-glyph from a glyph.** Mean |frame − plate| over
  the masked pixels: drones-on-OSD score 12.5–26.6, OSD detections 7.5–37.2. The
  distributions overlap almost completely, and in the wrong direction — the residual is
  dominated by how much the live scene differs from the median scene, not by the glyph.

### What is left

The 26 remaining losses are all the top-centre block, which the drone genuinely flies
through. Glyph-tight masking recovers none of them: that block's glyphs are dense enough
that tight and filled are the same mask. Separating a drone crossing it from the OSD needs
motion or track history, not geometry.

**And the general lesson:** the tool reports coverage, which is the wrong number.
13% of the frame sounded acceptable and was 20% of the ground truth. Score a mask against
labels before a run depends on it.

## EXP-021 — the 5-frame window: persistence works, the direction test's premise does not

- **Date:** 2026-09-29
- **Question:** EXP-019 left the epipole hopping 99–127 px between consecutive frames and
  the direction test running at only ×1.74 chance. Both point at the same fix — fit the
  epipole **once per window** instead of once per pair, and test each candidate's
  *accumulated* residual rather than one step's. Does that rescue the direction test?
- **Model / weights:** none. No learned component.
- **Data:** `data/processed/SOFA-O4/videos/first_catch.avi` frames 650–964, `tau = 1.661`,
  k = 5 with 4 appearances required, appearance radius 14 px.
- **Hardware / cost:** i7-1255U CPU, ~8 min for the span.
- **Scripts (gitignored):** `runs/sofa_o4/exp017_motion_first/window.py` (accumulation,
  chaining, pooled epipole, persistence; `py -3.13 -m window` self-checks it) and
  `overlay_window.py` (renderer and report). `overlay_video.py` stays as the single-pair
  baseline. Artifact: `window5_first_catch_650_964.mp4`.

### The accumulation does mechanically what it was designed to

Over 61 windows on frames 900–960: residual coherence `|Σμ| / Σ|μ|` **0.920** against
1.000 for pure parallax and 0.447 for a random walk; `|Σμ|` 3.97 px over five steps against
a per-step 0.93 px; direction anisotropy 0.389 windowed against 0.355 single-pair; and the
epipole's frame-to-frame jump falls to **40.9 px median** (p90 95.5) over the full span,
against EXP-019's 126.8 px. Every mechanical claim made for the window holds.

### And it still does not find a parallax field

| model, scored on the same accumulated field | 700–760 | 900–960 |
| --- | ---: | ---: |
| **A. uniform shift** — every residual points the same way | **37.2%** | **38.5%** |
| **B. parallax from a focus of expansion** | 30.3% | 23.8% |
| chance (`2·asin(0.35)/π`) | 22.8% | 22.8% |
| `\|μ\|` vs distance to the epipole, correlation | +0.171 | +0.061 |

**A uniform shift outscores a focus of expansion on both sub-spans.** Parallax means
`μ = γ·(e − x)`, so `|μ|` *must* grow with distance from the epipole; at +0.17 and +0.06 it
does not. The field is better described as a small near-uniform misalignment left by the
homography than as depth-induced parallax — equivalently, the epipole is at or near
infinity, where fitting it as a *point* is meaningless.

**One number in the run's own write-up is span-dependent and was generalised too far.** The
headline "the pooled epipole is at chance, 22.1%" is a 61-frame sub-span figure; the
full 315-frame span logged **35.4%** agreement, which is above the 22.8% floor, and
frames 700–760 gave 30.3%. The *comparative* conclusion — uniform beats FOE on every span
measured — is what survives, and it is the one the recommendation rests on. The absolute
claim "explains nothing" holds only on 900–960 and should not be restated unqualified.

### What persistence buys, and what the direction test costs

`first_catch` 900–964, appearance radius 14 px:

| | load/frame | drone kept |
| --- | ---: | ---: |
| single-pair, EXP-019 (650–964) | 59.0 | 82% |
| **persistence alone (≥ 4 of 5)** | **31.2** | **78%** |
| persistence + direction test | 25.7 | **45%** |

Over the full 650–964 span: 80.6% of 41,876 seeds are dropped as flashes, cutting 132.9
candidates/frame to 25.8 under persistence alone and 15.1 with the direction test added.
**Persistence roughly halves the load for ~4 points of recall; the direction test buys 5.5
more candidates/frame and costs 33 points of recall** — and since its rejections run at
chance, those drone frames are lost at random. The appearance radius saturates by 14 px
(8 px → 31% kept, 14 → 45%, 22 → 46%, 32 → 49%), which is why 14 is the O4 default; the
same physical radius on the 960-wide analog clip is 9 px.

### Three errors the self-test caught before any video ran

Each would have returned a null result indistinguishable from "the idea does not work".

1. **Pooling votes instead of accumulating per point.** Throwing all k pairs' votes into
   one fit gives k times as many votes of the same poor quality, so directional
   concentration never improves and the reliability gate still refuses — 5 of 20 pooled
   windows reliable against 9 of 24 single pairs. Accumulating each point's residual first
   is different in kind and fixed it.
2. **The synthetic's noise model was wrong.** It drew fresh noise per pair, but a real
   frame is measured *once*: its error enters one pair as the current endpoint and the next
   as the previous one, with opposite sign, so intermediate noise largely cancels under
   accumulation. Independent draws destroyed that cancellation.
3. **Selection bias in the comparison.** Estimators were scored only where each called
   itself reliable, so the per-pair fit was judged only on the windows it found easy.
   Scoring both on every window reversed the result.

After the fixes, at 0.8 px of tracking noise: per-pair reliable in **1 of 9** windows with
**121 px** of error; accumulated reliable in **9 of 9** with **5.7 px**. The machinery is
right; the footage does not contain the field it is designed to exploit.

### Standing recommendation

**Keep persistence, turn the direction test off.** Persistence rests only on "a real thing
is visible in consecutive frames", needs no geometry, and is the only part of stage 2
currently earning its cost.

## EXP-022 — stage 2, both branches: a sky silhouette detector and a two-plane scene test

> **Corrected 2026-09-30.** The "measured target size" quoted below is the DoG's
> detected scale, not the target: the labelled boxes are a median 71.6 px on O4 and
> 37.0 px on analog. Read the sky branch's recall as "a candidate landed in the grown
> box", and see *Correction to EXP-022 and EXP-023* at the end of this file.

- **Date:** 2026-09-29
- **Question:** run the plan's two-branch stage 2 over a whole span with an overview video.
  2a: multi-scale negative-polarity blob detection on sky, scored by the IRST local
  contrast `c = (median(ring) − min(core)) / σ_ring`. 2b: parallax-discounted motion — a
  layered homography, an epipolar direction rejector, and depth-aware rings.
- **Model / weights:** none. No learned component in either branch.
- **Data:** `data/processed/SOFA-O4/videos/first_catch.avi` frames 650–964 (315 frames, 257
  labelled) and `data/raw/SOFA-ANALOG/videos/catch_2.mp4` frames 441–800 (360 frames, 224
  labelled). `tau = 1.661`, k = 5 needing 4 appearances, `c >= 6`.
- **Hardware / cost:** i7-1255U CPU, ~14 min per span, both backgrounded.
- **Scripts (gitignored):** `runs/sofa_o4/exp017_motion_first/silhouette.py` (2a, 8 synthetic
  self-checks via `py -3.13 -m silhouette`), `layers.py` (2b, 10 checks via `py -3.13 -m
  layers`), `overlay_stage2.py` (the renderer and report), `probe_sky.py` (diagnostic).
  Artifacts: `stage2_first_catch_650_964.mp4` (315 frames, 45 MB),
  `stage2_catch_2_441_800.mp4` (360 frames, 31 MB).

### Two deviations from the plan, both forced by measurements already in this repo

**The sky branch runs ungated, not on stage-1 sky.** EXP-019 measured `skyline.split`
putting the labelled airborne target in `scene` in 257 of 257 O4 frames. Gated as the plan
specifies, 2a scores zero on O4 and the run says nothing about the detector. It therefore
runs over the whole valid frame with the result cut by stage-1 label.

**The epipolar direction test is off by default.** EXP-021 measured a uniform-shift model
outscoring a focus of expansion on every span. The plan's claim that this test "kills the
57.5% of analog false tracks at the horizon" is separately contradicted by EXP-019, which
located that failure in the tracker rather than in candidate generation. It is still
implemented and still measured, because this run changes its input — a two-plane residual
field instead of one.

### The premise holds: the regime inversion is real

σ_ring at random background points, by stage-1 label:

| | O4 sky | O4 scene | analog sky | analog scene |
| --- | ---: | ---: | ---: | ---: |
| σ_ring median (grey levels) | **1.48** | **11.86** | **1.48** | **7.41** |
| inversion | **8.0×** | | **5.0×** | |

**But sky's σ_ring is exactly 1.4826 on both clips — the MAD of a single grey level.** Sky
is flat to the 8-bit quantiser, so `c` on sky is contrast in units of 1.5 grey levels, not
in units of a measured noise distribution. The statistic is a usable ranking score; **no
Gaussian tail argument can be made from it**, and the plan's central claim — "a threshold
means a false-alarm rate on smooth sky" — does not survive contact with 8-bit video.

### And the false-alarm claim fails by six orders of magnitude

Measured false alarms per frame against the Gaussian rate the same threshold nominally buys:

| threshold | O4 measured | analog measured | Gaussian |
| --- | ---: | ---: | ---: |
| `c >= 6` | 14.65/frame | 5.16/frame | 6.5e-05/frame |
| `c >= 12` | 3.62/frame | 0.37/frame | 1.2e-28/frame |
| `c >= 20` | 1.39/frame | 0.03/frame | 1.8e-84/frame |

The synthetic control says why this is not a defect in the statistic: on pure Gaussian
noise sky the **maximum `c` observed anywhere was 1.8**. Everything above it on real
footage is *structure* — cloud edge, grain, compression blocking, analog chroma crawl — and
structure does not have a Gaussian tail. `c` ranks; it does not calibrate.

### The sky branch works, on the clip where the target is against sky

| | O4 | analog |
| --- | ---: | ---: |
| labelled frames | 257 | 224 |
| **sky branch (2a)** found the drone | 48 (**18.7%**) | 73 (**32.6%**) |
| scene branch (2b, persistence) found it | 79 (**30.7%**) | 22 (**9.8%**) |
| both | 31 (12.1%) | 19 (8.5%) |
| **either** | **96 (37.4%)** | **76 (33.9%)** |
| missed by both | 161 (62.6%) | 148 (66.1%) |
| σ_ring at the target | 7.41 | 5.93 |
| `c` at the target, median (p10) | 7.4 (6.3) | 9.4 (6.6) |
| measured target size | 3.1 px | 6.2 px |

**The two branches fail on different frames, which is the only reason to have two.** On O4
the scene branch is twice the sky branch and `either` adds 6.7 points over it; on analog the
ranking inverts entirely — the sky branch is 3.3× the scene branch. Neither clip is carried
by one branch.

**The sky branch's recall is bounded by where the target is, not by the detector.**
`probe_sky.py` shows the DoG finding the O4 target with a top-50 response in most frames
while σ_ring at the target runs 12–104 grey levels: the ring sits on tree line, so there is
nothing for the sky-noise normalisation to normalise by. This is EXP-019's stage-1
definition error appearing again, now in a second branch's own units.

### What does not work: the second plane

| | O4 two planes | O4 one plane | analog two | analog one |
| --- | ---: | ---: | ---: | ---: |
| epipole reliable | 91% | **94%** | 51% | **86%** |
| agreement with background | 0.294 | **0.309** | 0.246 | **0.277** |
| corr(\|μ\|, distance to epipole) | 0.104 | **0.135** | 0.104 | **0.174** |

**Fitting two planes makes the residual field *less* parallax-like on both clips, on every
diagnostic.** This is not a marginal call: on analog the epipole's reliability collapses
from 86% to 51%. Two distinct planes *are* found (98% of O4 frames, 82% of analog, median
separation 5.06 and 3.31 px, well clear of the guard), so the failure is not that the
second plane is imaginary — it is that splitting the points between two fits leaves each
fit less evidence, and the thing being fitted was never radial to begin with. **EXP-021's
verdict stands, now reproduced against an internal control on identical frames.**

Persistence reproduces EXP-021 exactly — 80.6% of O4 candidates dropped as flashes, 25.8
survivors/frame, drone kept in 79 frames — which is the check that this run's scene branch
is the same one.

### Four defects the self-checks caught before any long run

None would have raised an exception; each returns a plausible-looking number.

1. **σ_ring was measured on a blurred image.** A Gaussian blur divides white noise by
   ~`2√π·σ_blur`, which drove σ_ring under its floor at every realistic sky noise level.
   `c` then read the *floor* and stopped depending on sky noise at all — the one property
   it exists to have. Caught by tripling the synthetic sky noise and finding `c` unmoved
   (33.3 → 33.4).
2. **The candidate cap was acting as a rank budget**, returning exactly 600 peaks every
   frame — the failure mode `budget.py` was written to avoid.
3. **A frame-indexing error** paired the oldest window positions with the newest pair's
   homography and layer fit.
4. **"No opinion" was counted as "passed"**, putting unjudged candidates into the
   denominator of the rejection rate, so lift-over-chance measured how often the test
   declined rather than how well it discriminates.

### Next

1. **Keep 2a, drop the second plane.** 2a is the only thing in stage 2 that acquires from a
   single frame, and on analog it triples the motion branch. The layered homography is
   measurably worse than one plane on both clips and should not be carried forward.
2. **Quote `c` as a ranking score, never as a false-alarm rate.** Any operating point must
   come from the measured curve above, per clip.
3. **Redefine stage 1 as far/near before gating anything on it** — now blocking two branches
   rather than one.
4. **A second test after `c` is not optional**: 77–90% of clutter reaches the target's p10
   contrast, so `c` alone cannot separate target from clutter on either clip.

## EXP-023 — the depth-aware ring, and size as evidence

> **Corrected 2026-09-30.** The "measured target size" quoted below is the DoG's
> detected scale, not the target: the labelled boxes are a median 71.6 px on O4 and
> 37.0 px on analog. Read the sky branch's recall as "a candidate landed in the grown
> box", and see *Correction to EXP-022 and EXP-023* at the end of this file.

- **Date:** 2026-09-29
- **Question:** EXP-022 implemented the depth-aware ring, self-checked it, and then never
  passed it — the whole run used an undifferentiated ring. Turn it on and measure it. And
  answer a second question the user raised: why does `c` ignore the blob's *size*, and
  should small candidates be dropped?
- **Model / weights:** none.
- **Data:** the EXP-022 spans unchanged — O4 `first_catch` 650–964 (315 frames, 257
  labelled) and analog `catch_2` 441–800 (360 frames, 224 labelled). `c >= 6`, cap 4000.
- **Hardware / cost:** i7-1255U CPU. ~13 min per rendered span, plus four `--no-video`
  attribution passes.
- **Scripts (gitignored):** a **new folder**, `runs/sofa_o4/exp023_sky_branch/` (and the
  analog clipcfg in `runs/sofa_analog/exp023_sky_branch/`). EXP-022's
  `exp017_motion_first/` is left frozen as the baseline it is cited as.
  `silhouette.py` (10 synthetic self-checks), `overlay_sky.py`. Artifacts:
  `sky_first_catch_650_964.mp4`, `sky_catch_2_441_800.mp4`, and
  `candidates_*.csv` — 2,996 and 1,261 rows, one per kept candidate.

### The control first: the new code reproduces EXP-022 exactly

Run with `--plain-ring`, the whole-ring condition returns **48/257 at 14.65 FA/frame on O4
and 73/224 at 5.16 on analog** — EXP-022's numbers to the digit. Every comparison below is
therefore internal to one codebase, not against a remembered figure.

### Two changes, separated, because together they mislead

The first render changed the ring *and* began refusing candidates standing in stage 1's
`uncertain` band. A 17-frame probe made that look like an 87% false-alarm cut at no cost to
recall. **It is not.** Four attribution passes over the full spans:

| | O4 FA/frame | O4 recall | analog FA/frame | analog recall |
| --- | ---: | ---: | ---: | ---: |
| **A** whole ring (EXP-022) | 14.65 | 48 (18.7%) | 5.16 | 73 (32.6%) |
| **B** depth ring, band scored | 17.56 | **57 (22.2%)** | 4.79 | 71 (31.7%) |
| **C** depth ring, band refused | **9.35** | 40 (15.6%) | **3.30** | 49 (21.9%) |

**The depth-aware ring by itself (B vs A) is a recall win on O4** — 48 → 57 frames, +3.5
points — because a target beside the horizon no longer has its `sigma_ring` set by the step
instead of by the sky. The synthetic check measures the same effect directly: `c` 8.2 with
a whole ring, 16.4 with the label-restricted one. On analog it is neutral (73 → 71, and
slightly fewer false alarms).

**The band refusal is what moves the false-alarm rate, and it is not free.** It cuts FA by
47% on O4 and 31% on analog while costing 17 and 22 drone frames respectively.

### At matched false-alarm rate, the answer flips between clips

Raising B's threshold until its false-alarm rate equals C's — the only fair comparison,
computed from the dumps in seconds:

| at equal FA/frame | B (band scored) | C (band refused) | winner |
| --- | ---: | ---: | :--- |
| O4, 9.35 FA/frame | 30 frames (c≥9.3) | **40 frames** | **C** |
| analog, 3.30 FA/frame | **60 frames** (c≥6.8) | 49 frames | **B** |

**Neither policy dominates**, and the split is explained by where each clip's target lives:
EXP-019 measured the analog drone in the uncertain band in 22 of 224 frames, so refusing the
band throws real targets away there, while on O4 the band is mostly clutter. This is a
per-clip operating point to be read off the dump, not a constant to be fixed in code.

### Why `c` ignored size, and what replaced it

`c = (median(ring) − min(core)) / σ_ring` is a **per-pixel** contrast ratio: it uses the
detected scale only to place the core and ring, never as evidence. Two consequences.

It throws away information — matched-filter SNR goes as `Δ·√N/σ`, so a 12 px blob is ~2×
the evidence of a 3 px one at equal contrast, and `c` scores them identically (measured on
the synthetic: 4.5 px and 13.4 px blobs both at `c ≈ 16`).

Worse, **size was already leaking in backwards.** `min(core)` is a minimum over N pixels,
and the expected minimum drifts downward as N grows, so a larger core inflates `c` from
noise alone. That is the sub-proportionality EXP-022 recorded — tripling sky noise cut `c`
by only ~2 instead of 3.

`snr = (median(ring) − mean(core))·√n_core / σ_ring` is now computed alongside: the core's
*mean*, normalised by the standard error of that mean. Unbiased where `min` is not, and it
rises with size (56 vs 155 on those same two blobs). **It is not a clear win on separation**
— clutter reaching the target's p10 is 85.9% for `snr` vs 92.3% for `c` on O4, but 74.7% vs
71.9% on analog. Worth recording as a column; not worth replacing `c` with yet.

### A minimum size floor: right idea, and only on analog

| floor | analog clutter cut | analog target cut | O4 clutter cut | O4 target cut |
| ---: | ---: | ---: | ---: | ---: |
| 3–4 px | **−22.1%** | **−4.1%** | −66.1% | −58.0% |
| 6 px | −38.6% | −31.5% | −84.8% | −70.0% |
| 8 px | −48.3% | −90.4% | −94.7% | −90.0% |

**On analog a 4 px floor removes 22% of clutter for 4% of the target** — a real trade. **On
O4 it never pays**, because target and clutter have the *same* median diameter (3.1 px):
the O4 target sits at the bottom of the scale ladder, so size carries no information about
it at all. At 8 px both clips lose ~90% of the target. `--min-diameter` therefore exists,
is measured, and **defaults to off**.

Two practical notes: the scale ladder is geometric and discrete, so floors only bite at
ladder steps (3.1, 4.5, 6.2 … px) and 3.5 and 4.0 give identical results; and applied on
top of condition C on analog, a 3.5 px floor gives 2.57 FA/frame at 46 frames against
3.30 and 49 — roughly a wash once the threshold is free to move.

### Next

1. **Make the uncertain-band policy a per-clip operating point**, chosen from the dump.
   Neither setting dominates and the difference is 10 recall points either way.
2. **Fix stage 1 rather than working around it.** Both the ring restriction and the band
   refusal are compensations for a split that is `blue sky vs everything` where the branch
   needs `far vs near`. This is the third experiment to land on the same defect.
3. **Keep the per-candidate dump.** Every number in this entry after the renders was a
   re-cut of two CSVs, in seconds. EXP-022 had none and every question cost a 14-minute span.

## Correction to EXP-022 and EXP-023 — "target size" was the detected scale

- **Date:** 2026-09-30
- **Raised by:** the user, on reading EXP-023: *"How is the target less then 8 pixels? I am
  sure it has around 20 pixels."* Correct, and the entries above were wrong.

### What was wrong

EXP-022 and EXP-023 both report a **"measured target size"** of 3.1 px on O4 and 6.2 px on
analog. That is the **DoG's detected scale**, not the target. Read from the labels:

| | labelled box (max of w, h) | detected scale | ratio | ladder tops out at |
| --- | ---: | ---: | ---: | ---: |
| O4 650–964 | median **71.6 px** (17–142) | 3.1 px | **0.048** | 40 px |
| analog 441–800 | median **37.0 px** (18–80) | 6.2 px | **0.163** | 27 px |

Two compounding causes:

1. **The scale ladder tops out below the median target on both clips**, so the detector
   structurally cannot match the airframe as a blob. It fires on some small dark
   sub-feature instead.
2. **`on_drone` grows the box** by `max(10 px, 25%)` before testing, so a 3 px speck
   anywhere in a ~90×64 region on O4 is credited as a hit.

### What this invalidates

- **Every sky-branch recall figure in EXP-022 and EXP-023 means "a candidate landed inside
  the grown box", not "the drone was detected".** The numbers are reproducible and the
  A/B comparisons between conditions remain valid — both arms are credited the same way —
  but none of them is evidence that the branch detects the airframe.
- **The minimum-diameter analysis in EXP-023 cut on detected scale**, which is an artefact
  of the ladder. The tables are arithmetically correct about the detected-scale
  distribution and say nothing about target size. In particular the conclusion "the O4
  target is at the bottom of the scale ladder so size carries no information" describes
  the ladder's floor, not the drone.
- **`MAX_SCALE_FOR_TARGET` is mis-set**, at ~56 px of diameter against O4 targets reaching
  142 px. It would reject the real airframe at close range if the ladder were extended.
  Ceiling and ladder have to be fixed together, which is why no large-blob rule was
  changed here.

What is **not** affected: the σ_ring regime-inversion measurement (1.48 on sky against
11.86 and 7.41 on scene), the false-alarm curves, the c-vs-Gaussian comparison, and the
EXP-021 / EXP-022 scene-branch and layered-homography results. None of those depend on
target size.

### Two further defects fixed at the same time

- **The render was drawing rejected candidates.** ~190 per frame (141 `uncertain band`,
  50 `cloud`, 1.4 `bloom`) against ~10 kept, so the picture looked full of detections while
  the report said 5/frame — the user asked why, and the answer was that the two were
  describing different things. Rejections are now drawn only with `--show-rejected`.
- **Report sections printed for conditions the run did not apply.** With a whole ring there
  are no stage-1 labels, yet the per-label breakdown printed three zeros and the
  "what the depth-aware ring changed" block printed an identity — both reading as measured
  null results. Each section is now gated on the condition it describes.

### Rolled back, and why

The user asked to roll back the difference between EXP-022 and EXP-023. **Defaults now
reproduce EXP-022** — whole ring, uncertain band scored — with both EXP-023 changes behind
`--depth-ring` and `--refuse-uncertain`. The measurements justifying them stand: neither
wins on both clips at matched false-alarm rate, so neither is a safe default, and the
auditable default is the one already in the ledger. Verified: the rolled-back run returns
73/224 at 5.16 FA/frame on analog, EXP-022's numbers exactly.

### The candidate dump, as asked

Every candidate is now recorded, kept **and** rejected, with its `reason`, `kept` flag,
`sigma`, `response` and all ring/core quantities — the earlier dump held only kept rows, so
a threshold could be raised from the file but never lowered and no rejection could be
audited. Recording literally everything gave 530,273 rows and 59.6 MB on analog, 94.5% of
it `below threshold` at c ≈ 1.4 — beneath the **c = 1.8 ceiling measured on pure Gaussian
noise sky**, so those rows carry no information. `--dump-floor` (default 3.0, half the
operating threshold) always keeps structural rejections and anything on target regardless:
44,988 rows and 4.8 MB, a 12× reduction with nothing auditable lost.

### Next, in order

1. **Extend the scale ladder to cover 3–150 px and re-measure.** Nothing about the sky
   branch's recall means what it says until this is done, and no threshold, floor or
   ceiling should be tuned before it.
2. **Then revisit the size floor and the large-blob ceiling together**, from true sizes.
3. **Tighten `on_drone` for this branch**, or report the detected-scale-to-true-size ratio
   beside every recall figure. The run now prints the ratio and refuses to call it a
   detection when it is below 0.5.

### 2026-10-04 — rendered at the dump floor, c >= 3

The user asked for the sky-branch video without the c >= 6 cut, and chose the dump floor
(c >= 3) over no threshold (1473 candidates/frame) and over drawing only what c >= 6 drops.
`overlay_sky --contrast 3 --no-sky --no-truth` on `catch_2` 441-800, clean look:
`runs/sofa_analog/exp023_sky_branch/clean_c3_catch_2_441_800.mp4` with `clean_c3_analog.log`.

| catch_2 441-800 | kept/frame | drone frames (of 224) | `c` on drone, median |
| --- | ---: | ---: | ---: |
| c >= 6 (shipping) | 5.5 | 73 (32.6%) | 8.2 |
| **c >= 3 (this video)** | **45.5** | **130 (58.0%)** | 5.9 |

Lowering the cut buys 57 drone frames for 8x the load. Cloud rejections are unchanged
(75.55/frame), and the run's own false-alarm curve gives 5.16/frame at c >= 6, EXP-022's
figure exactly, so this is the same pass with a lower cut. Circles are drawn thicker at
twice the threshold, so **in this video the bold circles are exactly the c >= 6 set**, and
the thin ones are what the shipping cut removes. Recall still means "a candidate landed in
the grown box" (detected/true size 0.154), as corrected above.

## EXP-024 — window length: worse than 5 on O4, and no usable setting at all on analog
(where 1 frame in 6 is a duplicate, and the 2-frame sky branch beats it anyway)

- **Date:** 2026-10-01
- **Question:** the user asked for a 10-frame and a 15-frame persistence gate, after a
  design discussion in which **I recommended lengthening the window** on the grounds that
  a target's accumulated displacement grows with window length while its positional
  scatter does not. Does a longer window actually buy anything?
- **Model / weights:** none. No learned component.
- **Data:** `data/processed/SOFA-O4/videos/first_catch.avi` frames 650–964 (315 frames,
  257 labelled, the drone a candidate in 190 of them), `tau = 1.661`, appearance radius
  14 px, fb gate 1.00 px, **direction test off**. 41,876 seeds = 132.9 candidates/frame,
  identical across all three window lengths, so only the gate differs.
- **Hardware / cost:** i7-1255U CPU, ~6–9 min per rendered span, ~5 min per `--no-video`
  pass. Under an hour for everything below.
- **Scripts:** `experiments/exp024_window_length/` — `overlay_window.py` forked from
  EXP-017's so that experiment stays frozen, plus a `clipcfg.py` per clip under
  `o4_first_catch/` and `analog_catch_2/`. Two changes from EXP-017's renderer, documented
  in that folder's README: a `--direction` flag (default off, matching `overlay_stage2.py`)
  and an operating-point table printing **every** `min_appear` threshold from one pass.
  **Written before the 2026-10-01 restructure** (commits `0d5d729`, `0f98eb2`) that moved
  every experiment script out of gitignored `runs/` into tracked `experiments/`; the O4
  artifacts named in the first version of this entry — `window10_first_catch_650_964.mp4`
  and the other three — were deleted with the O4 run folders and can be regenerated from
  the commands in the folder README. Analog artifacts survive:
  `runs/sofa_analog/exp024_window_length/window{10,15}_catch_2_441_800.mp4` and seven logs.

### The whole curve, three window lengths, one span

`appearances` does not depend on `min_appear` — `Track.verdict` only compares against it —
so one render per window length measures the entire load-versus-recall curve. Drone recall
is out of the 190 labelled frames where the target is a candidate at all.

| load/frame | k=5 | k=10 | k=15 |
| ---: | :--- | :--- | :--- |
| ~133 | 1/5 → 100% | 1/10 → 100% | 1/15 → 100% |
| ~60–67 | 2/5 → **71%** | 2/10 → 72% | 2/15 → 72% |
| ~39–40 | 3/5 → **55%** | (3/10 → 56% at 47.6) | 4/15 → 45% |
| ~26–29 | 4/5 → **42%** | 5/10 → 33% | 6/15 → 29% |
| ~14 | 5/5 → **25%** | 8/10 → 19% | 10/15 → 17% |
| ~9.8 | *unreachable* | 9/10 → **17%** | 11/15 → 13% |
| ~3–6 | *unreachable* | 10/10 → 7% | 15/15 → 1% |

**k=5 is at least as good at every load it can reach, and strictly better below ~40.** At
matched load ~26–29 it holds 42% against 33% and 29% — nine and thirteen points. The three
converge only at the loosest setting (2/k, 71–72%), where the gate is barely filtering.

The one thing a longer window offers is **reach**: k=5 has five thresholds and 5/5 is its
floor at 14.3 candidates/frame. Below that only a longer window can operate — k=10 at 9/10
gives 17% at 9.8/frame. That is a granularity argument, not a motion argument, and it
arrives with less recall than k=5's floor.

### Why longer is worse, and why I predicted the opposite

The recommendation I gave before this run was argued from **motion-magnitude SNR**: a
target's net displacement grows roughly linearly with window length (16.8 px over 5 frames,
35.7 over 15, measured from the labels) while positional scatter stays ~10 px. That is true,
and it is about a test that **does not exist in the code**.

The gate that ships is persistence, and it degrades with window length for a reason that
runs the other way: a longer window requires the detector to fire on the target in *more
frames*, and those firings are intermittent. `4/k` returns the same recall at every k —
the same target frames satisfy it — while admitting steadily more noise that happened to
flash four times somewhere in a longer span. Lengthening the window therefore moves the
curve down and right. Two different tests; I applied one's reasoning to the other.

### The finding that outlives the comparison: LK tracks die at three frames

`usable residual steps per seed: median 3` at **k=5, k=10 and k=15 alike** — 47% of seeds
reach 4+ steps at k=5, 37% reach 9+ at k=10, 32% reach 14+ at k=15. Lengthening the window
does not lengthen the tracks.

So the accumulated residual a long window exists to compute is mostly unavailable, and the
motion-magnitude test proposed above cannot be built on `_track_back` at all. Its only
viable form is **chaining candidate peaks** by position across frames — which needs no
texture and no optical flow, and which `window._appears` already does for one frame at a
time without anything chaining it.

### Span dependence, stated because it nearly produced a wrong answer

On frames 900–964 alone — the easy end, target 70–107 px — the strict settings converge:
52% / 51% / 48% at load ~14 for k=5 / 10 / 15, a one-to-three frame spread on n=65. An
early draft of this conclusion rested on that sub-span and read it as "tied at the strict
end". On the full 315 frames, where the target is 19 px at frame 708, the same comparison
is 25% / 19% / 17%. **Any window-length claim from 900–964 is a claim about large targets.**
The same caution applies to EXP-021's headline 78% and 31.2/frame, which are that sub-span.

### The analog clip answers differently: nothing wins, because the curve collapses

`catch_2` 441–800 (360 frames, the drone a candidate in 174 of them), 60,338 seeds =
167.6/frame, appearance radius **9 px** — the same physical radius as O4's 14 px read on a
960 px picture. Stage-0 coverage reproduces EXP-021's analog run exactly (hud 11.64%,
prop 0.44%, ladder 3.21%, edge 7.63%, union 19.49%), which is what says the rebuilt clip
config is the same one.

| load/frame | k=5 | k=10 | k=15 |
| ---: | :--- | :--- | :--- |
| ~168 | 1/5 → 100% | 1/10 → 100% | 1/15 → 100% |
| ~18–23 | 2/5 → 34% | 2/10 → **37%** | 2/15 → **37%** |
| ~4–8 | 3/5 → 21% | 3/10 → 22% | 3/15 → 22% |
| ~1.1–3.6 | 4/5 → 13% | 4/10 → 17% | 4/15 → 17% |
| ~0.1–1.9 | 5/5 → 2% | 5/10 → 10% | 5/15 → 10% |

**At matched load the three window lengths are indistinguishable** — interpolating k=10 to
k=5's 18.1/frame gives ~33.5% against 34%. So the O4 conclusion ("longer is strictly
worse") does **not** transfer; on analog longer is merely *not better*.

What dominates instead is the shape of the curve. **1/k keeps 100% of the drone at 167.6
candidates/frame, and 2/k keeps 37% at ~20.** Two-thirds of the target is lost at the first
real threshold, and the default 4-of-5 sits at 13%. There is no operating region on this
clip where persistence both filters and keeps the target. EXP-021 published the 4-of-5
figure (1.1/frame, 13%) without the curve around it, so this is the first statement of how
steep the drop is.

### How much of that is the forward-backward gate, measured rather than assumed

`usable residual steps per seed: median 0` at k=5, 10 and 15 — against median 3 on O4. The
analog gate is `px(1.0) = 0.67 px` on a 960 px frame, and EXP-017's own `--fb-max` help
text warns it "on a noisy CVBS capture rejects nearly every track". That matters structurally:
`build_tracks` uses LK to find *where* to test `_appears`, so a track that dies at step 0
has a NaN position, scores no appearance, and is dropped as a flash however reliably the
detector fired on it. The 99.3% flash rate could therefore have been an optical-flow
artifact rather than target intermittency.

Re-run at `--fb-max 3.0`:

| | seeds at full depth | load/frame | drone kept |
| --- | ---: | ---: | ---: |
| k=5, fb 0.67 | 13% got 4+ | 1.1 | 13% |
| k=5, fb 3.0 | 28% got 4+ | 1.9 | 16% |
| k=10, fb 0.67 | 6% got 9+ | 2.9 | 17% |
| k=10, fb 3.0 | **16%** got 9+ | 5.9 | **25%** |

**The gate is throttling LK — and it is not the explanation.** Surviving tracks more than
double, but load rises with recall: at matched load ~6/frame it is 25% against 22%, about
three points. Median usable steps stays 0 at a 4.5× looser gate. So the analog collapse is
mostly the detector not firing on the target consistently, not flow failure. Three points
is still worth having, and `--fb-max` on analog should be revisited on its own rather than
inside a window-length question.

### What this pair of clips says together

Persistence is cheap and sound on O4 at k=5 and **has no usable setting on analog at any
window length**. That is a stage-1/stage-2 problem, not a window-length problem: the analog
target is not a candidate often enough for "appeared in m of k frames" to separate it from
noise, and no choice of k or m repairs that. The next move on analog is upstream — why the
detector misses the target in 186 of 360 frames — and, for both clips, the peak-chaining
association the O4 half of this entry argues for, which needs neither texture nor flow and
would make persistence independent of LK entirely.

### Added 2026-10-01: 5-of-7 and 5-of-10 on analog, rendered for viewing

The user asked for videos at 5/10 and 5/7. Same span, radius 9 px, fb gate 0.67 px, no
direction test, drawn in the clean look (red 10 px circles on kept tracks only). k=7 had
not been run before; its one pass gives the whole curve:

| need | k=7 load/frame | k=7 drone kept | k=10 load/frame | k=10 drone kept |
| ---: | ---: | :--- | ---: | :--- |
| 2 | 20.2 | 63 (36%) | 21.6 | 64 (37%) |
| 3 | 5.8 | 38 (22%) | 6.8 | 38 (22%) |
| 4 | 2.1 | 28 (16%) | 2.8 | 30 (17%) |
| **5** | **0.8** | **18 (10%)** | **1.3** | **18 (10%)** |

Out of the 174 labelled frames where the target is a candidate at all. **5/7 keeps the same
18 drone frames as 5/10 at 60% of the load**, and k=7 is otherwise k=10's curve shifted to
lower load. Neither changes the conclusion above: at 5 appearances either gate keeps the
target in one frame in ten. Artifacts in `runs/sofa_analog/exp024_window_length/`:
`window7_need5_catch_2_441_800.mp4` (+ `.log`) and `window10_need5_catch_2_441_800.mp4`, a
copy of the clean-redrawn `window10_` render, which already ran at need 5. Added
2026-10-04: `window5_catch_2_441_800.mp4` (+ `.log`) at the defaults, k=5 need 4 in the
clean look — the render `curve_k5.log` named but never wrote, having run `--no-video` —
and the per-seed dumps `seeds_k5_catch_2_441_800.csv` / `seeds_k7_...csv` behind the
contrast table below.

### 2026-10-04 — what survives scores what, and 1 frame in 6 is a duplicate

Asked: at k=5 against a two-frames-longer window, how many false alarms, and what does
the drone score. `overlay_window.py --dump` (added for this, documented in the folder
README) writes one row per seed per rendered frame with its contrast, so both are
readable at every threshold from a single pass. Both passes reproduce the earlier logs
exactly — 60338 seeds, identical operating points — and counting dump rows with
`appearances >= m` reproduces each OPERATING POINTS row, which is what ties the dump to
the pass that produced it.

| gate | kept/frame | **false/frame** | drone frames | drone `c` | FA `c` p50 | FA `c` p90 | drone rank | drone is #1 |
| ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| 3/5 | 4.25 | 4.14 | 36 (21%) | 3.89 | 2.61 | 4.48 | #3 | 28% |
| **4/5** | 1.09 | **1.03** | **22 (13%)** | **4.44** | 2.77 | 5.05 | **#1** | 50% |
| 5/5 | 0.11 | 0.10 | 3 (2%) | 4.37 | 2.69 | 5.13 | #1 | 67% |
| 4/7 | 2.08 | 2.00 | 28 (16%) | 4.37 | 2.72 | 4.90 | #2 | 39% |
| **5/7** | 0.84 | **0.79** | **18 (10%)** | **4.48** | 2.79 | 4.93 | **#1** | 56% |
| 6/7 | 0.30 | 0.26 | 14 (8%) | 4.50 | 2.79 | 5.28 | #1 | 86% |

`c` is median contrast; rank is the drone's median rank by contrast among that frame's
survivors; "drone is #1" is how often it is the strongest survivor in its frame.

**At matched false-alarm load the two windows are a wash.** Interpolating k=7 between 5/7
and 4/7 to k=5's 1.03 false alarms/frame gives ~12.0% against 4-of-5's 12.6% — a
difference of one frame in 174. k=5 is marginally ahead, the same direction as O4 but far
smaller. 5-of-7 is the tighter *absolute* setting: 0.79 false alarms/frame against 1.03,
bought with 4 drone frames.

**The drone's contrast is a detector property, not a window one** — 4.44 at 4-of-5 against
4.48 at 5-of-7. The window selects which seeds survive; it does not change their strength.
Against the false alarms it keeps, the drone sits at ~1.6x their median but just *below*
their p90, so roughly one false alarm in ten outscores it. Contrast narrows the field, it
does not finish the job.

**Ranking after persistence does not pay.** Keeping only the strongest survivor per frame
halves the false alarms and halves recall with it, landing on 6% at every setting tried
(0.39-0.73 false alarms/frame at 3/5, 4/5, 4/7 and 5/7 alike). Not an operating point.

#### 1 frame in 6 of catch_2 is a duplicate, and it caps every gate above

148 of 890 frames are near-duplicates of their predecessor — mean |diff| 0.32-0.94 against
8-26 for real frames — at `f % 6 == 4` without a single exception (147 gaps, all of them
6). That is 25 fps content resampled into a 30 fps container. **A duplicate pair yields
zero candidates**, and all 60 zero-seed frames in the k=5 dump are exactly those frames.

So the effective window is `k * 5/6`, and every `m/k` above is stricter than it reads:

| k | live frames per window | strictest reachable |
| ---: | :--- | :--- |
| 5 | 4 live in 300 of 360 windows, 5 live in 60 | 5/5 reachable in **17%** of windows |
| 7 | 6 live in 300 of 360 windows, 5 live in 60 | 7/7 reachable in **0%** |

**7-of-7 kept 0 of 60338 seeds** — that is arithmetic, not sampling: every 7-frame window
contains at least one duplicate. 5-of-5's 2% is the same effect. And 4-of-5 is secretly a
*perfect-run* requirement — 4 appearances out of 4 live chances in 83% of windows — which
is why it costs 9 points against 3-of-5 for a 4x load reduction.

**It is the analog capture chain, not this file.** First 400 frames of every clip:

| set | clips | duplicates | dominant gap |
| :--- | ---: | ---: | :--- |
| SOFA-ANALOG | 9 of 9 | 16.8-17.8% | 6 |
| SOFA-O4 | 0 of 6 | 0.0-0.8%, uncadenced | none |

So **O4's numbers are unaffected** — `first_catch` has 3 isolated repeats in 400 frames, no
cadence — and every analog clip has it. The *phase differs per clip* (`catch_2` repeats at
`f % 6 == 4`, `catch_6` at `% 6 == 2`), so a fix must **detect** duplicates rather than
assume a phase, and because it is the capture chain it belongs at decode in `src/data/`
rather than in an experiment.

This is the specific mechanism behind "the problem is upstream", and it is not the
detector's fault. **Deduplicate before windowing, or count `m` against live frames.** Until
then no analog persistence number should be read as if the window were k frames long.
Unmeasured: whether deduplicating recovers recall. It needs a change to how the window is
built, so it is a separate run, not a flag.

### 2026-10-04 — against EXP-023's 2-frame sky branch, at matched load

Asked for directly. Both on `catch_2` 441-800, both recut from dumps rather than re-run:
EXP-023's `candidates_catch_2_441_800.csv` (44988 candidates, floor c=-0.96, so the whole
curve is in it) and EXP-024's new `seeds_k5_` dump. The EXP-023 dump reproduces its
published operating point exactly -- 73 of 224 at c>=6 -- which is what licenses recutting it.

**Both recomputed out of the same 224 labelled frames.** EXP-024's own reports divide by
174, the frames where the target is a *motion* candidate at all; that is the window's
ceiling, not a denominator, and using it was what made the two look incomparable.

| load/frame | 5-frame window | 2-frame sky branch | sky `c >=` |
| ---: | ---: | ---: | ---: |
| 167.6 | 1/5 -> 174 (77.7%) | **223 (99.6%)** | floor |
| 18.1 | 2/5 -> 60 (26.8%) | **99 (44.2%)** | 4.10 |
| 4.25 | 3/5 -> 36 (16.1%) | **63 (28.1%)** | 6.58 |
| 1.09 | 4/5 -> 22 (9.8%) | **38 (17.0%)** | 9.26 |
| 0.11 | 5/5 -> 3 (1.3%) | **8 (3.6%)** | 15.89 |

**The 2-frame branch wins at every matched load, by about 1.7x throughout.** At its own
shipping point -- c>=6, 5.27 false alarms/frame, 32.6% -- the window needs 18.0 candidates
per frame, 3.2x the load, to reach a *lower* 26.8%. There is no load at which the 5-frame
window is the better instrument on this clip.

**Why, and it is not the gate:** the motion front-end makes the target a candidate in only
**174 of 224 labelled frames (77.7%)**, so that is the hard ceiling before persistence
rejects anything. The sky branch reaches **223 of 224 (99.6%)** at its floor. The window is
competing from 22 points behind with a gate that can only subtract.

**The duplicates hit only the motion branch.** The sky branch keeps 5.38 candidates/frame on
the 60 duplicate frames against 5.63 on live ones -- essentially unaffected, because
`silhouette.detect` reads the current frame alone and the pair is used only to warp the
stage-0 mask. It finds the target in **12 of those 60 frames, where the motion branch finds
nothing at all**. Single-frame appearance is structurally immune to a repeated frame.

**One thing the window does better.** Scale-free separation -- the share of clutter reaching
the target's p10, lower being better -- is **53.0% for the window's `c` at 4-of-5 against
82.0% for the sky's at c>=6**. What survives persistence is better ordered by contrast than
what survives the sky threshold. Read it as a hint and not more: the two statistics are
different quantities on different populations, and the window's is 22 target frames.

**Caveat on what this does and does not compare.** These are different detectors -- a motion
map at tau=1.661 versus single-frame appearance contrast -- so this is a pipeline comparison
at matched load, *not* a 5-frames-versus-2-frames ablation. It does not say short windows
beat long ones; EXP-024's own k=5/7/10/15 curves are the ablation, and they say window
length barely matters here. What it says is that the whole motion-plus-persistence path is
the weaker of the two on analog.

### Added 2026-10-04: the loose end — 2/3, 2/4, 2/5 and 3/5 on analog, rendered for viewing

The user asked for videos at these four gates. Same span, radius 9 px, fb gate 0.67 px, no
direction test, clean look. k=3 and k=4 had not been run before; each one pass gives its
whole curve, and their `--dump` CSVs reproduce those curves row for row. Drone frames are
shown out of 174 (the motion front-end's ceiling, as the rest of this entry uses) and out of
224 (every labelled frame, as the sky comparison above uses), with the sky branch recut from
EXP-023's dump to the same load:

| gate | load/frame | drone /174 | drone /224 | sky branch at that load |
| ---: | ---: | ---: | ---: | ---: |
| **2/3** | **13.8** | 57 (33%) | 25.4% | 95 (42.4%) at c>=4.46 |
| **2/4** | **16.4** | 60 (34%) | 26.8% | 98 (43.8%) at c>=4.22 |
| **2/5** | **18.1** | 60 (34%) | 26.8% | 99 (44.2%) at c>=4.09 |
| 3/4 | 3.1 | 32 (18%) | 14.3% | 56 (25.0%) at c>=7.09 |
| **3/5** | **4.2** | 36 (21%) | 16.1% | 63 (28.1%) at c>=6.58 |
| 3/3 | 1.5 | 17 (10%) | 7.6% | 42 (18.8%) at c>=8.51 |
| 4/4 | 0.4 | 8 (5%) | 3.6% | 13 (5.8%) at c>=12.05 |

Bold rows are the four rendered. **At two appearances, shortening the window is close to
free:** 2/3 sheds 24% of 2/5's load (13.8 against 18.1 per frame) for 3 drone frames
(57 against 60). The same target frames satisfy "twice" whatever k is, and a shorter window
gives clutter fewer chances to flash twice. That is the O4 mechanism above, here only at the
loose end. **3/4 against 3/5 is a straight trade**, 4 frames for 1.1 load/frame, with
nothing to choose between them.

**None of it changes the analog verdict.** The sky branch is still 1.6-1.8x the recall at
every one of these loads. The duplicates bite harder at short k by the same arithmetic as
before: a duplicate lands in 3 of 6 windows at k=3 and 4 of 6 at k=4, so 3/3 and 4/4 are
unreachable in half and two-thirds of windows respectively (not measured separately).
Artifacts in `runs/sofa_analog/exp024_window_length/`:
`window{3,4,5}_need2_catch_2_441_800.mp4` and `window5_need3_catch_2_441_800.mp4`, each with
its `_analog.log`, plus `seeds_k3_` / `seeds_k4_catch_2_441_800.csv`.

**Where the drone ranks among the survivors** (asked for as histograms, 2026-10-04).
`experiments/exp024_window_length/rank_hist.py` reads the dumps, ranks each frame's survivors
by the detector's own score, and drops each of the 224 labelled frames into one bar: the
drone's best rank, "filtered out" (a candidate on it existed, none survived) or "not a
candidate". Figure: `runs/sofa_analog/exp024_window_length/rank_hist_catch_2_441_800.png`.

| gate | survivors/frame | kept | #1 | top 3 | median rank | filtered out | not a candidate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2/3 window | 13.8 | 57 | 5 | 20 | 5 | 117 | 50 |
| 2/4 window | 16.4 | 60 | 4 | 20 | 5 | 114 | 50 |
| sky, c >= 6 | 5.5 | 73 | 31 | 61 | 2 | 150 | 1 |
| sky, c >= 4.46 (2/3's load) | 13.8 | 95 | 31 | 62 | 2 | — | 1 |

**When the window gate keeps the drone, it does not rank it.** Its ranks are flat from 1 to
8, #1 in under 1 frame in 10 of those it keeps, so the motion-peak height says almost
nothing about which survivor is the target. The sky branch keeps the drone at #1 in 31 of
73 and in the top 3 in 61. **Its ranks do not depend on the load:** loosened to the window's
13.8/frame, it gains 22 frames, only one of them in the top 3 (61 to 62), and its #1 count
does not move. Ranks are each detector's own `c`, so they compare across panels; the
scores do not.

**Ranked by the sky branch's `c` instead** (`rank_hist.py --rank-by sky`, figure
`rank_hist_skyc_catch_2_441_800.png`). The window still decides what survives, and each
survivor takes the `c` of the nearest EXP-023 candidate within 9 px. That dump holds every
blob down to c=-0.96. Blobs vetoed as `cloud` (large and soft-edged, a shape rule the branch
applies at any `c`) are left out. A survivor with no blob that close ranks last, behind any
clutter it ties with.

| gate, ranked by sky `c` | kept | #1 | top 3 | median rank | survivors with no sky blob |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2/3 window | 57 | **38** | 46 | 1 | 4506 of 4982 (90%) |
| 2/4 window | 60 | **39** | 49 | 1 | 5357 of 5906 (91%) |

From #1 in 5 frames to 38, but read it for what it is: **90% of the window's survivors have
no sky blob within 9 px**, so the ranking is mostly the intersection of the two detectors.
That intersection, unthresholded, is a candidate set in its own right, and **it beats the
sky branch alone at matched load**:

| | load/frame | drone frames /224 |
| --- | ---: | ---: |
| 2/3 window, with a sky blob within 9 px | 1.32 | **48** |
| sky branch alone, c >= 8.79 | 1.32 | 39 |
| 2/4 window, with a sky blob within 9 px | 1.52 | **52** |
| sky branch alone, c >= 8.47 | 1.52 | 42 |

Counted per blob: a frame counts when a blob on the drone is in the set, the same test both
rows use. Nine to ten frames at a load of about 1.4 per frame, on one clip, and recut from
dumps rather than run as a pipeline. The first version of this table kept cloud-vetoed blobs
(1.43 and 1.66/frame against 42 and 43). Dropping them removed only clutter: the drone's
frames and ranks did not change. That is the first evidence that motion persistence adds
something the sky branch's contrast does not. It is also a cheaper form of the open
follow-up, persistence on the sky branch: this AND needs no chaining. The drone is lost
from the intersection in 9 frames of 57 (2/3) where the window kept it but no sky blob lay
within 9 px.


### 2026-10-04 — would displacement magnitude be a better statistic than direction?

Asked: why not follow each candidate across the k frames and use how far it moved. The
pipeline already computes that — `Track.total_mu`, the ego-compensated displacement
accumulated into the reference frame — but its magnitude is only a veto
(`min_displacement = px(2.0)`, "displacement too small"); only its *direction* is tested.
So this measures whether promoting magnitude to the discriminator would be better. Sampled
on `catch_2` 441-800: label centres for the target, `accumulate_grid` for the background.

| displacement over 5 frames | n | p10 | median | p90 |
| :--- | ---: | ---: | ---: | ---: |
| target, raw image space | 52 frames | 6.29 | 18.51 | 36.87 px |
| target, ego-compensated (label centres) | 52 frames | **13.85** | **28.38** | 97.97 px |
| background grid, tracked all 5 frames | 19 windows | 2.38 | **3.18** | 5.60 px |
| target, from a full-depth track | **2 windows** | 100.67 | 107.46 | 114.25 px |

**Magnitude separates where direction does not.** The target's p10 (13.9 px) is above the
background's p90 (5.6 px) — non-overlapping tails, ~9x at the medians — while the epipolar
direction test on the same data runs at 30.7% against a 22.8% chance floor. Compensation
*amplifies* the target's motion (28 px against 18 px raw) because the camera is chasing it,
so removing ego-motion adds to the target's apparent travel. 100% of sampled frames clear
the 2 px veto and 92% clear 10 px.

**And it is unavailable.** The target had a full-depth track in **2 of 19 sampled windows**,
matching the k=5 log's "13% got 4+". The discriminator is strong and measurable about a
tenth of the time, which is the same LK-survival wall as everything else here. Those 2
windows read 107 px against the labels' 28 px, so they are not representative — likely
drifted tracks accumulating junk — and the ratio from them should not be quoted.

**The caveat that would decide it.** The background grid is *not* the false alarms:
candidates are peaks selected for high motion-map response, so surviving clutter is a
biased, higher-residual subset of the static field. Separation against grid points is the
optimistic case. Separation against surviving candidates is unmeasured, and is the number
that settles whether magnitude should replace direction. Measuring it needs `total_mu` in
the seed dump — a one-line change — plus a pass on O4, where tracks reach median depth 3
rather than 0.

**Forward vs backward tracking is a latency choice, not an information one.** Seeding from
the newest frame and tracking back yields a verdict for the current frame; seeding at `f`
and following forward yields it at `f + k` (167 ms at 30 fps) and only about candidates `k`
frames old. Same pairs, same information, and the window is buffered either way.

**Peak-chaining looks arithmetically viable as the replacement for LK.** The target's
frame-to-frame step is median 4.12 px, p90 12.67, so a ~13 px chaining radius catches 90% of
steps; at 167.6 candidates/frame on 960x720 that radius admits 0.13 spurious candidates per
step (a ~13% false-link rate). No texture, no flow, immune to the duplicate frames, and one
chain yields both the displacement and the appearance count.


### Standing recommendation

**On O4, keep k=5 with 4-of-5. Do not lengthen the window.** Revisit only if a downstream
stage needs under 14 candidates/frame, where k=10 at 9/10 is the only option and costs 8
points of recall against k=5's floor.

**On analog, do not tune the window at all — persistence has no usable setting there.**
Every window length keeps at most 37% of the target once the gate filters anything, and
the shipping 4-of-5 keeps 13%. Spending effort on k or m on this clip is spending it on
the wrong stage. Two things to do instead: raise `--fb-max` off its 0.67 px default (worth
~3 points of recall at matched load, measured above, and it should be decided on its own
terms), and look upstream at why the target is not a candidate in 186 of 360 frames.

**And before either, deduplicate.** 1 frame in 6 of this clip is a repeat carrying
no motion, so a nominally 5-frame gate is really a 4-frame one and the strictest
settings are partly unreachable by arithmetic. That is cheap to fix and every analog
persistence number above is measured through it.

**But the deeper point, measured 2026-10-04: on analog the whole motion-plus-persistence
path is dominated by EXP-023's 2-frame sky branch at every matched load** (~1.7x the recall
throughout, and 32.6% at 5.3 false alarms/frame against the window's 26.8% at 18.0). The
motion front-end tops out at 77.7% of labelled frames before the gate subtracts anything,
against the sky branch's 99.6%. So the analog question is not *which window* but **whether
persistence helps the sky branch** — the branch that is already winning, and that is immune
to the duplicate frames because it reads one frame at a time. That is the experiment to run
next on this clip, and it makes deduplication a prerequisite only for the motion path.

**Next, and separately from window length:** peak-chaining association, then accumulated
displacement and size growth measured off it. EXP-023 established size as evidence; the
labels give median growth +0.14 px over 5 frames against +6.34 over 15, so growth needs a
long baseline while persistence wants a short one. That tension — not the window length —
is the real open question.

## EXP-025 — the sky branch, top 3 per frame: half the load, 84% of the hits

- **Date:** 2026-10-01
- **Question:** the user asked for a video showing only each frame's top 3 targets by score,
  labelled so each can be told apart. If a downstream stage takes only a frame's N
  highest-contrast candidates, how often is the drone among them, and how often is it #1?
- **Model / weights:** none. EXP-023's sky branch at EXP-022 defaults (whole ring, c >= 6,
  uncertain band scored, no size floor), with a per-frame cap of 3 applied **after** the
  threshold and ranked by contrast `c`.
- **Data:** `data/raw/SOFA-ANALOG/videos/catch_2.mp4` frames 441–800 (360 frames, 224
  labelled).
- **Hardware / cost:** i7-1255U CPU, one render of the span (EXP-023's cost; the cap is free).
- **Scripts:** `experiments/exp025_top3/overlay_top3.py`. Artifacts in
  `runs/sofa_analog/exp025_top3/`: `top3_catch_2_441_800.mp4`, `top3_analog.log`, and
  `top3_catch_2_441_800.csv` (one row per ranked candidate: frame, rank, x, y, c, diameter,
  on_target).
- **Drawing:** the clean look. Each shown candidate is a red circle (at least 10 px across)
  tagged `#rank c`, with `#1` thicker. Nothing is drawn from the labels.

### Result

| | per frame | drone frames (of 224) |
| --- | ---: | ---: |
| every candidate at c >= 6 (= EXP-023) | 5.48 | 73 (32.6%) |
| **top 3** | **2.88** | **61 (27.2%)** |
| top 2 | ≤ 2 | 49 (21.9%) |
| top 1 | ≤ 1 | 31 (13.8%) |

Rank of the drone when it is shown: **#1 in 31 frames, #2 in 18, #3 in 12.** Before the cap
it reproduces EXP-023 to the digit (73/224, 5.5/frame), which is what makes the cap the only
difference.

**Capping at 3 cuts the load by 47% and keeps 61 of the 73 drone frames (84%).** The drone,
when it is a candidate at all, is usually near the top: it is #1 in 42% of the frames where
it is kept. But contrast is not a confident ranker. In more than half of those frames
(42 of 73) something else outranks it, so a top-1 policy would hand over clutter in most
frames.

### Read with

- **Same caveat as EXP-023:** "drone" means a candidate inside the label box grown by
  max(10 px, 25%), and the detector fires at ~0.16x the airframe's size. A hit says a top
  candidate was in the right place, not that the airframe was detected.
- **Comparable to EXP-024's window gate only after fixing the denominator**, which the
  2026-10-04 addendum to EXP-024 does: EXP-024's printed recall is out of the 174 frames
  where the target is a *motion* candidate, this is out of all 224 labelled frames. On the
  common 224 the sky branch beats the window at every matched load by ~1.7x, and the 174 is
  revealed as the motion front-end's 77.7% ceiling. Still different detectors, so it is a
  pipeline comparison and not a window-length one.
- A top-N cap is a *load* control, not a filter: it never removes a frame's best clutter, so
  the false-alarm count is at least min(3, kept) − 1 in every frame where the drone is shown.

### Next

Rank by a statistic that separates better than `c`. EXP-023 measured 80.9% of clutter
reaching the target's c p10 on this clip. The cap only helps as much as the ranking does,
and a statistic can be tried without a re-render once it
is in a candidate dump. EXP-023's
`runs/sofa_analog/exp023_sky_branch/candidates_catch_2_441_800.csv` already carries `snr`,
`c_plain`, `diameter` and `on_target` for every kept candidate, so re-ranking by any of them
and re-counting top-3 hits is a GROUP BY, not another span.

### Added 2026-10-04: the drone's score in the 163 frames it missed the top 3

Read from EXP-023's dump (`candidates_catch_2_441_800.csv`), which records every candidate
on the target whatever its `c`, so no re-render. It reproduces the counts above (73 kept,
61 in the top 3).

| why it missed | frames | the drone's best `c` there |
| --- | ---: | --- |
| kept (c >= 6) but ranked 4th or lower | 12 | 6.0–8.7, and within 0.0–0.6 of that frame's #3 in 7 of the 12 |
| a candidate, but below c = 6 | 150 | median **2.7** (p10 1.9, p90 5.1, max 5.9); median gap to #3 is 4.6 |
| not a candidate at all | 1 | none |

**The miss is the threshold, not the cap.** The cap costs 12 frames, most of them by a hair.
The other 150 are frames where the drone scores a median 2.7: barely above the 1.8 that
pure-noise sky reaches (EXP-023), and 4.6 below what the clutter in the same frame scores.
Of those, 16 sit at c 5–6 and 26 at c 4–6, so lowering the threshold recovers a few. The
bulk (93 frames at c < 3) is a target the contrast statistic does not see. That points
upstream, at the ranking statistic EXP-025 already names, not at N.

### Added 2026-10-04: a 4-of-5 persistence gate before the top 3 (EXP-025b)

The user asked for the top-3 video behind a 5-frame gate. They chose sky-branch persistence
at 4 of 5. `experiments/exp025_top3/overlay_gate.py` chains each kept candidate (c >= 6)
back through the 4 frames before it. At each step it takes the nearest kept candidate
within 9 px and moves onto it, so the chain follows a moving target. Only candidates with
4 or more appearances are ranked. Same span, same detector. One pass reports every
threshold, and `need 1/5` reproduces the table above to the digit (73 / 61 / 31-18-12).

| catch_2 441–800 | survivors/frame | shown/frame | drone in top 3 (of 224) | #1 / #2 / #3 |
| --- | ---: | ---: | ---: | :---: |
| no gate | 5.48 | 2.88 | 61 | 31 / 18 / 12 |
| 4 of 5, chain camera-compensated | 1.12 | 1.03 | **3** | 2 / 1 / 0 |
| **4 of 5, chain in picture coordinates** | **1.94** | **1.71** | **27** | **20 / 6 / 1** |
| ceiling: drone kept in >= 4 of its last 5 frames | | | 46 | |

**Compensating for camera motion is wrong on intercept footage.** I built the gate
camera-compensated first, which was the option offered and chosen. It kept the drone in 3
frames, because the camera follows the drone: from the labels, the drone's step between
frames is a median 4.1 px in the picture against 6.7 px after compensation, and over the
9 px radius in 20% of frames against 40%. Compensation suits static clutter, not a chased
target. `--coords image` chains in raw picture coordinates; it is the video to watch.

In picture coordinates the gate **cuts the load by 41% (2.88 → 1.71 shown/frame) and keeps
27 of the 46 frames any 4-of-5 gate could keep.** It also improves the ranking: the drone is
#1 in 20 of its 27 frames (74%), against 31 of 61 (51%) without the gate. In 3 frames
(524, 525, 572) it is in the gated top 3 although it was 4th or lower before, because the
gate removed the clutter above it. The cost is recall: 27 of 224 against 61. The drone's
c >= 6 firings come in short runs, broken mostly where it is not kept at all, and at 579/580
where it jumps 12 px in one step. The full per-frame table is in the folder README.

**Duplicate frames inflate this gate.** `catch_2` repeats 1 frame in 6 (EXP-024's 2026-10-04
addendum). The sky branch detects on one frame at a time, so a duplicate repeats the
previous frame's candidates (537/538) and counts as a free appearance. Two windows in three
contain a duplicate pair. This is the opposite of the motion branch, where a duplicate pair
produces nothing. Dropping duplicates before the window is built would make this gate
stricter and its numbers lower.

Artifacts in `runs/sofa_analog/exp025_top3/`: `gate4of5_image_top3_catch_2_441_800.mp4`
and `gate4of5_top3_catch_2_441_800.mp4` (camera-compensated), each with its `.log`, `.csv`
(every kept candidate with `appearances`) and `_drone_ranks.md` from `drone_ranks.py`.

### Added 2026-10-04: the 2-of-4 motion window, ranked by sky contrast, top 3 (EXP-025c)

The user asked for a video: EXP-024's 2-of-4 motion window deciding what survives, the sky
branch's `c` deciding the order, top 3 only. `experiments/exp025_top3/overlay_window_skyc.py`
draws it from two dumps, with no detector re-run: EXP-024's `seeds_k4_` and EXP-023's
candidate dump. A sky blob is **confirmed** when a 2-of-4 survivor lies within 9 px. Confirmed
blobs are ranked by `c` with **no threshold**, and blobs the sky branch vetoes as `cloud` are
dropped, as it drops them at any `c`. Each blob is drawn once. A window survivor with no blob
nearby cannot be ranked and is not drawn.

| catch_2 441–800 | ranked/frame | shown/frame | drone in top 3 (of 224) | #1 / #2 / #3 |
| --- | ---: | ---: | ---: | :---: |
| sky c >= 6, no gate (EXP-025) | 5.48 | 2.88 | **61** | 31 / 18 / 12 |
| sky 4 of 5 in picture coordinates (EXP-025b) | 1.94 | 1.71 | 27 | 20 / 6 / 1 |
| **2-of-4 motion window, ranked by sky c** | **1.52** | **1.29** | **49** | **39 / 8 / 2** |

The drone is confirmed in 52 frames and in the top 3 in 49. **It is #1 in 39, more than
EXP-025's 31, at 45% of the shown load** (1.29 against 2.88 per frame). It gives up 12
top-3 frames to EXP-025, mostly at #2 and #3. It beats the sky branch's own 4-of-5 gate
(EXP-025b) on every column. That gate needs c >= 6 in four of five frames; this one needs
motion in two of four, and no contrast floor.

Not independent of the EXP-024 rank histograms: it is the same intersection, which keeps the
drone in 52 frames at 1.52/frame against the sky branch alone's 42 at that load. The same
caveats apply too: one clip, recut from dumps, one 9 px radius.

Artifacts in `runs/sofa_analog/exp025_top3/`: `window2of4_skyc_top3_catch_2_441_800.mp4`,
its `.csv` (every shown blob: frame, rank, x, y, c, diameter, seeds confirming it,
on_target) and `window2of4_skyc_analog.log`.

### Added 2026-10-04: the frame split into sky and ground, one detector each, top 3 by c (EXP-025d)

The user asked for a video with the frame split into sky and ground: the sky branch on the
sky, the 2-of-4 motion window on the ground, and the top 3 chosen by the sky branch's `c`.
They chose **one pooled ranking per frame**, not a top 3 per section.
`experiments/exp025_top3/overlay_split.py` recomputes stage 1 (`skyline.split`, EXP-017) on
every frame. A blob on a stage-1 sky pixel belongs to the sky section; everything else is
ground, and the uncertain band goes to whichever side stage 1 called. The sky section is
EXP-025's detector (kept, c >= 6). The ground section is EXP-025c's (a non-cloud blob
confirmed by a 2-of-4 survivor within 9 px, any c). Both are drawn from the same two dumps,
so only stage 1 is new computation. Stage 1 called a median 22% of the frame sky, and no sky
at all in 66 of 360 frames.

| catch_2 441–800 | ranked/frame | shown/frame | drone in top 3 (of 224) | #1 / #2 / #3 |
| --- | ---: | ---: | ---: | :---: |
| sky c >= 6 everywhere (EXP-025) | 5.48 | 2.88 | 61 | 31 / 18 / 12 |
| 2-of-4 window everywhere, ranked by sky c (EXP-025c) | 1.52 | 1.29 | 49 | 39 / 8 / 2 |
| **sky branch on sky, 2-of-4 window on ground** | **2.00** | **1.66** | **66** | **52 / 10 / 4** |

**Better than either detector on its own: more top-3 frames than EXP-025 and more #1s than
EXP-025c, at 58% of EXP-025's shown load.** The drone is ranked in 69 frames.

Split by section (`split_rank_hist.py`):

| | sky section | ground section |
| --- | ---: | ---: |
| ranked/frame, shown/frame | 0.71, 0.66 | 1.29, 0.99 |
| drone ranked | 36 | 33 |
| drone in top 3 (#1 / #2 / #3) | 36 (36 / 0 / 0) | 30 (16 / 10 / 4) |
| drone dropped by its section | 9 | 145 |
| no blob on the drone | 0 | 1 |

**The sky branch's clutter was on the ground.** Restricted to stage-1 sky it ranks 0.71
blobs/frame, against 5.48 over the whole frame, and when it ranks the drone, the drone is #1
in all 36 frames. The ground section loses the drone in 145 frames. That is the motion
window's weakness (EXP-024), now confined to the frames where the drone is below the
horizon. Raising recall there is the ground detector's problem, not the ranking's.

Caveats: one clip, recut from dumps; "drone" is EXP-023's grown-box `on_target`. A dropped
drone is credited to the section of its best blob, and a frame with no blob to the section of
the label box's centre. The sky mask is stage 1's `blue sky vs everything` split, which
EXP-023 already named as the defect to fix. A frame where stage 1 finds no sky runs the
window alone.

Artifacts in `runs/sofa_analog/exp025_top3/`: `split_sky_window2of4_top3_catch_2_441_800.mp4`,
its `.csv` (every shown blob: frame, rank, section, x, y, c, diameter, on_target),
`_drone.csv` (per labelled frame: outcome, rank, section, c), `_rank_hist.png` and
`split_analog.log`.

### Added 2026-10-04: EXP-025d with blobs merged at 10 px (20 px across) — no change for the drone

The user asked for the distance threshold at 20 px: every blob within a 10 px radius
summed into one. `overlay_split.py --merge 10` does that **before either section's test**.
Strongest blob first, each anchor takes every remaining non-cloud candidate within 10 px.
The merged blob keeps the anchor's centre, c and section, and grows to cover its members.
It passes the sky test if any member has c >= 6, and the ground test if a 2-of-4 survivor
is within 9 px of any member. It is on the drone if any member is. With `--merge 0` the
script reproduces EXP-025d exactly.

| catch_2 441–800 | candidates folded | ranked/frame | shown/frame | drone in top 3 (of 224) | #1 / #2 / #3 | dropped (sky, ground) |
| --- | ---: | ---: | ---: | ---: | :---: | :---: |
| EXP-025d, no merge | 0 | 2.00 | 1.66 | 66 | 52 / 10 / 4 | 9, 145 |
| **merge 10 px** | **231 (0.64/frame)** | **1.99** | **1.65** | **66** | **52 / 10 / 4** | **9, 145** |

**The merge folds 231 candidates and changes the drone's outcome in no frame.** Ranked
blobs are rarely close to each other: of 509 shown blobs, 6 have another shown blob within
10 px and 41 within 20 px. A first version that merged after the tests, with a 20 px
radius, also left the drone's rank unchanged everywhere. The ground section's 145 drops
are not a fragmentation problem. In those frames the nearest 2-of-4 survivor is a median
79 px from the drone's blob, and 32 frames have no survivor at all. Even a 20 px
seed-to-blob radius would recover at most 13. The motion window does not fire on the drone
there.

Artifacts in `runs/sofa_analog/exp025_top3/`: `split_sky_window2of4_top3_merge10_catch_2_441_800.mp4`,
its `.csv` (adds `members`), `_drone.csv` and `_rank_hist.png`.

### Added 2026-10-04: merge at 20 px (40 px across) — one clean gain, two loose ones

The same run with `--merge 20`, and circles drawn at least 20 px across (`--min-draw 20`,
appearance only; scoring and merging are unaffected).

| catch_2 441–800 | candidates folded | ranked/frame | shown/frame | drone in top 3 (of 224) | #1 / #2 / #3 | dropped (sky, ground) |
| --- | ---: | ---: | ---: | ---: | :---: | :---: |
| EXP-025d, no merge | 0 | 2.00 | 1.66 | 66 | 52 / 10 / 4 | 9, 145 |
| merge 10 px | 231 (0.64/frame) | 1.99 | 1.65 | 66 | 52 / 10 / 4 | 9, 145 |
| **merge 20 px** | **1893 (5.26/frame)** | **1.95** | **1.62** | **68** | **53 / 10 / 5** | **9, 142** |

Four frames change. **Only one is a clean gain**: in 585 the drone's own blob anchors a
6-member merge and moves from #2 to #1. In 581 (ground, #3) and 590 (sky, #2) the anchor is
a clutter blob that absorbed a drone blob within 20 px. The merged blob counts as the drone,
and its grown circle covers the drone, but it is not centred on the drone. 782 moves from
dropped to #5. Counting only anchors on the drone, the top 3 holds it in 66 frames, as
without the merge, and #1 in 53. At 20 px the merge folds most candidates in a frame
(5.26/frame) but changes the shown load by only 0.04/frame. The ground section's drops fall
from 145 to 142, consistent with the motion window, not fragmentation, being the limit.

Artifacts in `runs/sofa_analog/exp025_top3/`: `split_sky_window2of4_top3_merge20_catch_2_441_800.mp4`,
its `.csv`, `_drone.csv` and `_rank_hist.png`.

## EXP-026 — the sky/ground split, #1 only

The user asked for EXP-025d's overlay (sky branch on the sky, 2-of-4 window on the ground)
showing only each frame's #1. `experiments/exp026_top1/overlay_top1.py` runs
`overlay_split.py` unchanged with `--top 1 --merge 20 --min-draw 20`. These are the latest
EXP-025d settings, with circles drawn at least 20 px across.

| catch_2 441–800 | ranked/frame | shown/frame | drone #1 (of 224) | sky / ground | shown blobs on the drone |
| --- | ---: | ---: | ---: | :---: | ---: |
| EXP-025d top 3, merge 20 | 1.95 | 1.62 | 53 | 36 / 17 | — |
| **top 1, merge 20** | **1.95** | **0.79** | **53** | **36 / 17** | **53 of 284** |

The cap does not change the ranking, so the #1 count is EXP-025d's to the digit. This
includes 585, the merge's one clean gain. The video halves the shown load, from 1.62 to
0.79 per frame, and 76 of 360 frames show nothing. Of the 284 circles drawn, 53 sit on
the labelled drone. The sky section is right in 36 of 131, the ground in 17 of 153.

Artifacts in `runs/sofa_analog/exp026_top1/`: `split_sky_window2of4_top1_merge20_catch_2_441_800.mp4`,
its `.csv`, `_drone.csv`, and `top1_analog.log`.

## EXP-027 — the OSD horizon dashes, dropped by the character grid

The user noticed the sky and window branches still firing on the white dots on the
operator's screen. These are the analog OSD's **artificial horizon**: a row of identical
dashes, one character column apart (`960 / 30 = 32 px`), sliding with pitch and roll.
Neither the static HUD mask nor EXP-017's `ladder_mask` covers them at that row. In
EXP-026's video, 120 of the 231 circles drawn off the drone sat in the dash band
(y 240–300, x 330–650).

**Measured before choosing the rule.** This covers all 17,790 non-cloud candidates in EXP-023's
catch_2 441–800 dump (1,633 on the drone), plus EXP-026's 284 shown blobs:

| Rule | on-drone candidates vetoed | shown drone #1 vetoed (of 53) | dash-band FA vetoed (of 120) | other FA vetoed (of 111) |
| --- | ---: | ---: | ---: | ---: |
| `has_twin` (GLAD's, anywhere 0.5–2.5 columns), 0.70 | 705 | 32 | 109 | 81 |
| a row of >= 3 candidate blobs 32 ± 3 px apart, dy <= 6 | 41 | 1 | 70 | 0 |
| **copies at 2 of ±1, ±2 columns (± 3 px), 0.70** | 186 | **0** | **93** | 16 |

- **GLAD's `has_twin` would have deleted the drone.** On 8–14 px blobs, a dark spot
  against plain sky correlates with any other dark spot within 64 px.
- **Polarity does not help.** The sky branch fires on the dash's black outline. All 2,034
  band candidates are darker than their ring, like the drone.
- **The grid does.** Copies have to sit at whole-column offsets, at two positions. A
  free-floating look-alike stops counting.

That rule is now `src.algo.masking.grid_twins` / `on_osd_grid` (unit-tested). It runs as
`overlay_split.py --osd-grid` on a square box of the blob's diameter, clamped to 8–14 px,
before merging and before either section's test. `experiments/exp027_osd_grid/overlay_grid.py`
runs it with EXP-026's settings. The 0.70 threshold is EXP-013's, out of sample, and it
was not re-tuned on this clip.

| catch_2 441–800, top 1, merge 20 | ranked/frame | shown/frame | drone #1 (of 224) | sky / ground | FA shown | in the dash band |
| --- | ---: | ---: | ---: | :---: | ---: | ---: |
| EXP-026 | 1.95 | 0.79 | 53 | 36 / 17 | 231 | 120 |
| **EXP-027 `--osd-grid`** | **1.16** | **0.64** | **57** | **36 / 21** | **175** | **49** |

- With the flag off, `overlay_split` reproduces EXP-026's `.csv` and `_drone.csv` byte for byte.
- The veto drops 16.05 candidates per frame, 307 of them on the drone (0.85 per labelled frame).
  These are low-ranked members. The drone gains #1 in 5 frames (663, 677, 705, 719, 759),
  where a dash had outranked it.
- **It loses one: 585**, where the drone flies through the dash row beside the centre
  marker. Its strongest blob (c 15.0) has dash-outline copies at grid positions. This is
  the cost the rule was expected to have, and it was EXP-025d's one clean merge gain.
- **What is left in the band (49):** mostly the **end dash** of the row (594, 602, 611,
  612, 634). The blob sits on the dash's edge and the analog blur softens it, so its copies
  score 0.55–0.70, just short. Also the **centre reticle's wing** at 651, just outside the
  static mask, and treeline. Next: test the end dashes at a lower score, but on a clip
  other than catch_2.
- The other recurring false alarms are the top corners (928,16) ×32 and (16,16) ×13. They
  are the picture rim, not OSD.

**Top 3** (`overlay_grid --top 3`, the user's follow-up). The candidates are the same, so
the veto counts are the same. Against EXP-025d's top 3 at merge 20:

| catch_2 441–800, top 3, merge 20 | shown/frame | drone in top 3 (of 224) | #1 / #2 / #3 | FA shown | in the dash band |
| --- | ---: | ---: | :---: | ---: | ---: |
| EXP-025d | 1.62 | 68 | 53 / — / — | 513 | 254 |
| **EXP-027 `--osd-grid`** | **1.08** | **67** | **57 / 6 / 4** | **318** | **92** |

A third of the shown load goes, and the dash band loses 162 of its 254 false alarms. The
drone enters the top 3 in 3 frames (703, 777, 782) and leaves it in 4:

- **590** is not a real loss. EXP-025d's circle there was a dash (4 grid copies) that the
  20 px merge credited to the drone because it absorbed a c 0.4 on-drone member.
- **499 and 585** are the drone in the dash row. At 499 its blob sits at y 286 with 4 grid copies.
- **660** is the one false veto: motion-blurred grass, where the drone's own blob (c 2.0)
  found copies at 2 grid positions.

Six rank moves are all upward: 2 → 1 (×4), 3 → 1, 3 → 2.

Artifacts in `runs/sofa_analog/exp027_osd_grid/`: `split_sky_window2of4_top1_merge20_osdgrid_catch_2_441_800`
`.mp4`, `.csv`, `_drone.csv`, and `grid_analog.log`. The same for `top3`, plus
`_rank_hist.png` and `grid_top3_analog.log`.

### Added 2026-10-04: the split on catch_4 and catch_5 — it only helps where stage 1 finds sky (EXP-025e)

The same overlay (`overlay_split.py --merge 20 --min-draw 20`) on the two other labelled
analog clips, by `experiments/exp025_top3/run_clips.py`. The clips are set in
`exp025_top3/clips/clipcfg.py`. The user asked for the unlabelled clips (catch_3/6/7/8,
miss_1/2, FIELD) to be dropped mid-run. The sky-branch candidates are EXP-023's own
dumps (catch_4 159–326, catch_5 69–444), with the same settings as catch_2's. They cover
every labelled frame (209–295 and 119–394). Only the 2-of-4 window and the overlay are new.
A first attempt that re-ran the sky branch from frame 2 was still on catch_4 after an
hour. The opening frames of these clips are slow for it (not profiled).

"Sky alone" is EXP-025's rule over the whole frame (kept, c >= 6, top 3 by c), counted from
the same dumps.

| clip | stage 1 sky, median / skyless frames | shown/frame | drone in top 3 | #1 / #2 / #3 | sky alone: top 3, #1 |
| --- | :---: | ---: | ---: | :---: | :---: |
| catch_2 441–800 (EXP-025d, merge 20) | 22% / 66 of 360 | 1.62 | 68 of 224 | 53 / 10 / 5 | 61, 31 |
| catch_4 159–326 | 0% / 110 of 168 | 1.30 | 14 of 82 | 8 / 3 / 3 | 33, 17 |
| catch_5 69–444 | 7% / 5 of 376 | 0.97 | 31 of 188 | 14 / 10 / 7 | 30, 14 |

**The catch_2 gain does not carry over.** On catch_5 the split ties the sky branch alone
(31 against 30 in the top 3, 14 #1 each) at about a third of its shown load. On catch_4 it
loses more than half (14 against 33). In both, stage 1 calls little or no sky, so nearly
every blob goes to the ground section, and the 2-of-4 window drops the drone there (66 of
82 frames on catch_4, 136 of 188 on catch_5). catch_4 is heavy analog breakup, with frames
of almost pure static (e.g. 260), and stage 1 finds no sky in 110 of 168 frames. On
catch_5, frame 135 has the drone in open sky ranked #1 by the *window*, tagged `gnd`:
stage 1 called only a strip near the tree sky. The split is only as good as the sky mask,
the defect EXP-023 already named. On these two clips the mask, not either detector, decides
the outcome.

Artifacts in `runs/sofa_analog/exp025_top3/{catch_4,catch_5}/`:
`split_sky_window2of4_top3_merge20_<clip>_<span>.mp4`, its `.csv`, `_drone.csv`,
`_rank_hist.png`, `seeds_k4_<clip>_<span>.csv`, and `window_`/`split_<clip>.log`.
`clips_summary.py` prints the per-clip table from the logs.

### Added 2026-10-04: the split on the FIELD capture — the sky branch carries it, the ground stays quiet (EXP-025e)

The same overlay (merge 20 px, circles at least 20 px) on all of
`captured_raw_20260616_040253_004` (frames 2–3600), at the user's request. There are no
labels, so there is no drone rank. Episodes are PROVENANCE's ranges (2–180, 1101–1553,
3140–3600), which come from where GLAD fired, not from ground truth.

**Stage 1 needed a per-clip setting.** "Sky is bright" (luminance above the frame's 55th
percentile) is false here: the deep blue sky is darker than the sunlit hillside, and the
drone in open sky landed in the ground section. `skyline.py` now reads `sky_luma_pctl`
and `sky_texture_pctl` from the clip config, with defaults unchanged, so analog does not
move. FIELD sets 0 and 60, and colour carries the call. Checked by eye on frames 50, 600,
1111, 1300, 1500, 3200 and 3300; the line follows the ridge.

| FIELD, frames | stage 1 sky, median | shown/frame, sky | shown/frame, ground |
| --- | :---: | ---: | ---: |
| all 3599 | 33% (no skyless frame) | 0.30 | 0.01 |
| episode 2–180 | 39% | 0.08 | 0.01 |
| episode 1101–1553 | 34% | 0.64 | 0.02 |
| episode 3140–3600 | 48% | 0.77 | 0.00 |
| outside the episodes (2506) | 29% | 0.17 | 0.02 |

The sky branch fires mostly inside the drone episodes: 0.64 and 0.77 shown/frame there,
against 0.17 outside. Frames 1300 and 3300 show the drone at #1 in the sky. The ground
section is nearly silent at 0.01/frame. The hillside gives ~1,400 candidates/frame, 978
of which the 20 px merge folds away, and the 2-of-4 window confirms almost none of the
rest. Whether the 0.17/frame outside the episodes is clutter or a drone that GLAD missed
needs labels (the open todo to annotate FIELD).

Cost: the sky branch is ~4 s/frame here against ~0.7 on analog, because of the
candidate count. `run_clips.py` runs it and the overlay in 4 parallel frame ranges and
joins them exactly. A 20-frame field test of the join matched a single pass, and the
chunked sky dump matched one byte for byte. A first single-pass overlay was killed at
the 2 h background limit with an unplayable video.

Artifacts in `runs/field/exp025_top3/`:
`split_sky_window2of4_top3_merge20_captured_raw_20260616_040253_004_2_3600.mp4`, its
`.csv`, `_frames.csv` (per-frame sky fraction and counts), the candidate and seed dumps,
and `split_field.log`.

## EXP-028 — the kinematic gate: answers held to a drone's top speed

The user asked for answers that do not pop up anywhere on screen: a real drone has a top
speed, so a candidate that moves like nothing a drone can do should be overruled. In
EXP-027's top 3, #1 moved more than 29 px between consecutive frames in **103 of 156**
frame pairs that both had a #1.

**Method.** `src/algo/kinematics.py` (new, unit-tested, numpy only):

- **`SpeedLimit` converts m/s to px/frame at an *assumed minimum range*.** It uses 40 m/s
  (edge-budget's closing-speed assumption) at 10 m, through the R1 Mini's 130° lens. On
  catch_2 (960 px wide, 30 fps) that is 29.8 px/frame, capped by a 25 px/frame ceiling
  (2× the labelled drone's p90 step of 12.67 px), plus 4 px of slack.
  - The range is not observable from one camera.
  - EXP-016's size-based range was inert, and the blob sizes are 0.05–0.16× the airframe
    (the 2026-09-30 correction). So the ceiling is what actually binds here.
- **`KinematicTracker` is a hysteresis track.** EXP-025b's hard 4-of-5 gate halved
  recall, so this design keeps tracks alive instead.
  - A strong candidate (the pool: sky c >= 6 or ground 2-of-4) starts a track, which is
    shown from its 2nd hit.
  - Once confirmed, any candidate at c >= `--c-keep` inside the reach continues it.
  - It coasts through up to 5 misses, with the reach growing by 25 px each frame.
  - A strong pop with no track in reach is overruled (`new`). A weak one never starts a track.
- **Displacement is the smaller of raw and camera-compensated.** Raw picture coordinates
  keep the target the camera follows (EXP-025b). The homography still excuses a host
  whip-turn.
- **Each track carries evidence**, the sum of c decayed by 0.8 per frame. `--rank-by track`
  orders the shown answers by evidence instead of this frame's c.

It runs as `overlay_split.py --kinematic`. `experiments/exp028_kinematic/overlay_kinematic.py`
runs it on top of EXP-027's top-3 settings. With the flag off, the CSVs are byte-identical
to EXP-027's (re-run and compared).

| catch_2 441–800, top 3, merge 20, osd-grid | shown/frame | FA shown | drone in top 3 (of 224) | drone #1 | #1 jumps > 29 px |
| --- | ---: | ---: | ---: | ---: | ---: |
| EXP-027 | 1.08 | 318 | 67 | 57 | 103 / 156 (66%) |
| gate, c-keep 3, rank by c | 2.47 | 796 | 89 | 72 | 165 / 318 (52%) |
| gate, c-keep 3, rank by track | 2.47 | 791 | 95 | 78 | 82 / 318 (26%) |
| gate, no weak continuation | 0.44 | 107 | 49 | 48 | 22 / 71 (31%) |
| gate, c-keep 6, rank by c | 0.93 | 264 | 69 | 58 | 65 / 165 (39%) |
| **gate, c-keep 6, rank by track** | **0.93** | **264** | **69** | **60** | **43 / 165 (26%)** |

**Reading.**

- **At c-keep 6 with track ranking, the gate is better than EXP-027 on every column at
  lower load:**
  - 14% fewer answers shown, and 17% fewer false alarms;
  - the drone in the top 3 in 69 frames (from 67), and #1 in 60 (from 57);
  - #1 jumps cut from 103 to 43.
  
  It is the default (`--c-keep 6 --rank-by track`).
- **The c-keep 3 rows are not comparable to EXP-027.** They show 2.3× the load, because
  weak continuation keeps persistent ground clutter alive (1,607 weak continuations).
  Their recall gain (95 in the top 3) is bought with load, and is stated only to show
  where that knob goes.
- **Most of the remaining flicker was ranking, not motion.** With c ranking, #1 changed
  track in 62 of 165 pairs. A single track cannot jump by construction, so those were
  swaps between two plausible objects whose c crossed. Evidence ranking cut that to 41.
  The 43 jumps left are swaps of this kind, for example a track dying and the next one up
  taking #1. A speed limit cannot remove them.
- **Cost: the gate held the drone back in 16 frames** (3 sky, 13 ground) as `new` or
  `unconfirmed`. These are frames where its track had just been born, or re-born after a
  break longer than 5 frames or a step larger than the reach.
- **What it does not do: persistent clutter passes**, because clutter that sits still
  moves plausibly by definition. This gate stops answers jumping. Clutter rejection is
  still the sections' tests' job.

**Caveats.**

- One clip, one span. The 25 px ceiling and c-keep 6 were chosen on this span, so the
  c-keep choice is in-sample. The physical constants are assumptions, not measurements;
  edge-budget lists closing speed as user-owned.
- The 1-in-6 duplicate frames are still in. A duplicate is a zero step and is harmless
  here, but it is a free hit toward confirmation.
- Not yet on catch_4/5, on O4, or wired into `baseline_detect`, `glad_detect` or `live_detect`.

Artifacts in `runs/sofa_analog/exp028_kinematic/`:

- `split_top3_merge20_osdgrid_kinematic_catch_2_441_800` is the default run, with
  `.mp4`/`.csv`/`_drone.csv`/`_frames.csv` and `kinematic_top3.log`.
- `v_ckeep3_c`, `v_track`, `v_noweak` and `v_ckeep6` are the other rows.
- `regress_exp027_top3` is the byte-identity check.

### Added 2026-10-05: out of sample on catch_4 and catch_5

The user asked for the overlays on the other two labelled analog clips. These are
`experiments/exp028_kinematic/run_clips.py` over EXP-025e's dumps, with no detector re-run.

The user named three of the six catch_2 variants: the default gate, c-keep 3, and no weak
continuation. A no-gate run was added at the same settings (`--osd-grid`), because
EXP-025e's catch_4/5 videos predate it and are not the comparison. All runs are top 3,
merge 20, and use the same constants. Nothing was re-tuned per clip.

| clip | run | shown/frame | FA shown | drone in top 3 | drone #1 | held back by gate | #1 jumps > 29 px |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| catch_4 (82 labelled) | no gate | 1.11 | 169 | 12 | 8 | – | 42 / 51 (82%) |
| catch_4 | **gate (default)** | **1.21** | **177** | **22** | **18** | 8 | **47 / 92 (51%)** |
| catch_4 | c-keep 3 | 2.46 | 376 | 35 | 15 | 6 | 67 / 147 (46%) |
| catch_4 | no weak continuation | 0.35 | 56 | 3 | 3 | 10 | 6 / 15 (40%) |
| catch_5 (188 labelled) | no gate | 0.71 | 235 | 29 | 16 | – | 67 / 94 (71%) |
| catch_5 | **gate (default)** | **0.37** | **100** | **37** | **31** | 16 | **20 / 66 (30%)** |
| catch_5 | c-keep 3 | 2.54 | 901 | 51 | 23 | 10 | 124 / 346 (36%) |
| catch_5 | no weak continuation | 0.19 | 54 | 16 | 11 | 16 | 15 / 34 (44%) |

**The default holds out of sample.**

- **catch_5 is a clean win.** The gate shows about half the answers (0.37 vs 0.71/frame,
  100 vs 235 FAs). It puts the drone in the top 3 in more frames (37 vs 29) and at #1 in
  almost twice as many (31 vs 16). #1 jumps fall from 71% to 30% of frame pairs.
- **catch_4 is not quite at matched load.** The gate shows 9% more (1.21 vs 1.11/frame,
  177 vs 169 FAs). For that, the drone is in the top 3 in 22 frames instead of 12 and at #1
  in 18 instead of 8, and the jump rate falls from 82% to 51%.
  - The absolute jump count rises (42 → 47) because #1 exists in nearly twice as many
    frame pairs (92 vs 51). The per-pair rate is the comparable figure.

**The other two settings fail the same way they did on catch_2.**

- **c-keep 3 is not comparable.** It shows 2.2–3.6× the load, because weak continuation
  keeps clutter tracks alive.
- **No weak continuation collapses recall.** catch_4's drone is in the top 3 in 3 frames,
  because its strong firings are too sparse to chain.

The gate's cost recurs on both clips: the drone is held back in 8 (catch_4) and 16
(catch_5) labelled frames, mostly ground section, as a new or unconfirmed track.

Caveat: these spans are short. catch_4 has 82 labelled frames, and its counts move in
single digits.

Artifacts in `runs/sofa_analog/exp028_kinematic/catch_4/` and `catch_5/`:
`{nogate,gate,ckeep3,noweak}_<clip>_<start>_<end>` with `.mp4`, `.csv`, `_drone.csv`,
`_frames.csv` and a log each. The batch log is `run_clips.log` one level up.

## EXP-029 — the moving factor, and where the direction test starts to earn its place

The user asked, after EXP-024, why a candidate's travel across the k frames is not used as
evidence rather than only the direction it points. Mostly it was already computed and
discarded: `window.Track.total_mu` is the ego-compensated displacement over the window, and
`criteria.epipolar_direction` reduces its magnitude to a 2 px veto (`min_displacement`)
before testing only direction. EXP-029 promotes the magnitude and measures what the
direction test still contributes.

- **Method.** `src/algo/kinematics.py` gains the **moving factor**, the exact complement of
  the speed limit already there — that docstring closed "a piece of clutter that stays put
  moves plausibly by definition, and this gate passes it". A confirmed track must now also
  have travelled `min_move` px against the static scene over `move_window` frames. Each
  track keeps its sightings as `anchors` and carries every one forward through each frame's
  homography, so differencing newest against oldest removes ego-motion. `min_move = 0` is
  inert and EXP-028's 32 tests pass untouched; 8 new tests cover clutter kept forever and
  the target dropped by premature judging or by being charged for the camera's own pan.
- **It runs on the sky branch, not the motion window**, because EXP-024 measured the window
  losing to it at every load and LK dying at a median 0 usable steps on analog. Association
  is EXP-028's `KinematicTracker` chaining candidate peaks: no texture needed, and a
  duplicated frame contributes no candidate rather than killing a tracker.
- **The direction half is in `experiments/exp029_moving_factor/moving.py`**, not `src/`,
  since it needs an epipole and `src/` must not depend on `experiments/`. `--mode` is
  `off` / `veto` / `require`; `veto` is the default because a test rejecting at chance must
  never reject *by default*, which is what `require` allows. `still`, `parallax`, `unjudged`
  and `moving` stay four outcomes — collapsing `unjudged` into `rejected` is how the
  direction test's cost stayed hidden in EXP-017.
- **Data.** `catch_2` 441–800, 360 frames, **224 labelled**. Labels on this clip run
  491–785. Sky strong at c >= 6, continued at c >= 3, speed limit 25 px/frame, confirm 2,
  coast 5, `--move-window 5`, no `--top` cap. `--min-move` **20** since 2026-10-07; the
  first runs used 8 and both are reported below.
- **Scripts.** `experiments/exp029_moving_factor/` (`moving.py` with a 9-of-9 self-check,
  `overlay_moving.py`, `analog_catch_2/clipcfg.py`). Artifacts in
  `runs/sofa_analog/exp029_moving_factor/`: `moving_m8_w5_veto_catch_2_441_800.mp4` (the
  clean look), `veto.log`, `off.log`, `require.log`, and `judged_veto.csv` /
  `judged_off.csv`, one row per judged track per frame with `moved`, `span`, `sin`,
  `outcome` and `on_target`, so any threshold recuts without re-running.

### Result

| mode | kept/frame | false alarms | drone kept (of 224) |
| :--- | ---: | ---: | ---: |
| `off` (magnitude alone) | 16.14 | 5626 | **105 (46.9%)** |
| `veto` (+ direction) | 14.59 | 5071 | 104 (46.4%) |
| `require` | 12.93 | 4494 | 96 (42.9%) |

**At `--min-move 8` the direction test contributes nothing at matched load.** The veto drops
555 false alarms for one drone frame, which looks like a gain until the loads are equalised:
magnitude alone at `--min-move 15` gives 14.57/frame and the **identical 104** drone frames.
`require` is worse on both axes, as EXP-021 predicted.

### Correction, 2026-10-07: that holds only at a threshold too low to matter

The paragraph above was measured at one `--min-move` and generalised, which was wrong. The
first version of this entry concluded "ship `--mode off`". Swept properly, the veto is worth
nothing up to 12 and then grows:

| `--min-move` | load/frame | veto drone | magnitude alone, same load | delta |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 14.59 | 104 | 104 (at 14.9) | **+0** |
| 12 | 13.76 | 104 | 104 (at 18.7) | +0 |
| 16 | 13.04 | 103 | 102 (at 22.9) | +1 |
| **20** | 12.33 | **103** | 99 (at 27.7) | **+4** |
| 24 | 11.80 | 100 | 90 (at 31.8) | +10 |
| 30 | 11.11 | 95 | 77 (at 36.9) | **+18** |
| 40 | 9.97 | 73 | 71 (at 45.6) | +2 |

**Why the two disagree.** At 8 the magnitude cut and the direction cut remove overlapping
sets, so the second is redundant. Past ~16 magnitude pushed harder starts cutting *the
target* — its travel distribution overlaps the clutter's, measured below — while the veto
removes clutter by a criterion uncorrelated with travel. The direction test does carry
information; it only becomes visible once magnitude is exhausted. Its epipole agreement of
28.9% against a ~23% floor is weak but not nothing, and this is where that shows up.

**The default moved from 8 to 20** (2026-10-07, at the user's suggestion). Recall is flat
across that move — 105 then 104 of 224 — while load falls 16.14 to 13.52/frame and false
alarms fall **5626 to 4686**. 940 fewer false alarms for one drone frame. With the veto on,
20 also dominates 16: the same 103 frames at 12.33/frame instead of 13.04.

### Against plain EXP-023 thresholding, at matched load

| load/frame | EXP-029 chain + factor | EXP-023 plain `c` | winner |
| ---: | ---: | ---: | :--- |
| 21.01 | **108 (48.2%)** | 102 (45.5%) | EXP-029 |
| 16.14 | **105 (46.9%)** | 97 (43.3%) | EXP-029 |
| 14.57 | **104 (46.4%)** | 96 (42.9%) | EXP-029 |
| 12.05 | **96 (42.9%)** | 94 (42.0%) | EXP-029 |
| 10.06 | 71 (31.7%) | **89 (39.7%)** | EXP-023 |
| 8.30 | 65 (29.0%) | **83 (37.1%)** | EXP-023 |
| 6.69 | 60 (26.8%) | **78 (34.8%)** | EXP-023 |

**The gain is real and in the wrong place.** Above ~12 candidates/frame the chain plus the
factor beats plain thresholding by 2–8 drone frames; below it the curve collapses. EXP-023
ships at 5.59/frame and EXP-028 at 0.93 shown/frame, both under the crossover, so **as built
this is not an improvement at any load the project wants.**

### Read with

- **The prediction was optimistic by the amount EXP-024 warned.** Measured travel over 5
  frames: on the drone p10 15.63, median 63.33; everything else p10 1.98, **median 19.97**.
  EXP-024 predicted the target's p10 13.85 against a *background-field* p90 of 5.60 — tails
  that do not overlap. They overlap heavily against the real false alarms. That is EXP-024's
  own filed caveat coming true: the background grid is not the clutter that survives.
- **Chained clutter genuinely moves**, from parallax and from association error. The
  25 px/frame speed limit permits 125 px of travel across a 5-frame window, so a chain that
  hops between two objects banks the hop as travel. The permissiveness that makes
  association work is what lets clutter accumulate displacement.
- **A floor of 5.42 kept/frame.** 1950 of 7564 confirmed-track rows (25.8%) have no span
  yet, and `off`/`veto` keep what they cannot judge, so `--min-move` cannot push the load
  below that. Those rows carry 42 of the drone frames, so refusing them is not free either.
- **Not comparable to EXP-028's 0.93 shown/frame**, which uses a `--top 3` cap this run does
  not. The `--top` flag exists and is untried.
- The epipole here is **one pair's**, not a window's; EXP-019 measured per-pair epipoles
  hopping 99–127 px. Pooling might rescue the direction test, but it would have to beat a
  statistic that already costs nothing to compute.

### Added 2026-10-07: stacked on EXP-028's gate, and now it wins

The user pointed out the stages are meant to come one on top of the other, not to compete.
`--min-move` is now a flag on `overlay_split.py` (default 0, **inert** — the shown and drone
CSVs at 0 are byte-identical to EXP-028's, the check EXP-028 used on itself), and
`experiments/exp029_moving_factor/overlay_stacked.py` runs EXP-028's defaults plus
`--min-move 20`. One `KinematicTracker`, the speed limit bounding motion from above and the
moving factor from below — the gap EXP-028's own docstring named.

| `--min-move` | shown/frame | FA shown | drone in top 3 | drone #1 | too-still/frame |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (= EXP-028) | 0.93 | 264 | 69 (30.8%) | 60 (26.8%) | 0.00 |
| 12 | 0.82 | 224 | 69 (30.8%) | 62 (27.7%) | 0.12 |
| **20** | **0.78** | **209** | **69 (30.8%)** | **62 (27.7%)** | 0.17 |
| 30 | 0.74 | 197 | 66 (29.5%) | 59 (26.3%) | 0.21 |
| 45 | 0.67 | 186 | 52 (23.2%) | 45 (20.1%) | 0.28 |

**At `--min-move 20` this is strictly better than EXP-028 on every column:** false alarms
264 → **209** (-21%), load 0.93 → **0.78** shown/frame (-16%), drone `#1` 60 → **62**, and
the drone in the top 3 unchanged at 69 of 224. 30 starts costing recall, 45 is well past the
knee — the same knee the standalone sweep found, at the same 20 px.

**This is the first EXP-029 configuration that improves the shipping pipeline**, and it
retires the entry's earlier conclusion that the moving factor only pays above ~12
candidates/frame. That conclusion was about `overlay_moving.py`, which rebuilt the sky
branch and in doing so discarded the merge, the OSD veto, c-keep and **evidence ranking**;
without the last of those it reproduced EXP-027's `#1` column (36 of 224 against EXP-028's
60) and lost on it. The statistic was fine; the pipeline around it was not. The standalone
measurements stay in this entry because they are what isolates the statistic from
everything else.

The `#1` jump rate is fractionally worse in proportion — 45 of 150 pairs (30%) against 43 of
165 (26%) — because the factor removes frames that had a `#1` at all; absolute jumps move by
two. Artifacts in `runs/sofa_analog/exp029_moving_factor/`:
`stacked_top3_merge20_osdgrid_kinematic_m20_catch_2_441_800.mp4` with its three CSVs (the
`_frames.csv` carries a new `too_still` column) and `stacked_m{12,20,30,45}.log`.


### Standing recommendation

**Keep the moving factor. Run it at `--min-move 20 --mode veto`** (both are now the
defaults): 12.33 kept/frame, 103 of 224 drone frames, **+9 over EXP-023 at matched load**.
Superseded 2026-10-07 — the first version of this line said `--mode off`, from a sweep taken
at `--min-move 8` only, where the veto is genuinely worth nothing. It is worth +4 frames at
20 and +18 at 30.

**Superseded 2026-10-07 by the stacked run above: `overlay_stacked.py --min-move 20` is
strictly better than EXP-028 on every column and is the configuration to ship.**

**Withdrawn 2026-10-07 by EXP-031.** That held on catch_2 only. On catch_5 `--min-move 20`
drops the drone from the top 3 in 13 frames (37 → 24), where the camera chases it and
compensation removes its travel. Once c-keep is calibrated it adds nothing pooled.

The paragraph this replaces said the gain was unusable because it lived above ~10-11
candidates/frame, and that the 5.42/frame floor of unjudged tracks kept EXP-029 away from
EXP-028's 0.93 shown/frame. Both were properties of `overlay_moving.py`'s standalone
rebuild, not of the moving factor: stacked inside EXP-028's pipeline the factor *lowers*
load to 0.78/frame. The standalone comparison against plain EXP-023 thresholding still
stands as written — it is simply the wrong pipeline to judge the statistic by.

**The next lever is the chain, not the statistic.** Clutter accumulates travel because
association is permissive; tightening it — a smaller reach, or requiring consistent
direction *along the chain* rather than against an epipole — attacks the measured cause.
Then re-run with `--top 3` so the numbers sit at EXP-028's operating point instead of 15x
above it.

## EXP-030 — merged objects scored by the sum of their members' c

The user asked whether `--merge 20` takes the max or the sum of the blobs it folds. It
took the max: the anchor's `c` alone. This run scores the object by the **sum** of every
member's `c` instead (`overlay_split.py --merge-score sum`).

**What is unchanged.**

- Anchors are still chosen by their own `c`.
- The sections' tests still pass on any member.
- The summed `c` feeds the ranking, the gate's c-keep 6 test and the track evidence.
- Every member counts, including sub-threshold blobs at c 3–4.

**Comparison.** Each clip was run four ways, max/sum with and without EXP-028's default
gate. The settings are EXP-028's: top 3, merge 20, circles >= 20 px, `--osd-grid`. The
`max_` runs re-create EXP-028's rows: all 18 CSVs are byte-identical. Runner:
`experiments/exp030_merge_sum/run_clips.py`.

| clip | run | shown/frame | FA shown | drone in top 3 | drone #1 | #1 jumps > 29 px |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| catch_2 (224 labelled) | max, no gate | 1.08 | 318 | 67 | 57 | 103 / 156 (66%) |
| catch_2 | sum, no gate | 1.08 | 318 | 67 | **58** | 103 / 156 (66%) |
| catch_2 | max, gate (EXP-028 default) | 0.93 | 264 | 69 | 60 | 43 / 165 (26%) |
| catch_2 | sum, gate | 1.23 | 353 | 88 | 72 | 57 / 200 (28%) |
| catch_4 (82 labelled) | max, no gate | 1.11 | 169 | 12 | 8 | 42 / 51 (82%) |
| catch_4 | sum, no gate | 1.11 | 167 | 13 | **11** | 40 / 51 (78%) |
| catch_4 | max, gate | 1.21 | 177 | 22 | 18 | 47 / 92 (51%) |
| catch_4 | sum, gate | 1.80 | 243 | 42 | 35 | 73 / 132 (55%) |
| catch_5 (188 labelled) | max, no gate | 0.71 | 235 | 29 | 16 | 67 / 94 (71%) |
| catch_5 | sum, no gate | 0.71 | 235 | 29 | **21** | 68 / 94 (72%) |
| catch_5 | max, gate | 0.37 | 100 | 37 | 31 | 20 / 66 (30%) |
| catch_5 | sum, gate | 0.64 | 173 | 66 | 40 | 31 / 96 (32%) |

**Reading.**

- **Without the gate, the sum is a clean win at identical load.** The pool's membership
  is unchanged, since the tests are, so the load is the same. Only the order changes. The
  drone is #1 in 90 frames across the three clips, against 81 with the max (58/57, 11/8,
  21/16). Top-3 recall is unchanged (one frame more on catch_4).
- **The reason is structural: the drone fragments, clutter does not.** In the shown sum
  objects, a drone object holds 2.2–3.0 blobs on average, and a false alarm holds
  1.1–1.6. The median lift from summing is +1.5 to +4.0 c on the drone, and 0 on false
  alarms on catch_2 and catch_4. catch_5's gated false alarms are the exception, with a
  +3.1 median lift.
- **With the gate, the sum rows are not comparable to EXP-028.** They show 32% (catch_2),
  49% (catch_4) and 73% (catch_5) more answers.
  - The mechanism is c-keep 6: a cluster of two or three sub-threshold blobs now clears
    it, so weak continuations rise from 178 to 280, 176 to 329 and 65 to 179.
  - The recall gain is large (drone in top 3: 69→88, 22→42, 37→66), but it is bought
    partly with load.
  - Top-3 hits per false alarm shown are about level: catch_2 0.26 vs 0.25, catch_4 0.12
    vs 0.17, catch_5 0.37 vs 0.38. So this is not yet evidence of a better operating
    point.
  - The gate also holds the drone back less often (16→11, 8→4, 16→12).
- **The #1 jump rate is unchanged within a couple of points.** Summing does not destabilise
  the ranking.

**Caveats.** The spans are short (catch_4 counts move in single digits). c-keep 6 was set
for a per-blob `c` and means something looser for a summed one. The sum also has no cap,
so a large cluster of weak clutter could in principle outrank a crisp drone; that has not
shown up on these clips.

**Next.** Compare the gated sum at matched load. Raise `--c-keep` under the sum until
shown/frame returns to EXP-028's (0.93 / 1.21 / 0.37), then read recall. If the sum still
wins there, make it the default.

Artifacts in `runs/sofa_analog/exp030_merge_sum/merge20/<clip>/`:
`{max,sum}_{nogate,gate}_<clip>_<start>_<end>` with `.mp4`, `.csv` (`c_max` on the sum
runs), `_drone.csv`, `_frames.csv` and a log each. The overlay is `sum_gate_*.mp4`. The
batch log is `runs/sofa_analog/exp030_merge_sum.log`.

### Added 2026-10-07: merge 30 (60 px across)

The user asked for the same four runs at a 30 px merge radius, in a separate subfolder
(`run_clips.py --merge 30`). The settings are otherwise identical. The `max_` rows are
merge 30's own baseline; EXP-028 has nothing at this radius.

| clip | run | shown/frame | FA shown | drone in top 3 | drone #1 | #1 jumps > 29 px |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| catch_2 (224 labelled) | max, no gate | 1.07 | 316 | 67 | 57 | 103 / 156 (66%) |
| catch_2 | sum, no gate | 1.07 | 316 | 67 | **60** | 99 / 156 (63%) |
| catch_2 | max, gate | 0.92 | 260 | 69 | 64 | 46 / 165 (28%) |
| catch_2 | sum, gate | 1.52 | 440 | 105 | 86 | 82 / 243 (34%) |
| catch_4 (82 labelled) | max, no gate | 1.10 | 166 | 13 | 9 | 40 / 49 (82%) |
| catch_4 | sum, no gate | 1.10 | 166 | 13 | **11** | 39 / 49 (80%) |
| catch_4 | max, gate | 1.20 | 175 | 22 | 18 | 45 / 90 (50%) |
| catch_4 | sum, gate | 2.00 | 280 | 46 | 36 | 78 / 140 (56%) |
| catch_5 (188 labelled) | max, no gate | 0.71 | 235 | 29 | 17 | 66 / 94 (70%) |
| catch_5 | sum, no gate | 0.71 | 233 | 30 | **23** | 72 / 94 (77%) |
| catch_5 | max, gate | 0.36 | 96 | 37 | 33 | 23 / 66 (35%) |
| catch_5 | sum, gate | 0.66 | 191 | 54 | 32 | 24 / 99 (24%) |

**Reading.**

- **The radius alone barely matters under the max.** Going from merge 20 to merge 30
  leaves the load within 0.01/frame. The drone is #1 in 83 frames vs 81 without the gate,
  and 115 vs 109 with it.
- **Without the gate, the sum at 30 px is the best #1 at identical load so far.** The
  drone is #1 in 94 frames across the three clips (60 + 11 + 23), against 90 for the sum
  at 20 px and 81 for EXP-027's max. Top-3 recall is 110, against 108 for EXP-027's max.
  - The wider bubble catches more of the drone's fragments: 3.8–4.0 members per drone
    object vs 2.2–2.6 at 20 px, and a median lift of +4.3 to +5.6 c.
  - Clutter in the no-gate rows still gains nothing (median lift 0).
- **With the gate, 30 px is worse than 20 px. The bubble starts adding up clutter too.**
  - The load rises further over the max: 65% (catch_2), 67% (catch_4) and 83% (catch_5),
    against 32/49/73% at 20 px.
  - Weak continuations rise to 407, 416 and 198.
  - Gated false alarms now gain from summing: their median lift is +3.6 on catch_4 and
    +3.4 on catch_5, with about 2 members each.
  - catch_5 is the out-of-sample warning. The drone is in the top 3 in 54 frames (66 at
    20 px), and #1 in 32, below the max gate's 33.
  - Top-3 hits per false alarm shown fall below the max gate on catch_2 (0.24 vs 0.27)
    and catch_5 (0.28 vs 0.39). catch_4 is the exception, at 0.16 vs 0.13.
- **The #1 jump rate stays within 7 points of the max rows**, except catch_5 with the gate,
  where it falls from 35% to 24%.

So the radius trades two ways. A bigger bubble collects more of the drone, which pays
when the ranking is the only consumer (no gate). It also lets a cluster of sub-threshold
clutter clear the gate's per-blob c-keep 6, which costs when the gate is on. The
matched-load comparison in `docs/todo.md` now applies to both radii.

Artifacts in `runs/sofa_analog/exp030_merge_sum/merge30/<clip>/`, named as for merge 20.
The batch log is `runs/sofa_analog/exp030_merge_sum/merge30.log`.

## EXP-031 — every stage at once, and EXP-030 at matched load

- **Date:** 2026-10-07
- **Question:** the user asked for the past experiments combined into one. EXP-025d to
  EXP-030 each added one flag to `overlay_split.py` and were measured one on top of the
  previous, but never all on together. EXP-029's moving factor had only been run on catch_2.
  Does the full stack beat EXP-028, and what does each stage add to it? This also answers
  EXP-030's open question: is the summed score still better once its load is brought back
  to EXP-028's?
- **Stages:** the sky/ground split with a pooled top 3 (EXP-025d), `--merge` (EXP-025d),
  `--osd-grid` (EXP-027), `--kinematic` with evidence ranking (EXP-028), `--min-move 20`
  (EXP-029) and `--merge-score sum` (EXP-030). Circles are drawn at least 20 px across.
- **Data:** the three labelled analog clips, catch_2 441–800 (224 labelled), catch_4
  159–326 (82) and catch_5 69–444 (188), 494 labelled frames in all. Merge radius 20 and 30.
- **Hardware / cost:** i7-1255U CPU, 72 overlays from EXP-030's dumps, with no detector
  re-run, 4 in parallel.
- **Scripts:** `experiments/exp031_combined/run_all.py`. Artifacts in
  `runs/sofa_analog/exp031_combined/merge{20,30}/<clip>/`: `{variant}_<clip>_<start>_<end>`
  with `.mp4`, `.csv`, `_drone.csv`, `_frames.csv` and `.log`. The full per-clip table is in
  `run_all.log` one level up.
- **Variants:** `exp028` (max score, gate), `exp029` (+ `--min-move 20`), `exp030`
  (+ `--merge-score sum`, no moving factor), `all` (both), and `exp030_ck{8,10,12,15}` and
  `all_ck{8,10,12,15}`, the last two at a stricter `--c-keep`.

**The reproduced rows are exact.** `exp028` and `exp030` match EXP-030's `max_gate` and
`sum_gate` byte for byte at both radii, on all three clips: 36 CSVs. The only difference is
the `too_still` column, which EXP-029 added to `_frames.csv` later. `exp029` on catch_2 at
merge 20 matches EXP-029's stacked run. Combining the stages changes none of them.

### Result — pooled over the three clips

One configuration for all three clips. Choosing a c-keep per clip would be tuning on the test.

| merge | run | shown/frame | FA shown | drone in top 3 (of 494) | drone #1 | top 3 per FA |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 20 | `exp028` (EXP-028 default) | 0.75 | 541 | 128 | 109 | 0.24 |
| 20 | `exp029` + moving factor | 0.65 | 468 | 115 | 102 | 0.25 |
| 20 | `exp030` + summed score | 1.09 | 769 | 196 | 147 | 0.25 |
| 20 | `all` | 0.95 | 671 | 171 | 140 | 0.25 |
| 20 | `exp030_ck8` | 0.68 | 461 | 141 | 120 | 0.31 |
| 20 | `all_ck8` | 0.62 | 422 | 128 | 111 | 0.30 |
| 20 | `exp030_ck10` | 0.56 | 373 | 120 | 104 | 0.32 |
| 30 | `exp028` | 0.74 | 531 | 128 | 115 | 0.24 |
| 30 | `exp030_ck8` | 0.78 | 530 | 165 | 142 | 0.31 |
| 30 | `all_ck8` | 0.70 | 474 | 146 | 133 | 0.31 |
| **30** | **`exp030_ck10`** | **0.65** | **434** | **142** | **123** | **0.33** |
| 30 | `exp030_ck12` | 0.53 | 353 | 124 | 117 | 0.35 |

The c-keep 12 and 15 rows of every sweep are in `run_all.log`.

**At matched load the summed score wins, pooled.** Raising c-keep from 6 to 8 or 10 removes
the load the sum added, and keeps most of its recall:

- **Merge 30, sum, c-keep 10 beats EXP-028 on every pooled column.** Load is 13% lower
  (0.65 against 0.75 shown/frame), false alarms 20% fewer (434 against 541), the drone is in
  the top 3 in 142 frames against 128 and #1 in 123 against 109.
- Merge 20, sum, c-keep 8 does the same at 0.68/frame: 461 false alarms, 141 / 120.
- Merge 30, sum, c-keep 8 has EXP-028's false-alarm count (530 against 541) and **+37 top-3
  and +33 #1 frames**. That is the largest gain at equal false alarms.

**Merge 30 beats merge 20 once c-keep is raised.** EXP-030 found the 30 px bubble worse
with the gate on, because clusters of weak clutter cleared c-keep 6. At c-keep 8 the two
radii tie on top-3 hits per false alarm (0.31), and from c-keep 10 up the 30 px row is ahead
(0.33, 0.35, 0.39 against 0.32, 0.33, 0.34). It also reaches more recall at each c-keep. The bubble still collects more of the drone's fragments, and the stricter
c-keep now stops the clutter.

**The moving factor adds nothing once c-keep is calibrated.** At merge 20, `all_ck8` sits at
0.62/frame with 128 / 111. The summed-score sweep without it, interpolated to the same
load, gives about 130 / 112. At merge 30, `all_ck8` (0.70/frame, 146 / 133) against about
151 / 130 interpolated. Raising c-keep removes load as well as the moving factor does, without
its out-of-sample failure below.

### Result — per clip, at the recommended setting

| clip | run | shown/frame | FA shown | drone in top 3 | drone #1 |
| --- | --- | ---: | ---: | ---: | ---: |
| catch_2 (224) | EXP-028 | 0.93 | 264 | 69 | 60 |
| catch_2 | **merge 30, sum, c-keep 10** | **0.75** | **189** | **82** | **78** |
| catch_4 (82) | EXP-028 | 1.21 | 177 | 22 | 18 |
| catch_4 | **merge 30, sum, c-keep 10** | **0.99** | **134** | **26** | **24** |
| catch_5 (188) | EXP-028 | 0.37 | 100 | 37 | 31 |
| catch_5 | merge 30, sum, c-keep 10 | 0.39 | 111 | 34 | **21** |

**Two clips win and one loses.** catch_2 and catch_4 improve on every column at lower load.
catch_5 loses 10 #1 frames at about the same load, and no summed-score setting near
EXP-028's load recovers them. At or below 0.37 shown/frame the best is 24 #1 (c-keep 12,
either radius), and 27 at 0.44 (merge 30, c-keep 8), against 31 under the max. Only c-keep 6
beats it (40 at merge 20), at 73% more load.
EXP-030 already measured why: catch_5 is the one clip where the gate's false alarms gain
from summing (median lift +3.1 c), so the sum lifts its clutter as much as its drone.

### The moving factor fails out of sample on catch_5

EXP-029's `--min-move 20` was chosen on catch_2. This is its first run elsewhere, on top of
EXP-028:

| clip | EXP-028: top 3 / #1 | + `--min-move 20`: top 3 / #1 | load |
| --- | :---: | :---: | :---: |
| catch_2 | 69 / 60 | 69 / 62 | 0.93 → 0.78 |
| catch_4 | 22 / 18 | 22 / 19 | 1.21 → 1.17 |
| catch_5 | 37 / 31 | **24 / 21** | 0.37 → 0.30 |

On catch_2 and catch_4 it gains nothing and loses nothing. On catch_5 the drone is overruled in
13 frames that EXP-028 ranked, all in one pass, frames 150–178, and gains none. From the
labels, the box centre moved 20–52 px in the picture across the 5-frame window in 7 of
those 13 frames (150–153, 158, 161, 162). The factor measures travel against the static
scene, after removing the camera's motion, and the camera is following the drone. That
removes most of the drone's own motion: **EXP-025b's lesson again**, that compensation
suits static clutter and not a chased target. In the other 6 (157, 173–178) the drone was
genuinely slow in the picture, 7–20 px.

### Read with

- **The c-keep choice is in-sample.** c-keep 8 and 10 were picked from these runs on these
  three clips, and they are every labelled analog clip there is. catch_3/6/7/8 and
  miss_1/2 are unlabelled. The pooled gain is measured, but the setting that achieves it has
  not been tested on anything held out.
- c-keep under the sum means something different from c-keep under the max: it thresholds a
  cluster's total, not a blob's.
- "Drone" is EXP-023's grown-box `on_target`, as in every entry since EXP-023. The spans are
  short: catch_4's counts move in single digits.
- The per-clip loads are not matched exactly: catch_5 runs 5% above EXP-028 at the
  recommended setting.

### Recommendation

- **The combined pipeline is EXP-028 plus `--merge 30 --merge-score sum --c-keep 10`,
  without the moving factor.** Pooled, it beats EXP-028 on every column at 13% less load,
  but catch_5 loses #1 frames. Defaults are left unchanged until it has been run on a
  held-out labelled clip.
- **Withdraw `--min-move 20` as a shipping setting.** EXP-029's recommendation came from
  catch_2 alone. Out of sample it costs a third of catch_5's drone frames and adds nothing
  that a c-keep rise does not. If it comes back, it should measure travel in raw picture
  coordinates, or take the larger of raw and compensated, the opposite of the speed limit's
  smaller-of rule.
- **Next: label a fourth analog clip** to test the c-keep 10 / merge 30 setting held out.

Video: `runs/sofa_analog/exp031_combined/merge30/<clip>/exp030_ck10_<clip>_<start>_<end>.mp4`
is the recommended setting. `all_<clip>_*.mp4` has every stage on at c-keep 6.

### Added 2026-10-07: `all` on the FIELD `.raw`, at the sensor's 4128x3008

The user added `data/raw/FIELD/videos/captured_raw_20260616_040253_004.raw` and asked for
the `all` row run on it. The file is the FIELD mp4's own recording before it was encoded:
headerless 8-bit BG Bayer, 4128x3008, mounted upside down, the same 3600 frames, aligned frame
for frame. The mp4 is this file shrunk 4x in each direction. `src/data/raw_bayer.py` reads it,
and `data/raw/FIELD/PROVENANCE.md` records how the layout was measured.

- **Spans:** the three episodes in PROVENANCE, 2–180, 1090–1553 and 3130–3600 (1,114 frames),
  at full resolution. The user chose this over the whole clip at 2064 wide.
- **Settings:** `all` exactly as `run_all.py` runs it: top 3, `--osd-grid`, `--kinematic`,
  `--min-move 20`, `--merge-score sum`, c-keep 6, at merge 20 and 30. Stage 1 uses FIELD's
  colour settings. The sky-branch and window dumps were built fresh, because no dump exists
  at this width.
- **Hardware / cost:** i7-1255U, 5 jobs, 13:41–18:55, **5 h 14 min**. The sky branch took
  about 55 s per frame per process. It is the cost, and its 4,000-candidate cap binds at this
  size.
- **Scripts:** `experiments/exp031_combined/run_field_raw.py`, with the clip config in
  `field_raw/clipcfg.py`. Artifacts in `runs/field/exp031_combined/raw4128/<a>_<b>/`:
  `all_merge{20,30}_<a>_<b>` with `.mp4`, `.csv`, `_frames.csv` and `.log`, the two dumps,
  and `first_by_run_<a>_<b>.png`. **The video is `_1080p.mp4`**, an H.264 copy at 1482x1080.
  The full-size `mp4v` at 4128x3008, which Windows players will not open, was deleted on
  2026-10-07. Delete the `_frames.csv` and re-run the driver to rebuild it.

**No labels, so no score.** Everything below is load, plus an eyeballed sheet.

| span | frames | shown/frame | frames with a #1 | #1 runs | #1 jumps > 29 px |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2–180 | 179 | 0.14 | 25 | 2 | 0 / 23 |
| 1090–1553 | 464 | 0.96 | 383 | 5 | 18 / 374 |
| 3130–3600 | 471 | 0.95 | 416 | 6 | 33 / 414 |

**Merge 20 and 30 are byte-identical in every shown CSV.** The merge radius is in absolute
pixels and is not rescaled by width, and at 4128 wide both radii sit well inside a target
that is 40–100 px across. So are `--kinematic`'s 25 px/frame ceiling, `--min-move 20` and
`--min-draw`. In mp4 pixels the ceiling is about 6 px/frame, four times stricter than on the
analog clips. Only the detectors' own constants (sky branch, window) scale with width/1440.
The gate held the tracks anyway, because at these ranges the drone moves slowly in the
picture.

**Eyeballed: the #1 is on the drone in 59 of 70 crops.** Each #1 run was cut at up to 6
evenly spaced frames, with no random seed, and every crop was taken from the `.raw` at
1:1. 12 of the 13 runs were sampled; the 3-frame run 3213–3215 was not. Judged by eye;
not a precision.

- **On the drone, 59 crops in 10 of the 12 runs sampled.** At this resolution the target is
  unmistakable: a hexacopter with a slung payload, rotors resolved, against sky and against
  the ridge (1517–1542, c 3–6, ground section). The three long runs follow it, 1113–1302 (189 frames),
  1331–1494 (164) and 3246–3556 (311). At 1098–1112 and 3216–3223 the circle sits on the
  payload or just off the body, because the drone fills most of a 192 px crop.
- **A tree, 5 crops, run 27–31** (ground section, c 9–10). The drone is in the same crops,
  dark against the slope just below and left of the circle, and is not the #1.
- **Beside a small object, 6 crops, run 3224–3245** (22 frames, mostly ground section, clear
  sky). The circle sits about 60 px from a small dark object that moves with it. The sheet
  cannot say whether that object is a bird or a second, distant drone. Undetermined.

By run length, about 794 of the 824 #1 frames belong to runs whose sampled crops are on the
drone. This is a judgement on sampled runs, not a count of frames.

**Read with:**
- The spans are where EXP-010's detector fired and was right on the mp4, so they are chosen
  by a detector. A quiet stretch outside them says nothing.
- No comparison to EXP-025e's FIELD mp4 run is valid: that run had a different width,
  had none of EXP-027 to EXP-030, and ran over the whole clip.
- At full resolution the target is large: the close passes are 60–100 px across. The small,
  distant case this project is built around appears only at the ends of the runs
  (3494–3556: about 20 px here, about 5 px in the mp4).

**Next:** labelling FIELD (the open todo) would turn this run into a score with no re-run.
The pixel constants in `overlay_split` (merge, ceiling, min-move, min-draw) should scale with
width like the detector's do, before any comparison across resolutions.
