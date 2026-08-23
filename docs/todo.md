# Todo

Captured tasks. Add via `/todo <description>`. Check an item off by moving it to Done
with the date it was finished. A task overtaken by a later decision moves to Superseded
rather than being deleted — the reasoning is why the current plan is the current plan.

Mission ids (M1–M7) refer to the approved evaluation plan: baseline through edge-ready
model. They are ordered — each assumes the previous one landed. Two constraints run
through all of them: the **15 official test videos only** until fine-tuning needs more,
and **field deployment is a hard requirement**, so every experiment from M3 on records
inference cost alongside accuracy.

`## Open` is ordered by mission id, with unscheduled backlog last. Dates are capture
dates, not priorities.

## Open

### M4b — generalisation: does GLAD hold up on video it has never seen?

- [ ] 2026-08-18 — [M4b] [algo] **Stretch, only after ARD100 lands: FL-Drones** (14
  videos / 38,948 frames, air-to-air, via the TransVisDrone repo). Genuinely held out —
  GLAD published on ARD-MAV and NPS-Drones but never on FL-Drones — but its confounds are
  *unbounded*, which is why it is second and not first: three annotation lineages, and
  752×480 against 1080p-tuned constants. Two blockers must be cleared before any
  FL-Drones number is believable:

  - **Resolution.** GLAD's motion module is tuned in absolute pixels for 1920×1080
    (`area 30–3000`, search region `a=160`, `dist_ref=200`, blur 11). Running as-is
    measures our failure to rescale, not GLAD's generalisation. Normalise by frame
    diagonal first: 752×480 → ×0.405 linear, ×0.164 for areas.
  - **Target size.** FL-Drones targets run up to 259×197 px while the motion branch caps
    blob area at 3000 px² (~55×55). Those targets are invisible to GMD/LMD *at any
    rescaling*, leaving only the YOLO appearance branch — and the motion branches carry
    recall (0.51→0.81 in the ablation). That is a structural handicap masquerading as a
    generalisation failure.

  Then sweep the IoU threshold, which M4a showed is the dominant variable.

### M5–M7

- [ ] 2026-08-20 — [M5-1] [algo] **Re-score EXP-001/002/003 to pick up `far`, `loc_err` and `loc_by_size`.** The metrics landed on 2026-08-20 and were applied to EXP-004 only; the three baseline runs still carry pre-M5 JSON, so the ledger's resize-vs-tile comparison has no false-alarm rate on either side. No inference — all three `detections.jsonl` are persisted. **The cost is labels, not scoring:** ~25 min wall-clock per criterion for the per-frame label read, so six re-scorings is an evening, and doing it *after* the per-video label store below turns it into minutes. Expect the localisation error to be the interesting column: these runs box 1.7x too large, so a large offset beside their near-zero recall would separate "never found it" from "found it and drew the wrong box".
- [ ] 2026-08-13 — [M7] [algo] [deploy] Fine-tune GLAD from its released weights on ARD-MAV's official training split, on a rented GPU (Kaggle 2×T4 free, or RunPod ~$1–2 for 3–4 h). Extract the 45 training videos **on the instance**, not locally. Score on the same held-out 15 videos; record whether it still fits the edge budget. Split across agents: `algo-agent` owns the training recipe and the ledger entry, `deploy-agent` owns provisioning, staging and the cost/wall-clock report.

- [ ] 2026-08-23 — [deploy] **Profile `GladPipeline.step` per stage on idle hardware.** `time.perf_counter()` around GAD, LAD, `MOD2_global`, `MOD2_local` and the classifier gate in `src/algo/glad/pipeline.py`, over ~2,000 contiguous frames. [edge-budget.md §2.3](edge-budget.md) currently *estimates* the split by solving `0.884*T_LAD + 0.116*(T_LAD + T_MOD) = 374 ms` and infers `T_MOD ~ 1.5 s`; that is arithmetic, not a profile, and it is the one number the whole edge budget rests on. **Could not be taken on 2026-08-23** — `exp005_glad_ard100` was occupying the CPU and a contended timing run violates the benchmarking rules in `deploy-agent`'s own charter. Cheap: minutes, no GPU. Also worth emitting p50/p95/p99 and the worst video rather than a run mean.
- [ ] 2026-08-23 — [deploy] **Export the two GLAD `yolov5s` checkpoints to ONNX and re-run through ONNX Runtime / OpenVINO on this host.** [edge-budget.md §3](edge-budget.md) item 1: expected 1.5-3x on the stage that dominates 88.4% of frames, and the ONNX artifact is the same one a Jetson TensorRT build would start from, so the work is not throwaway. **Not free of accuracy risk despite being fp32->fp32** — the NMS implementation changes — so it needs an `src.evaluate` re-score at the same criterion before the number is carried anywhere. Blocked on nothing.
- [ ] 2026-08-23 — [deploy] **Measure the pixel-scaling exponent of GLAD's motion path — no camera needed.** [edge-budget.md §4.2.4](edge-budget.md) prices a 12 MP Alvium by assuming every full-frame stage scales **linearly with pixel count (5.88x)**, and the entire "naive 12 MP is not viable" verdict rests on that one untested assumption. It is free to check: run `src.glad_detect` over a downscaled copy of two ARD-MAV test videos (960x540, i.e. 0.25x the pixels) and compare per-branch timings against 1080p. If the motion path does **not** fall ~4x, the 12 MP projection is wrong and must be redone. **Fold this into the per-stage profile above rather than running it separately** — same instrumentation, one extra resolution. Timings only: GLAD's constants are absolute pixels tuned for 1920x1080 ([glad-model.md §6](glad-model.md) item 4), so **the accuracy of a downscaled run is meaningless** and must not be recorded as a result.
- [ ] 2026-08-23 — [deploy] **Get the Alvium ROI mode-switch latency from Allied Vision.** [edge-budget.md §4.2.6](edge-budget.md) identifies sensor-side ROI readout as potentially better than the recommended hybrid — GLAD's local regime already *is* a 320x320 window, so reading only that window off the sensor would cut capture latency to near zero. Blocked on one number: how many frames a mode switch costs. EXP-004 says the regime is stable (acquisition fires 42 times in 28,337 frames), so even a several-frame switch may be affordable. A vendor email, not an experiment.
- [ ] 2026-08-23 — [deploy] [algo] **Decide 12 MP vs a narrow lens — needs one answer from the user: is the sensor cued or searching?** [edge-budget.md §4.2.8](edge-budget.md). A 28.6-degree lens on 1080p gives **identical pixels-on-target** to 12 MP at 60 degrees for **5.88x less compute, no retrain and no new camera** — it only costs field of view. If a bearing is handed over by radar/RF/GCS, 12 MP is buying FOV nobody uses. **Blocked on the user, and it is the cheapest open question in the project.** Now recorded as the third unresolved input alongside closing speed and persistence frames.


### Backlog — no mission, revisit when the trigger fires

- [ ] 2026-08-16 — [data] Derive per-frame **target motion** (fast/slow) from inter-frame box displacement in the ground truth, and add it as a second breakdown dimension alongside scene category. Nothing published provides this, so it has to be computed. Deferred from M2a, which delivered the scene-category half.
- [ ] 2026-08-13 — [data] Store labels per-video (one file, one row per frame) instead of one `.txt` per frame, and expand to the per-image tree only on the training instance. 28,337 tiny files cost minutes per full read — measured: two `find` calls and the MANIFEST regeneration all blew a 120 s timeout — and the per-image layout is only actually required by the ultralytics dataloader at M7, which runs on the rented GPU, not here. Space is not the issue (NTFS keeps sub-700-byte files resident in the MFT); per-file syscall latency is. **Trigger:** label reading starts dominating the M5 re-scoring loop, or the 45 training videos push the tree past ~100k files.

## Done

- [x] 2026-08-23 — [data] **Derived the `lighting` and `relative_range` axes for the
  ARD-MAV test split** — `py -3.13 -m src.data.scene_stats --processed
  data/processed/ARD-MAV --split test`, 28,337 frames, ~55 min. Open since 2026-08-18 and
  abandoned twice for competing with a scoring pass; run alone this time. `conditions.json`
  now carries all three axes (`scene_category` preserved), so `src.evaluate --conditions`
  breaks any ARD-MAV run down along them.

  **Done because M4b's backlit control needed it to be symmetric**, and it immediately
  corrected two things now fixed in [experiments.md](experiments.md) and
  [prepare_ardmav.md](prepare_ardmav.md):

  1. **ARD-MAV test is 18.6% `backlit`, not the "~0%" EXP-005 first assumed.** The
     0.0–0.2% figure was a per-video mean `pct_blown`; `backlit` is a per-frame test at 2%.
     Different quantities. Real gap against ARD100 is 18.6% vs 29.7%.
  2. **`backlit` is the *easiest* bucket on ARD-MAV** (R 0.9165, above its 0.8946 aggregate)
     and the second-hardest on ARD100 (0.6277), so the label does not carry an intrinsic
     difficulty.

  EXP-005's conclusion is unchanged: the symmetric non-backlit comparison is 0.8896 →
  0.7126, **17.7 points**, against 20.7 aggregate. Also settled that `relative_range` is
  **not** cross-dataset comparable — ARD-MAV has a `very far (>5x)` bucket (2,758 frames)
  that ARD100 lacks entirely, and 36.5% `far` against ARD100's 3.0%.

- [x] 2026-08-23 — [M4b] [algo] **Scored GLAD on ARD100's 15 unseen videos — M4b is
  answered. Recorded as EXP-005 in [experiments.md](experiments.md).** 34,287 frames /
  33,517 boxes, 2.65 h at 3.60 fps on CPU, byte-identical settings to EXP-004 so the one
  variable is the video content.

  **GLAD retains roughly three-quarters of its recall on unseen video from the same
  campaign.** At `centre@1x` — recall **0.895 → 0.688**, precision 0.993 → 0.946, false
  alarms **188 → 1,316** (0.0066 → 0.0384 per frame). At IoU@0.50, 0.834 / 0.606 / 0.702.
  The honest sentence is *"retains 77% of its recall on unseen video from the same
  campaign"*, never *"GLAD generalises"* — same lab, same rigs, likely the same capture
  campaign, so this is the optimistic end of any generalisation estimate.

  **Both confounds the plan warned about were checked, and neither explains the drop:**
  1. **Backlighting** (29.7% of frames, against ARD-MAV's ~0%) — cutting every backlit
     frame lifts recall only **2.5 points, 0.688 → 0.713**. An **18.2-point gap survives
     the control**, so exposure is ~1/8 of the effect.
  2. **Size composition** — recall falls in the **medium (32–96 px)** control bucket too,
     0.956 → 0.658. A composition shift reweights buckets; it cannot move them.

  **The mechanism is lock-loss, not localisation.** `global miss` rises 2.9% → 12.3% and
  `local yolo` falls 88.4% → 66.6%: GLAD loses lock four times as often and re-acquires
  worse. Meanwhile mean IoU on matched boxes is **0.6805 against 0.6829** — when it finds
  a drone it places the box exactly as well as on its training campaign. Range collapses:
  near 0.784 / mid 0.561 / **far 0.164**.

  **What this hands M7:** the lever is acquisition — GAD failing to re-acquire after lock
  loss, and at range — not a box-regression head and not a tighter matching rule.

  Two smaller notes carried into the ledger: ARD100 publishes no
  `ordinary`/`complex`/`small_mav` grouping, so EXP-004's headline cut has no counterpart
  and the comparison is aggregate plus **size in pixels** (never `relative_range`, which is
  scaled per split); and the motion module's 50-candidate cap
  (`third_party/GLAD/MOD2.py:60,183`, which returns an *empty* list rather than degrading)
  **fired 35 times** — never triggered on ARD-MAV, negligible here, but a silent recall
  floor on genuinely cluttered data.

- [x] 2026-08-23 — [M6] [deploy] **Added §4.2 to `docs/edge-budget.md`: the camera and the
  capture stage.** Answers "what FPS with a 12 MP Allied Vision 1800?" — and the honest
  answer is **it depends which of four cameras that is, and feeding GLAD the full 12 MP
  does not work.** The doc previously had **no camera model at all**, a genuine gap:
  capture is the first term in the end-to-end chain `deploy-agent`'s charter defines, and
  at 12 MP it stops being a rounding error (~30-35 ms, comparable to the whole inference —
  so §1's assumed 60 ms of pipeline overhead is now flagged as too low).

  **"12 MP Alvium 1800" is four products.** U-1240 (IMX226 rolling, USB3, **29 fps**),
  C-1240 (IMX226 rolling, CSI-2 4-lane, **41 fps**), U-1236 / C-1236 (IMX304 **global
  shutter**, **22-23 fps**). Budgeted the C-1240 as the compute worst case. **The U-1240's
  29 fps is a USB3 payload cap at 8-bit, not a sensor figure** — ask for RAW10 and it falls
  to ~23. On CSI-2 the same sensor needs 5.00 Gbit/s against a 10 Gbit/s 4-lane link, so
  there the sensor binds and the link does not.

  **Ingest is fine, and is the least risky part of the whole deployment:** Orin NX offers
  two 4-lane D-PHY groups at 2.5 Gbit/s per lane, its ISP runs 1.75 GPixel/s against the
  500 MPixel/s needed (29%), and Allied Vision ship a **JetPack 6.2 `.deb` driver covering
  all Orin modules** — the exact opposite of the dead TensorRT 7.2 path in §4. Caveat: the
  driver is **4-lane only**, no 2-lane fallback. USB3 would additionally tax 15-30% of a
  core on the branch that is **CPU-bound OpenCV**, so CSI-2 wins on two counts.

  **The crux, decomposed against EXP-004's branch split.** 12.20 MP is **5.88x** 1080p.
  The modal frame (88.4%, LAD on a fixed 320x320 crop) is **inference-cost-invariant to
  sensor resolution** — but its per-frame full-frame overhead is not, and the motion path
  (11.6%, all `O(pixels)`) takes the full 5.88x on the branch that was **already** the
  bottleneck. Projected Orin NX: **~13-14 fps sustained but 2.4-3.2 fps when the motion
  path fires**, which spends 38% of the whole engagement in latency. **That row fails.**
  Worse, GAD's 640 letterbox goes from a 3x reduction to **6.3x**, so a 21 px target
  reaches the acquisition detector at 3.3 px — the founding trap, doubled, on the branch
  EXP-004 already named as the real ceiling.

  **The upside is real and was stated fairly.** At the same FOV, 4024 px is **2.096x finer
  linearly**: first-detectable range moves **55 m -> 115 m**, the engagement window
  **1.4 s -> 2.9 s**, and — the number that matters — **the 0.98-recall boundary moves from
  ~34 m out to ~72 m.** 18,265 of ARD-MAV's 28,160 targets are under 16 px, so this attacks
  the project's central failure mode more directly than any architecture on the table.

  **Recommendation: never feed GLAD the full frame.** Option (c) — native-resolution 320
  crop for LAD, downsampled copy for the motion module *and* GAD — projects **~25 fps
  sustained / 7-9 fps hard scene** and is the only configuration that improves both range
  and the latency ratio. **Flagged loudly that this is a protocol change, not a config:**
  `yolov5s_GLAD-crop.pt` was trained on 320x320 crops from 1080p, so a native crop of a
  12 MP frame is a 2.1x input-scale change needing a **retrain, not a re-score** — and
  every absolute-pixel constant in `MOD2.py` breaks with it. No accuracy number in this
  repo may be read as applying to a 12 MP build, and that verdict is `algo-agent`'s.

  **Two findings that outrank the frame rate.** (1) **Rolling vs global shutter.** IMX226
  is rolling; GLAD compensates ego-motion with a **RANSAC homography**, which assumes all
  pixels were captured at one instant. Under rolling shutter with fast rotation that
  assumption fails and the residual — *which is exactly what GLAD thresholds as motion* —
  is corrupted, worst during hard manoeuvre. Counter-argument recorded honestly: ARD-MAV
  was shot on rolling-shutter Mavics, so it demonstrably works **on a gimbal**; nothing
  here measures hard-mounted. Since the pipeline sustains only ~13-25 fps anyway, **the
  C-1240's extra frame rate is unusable and the global-shutter C-1236 is probably the
  better buy** — its 1.1-inch sensor also attacks M2b's measured 19-point contrast loss.
  (2) **The lens is a competing answer and it is cheaper**: a 28.6-degree lens on 1080p
  gives identical pixels-on-target for 5.88x less compute. Which wins turns entirely on a
  question nobody has answered — **is the sensor cued or searching?** — now recorded as the
  **third unresolved user input** alongside closing speed and persistence frames.

  §8's three tables updated (camera/ingest specs as published, the 12 MP projections as
  [EXTRAPOLATED], six new assumptions including the untested linear-pixel-scaling
  exponent), §1 and §6 cross-linked. **No tests: docs-only, no behaviour or parameters
  under `src/`.** Three follow-ups filed — the pixel-scaling check (free, no camera), the
  ROI mode-switch latency (a vendor email), and the cued-vs-searching decision (blocked on
  the user, and the cheapest open question in the project).

- [x] 2026-08-23 — [M6] [deploy] **Wrote `docs/edge-budget.md`**, answering "this model is
  very slow, is there a faster one I can deploy?" — and the answer is **no, and you do not
  need one.** The slowness decomposes almost entirely onto the **host**: GLAD is published
  at 23.6 FPS on a 2019 Jetson Xavier NX and 146.5 FPS on an RTX 3070 against our **2.67
  fps** sustained (EXP-004, 28,337 frames, fp32 PyTorch, CPU-only laptop) — **~9x and ~55x,
  closed by a $249 board, not by research.** Protocol contributes ~1x (GLAD has no tile or
  resize switch, and striding is incoherent for a pipeline that differences consecutive
  frames) and the model ~1x (`yolov5s` at 640 is already the cheap end of what works).

  **The premise inverts on the measured numbers.** GLAD is the *second-fastest* of the five
  configurations this project has timed and carries 7-70x the recall of every one of them;
  the two fastest score exactly **zero** tiny-target recall. There is no speed/accuracy
  trade available because the fast end of the curve is empty.

  **Two premises in the prompt were corrected in the doc.** The 11 fps / 0.72 fps pair in
  `CLAUDE.md` is `src.baseline_detect` with **`yolov8n` at 720p** — a different CLI, model
  and resolution, not GLAD's number. And GLAD's own two figures disagree: **4.8 fps spot vs
  2.67 fps sustained**, with `exp005` observed at 3.16-3.92 — the mean-vs-worst-case rule
  applying to our own measurement, not just to vendors'.

  Recommendation: **Jetson Orin Nano Super 8 GB (~$249, 67 TOPS, 102 GB/s, 15 W)**, GLAD
  unchanged, `.pt` -> ONNX -> **TensorRT 10 FP16**, stopping short of INT8 (1.2x for a
  3-7 point mAP risk landing in exactly the smallest size bands, where 18,265 of ARD-MAV's
  28,160 targets live). Confirmed a **toolchain break** that makes this a port and not an
  export: the released `detector*_trt.py` deserialises **TensorRT 7.2** engines, and TRT
  engines do not load across JetPack versions — JetPack 6 ships TRT 10.x with a changed
  API, so the engines and the `libmyplugins.so` plugin must be rebuilt. Six explicit
  change-my-mind triggers recorded, including **multi-intruder, which architecturally
  disqualifies GLAD** (single target by construction) and promotes YOLOMG regardless of
  speed.

  Alternatives surveyed with board/precision/resolution/batch/content on every figure and
  the non-transferable ones marked: YOLOMG (133/35 FPS but on an **RTX 2080Ti**, different
  dataset *and* split from EXP-004, GPL-3.0, no pretrained weights), A2A-YOLO (15 FPS on
  RK3588 but Det-Fly's 4K large targets, appearance-only), TransVisDrone, and plain
  YOLO26n/YOLO11n at 3.80 ms INT8 on Orin — which is the "fast model" the question reaches
  for and which EXP-001-003 already measured at **AP@0.5 0.006-0.025**.

  **The whole document is marked as assumption where it is one.** Closing speed and
  persistence frames remain **the user's calls and are still unsupplied**, so §1 is worked
  with invented values and labelled as an illustration; §8 separates measured / published-
  under-their-conditions / assumed line by line. No edge hardware has been benchmarked and
  the doc says so at the top. Every accuracy claim about a swapped or quantised model is
  flagged unproven until re-scored through `src.evaluate` at the same criterion — that
  verdict is `algo-agent`'s property, not `deploy-agent`'s.

  **No tests: docs-only change, no behaviour and no parameters under `src/`.** Two
  follow-ups filed under Open — the per-stage profile (which could not be taken because
  `exp005` was holding the CPU) and the ONNX/OpenVINO export.

- [x] 2026-08-20 — [meta] [deploy] **Added `deploy-agent`, a third boundary beside data
  and model.** `.claude/agents/deploy-agent.md`, registered in `CLAUDE.md`. M6 and M7 were
  both open and **neither existing agent owned them**: `dataset-agent` opens with "models
  are somebody else's problem", and `algo-agent`'s only contact with hardware is one bullet
  telling it to quote GPU cost. Nobody owned *does it run on the target board at the target
  frame rate*. The new agent owns `docs/edge-budget.md`, latency/power budgets, export and
  quantisation, board selection, and renting the GPU that training runs on — while the
  accuracy ledger stays `algo-agent`'s property, so the M6/M7 split is stated in both
  agents' terms rather than left to be discovered.

  **Three project-specific traps written into it**, all of which would otherwise have been
  rediscovered on the hardware:

  1. **A mean frame rate is not a guarantee.** GLAD's 23.6 FPS on a Xavier NX is an
     *average* over a pipeline whose expensive motion path fires only when appearance
     detection fails — GMD alone is 5.1 FPS on that board. Throughput collapses ~4.6x
     exactly when the scene is hard, so the agent reports worst case and percentiles.
  2. **Export is a protocol change, not packaging.** FP32→FP16→INT8 is the `imgsz` rule
     `algo-agent` already enforces, wearing a deployment badge — and the damage lands in
     the smallest size bands, where a 10–30 px target is a handful of activations.
     Re-score through `src.evaluate`; the size cut is a re-cut of a persisted dump, not a
     new experiment.
  3. **Every number in the ledger came from our CPU shim, not from TensorRT.** EXP-004 ran
     `src/algo/glad/`; the released entry point deserialises TRT 7.2 engines through
     `detector*_trt.py`. A Jetson means returning to a code path **no measurement in this
     project was taken on** — a new implementation to validate, not an export.

  Also derives "real-time" from closing speed and persistence frames rather than inheriting
  30 FPS, and separates detection latency from inference latency. No tests: an agent
  definition is markdown config, with no behaviour under `src/` to cover.

- [x] 2026-08-20 — [algo] **Bin false alarms by distance from the nearest real drone.**
  `src/eval/alarms.py` + CLI `src/alarm_eval.py`, documented in
  [alarm_eval.md](alarm_eval.md); `nearest_gt_dist` / `nearest_gt_dist_rel` /
  `nearest_gt_size` added to the dump in `src/eval/records.py`, computed by the new
  `metrics.nearest_target`. Separates the two failures `far` pools: a box on the drone
  versus a box on a rooftop. Zero cost — dumps written before the columns landed are
  re-derived from the frame's own rows, so no run needed re-scoring. Result in
  [experiments.md](experiments.md): **all 3,470 alarms IoU@0.50 adds over centre matching
  are inside one target size** (median 2.2 px), while GLAD's 188 real alarms have **zero**
  inside one target size and a median of 100 px. Tabled in both units (`--unit px` added
  to the ledger 2026-08-20, `runs/exp004_glad/alarm_distance_px.csv`); the pixel ladder
  smears the criterion boundary across three bins because the boundary is size-relative,
  which is why `rel` is the default. The baselines are 65–78% beyond 32 target
  sizes at a ~600–870 px median — their failure was never localisation.

- [x] 2026-08-20 — [algo] **Cross-cut the scoring dump by target size *and* capture
  condition at once.** `src/eval/crosscut.py` + CLI `src/cross_eval.py`, documented in
  [cross_eval.md](cross_eval.md). Answers the conjunction the metric block's one-way
  blocks cannot — the marginals are not multipliable, since 5,032 of ARD-MAV's 5,677
  sub-8 px targets are `small_mav`. Cells are built frame-by-frame so a false alarm
  inherits the band of the frame it fired in, which is what makes per-cell `far`
  comparable to the metric block's. Zero cost: a `GROUP BY` over persisted dumps, and
  pooling every cell reproduces EXP-004's headline exactly. Result in
  [experiments.md](experiments.md): the sub-8 px / `complex` cell is Pd 0.8625 at 0.0031
  FA/frame under centre matching and Pd 0.5844 at 0.2812 under IoU@0.50 — and 562 of its
  640 targets are one video (phantom58), so the background label is not what the number
  is about.

- [x] 2026-08-20 — [tools] [algo] **`src.render_video`: a scored run drawn back onto its source video.** Both boxes on every frame — ground truth and the detector's claim — coloured by the match outcome, with a nearest-neighbour magnified inset (a 9 px drone is otherwise a speck) and a caption strip carrying the frame key, the frame's TP/FP/missed tally, the matching rule and whatever the run recorded per frame (`branch`, for GLAD). **Verdicts come from `match_frame`**, the same function `records` and `curves` call, so the picture cannot disagree with the ledger; nothing is re-inferred, the boxes come from the persisted `detections.jsonl`. Rendered `phantom19` from EXP-004 to `runs/exp004_glad/examples/phantom19_overlay.mp4` (2,158 frames, 251 MB, not committed — `runs/` is gitignored). New `src/output/overlay.py` and `src/output/video.py` (`LazyVideoWriter`, now shared with `annotate.VideoSink`); `load_frames` gained a `key_filter` so one video does not pay 28,337 label reads. 32 new tests; reference in [render_video.md](render_video.md).

- [x] 2026-08-20 — [M5] [algo] **`src.evaluate` now reports FAR and a size-normalised localisation error per size bin.** `far` (FP/frame), `loc_err` + `loc_err_p90` (centre offset in multiples of the target's own size) and `loc_by_size` on `Metrics`, `far` on `ConditionScore`, `loc_error_by_size` in `src.eval.curves` (new `ErrorCurve`, written into the `plot_eval` sidecar). All appended after `criterion`, so the `--json-out` key order the ledger cites is unchanged. **No separate `Pd` field: Pd *is* recall** — ARD-MAV carries ~1 target per frame so the per-target and per-frame readings coincide, and a duplicate field would be two names for one number that could drift; the report labels it `recall (Pd)`. **Applied to EXP-004 (M4a) only** — see [M5-1] for the three baselines. Both criteria re-scored from the persisted JSONL with every headline number reproducing exactly. **The finding:** relative localisation error degrades 5x monotonically as targets shrink (0.051 target-widths at 32-48 px to 0.255 below 8 px), which is the measured mechanism behind the IoU/centre gap this project has been asserting from P/R since EXP-004 — at 0.255 no pair can clear IoU 0.50. FAR 0.0066/frame centre vs 0.1291 IoU, and **69% of the centre-matched false alarms are one video** (phantom63, 129 of 188), so the aggregate rate is not a property of the detector. 11 new tests; full write-up in docs/experiments.md.
- [x] 2026-08-20 — [meta] **Make the definition of done enforceable instead of repeatable.** The instruction "commit, push, test, document, tick the todo" was typed again on 2026-08-20 despite already being three separate bullets in `CLAUDE.md` — and tests were never in there at all. Now one `## Definition of done` section in `CLAUDE.md` (four numbered gates), plus `.claude/hooks/definition_of_done.py` wired in a new `.claude/settings.json`: `PostToolUse` on `Write|Edit` records which repo file the session touched (async, so it adds no latency to an edit), and `Stop` blocks the turn while any of those is uncommitted or unpushed. **Tracks the session's own edits, not `git status`** — parallel sessions each see the others' work in progress as a dirty tree, and a hook that cries wolf gets switched off. Blocks only on what is machine-checkable; tests/docs/todo ride in the message. Guarded against looping via `stop_hook_active`, and silent on gitignored paths, paths outside the repo, malformed stdin and a missing upstream. Verified by piping synthetic payloads through both modes. **Two things to know:** `.claude/settings.json` did not exist when the current sessions started, so the hook only takes effect in sessions opened *after* this commit; and the file carries hooks only — `/fewer-permission-prompts` still has the permission allowlist to write.
- [x] 2026-08-16 — [M4a] [algo] Run GLAD's released weights (MIT, no training) over the ARD-MAV 15 test videos and compare against their published P/R/F1 — 0.99/0.96/0.97 ordinary, 0.94/0.86/0.90 complex, 0.82/0.67/0.73 small MAV. **This is a test of our pipeline, not of GLAD.** GLAD's weights were trained on ARD-MAV's other 45 videos and its architecture was tuned against this very split, so the number is optimistic by construction and must never be reported as "GLAD scores X for us". Reproducing their per-category rows is the only independent confirmation we have that M1's harness and M2's conversion are correct; a large gap means we have a bug. **Needs contiguous frames** — GLAD's motion modules difference consecutive frames, so run it on the original `.mp4`s, not M3's stride-10 stills. **Two corrections from the 2026-08-16 model survey ([glad-model.md](glad-model.md)):** (a) the reproduction target is the *ablation* row **P 0.91 / R 0.81 / F1 0.86**, not full GLAD — the released code has no Kalman filter and no adaptive search region (fixed 320×320), so the per-condition figures above are an upper bound; (b) it **runs on this laptop** — `GLAD.py` wants TensorRT engines, but `yolov5s_GLAD.pt` / `-crop.pt` ship alongside and load on CPU via `third_party/yolov5` (v6.0, `nc=1`, `names=['Drone']`, verified). Needs a shim replacing `detector{1,2,3}_trt` with yolov5 CPU inference. Score on **P/R/F1 only** — GLAD emits no confidences, so any AP for it is not comparable. **Done 2026-08-16 — EXP-004.** CPU port at `src/algo/glad/` driven by `src.glad_detect`; all 28,337 frames in 2.9 h at 2.67 fps. **The harness is validated:** `ordinary` came back 0.987/0.965 against the published 0.99/0.96, and the aggregate 0.856/0.771/0.811 sits within 0.054 P and 0.039 R of the ablation target. The residual is concentrated entirely in `small_mav` and is a **scoring threshold, not a detector** — re-scored at IoU 0.40 our numbers bracket the published ones in all three categories. Three upstream defects found; the letterbox padding one is fixed behind `--pad` and measured to be worth nothing. Full write-up in docs/experiments.md.
- [x] 2026-08-16 — [M2a] [data] Wired `conditions.json` into `src.evaluate` via `--conditions`: per-scene-category precision/recall/F1/AP, which is the only form comparable to GLAD's published per-category figures. Re-scored EXP-001/002/003 for free from the persisted JSONL. Immediately paid off — it exposed that precision on *ordinary* backgrounds is 5x that on *complex* (the domain-shift signature), and that `small_mav` is a total wipeout (2 correct out of 935) rather than merely weak, both of which the aggregate hid. Empty categories report NaN, not 0.0.
- [x] 2026-08-16 — [M3] [algo] Baseline established: EXP-001/002/003, one set of off-the-shelf weights at three effective resolutions (6.7 / 13.3 / 20 px on target). AP@0.5 0.0058 / 0.0219 / 0.0251. Recall rises 10x with resolution but precision collapses to 3% (11,353 false positives when tiled), and 97% of tiny targets are missed even with no downscaling at all. Resolution is the binding constraint but is not sufficient — the ceiling is the model. False positives land on window recesses and AC units, and predicted boxes run 1.7x too large. Ran three configs rather than the planned two so the medium bucket could act as a control. Full write-up in docs/experiments.md.
- [x] 2026-08-13 — [M2] [data] Prepared the ARD-MAV test split: 28,337 frames / 28,160 boxes from the 15 official test videos, VOC XML → YOLO, with `data.yaml`, `conditions.json` and `MANIFEST.md`. Validation battery clean — no size mismatches, no out-of-range coordinates, no degenerate boxes, only `Drone` as a class name. Labels visually verified across three videos and two scene categories. **Two things I had wrong going in:** `CAP_PROP_FRAME_COUNT` overstates by 307 (header metadata, not decodable frames — every decodable frame *is* annotated), and a missing XML means *unannotated*, not *negative*; the 177 genuine negatives are frames whose XML contains zero objects, clustered in phantom47/08/09. Size distribution: 64.8% tiny, 28.2% small, 7.0% medium, **0% large**.
- [x] 2026-08-13 — [M1] [test] Verify the evaluation math. 50 tests over `src/eval/` covering IoU, greedy confidence-ordered matching, one-claim-per-target, class-awareness, AP endpoints and the CLI contract. All 8 deliberate mutations (threshold `>=`→`>`, ascending sort, repeated claims, dropped class check, IoU union, AP envelope, YOLO half-extent, corner/centre swap) were caught.

## Superseded

Not to be worked. Kept because each records why a plausible option was rejected —
deleting them invites re-proposing the same dataset next month.

- [~] 2026-08-18 — [M4b-1-alt] [algo] **Download NPS-Drones and score GLAD on its published 10-video test split.** *Superseded 2026-08-18: chosen against the wrong criterion.* The reasoning was that NPS is the only freely downloadable set carrying a published GLAD number in metrics we can compute (Table V: **P 0.92 / R 0.95 / F1 0.93**); Drone-vs-Bird was ruled out because GLAD published **AP only** there (0.701) and our port emits no confidence scores. But the anchor for M4b is EXP-004, our own run, so protocol identity is what matters and an external published number is not. **The fatal defect:** NPS targets are **delta-wing fixed-wing UAVs, not multirotors** — the source paper attributes its detection signature to "the delta wing shape of the UAVs", and the max target box is 65×21, a 3:1 planform. Useful as generic small-object-against-sky pretraining only. Also, GLAD trained on 40 of the 50 videos, so it says nothing about held-out performance. Detail retained if it is ever revived: direct HTTP, BSD-3, no registration (`curl -O https://engineering.purdue.edu/~bouman/UAV_Dataset/{Videos.zip,Video_Annotation-v2.zip}`), ≈1.5 h CPU for the 10 test videos (~14k frames at 2.67 fps); annotation **v2** only, the published last-10 split, P/R/F1 never AP, and NPS ships **two resolutions** (1920×1080 and 1280×960) against 1080p-tuned constants, so restrict to 1080p or rescale by frame diagonal and report which.
- [~] 2026-08-18 — [M4b-0] [algo] **Run GLAD over 15 of its own 45 training videos and compare against EXP-004's test numbers.** *Declined 2026-08-18.* The idea: all 60 raw ARD-MAV videos are on disk, so the train/test gap is measurable tonight for ~2.9 h CPU with zero new code, and it bounds how optimistic ARD100-extra will be — if train ≈ test (~0.86 F1) GLAD is not overfit to this campaign; if train ≫ test, the optimism is quantified in advance. Bucketing was to use `src.data.scene_stats`, which derives `lighting` and `relative_range` from imagery and labels directly. Declined in favour of spending the effort on acquiring genuinely unseen video instead.
- [~] 2026-08-17 — [M4b] [data] **MOT-FLY as a third test set.** *Withdrawn 2026-08-17: unobtainable.* Its Google Drive link returns "file does not exist", and repo issue #1 (link expired) has sat unanswered since 2024-12-27 with no mirror anywhere. Only the untested Baidu link (`pe53`) and an email to `3120210041@bit.edu.cn` remain, so it cannot be planned around.
- [~] 2026-08-16 — [M4b] [data] **Det-Fly and Anti-UAV300 as test sets.** *Rejected.* Det-Fly is sparse stills with no stated frame ordering, so GLAD's motion branch cannot run and it would benchmark the appearance path alone — breaks criteria 1 and 5 above. Anti-UAV300 is shot from a static ground camera, which makes motion compensation trivial and would score GLAD artificially high — breaks 2 and 4.
