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

- [ ] 2026-08-18 — [data] Derive the **lighting** condition axis for the ARD-MAV test split: `py -3.13 -m src.data.scene_stats --processed data/processed/ARD-MAV --split test`. It merges `lighting` and `relative_range` into `conditions.json`, after which `src.evaluate --conditions` breaks every run down along them with no further work. **Budget ~1 h wall-clock and run it alone** — it decodes all 28,337 JPEGs and measured ~100 frames/min while competing with a scoring pass, which is why it was abandoned twice on 2026-08-18. `relative_range` is already available indirectly from `gt_size` in a `--dump` CSV; `lighting` is the part that needs the pixels.
### M4b — generalisation: does GLAD hold up on video it has never seen?

- [ ] 2026-08-18 — [M4b] [algo] **Get ARD100 and score GLAD on 10–15 of its videos that
  are not in our local 60.** ARD100 is 100 videos / 202,467 frames from the same lab as
  ARD-MAV; excluding by filename whatever overlaps our 60 leaves an unseen remainder
  without needing to know the overlap in advance.

  **Why ARD100 and not a more independent dataset.** The comparison anchor is EXP-004,
  our own run — not a published table — so what matters is *protocol identity with
  EXP-004*, and M4b must change exactly one thing: the video content. M4a and M2b both
  showed that uncontrolled variables dominate the effect being measured — moving the IoU
  threshold 0.50→0.40 shifted `small_mav` precision 23 points, and target/background
  contrast alone swings small-target recall 19 points. Both exceed any plausible
  generalisation drop. Six things must therefore be held fixed:

  1. **1920×1080** — GLAD's constants are absolute pixels.
  2. **Same annotation standard** — box-tightness convention alone can outweigh the
     effect being measured (see the 23-point swing above).
  3. **Multirotor targets.**
  4. **Air-to-air moving camera** — otherwise motion compensation is exercised
     differently.
  5. **Full-rate contiguous video** — the motion branches difference consecutive frames.
  6. **No new code path.**

  ARD100 is ✓ on all six: same lab, same VOC XML, same Mavic2/M300 rigs, distributed as
  `.mp4`, so `prepare_ardmav` runs unmodified and every hardcoded pixel constant stays
  valid. It is also genuinely *harder* — the smallest average target of any published set
  (0.01% of frame vs ARD-MAV's 0.02%), plus low light and abrupt camera movement — so a
  drop is informative. Its weakness, same lab and likely the same capture campaign, is
  **bounded and statable**: the number is optimistic, and "GLAD retains X on unseen video
  from the same campaign" is still a true and useful sentence.

  **Routes, in order.** (1) Try the Baidu share link in a browser:
  `https://pan.baidu.com/s/1ycAoKbzQ1rlzvKr8VRakgw?pwd=1x2z`. (2) Email
  `guohanqing@westlake.edu.cn` — GLAD's lead author, same lab, already mirrors ARD-MAV on
  Google Drive. All 60 local filenames are known, so ask for a specific ~40-video list
  rather than "the dataset", and offer to accept any host or a 10-video subset. The cost
  here is Baidu friction, not validity.

  **Constraints on the run.** 10–15 unseen videos carry the same statistical weight as
  EXP-004's 15, so the full 100 is unnecessary. If disk is tight, subsample by whole
  videos at **native resolution** — never downscale, since shrinking targets destroys
  exactly the small-object performance being measured — and never stride frames.

  **If a frames-based dataset is ever substituted:** `src.glad_detect` reads `.mp4`
  through `cv2.VideoCapture` only. The video/images handling in `src/data/sources.py` is
  wired into `baseline_detect`, not into it, so a frame-sequence source needs an
  ordered-file reader added to `run_video` first — modest, but a new code path to
  validate before it can carry a comparison, and it breaks criterion 6 above.

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

- [ ] 2026-08-13 — [M5] [algo] Extend `src.evaluate` with Pd and FAR plus a size-normalised localisation error (bbox-to-truth offset scaled by true target size), reported per size bin. Re-scoring is free — `detections.jsonl` is persisted — so amend EXP-001/002/003 rather than re-running inference.
- [ ] 2026-08-13 — [M6] [deploy] Write `docs/edge-budget.md`: target FPS, resolution, power and latency ceiling, and what "real-time" means for closing speed. Add FPS-on-target as a required ledger field, recommend a board (Jetson Orin Nano 8 GB, ~$250 — GLAD's published 23.6 FPS on the older Xavier NX is a floor), and plan the TensorRT export path. On-device benchmarking deferred until hardware exists.
- [ ] 2026-08-13 — [M7] [algo] Fine-tune GLAD from its released weights on ARD-MAV's official training split, on a rented GPU (Kaggle 2×T4 free, or RunPod ~$1–2 for 3–4 h). Extract the 45 training videos **on the instance**, not locally. Score on the same held-out 15 videos; record whether it still fits the edge budget.

### Backlog — no mission, revisit when the trigger fires

- [ ] 2026-08-16 — [data] Derive per-frame **target motion** (fast/slow) from inter-frame box displacement in the ground truth, and add it as a second breakdown dimension alongside scene category. Nothing published provides this, so it has to be computed. Deferred from M2a, which delivered the scene-category half.
- [ ] 2026-08-13 — [data] Store labels per-video (one file, one row per frame) instead of one `.txt` per frame, and expand to the per-image tree only on the training instance. 28,337 tiny files cost minutes per full read — measured: two `find` calls and the MANIFEST regeneration all blew a 120 s timeout — and the per-image layout is only actually required by the ultralytics dataloader at M7, which runs on the rented GPU, not here. Space is not the issue (NTFS keeps sub-700-byte files resident in the MFT); per-file syscall latency is. **Trigger:** label reading starts dominating the M5 re-scoring loop, or the 45 training videos push the tree past ~100k files.

## Done

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
