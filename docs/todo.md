# Todo

Captured tasks. Add via `/todo <description>`. Check an item off by moving it to Done
with the date it was finished.

Mission ids (M1–M7) refer to the approved evaluation plan: baseline through edge-ready
model. They are ordered — each assumes the previous one landed. Two constraints run
through all of them: the **15 official test videos only** until fine-tuning needs more,
and **field deployment is a hard requirement**, so every experiment from M3 on records
inference cost alongside accuracy.

## Open

- [ ] 2026-08-13 — [M2] [data] Prepare the ARD-MAV test split: relocate to `data/raw/`, extract frames from the 15 official test videos, convert VOC XML → YOLO txt, write `data.yaml` + `MANIFEST.md` with the official 45/15 split, run the validation battery, and visually verify ~20 rendered frames. Watch the two silent failure modes: 1-based 4-digit frame numbering (`phantom02_0001`) must match exactly, and a missing XML is a negative, not an error.
- [ ] 2026-08-13 — [M2a] [data] Tag each eval frame with background class (sky/urban/trees/mixed) and target motion (fast/slow), so `src.evaluate` can break metrics down by condition as it already does by target size. Worth doing alongside M2: GLAD publishes its results split by scene category (ordinary / complex background / small MAV), so without condition tags the M4 comparison against those published numbers cannot be made properly.
- [ ] 2026-08-13 — [M3] [algo] Evaluate an off-the-shelf single-class drone detector on our footage as-is, no fine-tuning, whole-frame and `--tile` — EXP-001 / EXP-002. Identical weights, threshold and split; only tiling differs, so the gap isolates resolution loss from genuine small-object weakness. Expect ~AP 0.53 (YOLOv5's published ARD100 number); that is the figure GLAD has to beat.
- [ ] 2026-08-13 — [M4] [algo] EXP-003: run GLAD's released weights (MIT, no training) over the same 15 test videos, scored with our harness on our split. Also check against GLAD's published P/R/F1 on this set — 0.82/0.67/0.73 on small MAVs. Reproducing that row validates the whole pipeline; a large gap means we have a bug, not a discovery.
- [ ] 2026-08-13 — [M5] [algo] Extend `src.evaluate` with Pd and FAR plus a size-normalised localisation error (bbox-to-truth offset scaled by true target size), reported per size bin. Re-scoring is free — `detections.jsonl` is persisted — so amend EXP-001/002/003 rather than re-running inference.
- [ ] 2026-08-13 — [M6] [deploy] Write `docs/edge-budget.md`: target FPS, resolution, power and latency ceiling, and what "real-time" means for closing speed. Add FPS-on-target as a required ledger field, recommend a board (Jetson Orin Nano 8 GB, ~$250 — GLAD's published 23.6 FPS on the older Xavier NX is a floor), and plan the TensorRT export path. On-device benchmarking deferred until hardware exists.
- [ ] 2026-08-13 — [M7] [algo] Fine-tune GLAD from its released weights on ARD-MAV's official training split, on a rented GPU (Kaggle 2×T4 free, or RunPod ~$1–2 for 3–4 h). Extract the 45 training videos **on the instance**, not locally. Score on the same held-out 15 videos; record whether it still fits the edge budget.

## Done

- [x] 2026-08-13 — [M1] [test] Verify the evaluation math. 50 tests over `src/eval/` covering IoU, greedy confidence-ordered matching, one-claim-per-target, class-awareness, AP endpoints and the CLI contract. All 8 deliberate mutations (threshold `>=`→`>`, ascending sort, repeated claims, dropped class check, IoU union, AP envelope, YOLO half-extent, corner/centre swap) were caught.
