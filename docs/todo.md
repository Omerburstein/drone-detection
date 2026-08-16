# Todo

Captured tasks. Add via `/todo <description>`. Check an item off by moving it to Done
with the date it was finished.

Mission ids (M1–M7) refer to the approved evaluation plan: baseline through edge-ready
model. They are ordered — each assumes the previous one landed. Two constraints run
through all of them: the **15 official test videos only** until fine-tuning needs more,
and **field deployment is a hard requirement**, so every experiment from M3 on records
inference cost alongside accuracy.

## Open

- [ ] 2026-08-13 — [M2a] [data] Wire `conditions.json` into `src.evaluate` so metrics break down by scene category the way they already do by target size — that breakdown is what the M4 comparison against GLAD's published per-category P/R/F1 needs. Scene category itself is **done** (M2 captured GLAD's official ordinary/complex/small_mav grouping, so no hand-labelling of backgrounds is needed). Still open: per-frame **target motion** (fast/slow), which nothing published provides and which would have to be derived from inter-frame box displacement.
- [ ] 2026-08-13 — [M3] [algo] Evaluate an off-the-shelf single-class drone detector on our footage as-is, no fine-tuning, whole-frame and `--tile` — EXP-001 / EXP-002. Identical weights, threshold and split; only tiling differs, so the gap isolates resolution loss from genuine small-object weakness. Expect ~AP 0.53 (YOLOv5's published ARD100 number); that is the figure GLAD has to beat.
- [ ] 2026-08-13 — [M4] [algo] EXP-003: run GLAD's released weights (MIT, no training) over the same 15 test videos, scored with our harness on our split. Also check against GLAD's published P/R/F1 on this set — 0.82/0.67/0.73 on small MAVs. Reproducing that row validates the whole pipeline; a large gap means we have a bug, not a discovery.
- [ ] 2026-08-13 — [M5] [algo] Extend `src.evaluate` with Pd and FAR plus a size-normalised localisation error (bbox-to-truth offset scaled by true target size), reported per size bin. Re-scoring is free — `detections.jsonl` is persisted — so amend EXP-001/002/003 rather than re-running inference.
- [ ] 2026-08-13 — [M6] [deploy] Write `docs/edge-budget.md`: target FPS, resolution, power and latency ceiling, and what "real-time" means for closing speed. Add FPS-on-target as a required ledger field, recommend a board (Jetson Orin Nano 8 GB, ~$250 — GLAD's published 23.6 FPS on the older Xavier NX is a floor), and plan the TensorRT export path. On-device benchmarking deferred until hardware exists.
- [ ] 2026-08-13 — [M7] [algo] Fine-tune GLAD from its released weights on ARD-MAV's official training split, on a rented GPU (Kaggle 2×T4 free, or RunPod ~$1–2 for 3–4 h). Extract the 45 training videos **on the instance**, not locally. Score on the same held-out 15 videos; record whether it still fits the edge budget.

## Done

- [x] 2026-08-13 — [M2] [data] Prepared the ARD-MAV test split: 28,337 frames / 28,160 boxes from the 15 official test videos, VOC XML → YOLO, with `data.yaml`, `conditions.json` and `MANIFEST.md`. Validation battery clean — no size mismatches, no out-of-range coordinates, no degenerate boxes, only `Drone` as a class name. Labels visually verified across three videos and two scene categories. **Two things I had wrong going in:** `CAP_PROP_FRAME_COUNT` overstates by 307 (header metadata, not decodable frames — every decodable frame *is* annotated), and a missing XML means *unannotated*, not *negative*; the 177 genuine negatives are frames whose XML contains zero objects, clustered in phantom47/08/09. Size distribution: 64.8% tiny, 28.2% small, 7.0% medium, **0% large**.
- [x] 2026-08-13 — [M1] [test] Verify the evaluation math. 50 tests over `src/eval/` covering IoU, greedy confidence-ordered matching, one-claim-per-target, class-awareness, AP endpoints and the CLI contract. All 8 deliberate mutations (threshold `>=`→`>`, ascending sort, repeated claims, dropped class check, IoU union, AP envelope, YOLO half-extent, corner/centre swap) were caught.
