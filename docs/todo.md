# Todo

Captured tasks. Add via `/todo <description>`. Check an item off by moving it to Done
with the date it was finished.

## Open

- [ ] 2026-08-13 — [algo] Evaluate an off-the-shelf single-class drone detector on our footage as-is, no fine-tuning, whole-frame and `--tile`
- [ ] 2026-08-13 — [algo] Extend `src.evaluate` with Pd and FAR plus a size-normalised localisation error (bbox-to-truth offset scaled by true target size), reported per size bin
- [ ] 2026-08-13 — [data] Tag each eval frame with background class (sky/urban/trees/mixed) and target motion (fast/slow), so `src.evaluate` can break metrics down by condition as it already does by target size

## Done
