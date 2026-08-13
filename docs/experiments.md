# Experiment ledger

Every run gets an entry — **including failed and abandoned ones.** Negative results are
what stop the project re-treading dead ends.

An entry missing weights, split rule, or the exact command is incomplete. Mark unknown
fields `UNKNOWN` explicitly rather than omitting them.

Maintained by `algo-agent`. Metrics come from `scripts/evaluate.py` (see
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

*No experiments recorded yet. The first will be the off-the-shelf baseline from step 1
of the plan — see [research-notes.md](research-notes.md).*

Two entries are expected for that baseline, differing only in the tiling switch, so the
gap between them isolates how much of the failure is resolution loss:

- `EXP-001` — whole-frame
- `EXP-002` — tiled, same weights, same threshold, same split
