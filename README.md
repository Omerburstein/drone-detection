# drone-detection

Air-to-air detection of multirotor drones — camera mounted on a flying drone, targets
small and fast against sky and clutter.

- **[docs/research-notes.md](docs/research-notes.md)** — dataset and model survey; the
  plan this repo is built around.
- **[docs/hardware.md](docs/hardware.md)** — what this machine can and can't do, and
  where to rent a GPU for the training step.
- **[docs/baseline_detect.md](docs/baseline_detect.md)** — full reference for every
  parameter of the baseline script, the output format, and how to read the metrics.
- **[docs/evaluate.md](docs/evaluate.md)** — evaluation parameters and a glossary of
  every metric reported.
- **[docs/experiments.md](docs/experiments.md)** — the experiment ledger. Every run,
  including the failed ones.
- **[docs/prepare_ardmav.md](docs/prepare_ardmav.md)** — dataset conversion: the official
  split, the output layout, and the two conventions that fail silently.
- **[docs/todo.md](docs/todo.md)** — the mission sequence, M1–M7.

## Setup

Use **Python 3.13** (`py -3.13`). 3.14 is installed on this machine but runs ahead of
the torch/ultralytics wheel matrix.

```
py -3.13 -m pip install -r requirements.txt
```

Most of it is already present globally at the required versions.

## Step 1 — baseline

Establish how far off-the-shelf weights get before training anything. Runs on CPU.

Fetch single-class drone weights (see [docs/research-notes.md](docs/research-notes.md)
for the shortlist) into `weights/`, then:

```
py -3.13 -m src.baseline_detect \
    --weights weights/<drone-weights>.pt \
    --source data/ARD-MAV/video01.mp4 \
    --stride 5 --conf 0.15
```

Outputs `runs/baseline/detections.jsonl` plus an annotated video. The headline number
is the **empty-frame rate** — on air-to-air datasets, where nearly every frame contains
exactly one drone, that approximates the miss rate.

### On tiling

**Tiling is the default.** The detector runs over overlapping native-resolution crops,
merged with class-aware NMS, and nothing is rescaled. This matters because an air-to-air
target is often 10–30 px; a 640 px letterbox of a 4K frame destroys it before the
detector sees it.

`--no-tile` letterboxes the whole frame to `--imgsz` instead. Run both ways — the gap
between them tells you how much of the baseline's failure is resolution loss versus
genuine small-object weakness.

Tiling is much slower. Measured here on 720p with `yolov8n`: **11 fps whole-frame vs
0.72 fps tiled**. Use `--stride` and `--max-frames` to sample rather than grind.

## Layout

```
src/        the package — run entry points with `py -3.13 -m src.<module>`
  data/       source classification, frame decoding, striding, frame budgets
  algo/       detector config, tiled inference, NMS merging, result type
  output/     JSONL record, run counters, annotated media
data/       datasets (gitignored — never commit)
weights/    model checkpoints (gitignored)
runs/       inference output (gitignored)
docs/       research notes, hardware constraints
.claude/    project skills and agents (committed)
```

Note `src/data/` (code) and `data/` (datasets) are different things. The gitignore
rules are root-anchored so the former stays tracked.

`src` splits along the same data/model boundary the two agents own, so
`dataset-agent` and `algo-agent` can work without colliding. Nothing in `algo/`
touches the filesystem — it takes arrays and returns `Detections`, so a model
experiment can drive it directly without going through the CLI.

## Claude Code tooling

Skills — invoke with `/<name>`, both take an optional file or folder to scope to:

| Skill | Does |
| --- | --- |
| `/clean-up` | Unused imports, duplication, oversized/multi-purpose functions and files, OOP structure, missing docstrings. Quality only — not a bug hunt. |
| `/test-creation` | Audits unit + integration coverage and writes the missing tests. |
| `/eval` | Scores a run against ground truth and interprets the result. |

Agents — delegated with the Agent tool:

| Agent | Owns |
| --- | --- |
| `dataset-agent` | Ingest, format conversion, label validation, splits, dataset stats. Enforces **sequence-level** splits. |
| `algo-agent` | Model registry, the `docs/experiments.md` ledger, and next-step recommendations. Enforces comparability between runs. |

The two agents split cleanly at the data/model boundary, so they can run against the
same project without stepping on each other.

## Step 2 — evaluate

```
py -3.13 -m src.evaluate --pred runs/exp001/detections.jsonl \
    --labels data/processed/ARD-MAV/labels/val --json-out runs/exp001/metrics.json
```

Reports AP@0.50, mAP@0.50:0.95, precision/recall, mean IoU, and **recall broken down by
target size** — the most diagnostic block in the report. See
[docs/evaluate.md](docs/evaluate.md) for the metric glossary.

Requires ground-truth labels, so it comes after `dataset-agent` has produced a
validated, sequence-level split.

## Tests

```
py -3.13 -m pytest              # full suite
py -3.13 -m pytest -m "not slow"
```

The eval math is covered by mutation-verified tests: every deliberate breakage of the
IoU, matching, and AP code is caught. See [docs/todo.md](docs/todo.md) M1.

## Status

- [x] Repo scaffold, baseline inference tooling
- [x] Evaluation harness (IoU matching, AP/mAP, size breakdown)
- [x] ARD-MAV downloaded — 60 videos, 100,423 VOC annotations
- [x] **M1** — eval math tested and mutation-verified
- [ ] **M2** — convert + validate the official 15-video test split
- [ ] **M3** — EXP-001 / EXP-002: off-the-shelf detector, whole-frame and tiled
- [ ] **M4** — EXP-003: GLAD released weights, no training
- [ ] **M5** — Pd / FAR / size-normalised localisation error
- [ ] **M6** — edge budget and board decision
- [ ] **M7** — fine-tune GLAD on a rented GPU
