# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Air-to-air detection of multirotor drones — camera mounted on a *flying* drone, targets
small (often 10–30 px) and fast, against sky and ground clutter. This framing drives
nearly every technical decision below.

## Commands

Always use **Python 3.13** via `py -3.13`. Python 3.14 is also installed on this
machine but runs ahead of the torch/ultralytics wheel matrix.

```bash
py -3.13 -m pip install -r requirements.txt

# Baseline inference (see docs/baseline_detect.md for every parameter)
py -3.13 scripts/baseline_detect.py --weights <path> --source <path> [--tile] [--stride N]

# Tests (pytest is not yet installed: py -3.13 -m pip install pytest)
py -3.13 -m pytest
py -3.13 -m pytest -m "not slow"              # skip real-model integration tests
py -3.13 -m pytest tests/unit/test_x.py::test_y -v   # single test

# Lint / dead code
py -3.13 -m ruff check --select F401,F841,ARG,ERA .
```

## Hardware reality — check before proposing anything

This machine is **CPU-only**: i7-1255U, 16 GB RAM, Intel Iris Xe, no CUDA
(`torch` is the `+cpu` build, and `torch.xpu` is unavailable too).

- **Inference runs locally.** Training does **not** — fine-tuning on ARD100's 202k
  frames here would take weeks. Route every training step to a rented GPU
  (Kaggle 2×T4 free tier, RunPod/Vast ~$0.25/hr). See `docs/hardware.md`.
- Never propose a local training run. Always include GPU cost and wall-clock in a
  training proposal.
- Pull large datasets **directly onto the GPU instance**. Det-Fly alone is ~50 GB at
  4K; staging locally then re-uploading wastes hours.

## Architecture

Currently one script plus a documentation and agent layer. `scripts/baseline_detect.py`
is structured as:

- `InferenceConfig` — frozen dataclass holding detector settings, built via
  `from_args()` but constructible directly, so the pipeline is callable without a
  command line.
- `detect_frame()` → dispatches to whole-frame or `detect_tiled()`. Both return the
  same `(xyxy, conf, cls)` tuple in **absolute original-frame pixels**, so the two
  modes produce directly comparable output.
- `RunRecorder` — context manager owning JSONL output and the run's counters. Both the
  video and image paths funnel through it, so the record schema and statistics are
  defined in exactly one place. Extend it rather than writing output elsewhere.

### Tiled inference — why it exists

A 640 px letterbox of a 4K frame shrinks a 20 px drone to ~3 px, destroying it before
the detector runs. `--tile` runs the detector over overlapping native-resolution crops
and merges with class-aware NMS. **Always run both ways when benchmarking**: the gap
between whole-frame and tiled separates "the model is weak on small objects" from "the
resize threw the target away." Tiling costs roughly (tile count)× the time — measured
11 fps vs 0.72 fps at 720p — so pair it with `--stride`/`--max-frames`.

## Project-specific traps

These have already cost real effort. Do not rediscover them.

- **Split by sequence/video, never by frame.** These datasets are video; adjacent
  frames are near-identical, so a random frame-level split leaks near-duplicates into
  validation and inflates mAP by tens of points. Every downstream number becomes
  fiction.
- **VisDrone is not a drone-detection dataset.** It is cars and pedestrians shot *from*
  a drone. Search engines conflate it constantly. Ignore it and any
  "VisDrone-pretrained" checkpoint.
- **COCO weights have no `drone` class.** Nearest are `airplane` (4) and `bird` (14),
  and neither fires reliably on a quadcopter. Use single-class drone weights.
- **Runs at different `imgsz`, thresholds, or split rules are not comparable.** Small
  targets make mAP unusually sensitive to input resolution — a resolution change alone
  can move mAP@0.5 by tens of points and masquerade as an architectural win. Say a
  comparison is invalid rather than producing a table with a caveat under it.
- **Empty-frame rate is a recall proxy, not accuracy.** A model boxing random sky
  scores 0%. Read it alongside detections-per-frame.
- **Drone-vs-Bird requires a signed DUA and is non-commercial.** YOLOMG is GPL-3.0;
  GLAD and DUT are MIT. Flag licensing when the use case might be commercial.

## Conventions

- `data/`, `weights/`, and `runs/` are gitignored — **never commit datasets or
  checkpoints**. `.claude/` and all docs *are* committed.
- `data/raw/` is immutable. Transforms read from it and write to `data/processed/`, so
  any step can be re-derived.
- Give every experiment its own `--out` directory; runs overwrite in place otherwise.
- **Commit and push to `origin main` after each unit of work** — this is standing
  instruction from the user, no need to ask. `gh` CLI is not installed; plain HTTPS
  push works.
- Docs live in `docs/`. When a script's parameters change, update its reference doc in
  the same commit — the user relies on these instead of re-reading the source.

## Agents and skills

`.claude/agents/` — `dataset-agent` owns ingest, label validation, and splits;
`algo-agent` owns the model registry, the `docs/experiments.md` ledger, and next-step
recommendations. They split at the data/model boundary.

`.claude/skills/` — `/clean-up` (quality only, not a bug hunt), `/test-creation`,
`/eval`.
