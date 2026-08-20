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
# Run from the repo root -- that is what puts `src` on the import path.
py -3.13 -m src.baseline_detect --weights <path> --source <path> [--tile] [--stride N]

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

The `src` package splits along the same data/model boundary the two agents own,
so dataset work and model work do not collide:

```
src/data/     frames.py     FrameSource -> Frame; video decode, striding, budgets
              sources.py    classify --source as video or images
src/algo/     config.py     InferenceConfig
              detector.py   load_model, detect_frame, detect_tiled
              tiling.py     tile_origins, crop_grid, merge_boxes
              detections.py Detections
src/output/   recording.py  RunRecorder (JSONL + counters)
              annotate.py   AnnotationSink -> VideoSink / ImageDirSink / NullSink
              video.py      LazyVideoWriter: an mp4 sized by its first frame
              overlay.py    ground truth + prediction on one frame, coloured by outcome
src/eval/     labels.py     EvalFrame; ground truth paired with recorded preds
              metrics.py    matching, AP, the Metrics record
              conditions.py Axis; grouping frames by capture conditions
              report.py     the printed metric block
              results.py    the append-only results log (settings + metrics)
              records.py    the per-object dump: one row per tp / fp / fn
              curves.py     binning that dump into precision/recall vs size
src/baseline_detect.py      CLI: inference — parser, run loop, wiring
src/evaluate.py             CLI: scoring a recorded run against labels
src/plot_eval.py            CLI: precision-against-size figure from a dump
src/render_video.py         CLI: a scored run drawn back onto its source video
```

**`src/data/` is source code, not a dataset.** The gitignore rules for `data/`,
`weights/` and `runs/` are anchored with a leading slash for exactly this reason —
an unanchored `data/` matches at any depth and silently drops the package. Do not
remove those slashes.

Load-bearing points:

- `Detections` — frozen dataclass of `(boxes, scores, classes)`, always in **absolute
  original-frame pixels**. Tiled results are mapped back before they leave `algo`, so
  tiled and whole-frame output are directly comparable. Pass this, not loose tuples.
- `InferenceConfig` — frozen dataclass built via `from_args()` but constructible
  directly, so inference is callable from a notebook or test without a command line.
  Nothing under `algo/` touches the filesystem.
- `FrameSource` — video and stills differ in striding, progress cadence, and JSONL key;
  each source owns those decisions and yields a fully-described `Frame`. That is what
  lets **one** run loop serve both. Put per-source behaviour here, not in the loop.
- `RunRecorder` — context manager owning JSONL output and the run's counters. Every
  frame funnels through it, so the record schema and statistics are defined in exactly
  one place. Extend it rather than writing output elsewhere.
- `AnnotationSink` — owns drawing as well as writing, so `--no-save-frames` skips the
  draw work entirely instead of rendering frames nobody sees.
- `Detections.to_records` / `.from_records` — the `detections.jsonl` row schema, defined
  once. `RunRecorder` writes through the first, `src.eval.labels` reads through the
  second. Change the row format here, not at either end.
- `Metrics` — frozen dataclass whose **field order is the `--json-out` schema** the
  ledger cites. Append fields; never reorder them.
- `records.write_dump` — the per-object CSV every later cut is taken from. Every
  prediction and every target appears **exactly once**, so counting rows reproduces the
  metric block rather than approximating it; `tests/unit/test_records.py` pins that.
  Add a column here rather than re-deriving one downstream.
- `MatchCriterion` — the one place a match is decided. `records`, `curves` and the
  `overlay` renderer call `match_frame`, they do not reimplement it, so neither a dump's
  `outcome` nor the colour of a box on a rendered video can disagree with the metric
  block it explains.

`src/evaluate.py` is the second CLI: it reads a run's JSONL and scores it against
labels. It shares the package but not the inference path — `src/eval/` imports only
numpy, PIL and `algo.detections`, so evaluation stays runnable without torch or a
checkpoint present. Keep it that way.

Note `src.data.frames.Frame` (a frame going *into* the detector) and
`src.eval.labels.EvalFrame` (one coming back out with labels attached) are different
types on purpose.

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
- **A published P/R is meaningless without its IoU threshold.** Measured in EXP-004:
  moving the match threshold from 0.50 to 0.40 moves `small_mav` precision by 23 points
  and recall by 19 — larger than any architectural difference this project has measured.
  Tiny targets are why: at 12 px a 2 px centre offset drops IoU below 0.5 while the
  detection is plainly correct. Before concluding "we are worse than the paper", re-score
  the same JSONL across thresholds; if the gap closes and scales with target size, it is
  the protocol, not the detector. State the threshold in any comparison or mark it
  uncertain.
- **Empty-frame rate is a recall proxy, not accuracy.** A model boxing random sky
  scores 0%. Read it alongside detections-per-frame.
- **Drone-vs-Bird requires a signed DUA and is non-commercial.** YOLOMG is GPL-3.0;
  GLAD and DUT are MIT. Flag licensing when the use case might be commercial.

## Definition of done — every task, not just M-numbered ones

Standing instruction from the user, restated 2026-08-20. A task is not finished when
the code works; it is finished when these four are true. Do all four **without being
asked**, in the *same commit* as the work.

1. **Tests** — cover changed behaviour under `src/` and run them:
   `py -3.13 -m pytest -m "not slow"`. Unit tests for functions, integration tests for
   the wiring. If tests genuinely do not apply — a docs-only or config-only change —
   say so in one line rather than skipping silently.
2. **Docs** — if a script's parameters changed, update its reference in `docs/` in the
   same commit. The user relies on these instead of re-reading the source, so a stale
   doc is a broken one.
3. **The record** — move the finished task to `## Done` in `docs/todo.md` with the
   date. If the work produced numbers, add the run to `docs/experiments.md`. **An
   unrecorded mission gets re-asked and re-run**: M2 landed on 2026-08-16 and had to be
   chased because the todo still showed it open.
4. **Commit and push to `origin main`** — standing approval, never ask for it. `gh` CLI
   is not installed; plain HTTPS push works.

Only items 1–2 are ever optional, and only when the change genuinely has no behaviour
and no parameters. Items 3 and 4 are not.

**This is enforced, not just documented.** `.claude/hooks/definition_of_done.py` runs on
`Stop` and refuses to end a turn while a file *this session edited* is uncommitted or
unpushed. It tracks the session's own edits rather than the whole tree because several
sessions run against this repo at once — so if it fires, it is about your work. It
checks only what a machine can check; items 1–3 ride in its message and are on you.

When it does fire, **stage only the paths your task touched** (`git add <paths>`, never
`git add -A`) — another session very likely has unrelated work in the tree.

## Conventions

- `/data/`, `/weights/`, and `/runs/` are gitignored — **never commit datasets or
  checkpoints**. `.claude/` and all docs *are* committed. The rules are root-anchored,
  so `src/data/` is tracked normally.
- `data/raw/` is immutable. Transforms read from it and write to `data/processed/`, so
  any step can be re-derived.
- Give every experiment its own `--out` directory; runs overwrite in place otherwise.
- **Say when a prompt could have been routed better.** Standing request from the user
  (2026-08-19): if a skill or agent should have caught the task, if two live sessions are
  covering the same ground, or if a long job should have been backgrounded and resumable,
  add one sentence at the end of the reply and move on. `docs/working-with-claude.md` is
  the collected version. Do not moralise, and do not repeat a point already made.

## Agents, skills and hooks

`.claude/agents/` — `dataset-agent` owns ingest, label validation, and splits;
`algo-agent` owns the model registry, the `docs/experiments.md` ledger, and next-step
recommendations. They split at the data/model boundary.

`.claude/skills/` — `/clean-up` (quality only, not a bug hunt), `/test-creation`,
`/eval`, `/todo` (captures a task in `docs/todo.md`; capture only, does not do the work).

`.claude/hooks/` — `definition_of_done.py`, wired in `.claude/settings.json` as two
hooks sharing one script: `record` (`PostToolUse` on `Write|Edit`, async) notes which
repo file the session touched, and `check` (`Stop`) blocks the turn while any of those
is uncommitted or unpushed. See the definition-of-done section above. It is deliberately
quiet — no state, no output — until it has something real to say, and it never blocks
twice in a row, so it cannot trap a session.
