# `src.data.seed_track` — reference

Turns one hand-placed box into a labelled segment, so footage nobody annotated can be
scored. Built for our own clips — `data/raw/FIELD/` and `data/raw/SOFA-O4/` — which have
no ground truth and therefore no precision, recall or AP.

```
py -3.13 -m src.data.seed_track --video <mp4|avi> --seed FRAME X Y W H
    --labels-out <dir> --verified-out <jsonl> [--review-out <dir>]
```

> **A tracker, not a detector.** The detector is the thing under test; labelling its test
> set with it would measure nothing. Normalised cross-correlation knows only what the seed
> box looked like.

---

## Parameters

| Parameter | Default | What it does |
| --- | --- | --- |
| `--video` | required | Source clip. Use the **processed** one the run was made on, so the boxes share its coordinate system. |
| `--seed FRAME X Y W H` | required | A box placed by eye on one 1-based frame. **Repeatable** — give one per appearance of the target. |
| `--negatives START-END` | none | Frame ranges confirmed by eye to hold **no** target. Repeatable. |
| `--labels-out` | required | Label directory, e.g. `data/processed/SOFA-O4/labels/test`. |
| `--verified-out` | required | JSONL of every adjudicated frame, for `src.evaluate --keys-from`. Merged across runs. |
| `--review-out` | none | Directory for the proposal contact sheet. Optional, but review it before trusting a number. |
| `--images-dir` | labels' sibling `images/<split>` | Path the verified rows are keyed by. Nothing is read from it. |
| `--forward` / `--backward` | 400 / 400 | Frames to track after / before each seed. |
| `--min-score` | **0.45** | Correlation below which a segment ends. |
| `--search` | 4.0 | Search window, in target sizes. |
| `--update` | **0.0** | Template blend rate. See the warning below before raising it. |

---

## Two outputs, and the split is the point

- **`labels/test/<stem>_<frame>.txt`** — YOLO boxes for frames holding a target.
- **`verified.jsonl`** — every frame a human has adjudicated, target present *or* absent.

A missing label file means "empty frame" to `src.eval.labels`, which is correct for a
prepared dataset where every frame was annotated, and **badly wrong here**: a frame nobody
looked at would be scored as a confirmed negative and every detection in it counted a false
alarm. `verified.jsonl` is the fix — pass it to `src.evaluate --keys-from` and only
adjudicated frames are scored.

```
py -3.13 -m src.evaluate --pred runs/sofa_o4/exp011_sofa_o4_glad/detections.jsonl
    --labels data/processed/SOFA-O4/labels/test
    --keys-from data/processed/SOFA-O4/verified.jsonl
    --frame-size 1440 1080
```

**`--negatives` is what makes precision real.** Without it only frames containing a drone
are scored, and a detector that fires constantly looks perfect.

---

## Two things measured on real footage

### Template blending is a trap — it is off by default

Blending the template toward each new match sounds like the obvious way to follow a target
whose appearance changes. On `first_catch` it did this: the track slipped off the drone
onto terrain, adopted the terrain as its template, and then matched it at **0.99**. The
highest-confidence proposals on the review sheet were the worst ones — the exact inversion
of what the sheet is for.

Pinned to the seed appearance (`--update 0`, the default), a lost track dies within a few
frames and the score says so. Raise it only for a target whose appearance changes slowly.

### Expect tens of frames per seed, not hundreds

An intercept target grows and rotates fast. On `first_catch`, one seed at frame 962 covered
**10 frames** at `--min-score 0.45`. Dropping to 0.35 bought 118 frames — and the review
sheet showed them sitting on trees and rock.

**Place another seed rather than lowering the threshold.** The threshold is not a
sensitivity dial; below ~0.45 it buys frames by admitting terrain.

---

## The review pass is not optional

`--review-out` writes `<stem>_proposals.png`: every proposed box as a 132 px crop,
**ordered by score ascending**, so the frames where the tracker let go are in the first
row. Read it before scoring anything. Then re-run with tighter bounds, more seeds, or a
higher `--min-score`.

---

## Performance

Tracking reads **sequentially**; only entering a segment seeks. Measured on
`first_catch.avi`: a seek costs **1,109 ms** against **26 ms** for the next sequential
frame — 40×. A per-frame-seek implementation took minutes where this takes seconds.

The backward half buffers the frames it revisits, in greyscale — about 1.5 MB per frame at
1440×1080, so `--backward 200` costs roughly 300 MB.

> Exact random access works here because the processed clips are **FFV1, which is
> all-intra**: every frame is a keyframe. On a long-GOP source a seek lands on the nearest
> keyframe and every label in the segment is off by frames.
