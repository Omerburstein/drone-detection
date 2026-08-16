# `src.glad_detect` — reference

Runs GLAD's released detection pipeline over ARD-MAV video and writes a
`detections.jsonl` that `src.evaluate` scores like any other run.

```
py -3.13 -m src.glad_detect [options]
```

> **What this measures is our harness, not GLAD.** GLAD's weights were trained on
> ARD-MAV's other 45 videos and its architecture was tuned against this very split, so
> the result is optimistic by construction. Never report it as "GLAD scores X for us".
> Its one job is to check M1's evaluation math and M2's VOC→YOLO conversion against a
> published number: a large gap means *we* have a bug. See [glad-model.md](glad-model.md)
> for the model itself.

## Parameters

| Parameter | Default | What it does |
| --- | --- | --- |
| `--videos` | `data/raw/ARD-MAV/videos` | Directory of source `.mp4` files. |
| `--labels` | `data/processed/ARD-MAV/labels/test` | Label directory for the split. Frames with no label file are **processed but not recorded** — see "Which frames are scored" below. |
| `--images` | `data/processed/ARD-MAV/images/test` | Directory the JSONL rows are keyed by. Nothing is read from it; it is what lets `src.evaluate` resolve labels exactly as for a stills run. |
| `--video-names` | official test 15 | Videos to run, without the `.mp4`. |
| `--out` | `runs/glad` | Output directory. Give every experiment its own. |
| `--max-frames-per-video` | none | Stop each video after N frames. A **contiguous prefix**, so the motion branches still work — for smoke tests, not for results. |
| `--glad-repo` | `third_party/GLAD` | Clone of the GLAD release. Its `weights/` must hold `yolov5s_GLAD.pt`, `yolov5s_GLAD-crop.pt` and `Net_best.pth`. |

**There is no `--stride`, `--conf`, `--imgsz` or `--tile`, deliberately.** Every threshold
is fixed at the value in the released source, because the point is to reproduce it. And
striding is not merely discouraged but incoherent here: both motion branches difference
the current frame against the previous one, so a strided sample measures a different
algorithm.

## Scoring a run

Every row is keyed by an image path, so no run-specific flags are needed:

```
py -3.13 -m src.evaluate \
    --pred runs/exp004_glad/detections.jsonl \
    --labels data/processed/ARD-MAV/labels/test \
    --conditions data/processed/ARD-MAV/conditions.json \
    --frame-size 1920 1080 \
    --json-out runs/exp004_glad/metrics.json
```

`--frame-size` is optional — all 15 test videos are 1920×1080, and passing it skips
28,337 image-header reads. `--conditions` is not optional for this run: GLAD publishes
per scene category and not in aggregate, so the per-category rows are the only
like-for-like comparison available.

**Read precision, recall and F1. Ignore the AP.** GLAD emits no confidence — its output
rows are `[frame, x, y, w, h]`, and its branches are not on a common scale — so every box
is recorded at 1.0. A ranking metric over a constant score is degenerate. This is a
single-operating-point pipeline and it can only be scored honestly as one.

## Which frames are scored

Frames whose label file is missing were never annotated, so a detection there would count
as a false positive and understate precision. They are **processed** — dropping them
would break frame-to-frame differencing — but not recorded. On the test 15 this never
fires: every decodable frame is annotated.

The first frame of every video is recorded with no detection. Upstream skips it outright
(there is no previous frame to difference against), but it carries ground truth, so
omitting it would quietly inflate recall by 15 frames.

## The `branch` field

Each JSONL row carries the branch that produced it, which makes the paper's ablation
table visible in our own run — how much of the recall is appearance and how much is
motion. The summary is printed at the end of a run.

| Value | Meaning |
| --- | --- |
| `first frame` | No previous frame; no detection possible. |
| `global yolo` | GAD found it in the full frame. |
| `global mod` | GAD missed, GMD proposed a candidate, LAD confirmed it. |
| `local yolo` | LAD found it inside the 320×320 search region. |
| `local mod` | LAD missed, LMD found it in the same region. |
| `global miss` | Nothing found while unlocked. |
| `local miss` | Nothing found while locked. 30 in a row fall back to global. |

## What this port changes, and what it does not

The released entry point cannot run on this machine: it deserialises TensorRT 7.2 engines
onto a hardcoded *second* CUDA device. The PyTorch checkpoints ship in the same folder and
load on CPU, so only the runtime had to be replaced.

**Replaced:** the three TensorRT detector classes (`src/algo/glad/yolo.py`, yolov5 v6.0 on
CPU with tensorrtx-compatible pre- and post-processing); the classifier's per-candidate
checkpoint reload (`src/algo/glad/classifier.py`); the `imshow` display loop
(`src/algo/glad/pipeline.py`).

**Unchanged:** `MOD2.py`'s two motion modules, imported and called verbatim; every
threshold, region size, selection rule and state transition.

Three upstream defects are reproduced rather than fixed, because fixing any of them would
change what is being measured:

1. **The letterbox pads black, not grey.** Upstream writes
   `copyMakeBorder(img, t, b, l, r, BORDER_CONSTANT, (128, 128, 128))`, but that
   function's seventh positional parameter is `dst`, not `value` — the tuple is discarded
   and the border defaults to 0. At 640×640 a 1080p frame is 44% padding, and yolov5
   trained these weights against 114 grey, so the global detector runs on a train/test
   mismatch over nearly half its input. The local detectors take square 320×320 crops and
   are unaffected.
2. **The tracked position is not re-based.** After a local hit the search region recentres
   on the new box, but the stored relative position still refers to the *old* region, so
   the next frame's anchor is stale by one frame of target motion. Small against radii of
   50 and 200 px, but real. Upstream re-bases correctly in both global branches and not in
   either local one, which is what marks it as an oversight.
3. **`ratio_theta = std(theta) / mean(theta)`** in both motion modules, with `theta` in
   degrees over (−180, 180]. The mean passes through zero and the wrap at ±180° gives a
   leftward-moving target a huge spurious standard deviation, so **leftward motion is
   penalised** by a rejection test. Untouched here; see
   [glad-model.md §6](glad-model.md).

Two differences from the engines are unavoidable: fp32 on CPU against whatever precision
the engines were built at, and tensorrtx's 1000-box output cap, which a single-drone frame
never approaches. Neither is expected to move a detection.

## Cost

Measured on the i7-1255U, CPU only: **~4.8 fps**, so the full 28,337-frame test split is
roughly 1.6 hours. Throughput is content-dependent, not fixed — the expensive motion path
only runs when appearance detection fails, so an easy sequence runs faster than a hard
one. The same asymmetry is why the paper's 146.5 FPS on an RTX 3070 sits so far above its
own GMD-only figure of 41.3.
