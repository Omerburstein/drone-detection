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
| `--pad` | `trained` | Letterbox fill for the global detector. `trained` (114) is the value yolov5 v6.0 trained these weights against. `released` (black) reproduces upstream including its padding bug. `tensorrtx` (128) is what upstream intended. See "The letterbox fill" below. |

**There is no `--stride`, `--conf`, `--imgsz` or `--tile`, deliberately.** Every threshold
is fixed at the value in the released source, because the point is to reproduce it. And
striding is not merely discouraged but incoherent here: both motion branches difference
the current frame against the previous one, so a strided sample measures a different
algorithm.

`--pad` is the one exception, and it exists because the released value is a defect rather
than a choice.

## The letterbox fill

A 1920×1080 frame cannot enter a 640×640 network, so it is scaled to 640×360 and 140 blank
rows are added above and below. **That is 44% of the input**, and the colour of those rows
has to be one the network learned to ignore.

Upstream writes:

```python
cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, (128, 128, 128))
```

OpenCV's signature is `copyMakeBorder(src, top, bottom, left, right, borderType[, dst[,
value]])`. The seventh positional parameter is **`dst`**, not `value` — the tuple is
discarded, `value` falls back to its default of 0, and the bars come out **black**. There
is no error and no warning.

The right fill is not the 128 upstream intended either. These weights were trained by
yolov5 v6.0, which fills with **114** in `letterbox`, in the mosaic base image and in the
warp border alike. So `trained` is the correct value and is the default; `tensorrtx` is
provided only to separate "upstream's intent" from "what the weights actually saw".

Two further traps in the same call, both silent: passing the fill positionally (the
upstream bug) and passing it as a bare scalar, which OpenCV widens to `(v, 0, 0)` and uses
to tint the bars. `tests/unit/test_glad_geometry.py` pins both across all three styles.

> **A run compared against the paper must use `--pad released`.** The published numbers
> were produced by the released code, black bars included.

Only the global detector is affected. The local detectors take square 320×320 search-region
crops, which scale to 640×640 with no padding at all.

### What the fix is worth: nothing measurable

Measured directly, GAD alone over every 60th frame of the test split (473 frames,
468 targets, IoU 0.50):

| Fill | P | R | F1 | TP / FP |
| --- | --- | --- | --- | --- |
| `released` (0) | 0.717 | 0.152 | 0.250 | 71 / 28 |
| `trained` (114) | 0.726 | 0.147 | 0.245 | 69 / 26 |
| `tensorrtx` (128) | 0.719 | 0.147 | 0.245 | 69 / 27 |
| *paper, `GAD only`* | *0.76* | *0.17* | *0.28* | — |

Two detections separate the three, out of 468 targets. **The defect is real and the fix is
correct, but it buys nothing** — the reasonable-sounding argument that a 44% train/test
mismatch must cost recall does not survive contact with the measurement. Kept as the
default anyway, because it is the correct preprocessing and costs nothing to run, and
because M7 will fine-tune from these weights and should not inherit a defect.

The last row is the more valuable one: **GAD alone lands within 0.04 precision and 0.02
recall of the paper's own ablation figure**, which is independent evidence that the port,
M1's evaluation math and M2's conversion are all sound. Per-category recall is
0.42 ordinary / 0.03 complex / 0.01 small_mav — the same shape the full pipeline is
expected to show, and a reminder that GAD's job is acquisition, not detection.

## Scoring a run

Every row is keyed by an image path, so no run-specific flags are needed:

```
py -3.13 -m src.glad_detect --pad released --out runs/exp004_glad   # to match the paper

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
is recorded at 1.0. A ranking metric over a constant score is degenerate: on EXP-004 the
reported AP@0.50 of 0.705 is just P×R (0.856 × 0.771 = 0.660) plus interpolation slack,
carrying no information the other two columns do not. This is a single-operating-point
pipeline and it can only be scored honestly as one.

**Sweep `--iou` before concluding anything from a gap.** The paper does not state its
matching threshold, and on targets this small the choice dominates. From EXP-004, the same
`detections.jsonl` re-scored:

| Category | @0.50 | @0.40 | @0.30 | Published |
| --- | --- | --- | --- | --- |
| ordinary | .987/.965 | .995/.973 | .997/.975 | 0.99/0.96 |
| complex | .907/.828 | .975/.890 | .993/.907 | 0.94/0.86 |
| small_mav | .642/.522 | **.869/.707** | .955/.777 | **0.82/0.67** |

23 points of precision in `small_mav` sit between two defensible thresholds. Re-scoring is
free — the JSONL is persisted — so there is no reason not to look.

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

Three upstream defects were found while porting. The first is fixed and selectable via
`--pad`, because it is a plain mistake with a correct answer. The other two are reproduced
rather than fixed, because fixing them would change what is being measured and neither has
an obviously right replacement:

1. **The letterbox padded black, not grey** — fixed, see "The letterbox fill" above.
   `--pad released` restores the released behaviour for comparisons against the paper.
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
