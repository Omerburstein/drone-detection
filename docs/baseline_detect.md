# `src.baseline_detect` — reference

Runs a pretrained detector over video or images and records what it finds. This is
step 1 of the plan: measure how far off-the-shelf weights get *before* training
anything, so later work has a number to beat.

```
py -3.13 -m src.baseline_detect --weights <path> --source <path> [options]
```

Run it from the repository root, which is what puts `src` on the import path.
The implementation is split across `src/data` (frame sources), `src/algo`
(inference and tiling) and `src/output` (JSONL and annotated media); this page
documents the CLI, which is the stable surface.

---

## Parameters

### `--weights` (required)

Path to a `.pt` checkpoint, or a bare ultralytics model name (e.g. `yolov8n.pt`, which
auto-downloads).

**Use single-class drone weights.** COCO-trained models like `yolov8n.pt` have no
`drone` class — the nearest are `airplane` (4) and `bird` (14), and neither fires
reliably on a quadcopter. The script prints the class list on load and warns if it
sees more than 10 classes. See [research-notes.md](research-notes.md) for the
shortlist of drone-specific weights.

### `--source` (required)

One of:
- a video file (`.mp4 .avi .mov .mkv .m4v .wmv`)
- a single image
- a directory — searched **recursively** for images

### `--out` (default `runs/baseline`)

Output directory. Created if missing. Gitignored.

> Each run **overwrites** the previous output in the same directory. Pass a distinct
> `--out` per experiment, e.g. `runs/exp003_tiled_1280`, or you will lose the
> comparison you were trying to make.

### `--imgsz` (default `640`)

The square size every frame is letterboxed to before inference. **The single most
important knob in this project.**

A 1080p frame at `--imgsz 640` is scaled by 0.59, so a 20 px drone becomes 12 px. At
4K the same drone becomes 3 px — gone. Raising this preserves small targets at
quadratic cost in compute.

| Value | Effect |
| --- | --- |
| 640 | Fast, loses small targets. Fine for a first look. |
| 1280 | Usual sweet spot for air-to-air. ~4× the compute of 640. |
| 1920+ | Diminishing returns unless the source is 4K. Slow on CPU. |

Must be a multiple of 32. Ultralytics silently rounds up if not.

### `--conf` (default `0.25`)

Minimum confidence to keep a detection.

For a *baseline* run, drop this to **0.10–0.15**. The question you are asking is "does
the model see the drone at all?" — a target found at 0.18 confidence is a very
different situation from one not found at all, and the default threshold hides that
distinction. Raise it later, once you are tuning for precision.

### `--iou` (default `0.5`)

IoU threshold for non-max suppression: two boxes overlapping more than this are
merged, keeping the higher-confidence one. Also used to merge detections **across
tiles** when `--tile` is on.

Rarely needs changing. Lower it if you see duplicate boxes stacked on one drone; raise
it if genuinely distinct nearby drones are being merged into one.

### `--classes` (default: all)

Space-separated class ids to keep, e.g. `--classes 4 14`. Filters everything else out.

Not needed for single-class drone weights. Useful when experimenting with COCO weights
(`--classes 4 14` for airplane + bird).

### `--stride` (default `1`)

Process every Nth frame. `--stride 5` on 30 fps video samples 6 frames/sec.

This is your main CPU-budget lever. Consecutive video frames are near-identical, so
striding costs you very little information for a large speedup. Use `5`–`10` freely
for baseline measurement.

> Note the asymmetry: for **video**, `--stride` samples frames and `--max-frames` caps
> the total. For **image directories**, striding is applied only when `--max-frames`
> is also set. Directories are usually already sparse samples, not dense video.

### `--max-frames` (default: unlimited)

Stop after this many processed frames. Combine with `--stride` for a quick
representative sample: `--stride 10 --max-frames 200` covers 2000 frames of source.

### `--tile` (flag, default off)

Run the detector over **overlapping crops at native resolution** and merge the results
with class-aware NMS, instead of letterboxing the whole frame.

This exists because a whole-frame run conflates two very different failures: the
detector being weak on small objects, and the target being destroyed by the resize
before the detector ever saw it. **Run both ways.** The gap between them tells you
which problem you actually have — and if the gap is large, the fix is resolution, not
a bigger model.

Costs roughly (number of tiles) × the single-pass time. Measured on this machine, 720p
with `yolov8n`: **11 fps whole-frame vs 0.72 fps tiled.** Always pair `--tile` with
`--stride` and `--max-frames`.

### `--tile-size` (default `640`)

Edge length in pixels of each crop. Should match the `--imgsz` the weights were trained
at — crops are fed at native resolution, so a 640 crop into a model expecting 640 means
no rescaling at all, which is the entire point.

### `--tile-overlap` (default `0.2`)

Fractional overlap between adjacent tiles. At 0.2 with `--tile-size 640`, tiles step
512 px.

Overlap exists so a drone straddling a tile boundary lands whole inside at least one
crop. Raise toward 0.3 if targets are being clipped at seams; lower toward 0.1 to cut
tile count and run faster. Tile count — and therefore runtime — rises sharply as this
approaches 0.5.

### `--tile-batch` (default `8`)

How many crops are fed to the model per call. Caps peak memory: a 4K frame generates
~32 tiles, and batching all of them at once is wasteful on a 16 GB machine.

Lower to 4 if you hit memory pressure; raising it above 8 gains little on CPU.

### `--no-save-frames` (flag, default off)

Write only `detections.jsonl`, skip annotated video/images. Meaningfully faster and
much smaller on disk. Use it once you trust the pipeline and only want numbers.

---

## Outputs

### `detections.jsonl`

One JSON object per processed frame. Video runs key on `frame`, image runs on `image`:

```json
{"frame": 40, "detections": [{"bbox": [1201.4, 388.0, 1229.8, 410.2], "conf": 0.4127, "cls": 0}]}
{"image": "data/Det-Fly/images/00312.jpg", "detections": []}
```

- `bbox` — `[x1, y1, x2, y2]` in **absolute pixels of the original frame**, top-left
  origin. Tiled runs are already mapped back to full-frame coordinates, so both modes
  produce directly comparable output.
- `conf` — confidence, 0–1.
- `cls` — integer class id; resolve against the class map printed at load.
- An empty `detections` list means nothing was found in that frame. These entries are
  written, not skipped — the misses are the point of a baseline.

### `annotated.mp4` / `annotated/`

Green boxes with class and confidence. Video runs write a single `annotated.mp4` at
`source_fps / stride`, so it plays back at real-world speed. Image runs write one file
per input into `annotated/`.

---

## Reading the results

The script prints three things at the end:

```
412 frames in 96.3s (4.28 fps)
389 detections, 0.94 per frame
Frames with no detection: 118/412 (28.6%)
```

**The empty-frame rate is the headline number.** On air-to-air datasets nearly every
frame contains exactly one drone, so it approximates the miss rate directly. Track it
across runs.

Two caveats before trusting it:

- It counts *any* detection as a hit, including a false positive on a cloud. It is a
  recall proxy, not accuracy — a model that boxes random sky scores 0% empty frames.
- `detections per frame` meaningfully above 1.0 on air-to-air data means false
  positives. Read it alongside the empty-frame rate; the two together tell you whether
  you have a recall problem or a precision problem, which need opposite fixes.

For real mAP against ground truth you need labels and an evaluation pass — that comes
after `dataset-agent` has produced a validated split.

---

## Recipes

```bash
# Quick look — is anything working at all?
py -3.13 -m src.baseline_detect --weights weights/drone.pt \
    --source data/ARD-MAV/video01.mp4 --stride 10 --max-frames 100 --conf 0.15

# Honest whole-frame baseline at a sane resolution
py -3.13 -m src.baseline_detect --weights weights/drone.pt \
    --source data/ARD-MAV/video01.mp4 --stride 5 --imgsz 1280 --conf 0.15 \
    --out runs/exp001_whole_1280

# Tiled comparison — same video, same threshold, only the strategy differs
py -3.13 -m src.baseline_detect --weights weights/drone.pt \
    --source data/ARD-MAV/video01.mp4 --stride 20 --max-frames 100 --conf 0.15 \
    --tile --tile-size 640 --out runs/exp002_tiled

# Numbers only, no annotated output
py -3.13 -m src.baseline_detect --weights weights/drone.pt \
    --source data/Det-Fly/images --conf 0.15 --no-save-frames \
    --out runs/exp003_detfly
```

Change **one variable at a time** between runs, and give each its own `--out`.
Comparing a tiled 1280 run against a whole-frame 640 run tells you nothing about
which change mattered.
