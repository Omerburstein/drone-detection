# `src.glad_detect` — reference

Runs GLAD's released detection pipeline over ARD-MAV or ARD100 video and writes a
`detections.jsonl` that `src.evaluate` scores like any other run.

```
py -3.13 -m src.glad_detect [--dataset ARD-MAV|ARD100] [options]
```

> **On ARD-MAV, what this measures is our harness, not GLAD.** GLAD's weights were trained on
> ARD-MAV's other 45 videos and its architecture was tuned against this very split, so
> the result is optimistic by construction. Never report it as "GLAD scores X for us".
> Its one job is to check M1's evaluation math and M2's VOC→YOLO conversion against a
> published number: a large gap means *we* have a bug. See [glad-model.md](glad-model.md)
> for the model itself.
>
> **On ARD100 it measures GLAD**, on 15 videos it never trained on — bounded by the fact
> that they come from the same lab and likely the same campaign. See "Running it on
> ARD100 (M4b)" below for what may and may not be concluded from it.

## Parameters

| Parameter | Default | What it does |
| --- | --- | --- |
| `--dataset` | `ARD-MAV` | Which prepared dataset to run over: `ARD-MAV` (M4a, the harness check) or `ARD100` (M4b, unseen video). Sets the defaults for the four parameters below. |
| `--split` | `test` | Split whose labels are scored. Selects `labels/<split>` and `images/<split>`. |
| `--videos` | the dataset's `videos/` | Directory of source `.mp4` files. |
| `--labels` | the dataset's `labels/<split>` | Label directory for the split. Frames with no label file are **processed but not recorded** — see "Which frames are scored" below. |
| `--images` | the dataset's `images/<split>` | Directory the JSONL rows are keyed by. Nothing is read from it and **it need not exist** — a labels-only tree (`prepare_ardmav --no-images`) runs fine. It is what lets `src.evaluate` resolve labels exactly as for a stills run. |
| `--video-names` | the dataset's test 15 | Videos to run, without the `.mp4`. |
| `--crop` | none | `X,Y,W,H` — detect on this rectangle of each frame instead of the whole one, for sources that are not all picture. Boxes are recorded in **cropped** coordinates. See "Sources that are not all picture" below. |
| `--motion-profile` | none | Tuning for the motion branches: `upstream` (the port at upstream's constants) or `clutter` (keeps ranked candidates instead of discarding all of them). Omitting it runs the **vendored** `MOD2` itself. See "When the motion branches switch themselves off" below. |
| `--hud-mask` | none | PNG from [`src.data.hud_mask`](hud_mask.md). A box lying mostly on a burned-in overlay is neither emitted nor locked onto. Goggles recordings only. |
| `--osd-twins` | off | Also reject a box with an identical copy 0.5–2.5 OSD character columns (`width / 30`) to either side on the same frame. This handles the **moving** HUD a static `--hud-mask` cannot hold, such as the analog artificial horizon. Applied at every emit and lock point, like the mask. Goggles recordings only. See [hud_mask.md](hud_mask.md#the-hud-that-moves). |
| `--record-all` | off | Record **every** processed frame, not only ones with a label file. For unlabelled footage — our own field capture. See "Running it on footage nobody has labelled" below. |
| `--out` | `runs/glad` | Output directory. Give every experiment its own. |
| `--max-frames-per-video` | none | Stop each video after N frames. A **contiguous prefix**, so the motion branches still work — for smoke tests, not for results. Counts **decoded** frames, so it covers the same span of video under every `--sample` mode. |
| `--sample` | `every` | Duty-cycle policy: `every` (what EXP-004 ran), `nth`, or `burst`. See "Duty cycling" below. **Not a stride.** |
| `--sample-n` | `2` | For `--sample nth`: process every Nth frame. `2` is 15 fps from a 30 fps source. |
| `--burst-length` | `2` | For `--sample burst`: frames per burst. The first frame of a burst follows a reset and can never detect, so a burst of K gives **K−1** chances. Must be ≥ 2. |
| `--burst-period` | `60` | For `--sample burst`: frames between burst starts. 60 is every 2 s at 30 fps. Sets detection **latency**, not per-burst probability — see below. |
| `--glad-repo` | `third_party/GLAD` | Clone of the GLAD release. Its `weights/` must hold `yolov5s_GLAD.pt`, `yolov5s_GLAD-crop.pt` and `Net_best.pth`. |
| `--pad` | `trained` | Letterbox fill for the global detector. `trained` (114) is the value yolov5 v6.0 trained these weights against. `released` (black) reproduces upstream including its padding bug. `tensorrtx` (128) is what upstream intended. See "The letterbox fill" below. |
| `--scale` | `1.0` | Resize each frame by this factor before the detector, mapping boxes back so they are recorded in **original** pixels. `auto` picks the factor that puts the frame at 1920×1080's diagonal. `1.0` is a no-op. See "Footage that is not 1080p" below. |
| `--invert` | off | Feed the detector `255 - pixel`. A **diagnostic**, not a fix — it tests GLAD's appearance prior. Geometry is untouched. See "Testing the appearance prior" below. |

**There is no `--stride`, `--conf`, `--imgsz` or `--tile`, deliberately.** Every threshold
is fixed at the value in the released source, because the point is to reproduce it. And
striding is not merely discouraged but incoherent here: both motion branches difference
the current frame against the previous one, so a strided sample measures a different
algorithm.

There are two exceptions, and neither is a tuning knob:

- **`--pad`**, because the released value is a defect rather than a choice.
- **`--scale`**, because the released thresholds are absolute pixels at a resolution we do
  not shoot at. It does not change a threshold; it puts the frame where the thresholds
  already are. See below.

## Footage that is not 1080p

Every constant in the motion branch is absolute pixels tuned at 1920×1080 — blob area
30–3000 and the 50-blob cap in `MOD2_global`, `dist_ref = 200` and the 30-blob cap in
`MOD2_local`, the Gaussian kernel 11, `REGION_HALF = 160`, `MAX_DISTANCE` 50 and 10. None
is a ratio. On a smaller frame every one of them is being applied to a target a fraction of
the size it was calibrated for, and the blob-area gate worst of all, since area falls with
the **square** of the linear scale.

Our FIELD capture is 1032×752: diagonal 1277 against 2203, so **0.579× linear and 0.335× in
area**. EXP-010 ran it natively and therefore measured our failure to rescale alongside the
detector. `--scale auto` is the correction:

```
py -3.13 -m src.glad_detect --videos data/raw/FIELD/videos     --video-names captured_raw_20260616_040253_004 --record-all     --images data/processed/FIELD/images/test --pad released --scale auto     --out runs/field/exp011_field_glad_scaled
```

Four things about it:

- **Boxes are recorded in original coordinates.** Unlike `--crop`, which declares its
  coordinate change and requires the same flag at render time, `--scale` is invisible in the
  record: a scaled run's `detections.jsonl` is directly comparable to an unscaled one's, and
  `src.render_video` needs no matching flag.
- **Aspect ratio is preserved.** Upstream's own attempt is a commented-out
  `cv2.resize(frame, (1920, 1080))` in `GLAD.py`, which would distort anything not 16:9 —
  and this capture is 1.372:1. MOD2's optical-flow coherence tests threshold on the *spread*
  of flow angles and distances, exactly what anisotropic scaling perturbs. Matching the
  reference *diagonal* keeps the scaling isotropic.
- **It costs pixels.** `auto` on the FIELD capture is 1.725× linear, so **2.98× the pixels**
  through MOD2 and the ego-motion homography. Measured 2.15 fps against 3.05 native. The
  YOLO branches letterbox to 640 regardless and the local branch works on fixed 320×320
  crops, so only the global path pays.
- **A run with `--scale` other than 1.0 is not directly comparable to EXP-004–010.** Say the
  factor in any comparison, the same way `--pad` and the match threshold have to be stated.

It composes with `--crop`: the crop is applied at decode, the scale to the cropped frame.
Scaling **down** is allowed and warns — it destroys the small targets this project exists to
detect.

The better fix is to make the constants themselves scale-relative, which
[glad-model.md](glad-model.md) ranks as improvement #4. That is not available cheaply: they
are function-local literals inside the vendored, gitignored `third_party/GLAD/MOD2.py`, with
no parameter and nothing `import_motion` can rebind. Scaling the frame buys the same
relationship between target and threshold without touching the vendored tree.

## Testing the appearance prior

`--invert` exists for one experiment and should not be used for anything else.

GLAD's weights were fitted on ARD-MAV: **white DJI Phantoms shot against ground** — roads,
concrete, grass, buildings. Measured on the test split, **84.4% of its targets are brighter
than their background**. Our own field capture is the opposite on both axes at once — a dark
airframe silhouetted against bright sky, **98.5% darker than background** — while the ground
clutter it false-alarms on is **99.1% brighter**, which is to say the false alarms sit in the
training distribution more comfortably than the real target does.
[glad-model.md §5b](glad-model.md) has the measurement.

Inverting swaps both polarities at once: the target moves onto the training side of the
distribution and the pale clutter moves off it. If the prior is what drives the behaviour,
cold acquisition (`global yolo`) should rise and the clutter locks should not form.

```
py -3.13 -m src.glad_detect --videos data/raw/FIELD/videos     --video-names captured_raw_20260616_040253_004 --record-all     --images data/processed/FIELD/images/test --pad released --invert     --out runs/field/exp012_field_glad_inverted
```

- **Geometry is untouched.** Inversion is pointwise, so boxes are recorded exactly as they
  would be otherwise and `src.render_video` needs no matching flag — unlike `--crop`, and
  unlike `--scale`, which has to map back.
- **It composes with `--crop` and `--scale`.** Applied after the crop; pointwise, so it
  commutes with the letterbox and with `--scale`'s interpolation.
- **A run made this way is not comparable to any un-inverted run**, including EXP-004–011.
  It is a probe of the model, not a measurement of the detector.
- **It is not a deployment fix.** Even if it works it would only say the prior is real. The
  actual remedy for an appearance mismatch is fine-tuning on our own footage, which needs
  labels.

## Duty cycling — running on fewer frames

A camera delivers 30 fps and GLAD sustains 2.67 on this host, so processing fewer frames
is the obvious lever. *Which* fewer is not a free choice.

**`--sample` is not `--stride`, and the difference is the whole point.** Striding hands the
motion module frame *n* and frame *n+10*, a third of a second apart; the differencing is
then between two effectively unrelated views and the run measures a different algorithm.
Both `--sample` modes keep a differenceable predecessor:

| Mode | What it does | What the motion branch sees |
| --- | --- | --- |
| `nth` | Runs the whole pipeline at 1/N of the source rate | Adjacent *processed* frames, still adjacent to each other — just 2/30 s apart at N=2 instead of 1/30 s |
| `burst` | Processes K consecutive frames, then sleeps until the next period | Frames **inside** a burst, a true 1/30 s apart. The sleep costs opportunity, not coherence |

Each fails differently, and knowing which failure to look for is most of the analysis:

- **`nth` doubles apparent motion.** `TrackingDetector.MAX_DISTANCE` is 50 px from the last
  box centre, and the motion module's constants (`dist_ref=200`, blob area 30–3000) are
  absolute pixels tuned for 30 fps at 1080p. Lowering the rate pushes against all of them
  at once. Expect this as the **local regime losing lock** — a gate, not a gradient — while
  acquisition, which is per-frame, is indifferent.
- **`burst` never enters the local regime at all.** A lock does not survive the sleep, so
  every burst is spent in the global regime: GAD, then GMD, then LAD confirmation. Burst
  mode therefore measures **acquisition probability**, not tracking recall, and it runs the
  expensive branch every time it runs at all.

**The pipeline is reset at the head of every burst**, and this is load-bearing rather than
tidy. Without it, `_prev` holds a frame from the previous burst — seconds of ego-motion
ago — and `MOD2_global` differences two unrelated scenes into a field of blobs. Those blobs
are an artefact of the harness, not a property of the scheme. The guard is the branch
summary: a burst run must show **zero `local yolo`** and roughly `1/K` `first frame`.
Anything else means the reset is misfiring.

**Every frame is still decoded.** That cost is real on a live camera and is not what a duty
cycle saves; the run prints processed-against-decoded so the two are never conflated.

### Choosing the burst period

The period sets detection **latency**. It does not change the per-burst detection
probability, because each burst is an independent acquisition attempt on an arbitrary
frame. So **measure dense and deploy sparse**: a 900-frame period (30 s) yields ~63 bursts
across the test split, too thin to measure anything, while a 60-frame period yields ~470
attempts at the identical quantity. Say which was measured in any number that is reported,
or it reads as a result about a 2-second period.

The latency consequence has to be stated next to the recall, because it is what decides
the scheme. At the edge budget's illustrative 40 m/s closing speed, a 30-second period is
**up to 1,200 m of approach unnoticed**. A scheme that keeps its recall for 0.2% of the
compute is still unusable if the target crosses the whole engagement envelope between
bursts.

### Scoring a duty-cycled run

`src.eval.labels` iterates the *prediction* rows, not the label directory, so a run that
recorded only the frames it processed is scored on exactly those frames — the skipped ones
are absent, not counted as misses. That makes the raw number meaningful but **not
comparable to a full-rate run**, which covers a different frame population.

Use `src.evaluate --keys-from` for the control. It restricts a dense run to the sparse
run's frame set, so both sides carry identical weights, thresholds, resolution and match
criterion, and the only variable is the duty cycle:

```
py -3.13 -m src.glad_detect --sample nth --sample-n 2 --out runs/exp006_half_rate

py -3.13 -m src.evaluate --pred runs/exp006_half_rate/detections.jsonl \
    --labels data/processed/ARD-MAV/labels/test --frame-size 1920 1080 \
    --json-out runs/exp006_half_rate/metrics.json

# The control: EXP-004 over the same frames. No inference — the JSONL is persisted.
py -3.13 -m src.evaluate --pred runs/exp004_glad/detections.jsonl \
    --keys-from runs/exp006_half_rate/detections.jsonl \
    --labels data/processed/ARD-MAV/labels/test --frame-size 1920 1080 \
    --json-out runs/exp006_half_rate/control_metrics.json
```

A burst run's rows additionally carry `burst` and `in_burst`, which `src.evaluate --dump`
passes through, so "second frame of each burst only" is a filter on a CSV rather than a
second run.

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

## Running it on ARD100 (M4b)

`--dataset ARD100` points the same pipeline at 15 videos GLAD has never seen — ARD100's
own test split intersected with the videos absent from our local ARD-MAV 60. Prepare the
labels once, then run:

```
py -3.13 -m src.data.prepare_ardmav --dataset ARD100 --split test --no-images

py -3.13 -m src.glad_detect --dataset ARD100 --pad released --out runs/exp005_glad_ard100

py -3.13 -m src.evaluate \
    --pred runs/exp005_glad_ard100/detections.jsonl \
    --labels data/processed/ARD100/labels/test \
    --conditions data/processed/ARD100/conditions.json \
    --frame-size 1920 1080 \
    --dump runs/exp005_glad_ard100/matches.csv \
    --json-out runs/exp005_glad_ard100/metrics.json
```

`--no-images` is deliberate: nothing on this path opens an extracted frame, and the JPEGs
would cost 30 GB. See [prepare_ardmav.md](prepare_ardmav.md) § "Labels only".

**Match EXP-004 on every setting or the comparison is not a comparison.** Same `--pad`,
same IoU threshold, same full-rate contiguous decode, same 1920×1080. The one variable is
the video content — that is the entire experiment, and this dataset was chosen over more
independent ones (FL-Drones) precisely because it changes nothing else.

Two differences are unavoidable and must be stated with any number that comes out:

- **No per-category rows.** ARD100 publishes no `ordinary` / `complex` / `small_mav`
  grouping, so `--conditions` gives `lighting` and `relative_range` only. Since EXP-004's
  headline is per category, the like-for-like cut is aggregate P/R/F1 plus the size
  breakdown — and `relative_range` buckets are scaled per split, so compare `gt_size` in
  pixels from the `--dump`, not range labels.
- **Same lab, likely the same campaign.** The result is optimistic as a generalisation
  measure. The honest sentence is "GLAD retains X on unseen video from the same
  campaign", not "GLAD generalises".

**This has been run — EXP-005, 2026-08-23**, full entry in
[experiments.md](experiments.md). Recall **0.895 → 0.688** at `centre@1x`, precision
0.993 → 0.946, false alarms 188 → 1,316. The backlit control the section above demands was
applied and **does not explain the gap**: dropping every backlit frame recovers 2.5 points
of the 20.7, leaving 18.2 attributable to the video content. Neither does size composition
— the medium (32-96 px) control bucket falls 0.956 → 0.658. Localisation is untouched
(mean IoU 0.6805 against 0.6829), so the deficit is acquisition, not regression.

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

## Sources that are not all picture

A DJI goggles screen recording is **2520x1080 with a 1440x1080 picture in the middle**;
the rest is pillarbox with HUD glyphs drawn on it. Handing that to the detector whole is
not merely wasteful — the global detector letterboxes the frame's **longest side** to 640,
so the bars are paid for in target resolution:

| | Scale to 640 | A 20 px drone arrives as |
| --- | --- | --- |
| Whole 2520x1080 frame | 0.254 | **5 px** |
| Cropped 1440x1080 | 0.444 | **9 px** |

There are two ways to deal with it, and **the first is the project's normal path.**

### Materialise it into `data/processed/` (what SOFA-O4 does)

`data/raw/` is immutable and transforms write to `data/processed/`, so the crop is a
derived video with a manifest recording how it was made:

```
data/raw/SOFA-O4/videos/first_catch.mp4        2520x1080, untouched
data/processed/SOFA-O4/videos/first_catch.avi  1440x1080, FFV1, bit-exact
data/processed/SOFA-O4/MANIFEST.md             the crop, the codec, the evidence
```

Runs then point at `data/processed/SOFA-O4/videos` and need no `--crop` at all. The
transform is re-derivable, auditable, and paid for once rather than on every pass.

**Encode it losslessly.** The targets are 10–30 px and a lossy generation between
`data/raw/` and every number taken from it smears exactly what is being measured. Measured
on the same cropped frames: `mp4v` 37.5 dB, `avc1` 37.8 dB, `MJPG` 41.2 dB, **`FFV1`
lossless** at 5.7 GB for 7,386 frames. MP4 cannot carry FFV1, so these are `.avi` —
`src.data.sources.resolve_video` finds a stem in any known container and still prefers
`.mp4`, so ARD-MAV and ARD100 resolve exactly as they did for EXP-004.

### Or crop at decode with `--crop`

```
--crop 540,0,1440,1080
```

No intermediate file, nothing re-encoded. Use it for a quick look, or when the disk cost
of a derived copy is not worth paying. What it gives up is the audit trail: the run's
geometry lives in a flag in the shell history rather than in a manifest beside the data.

- **Boxes are recorded in cropped coordinates**, because that is the frame the detector
  saw. Pass the **same** `--crop` to `src.render_video`, or every box lands offset by the
  crop origin — plausible-looking and all wrong.
- **A crop that runs off the source is refused before the decode starts.**

### What neither fixes: the HUD inside the picture

The ladder marks, arrows and the bottom telemetry strip are burned into the video region
and the detector sees them. They are mostly static, so the motion branches difference them
away, but the appearance branch has no such protection — check where detections land
before trusting a count.

## When the motion branches switch themselves off

`MOD2` discards **every** candidate when too many survive its blob filter:

```python
if len(rect_merge) > 50:      # 30 in the local variant
    print('too much bboxes')
    return []
```

| Run | Frames | `too much bboxes` |
| --- | ---: | ---: |
| EXP-010 FIELD | 3,600 | 23 |
| EXP-011 SOFA-O4 | 7,386 | **3,595** |

Low-altitude flight over close, textured ground through a wide lens leaves a large residual
after a single-homography compensation, so the difference image fills with blobs. Measured
on `first_catch` frames 901-907: **105-170 candidates against a cap of 50** — every frame
discarded everything.

That matters more than it sounds. GLAD's published ablation puts recall at 0.51 without the
motion branches and 0.81 with them, and **acquisition runs through `MOD2_global`** — the
full-frame appearance detector fired 15 times in 7,386 frames. Half our run was a pipeline
with its acquisition path switched off.

### The profiles

| Profile | What it is |
| --- | --- |
| *(omitted)* | The **vendored** `MOD2`, unmodified. The default, and what EXP-004, EXP-005 and EXP-010 ran. |
| `upstream` | `src.algo.glad.motion` at upstream's constants. Equivalent to the above — that is exactly what `tests/integration/test_motion_equivalence.py` pins. |
| `clutter` | Keeps ranked candidates when crowded, and raises the blob-area ceiling from 3,000 px² to 12,000. |

**A run using `clutter` is a variant of GLAD and must not be reported as GLAD.**

### Why ranking rather than simply raising the cap

The guard exists for a real reason: the per-candidate loop runs corner detection, optical
flow and a CNN gate, so hundreds of candidates are genuinely unaffordable. Ranking keeps
that budget — score every candidate by how target-like its *shape* is (how much of its
bounding box the contour fills, times how close to square it is), keep the best 50, and
spend exactly what upstream was willing to spend. The score decides **order only**; every
kept candidate still faces the same optical-flow and CNN tests.

### Why the area ceiling moves

`MOD2` accepts blobs of 30-3,000 px², about 55x55. A target at the moment of a catch is
larger than that and becomes **invisible to the motion branches at any threshold** — the
confirmed drone in EXP-011 is 80x34 = 2,720 px², already within a few percent of the old
ceiling. This is the same structural handicap [todo.md](todo.md) raises for FL-Drones.

### What it does not fix

`Functions.motion_compensate` discards any grid point moving **more than 50 px between
frames** and then fits a single homography. At our altitude and speed that throws away the
near-field points that dominate the flow, and a homography cannot describe close 3D terrain
through a fisheye anyway. Fewer blobs would be better than better handling of many. That
constant lives in the vendored `Functions` and is not ported yet.

## Running it on footage nobody has labelled

`--record-all` records every processed frame regardless of whether a label file exists.
It is for **our own capture** — `data/raw/FIELD/` — where there is no ground truth at all
and the default gate above would write an empty JSONL.

```
py -3.13 -m src.glad_detect     --videos data/raw/FIELD/videos     --video-names captured_raw_20260616_040253_004     --record-all --pad released     --images data/processed/FIELD/images/test     --out runs/field/exp010_field_glad
```

Three things to hold onto when reading such a run:

- **It is not a score.** No labels means no precision, recall or AP — `src.evaluate`
  has nothing to match against. What the run gives you is how many boxes fired, where,
  and via which branch. Never put those counts in a table beside EXP-004–009.
- **Key it where the labels will land.** `--images` sets the JSONL row key and nothing
  reads it, so pointing it at `data/processed/FIELD/images/test` costs nothing now and
  means that if the footage is ever annotated, `src.evaluate` scores **this** JSONL
  with no second inference run.
- **Never pass it on a labelled split.** It would add unannotated frames to the scored
  set and understate precision — exactly the failure the default gate exists to prevent.

Resolution is the other caveat, and it is not specific to this flag: GLAD's motion
constants are absolute pixels tuned for 1920×1080, so any source of another size is
measuring our failure to rescale alongside the detector. `data/raw/FIELD/PROVENANCE.md`
carries the arithmetic for the 1032×752 capture (0.579× linear).

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

Measured on the i7-1255U, CPU only. Throughput is content-dependent, not fixed — the expensive motion path
only runs when appearance detection fails, so an easy sequence runs faster than a hard
one. The same asymmetry is why the paper's 146.5 FPS on an RTX 3070 sits so far above its
own GMD-only figure of 41.3.

| Run | Frames | fps | Wall-clock |
| --- | --- | --- | --- |
| EXP-004, ARD-MAV test 15 | 28,337 | 2.67 | 2.9 h |
| EXP-005, ARD100 test 15 | 34,287 | 3.60 | 2.65 h |

**A faster run is not good news.** EXP-005 is 35% faster per frame than EXP-004 on the same
machine and the same code because it *detects less*: `global miss` rises 2.9% → 12.3%, and a
pipeline that has lost lock and found nothing skips the expensive confirmation path. Read
fps alongside the branch summary, never on its own. A short smoke run over the first frames
of each video is a poor predictor for the same reason — it measured 4.81 fps against the
full run's 3.60.
