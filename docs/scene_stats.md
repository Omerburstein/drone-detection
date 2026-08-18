# `src.data.scene_stats` — reference

Measures **lighting** and **relative range** per frame and writes them into
`conditions.json` as extra condition axes, so `src.evaluate --conditions` can break a run
down by them the same way it already does by scene category.

```
py -3.13 -m src.data.scene_stats --processed data/processed/ARD-MAV --split test
```

`src.data.prepare_ardmav` calls the same code inline, from the frame it has already
decoded, so a fresh prepare run produces the axes automatically. This CLI exists to
**retrofit an already-extracted tree** without re-decoding the source videos.

## Parameters

| Parameter | Default | What it does |
| --- | --- | --- |
| `--processed` | required | Processed dataset root, e.g. `data/processed/ARD-MAV`. |
| `--split` | `test` | Which split under `images/<split>/` to measure. |

Runs at roughly 45 frames/s on this machine — about 10 minutes for the 28,337-frame
ARD-MAV test split. It reads images and rewrites only `conditions.json`.

## What it measures

### `lighting` — target-versus-background separation

The mean luminance inside each ground-truth box, minus the mean of the annulus around it
(out to **3× the box**), absolute value. Not global brightness: a drone at 132 mean
luminance against clutter at 130 is invisible however well exposed the frame is, and the
frame-wide histogram cannot see that.

| Bucket | Separation |
| --- | --- |
| `invisible (<5)` | under 5 grey levels |
| `low (5-15)` | 5–15 |
| `moderate (15-30)` | 15–30 |
| `strong (>=30)` | 30 and up |
| `backlit` | **override** — more than 2% of frame pixels above 250 |
| `no_target` | frame has no measurable ground-truth box |

`backlit` overrides the contrast bucket because blown highlights are a distinct optical
regime rather than a point on the same scale — shooting toward the sun. The 2% threshold
is well clear of the 0.0–0.2% that cleanly exposed ARD-MAV videos sit at, while catching
`phantom86` (12.2% blown) and `phantom47` (5.5%).

The ring is local on purpose. A drone against bright sky and the same drone against a
roofline are different detection problems at identical global exposure, and only a local
comparison distinguishes them.

### `relative_range` — apparent size, expressed as range

Every ARD-MAV target is the same airframe (a DJI Phantom, inferred from the video
naming), so real-world size is constant and the pinhole relation

```
d = f_px × W_real / w_px
```

collapses to **`d ∝ 1/size`**, where size is `sqrt(w × h)` of the ground-truth box.

| Bucket | Range |
| --- | --- |
| `near (<2x)` | under 2× the closest approach |
| `mid (2-3x)` | 2–3× |
| `far (3-5x)` | 3–5× |
| `very far (>5x)` | beyond 5× |
| `no_target` | no measurable box |

> **This is relative range only, never metres.** The raw ARD-MAV download ships no
> camera intrinsics, so `f_px` is unknown; assuming a 60°/90°/120° horizontal field of
> view moves the implied median range from 39 m to 22 m to 13 m. Buckets are therefore
> stated as multiples of the split's own closest approach — the 95th percentile of
> apparent size, recorded in the file as `reference_size_px`. A percentile rather than
> the maximum, so one oversized mislabelled box cannot rescale the whole axis.
>
> For absolute range you need a dataset that labels telemetry-derived distance. LRDDv2/v3
> are the only ones that do; see [datasets.md](datasets.md).

Two further limits worth knowing before leaning on this:

- **Quantisation dominates at long range.** Labels are integer pixels, so at 6 px a
  one-pixel error is ±17% range; at 40 px it is ±2.5%. Precision is worst exactly where
  range matters most.
- **`W_real` is itself ambiguous by 1.7×** — a Phantom is 350 mm motor-to-motor but
  ~590 mm across prop tips, and whether the box includes spinning props varies with blur.
  This is a constant scale error, so it cancels in *relative* range and would not in
  absolute range.

## Multi-target frames

A frame is labelled by its **worst** target for lighting and its **nearest** (largest)
target for range: a frame is only as easy as its hardest target and only as near as its
nearest one. On ARD-MAV this is nearly moot — 28,160 boxes across 28,337 frames, so
essentially one target per frame — but the rule matters for datasets where it is not.

## Output

Merged into `conditions.json`, alongside the existing keys rather than replacing them:

```json
{
  "scene_category": { "phantom05": "complex", ... },
  "categories": { ... },
  "axes": {
    "scene_category":  { "level": "video", "labels": { "phantom05": "complex", ... } },
    "lighting":        { "level": "frame", "order": [...],
                         "labels": { "phantom05_0001": "moderate (15-30)", ... } },
    "relative_range":  { "level": "frame", "order": [...], "reference_size_px": 31.0,
                         "labels": { "phantom05_0001": "mid (2-3x)", ... } }
  },
  "lighting_summary": {
    "phantom05": { "frames": 1799, "brightness": 128.3, "pct_blown": 0.19,
                   "contrast_p50": 21.2, "frac_below_5": 0.094 }
  }
}
```

`scene_category` is **video**-level; the two derived axes are **frame**-level, because
both genuinely vary inside a sequence — one video spans a 5.7× range change, another
blows its highlights only partway through. Averaging either to a per-video label would
discard the variation that makes it worth measuring.

The legacy top-level `scene_category` map is left in place, so a reader written against
the original file shape keeps working.

## Reading the result

Pass the file to `src.evaluate --conditions` and you get one table per axis. Do **not**
sum across axes — every axis covers all the frames, so the counts partition *within* an
axis and double-count across them. That is why the report splits the table rather than
pooling it.

Findings from the first pass over ARD-MAV test are recorded in
[experiments.md](experiments.md) under "`small_mav` is two handicaps, not one": lighting
and target size are near-independent (r = 0.071) and compound, and poor lighting costs
about three times more recall at long range than at short range.
