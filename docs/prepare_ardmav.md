# `src.data.prepare_ardmav` — reference

Converts the raw ARD-MAV download into the canonical YOLO layout that
`src.baseline_detect` and `src.evaluate` consume.

```
py -3.13 -m src.data.prepare_ardmav [--split test]
```

Re-runnable. `data/raw/` is never modified, so the processed tree can be deleted and
rebuilt at any time.

## Parameters

| Parameter | Default | What it does |
| --- | --- | --- |
| `--raw` | `data/raw/ARD-MAV` | Source tree, containing `videos/` and `Annotations/`. |
| `--out` | `data/processed/ARD-MAV` | Destination for images, labels, and metadata. |
| `--split` | `test` | Split name. Written as `images/<split>/` and `labels/<split>/`. |
| `--videos` | official test 15 | Override the video list. Mostly for smoke-testing one sequence. |
| `--verify-sample` | `20` | Frames rendered with boxes into `_verify/` for visual checking. |
| `--seed` | `42` | Seeds the verify sample, so re-running inspects the same frames. |

## Output

```
data/processed/ARD-MAV/
  images/test/phantom05_0001.jpg     # 1920x1080, JPEG quality 95
  labels/test/phantom05_0001.txt     # YOLO: 0 <cx> <cy> <w> <h>, normalised
  data.yaml                          # ultralytics dataset spec
  conditions.json                    # condition axes: scene category, lighting, range
  MANIFEST.md                        # provenance, counts, split rule, known issues
  _verify/                           # sampled frames with labels drawn on
```

`conditions.json` carries three axes. `scene_category` is GLAD's published per-video
grouping; `lighting` and `relative_range` are **measured per frame** during extraction,
from the frame already decoded, so they cost no extra decode. See
[scene_stats.md](scene_stats.md) for what they mean and
`py -3.13 -m src.data.scene_stats` for regenerating them on a tree that already exists.

## The official split

The 15 test videos are **GLAD's published split, used verbatim** — 05, 08, 09, 10, 19,
30, 41, 43, 46, 47, 58, 63, 65, 70, 86. Inventing our own split would make our numbers
incomparable to the paper's, which is the main reason to run this dataset at all.

GLAD also reports results by scene category, recorded in `conditions.json`:

| Category | Videos |
| --- | --- |
| ordinary | 09, 10, 30, 47, 70 |
| complex | 05, 08, 58, 65, 86 |
| small_mav | 19, 41, 43, 46, 63 |

Splitting is **by whole video**, never by frame. Adjacent frames are near-identical, so
a frame-level split would leak near-duplicates into validation and inflate every metric.

## Two conventions that fail silently

**Frame numbering is one-based.** Annotations are `<video>_0001.xml`, four-digit padded.
Decoded frame *i* (zero-based) pairs with `i + 1`. An off-by-one still puts boxes near
the drone — consecutive frames barely differ — so no numeric check catches it. That is
what `_verify/` is for; look at the renders.

**Unlabelled frames are not negatives.** A frame with no XML was never annotated rather
than confirmed empty, so emitting it with an empty label would count any detection there
as a false positive and understate precision. Extraction skips it.

On the test 15 this guard never actually fires — and the reason is worth recording:

| | Header claims | XMLs | Actually decoded | Written |
| --- | --- | --- | --- | --- |
| Test 15 total | 28,644 | 28,337 | 28,337 | 28,337 |

`CAP_PROP_FRAME_COUNT` reads the container header and **overstates by 307**. Every frame
that actually decodes has an annotation. Reconcile against the decoder, never the header.

Distinct from that: **177 frames have an XML containing zero objects.** Those are genuine
negatives and are kept, with an empty label file.

## Validation battery

Printed at the end of every run; a failure exits non-zero.

- **Size mismatch** — decoded frame dimensions against the XML's own `<size>`. Boxes are
  normalised by the XML size, so a mismatch silently rescales every label.
- **Coordinates out of [0, 1]** after normalisation.
- **Degenerate boxes** — zero or negative width/height. Skipped, and reported.
- **Class names** — anything other than `Drone` is flagged; ARD-MAV should contain only
  that one.
- **Frames with no box** and **trailing frames skipped**, reported separately because
  they mean different things.
- **Target size histogram**, bucketed to match `src.eval.metrics.AREA_BUCKETS`.

### What the size histogram showed

Measured over the full test split — 28,160 boxes across 28,337 frames:

| Bucket | Count | Share |
| --- | --- | --- |
| tiny (<16 px) | 18,261 | 64.8% |
| small (16–32 px) | 7,931 | 28.2% |
| medium (32–96 px) | 1,968 | 7.0% |
| large (>96 px) | 0 | 0% |

**93% of targets are under 32 px and nothing exceeds 96 px.** This is the strongest
available confirmation that the project's difficulty is small-object detection
specifically, and it means an aggregate mAP on this split is largely reporting
tiny-target performance under another name — read the size breakdown, not the headline.

The 7% medium bucket is useful rather than noise: it is the control. If a model scores
well there and collapses on tiny, the limit is resolution, not architecture.
