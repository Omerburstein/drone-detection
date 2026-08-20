# `src.data.prepare_ardmav` — reference

Converts a raw ARD-MAV-shaped download into the canonical YOLO layout that
`src.baseline_detect`, `src.glad_detect` and `src.evaluate` consume.

```
py -3.13 -m src.data.prepare_ardmav [--dataset ARD-MAV|ARD100] [--split test] [--no-images]
```

Re-runnable. `data/raw/` is never modified, so the processed tree can be deleted and
rebuilt at any time.

## Parameters

| Parameter | Default | What it does |
| --- | --- | --- |
| `--dataset` | `ARD-MAV` | Which download to convert. Sets the defaults for `--raw`, `--out` and `--videos`. See "Two datasets, one converter" below. |
| `--raw` | the dataset's own | Source tree, containing `videos/` and `Annotations/`. |
| `--out` | the dataset's own | Destination for images, labels, and metadata. |
| `--split` | `test` | Split name. Written as `images/<split>/` and `labels/<split>/`. |
| `--videos` | the dataset's test split | Override the video list. Mostly for smoke-testing one sequence. |
| `--no-images` | off (images written) | Write labels and metadata but no JPEGs. See "Labels only" below. |
| `--verify-sample` | `20` | Frames rendered with boxes into `_verify/` for visual checking. |
| `--seed` | `42` | Seeds the verify sample, so re-running inspects the same frames. |

## Two datasets, one converter

`src/data/datasets.py` holds a `DatasetSpec` per download: raw and processed roots, the
test videos, the published scene grouping if there is one, and the provenance its
`MANIFEST.md` records. **The conversion code itself does not branch on the dataset.** That
is deliberate and it is what M4b rests on — the question there is how much GLAD loses on
video it has never seen, and if the extractor differed between the two runs, part of the
answer would be about the extractor.

| | ARD-MAV | ARD100 |
| --- | --- | --- |
| Raw | `data/raw/ARD-MAV` | `data/raw/ARD100` |
| Processed | `data/processed/ARD-MAV` | `data/processed/ARD100` |
| Test videos | GLAD's published 15 | 15 — ARD100's own test split ∩ "not in our local 60" |
| Resolution / rate | 1920×1080 @ 30 fps | identical |
| Annotation | VOC XML, one-based, class `Drone` | identical |
| `scene_category` axis | ✅ published by GLAD | ❌ none published |
| License | MIT | CC-BY-4.0 |

Two consequences for ARD100 specifically, both recorded in its generated `MANIFEST.md`:

- **No `scene_category`.** GLAD's `ordinary` / `complex` / `small_mav` grouping is from
  the paper and has no ARD100 equivalent. `conditions.json` therefore carries only the
  measured `lighting` and `relative_range` axes, and the EXP-004 comparison is available
  in aggregate and along those, not per published category. Inventing a grouping of our
  own would look like the same axis while comparing nothing.
- **`relative_range` is not comparable across datasets.** It is scaled to *each split's*
  own closest approach, so ARD100's "near" is not ARD-MAV's "near". Compare `gt_size` in
  pixels from a `--dump` CSV instead.

## Labels only

`--no-images` skips the JPEGs and the `data.yaml` that points at them. It exists because
the entire GLAD path needs no extracted pixels:

- `src.glad_detect` reads the source `.mp4` — it must, the motion branches difference
  consecutive frames — and opens no extracted frame. It uses `--labels` only to decide
  which frames were annotated, and `--images` only as the JSONL row key.
- `src.evaluate --frame-size 1920 1080` takes the frame size from the flag instead of an
  image header.

At ~900 KB a frame that is 25 MB against 30 GB on a 34k-frame split. A stills run
(`src.baseline_detect`) does need the images — re-run without the flag to add them, since
`data/raw/` is untouched either way. `_verify/` renders are still produced: they are
encoded from the decoded frame during extraction rather than read back from the tree.

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

With `--no-images` the same tree appears without `images/` and without `data.yaml`;
everything else, `_verify/` included, is identical.

`conditions.json` carries three axes on ARD-MAV and two on ARD100. `scene_category` is
GLAD's published per-video grouping and exists only where the release publishes one;
`lighting` and `relative_range` are **measured per frame** during extraction,
from the frame already decoded, so they cost no extra decode. See
[scene_stats.md](scene_stats.md) for what they mean and
`py -3.13 -m src.data.scene_stats` for regenerating them on a tree that already exists.

## The official split (ARD-MAV)

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

Measured over ARD-MAV's full test split — 28,160 boxes across 28,337 frames:

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
