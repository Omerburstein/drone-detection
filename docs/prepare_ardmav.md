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
  conditions.json                    # scene category per video
  MANIFEST.md                        # provenance, counts, split rule, known issues
  _verify/                           # sampled frames with labels drawn on
```

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

**Unlabelled frames are not negatives.** Each video's XMLs run contiguously from 1 to N,
where N is at or below the frame count. The shortfall is always *trailing* frames:

| | Frames | Annotated | Trailing |
| --- | --- | --- | --- |
| Test 15 total | 28,644 | 28,337 | 307 |

Those frames were never annotated, so they are excluded rather than emitted with empty
labels. Treating them as confirmed-empty would count any detection there as a false
positive and understate precision.

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

Measured over the test split, **every target is under 32 px**; roughly 59% are under
16 px. There are no medium or large targets at all. This is the strongest available
confirmation that the project's difficulty is small-object detection specifically —
and it means any evaluation reported only as an aggregate mAP is really reporting
tiny-target performance under another name.
