"""Convert the raw ARD-MAV download into the canonical YOLO layout.

Reads `data/raw/ARD-MAV/{videos,Annotations}` and writes
`data/processed/ARD-MAV/{images,labels}/<split>/` plus `data.yaml`,
`conditions.json` and `MANIFEST.md`. Re-runnable: raw is never modified, so any
step can be re-derived by deleting the processed tree and running again.

Two conventions in the source data drive the whole design, and both fail
silently if assumed wrong:

**Frame numbering.** Annotations are named `<video>_0001.xml`, one-based and
four-digit. Decoded frame *i* (zero-based) therefore pairs with `i + 1`. An
off-by-one here still produces boxes that land near the drone -- adjacent
frames are near-identical -- so it cannot be caught numerically. That is what
`--verify` renders exist for.

**Unlabelled frames are not negatives.** Each video's XMLs run contiguously
from 1 to N. A frame with no XML was never annotated, so emitting it with an
empty label would count any detection there as a false positive and understate
precision; extraction skips it instead.

On the test 15 this guard never fires: every decodable frame has an
annotation. `CAP_PROP_FRAME_COUNT` claims 28,644 frames against 28,337 XMLs,
but that is container metadata and it overstates -- decoding yields exactly
28,337. Trust the decoder, not the header. The guard stays because the
training videos have not been checked and it costs nothing.

Example
-------
    py -3.13 -m src.data.prepare_ardmav --split test
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .scene_stats import FrameStats, measure_frame, update_conditions
from .voc import parse_voc, to_yolo

# The official split published with GLAD. Using it verbatim is what makes our
# numbers comparable to the paper's; a split of our own invention would not be.
OFFICIAL_TEST_VIDEOS = (
    "phantom05", "phantom08", "phantom09", "phantom10", "phantom19",
    "phantom30", "phantom41", "phantom43", "phantom46", "phantom47",
    "phantom58", "phantom63", "phantom65", "phantom70", "phantom86",
)

# GLAD reports results split three ways, so the same grouping is recorded here.
# Without it the M4 comparison against their published per-category P/R/F1
# cannot be made.
SCENE_CATEGORIES = {
    "ordinary": ("phantom09", "phantom10", "phantom30", "phantom47", "phantom70"),
    "complex": ("phantom05", "phantom08", "phantom58", "phantom65", "phantom86"),
    "small_mav": ("phantom19", "phantom41", "phantom43", "phantom46", "phantom63"),
}

CLASS_NAMES = ("drone",)
SOURCE_CLASS = "Drone"  # the single `<name>` used throughout ARD-MAV's XML
JPEG_QUALITY = 95
PROGRESS_EVERY = 250


@dataclass
class Stats:
    """Counters and anomalies gathered while converting."""

    frames: int = 0
    boxes: int = 0
    empty_labels: int = 0
    unannotated_skipped: int = 0
    areas: list[float] = field(default_factory=list)
    class_names: Counter = field(default_factory=Counter)
    size_mismatch: list[str] = field(default_factory=list)
    out_of_range: list[str] = field(default_factory=list)
    degenerate: list[str] = field(default_factory=list)
    # Per-frame lighting and apparent size, measured from the frame already
    # decoded here rather than by re-reading the JPEGs afterwards.
    scene: list[FrameStats] = field(default_factory=list)

    def merge(self, other: Stats) -> None:
        """Fold one video's counters into the run total."""
        self.frames += other.frames
        self.boxes += other.boxes
        self.empty_labels += other.empty_labels
        self.unannotated_skipped += other.unannotated_skipped
        self.areas += other.areas
        self.scene += other.scene
        self.class_names += other.class_names
        self.size_mismatch += other.size_mismatch
        self.out_of_range += other.out_of_range
        self.degenerate += other.degenerate


def annotation_path(raw: Path, video: str, frame_number: int) -> Path:
    """Path of the XML for a one-based frame number."""
    return raw / "Annotations" / video / f"{video}_{frame_number:04d}.xml"


def convert_video(raw: Path, video: str, images_dir: Path, labels_dir: Path) -> Stats:
    """Extract and convert one video's annotated frames.

    Decoding is sequential -- seeking per frame is far slower and, on some
    codecs, inexact.
    """
    stats = Stats()
    capture = cv2.VideoCapture(str(raw / "videos" / f"{video}.mp4"))
    if not capture.isOpened():
        sys.exit(f"Could not open {video}.mp4")

    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1  # XML numbering is one-based

            xml = annotation_path(raw, video, frame_index)
            if not xml.exists():
                # Unannotated, which is not the same as confirmed-empty, so the
                # frame is dropped rather than emitted with an empty label.
                stats.unannotated_skipped += 1
                continue

            stem = f"{video}_{frame_index:04d}"
            _write_frame(frame, xml, stem, images_dir, labels_dir, stats)

            if stats.frames % PROGRESS_EVERY == 0:
                print(f"    {video} {stats.frames} frames", flush=True)
    finally:
        capture.release()
    return stats


def _write_frame(frame, xml: Path, stem: str, images_dir: Path, labels_dir: Path,
                 stats: Stats) -> None:
    """Write one image and its YOLO label, recording any anomaly found."""
    annotation = parse_voc(xml)
    height, width = frame.shape[:2]
    if (width, height) != (annotation.width, annotation.height):
        # Normalising against the wrong size silently rescales every box.
        stats.size_mismatch.append(
            f"{stem}: frame {width}x{height} vs xml {annotation.width}x{annotation.height}")

    lines, boxes = [], []
    for obj in annotation.objects:
        stats.class_names[obj.name] += 1
        cx, cy, box_w, box_h = to_yolo(obj, annotation.width, annotation.height)

        if not all(0.0 <= v <= 1.0 for v in (cx, cy, box_w, box_h)):
            stats.out_of_range.append(f"{stem}: {cx:.4f} {cy:.4f} {box_w:.4f} {box_h:.4f}")
        if obj.width <= 0 or obj.height <= 0:
            stats.degenerate.append(f"{stem}: {obj.width}x{obj.height}")
            continue

        stats.areas.append(obj.width * obj.height)
        boxes.append([obj.xmin, obj.ymin, obj.xmax, obj.ymax])
        lines.append(f"0 {cx:.6f} {cy:.6f} {box_w:.6f} {box_h:.6f}")

    # Measured before the JPEG is written, so the numbers describe the decoded
    # frame rather than what survived compression.
    stats.scene.append(measure_frame(frame, np.array(boxes, dtype=float).reshape(-1, 4),
                                     stem))

    cv2.imwrite(str(images_dir / f"{stem}.jpg"), frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    # Explicit encoding on every write: Path.write_text defaults to the system
    # ANSI codepage on Windows, which mangles non-ASCII (it corrupted the em
    # dashes in a generated MANIFEST before this was pinned).
    (labels_dir / f"{stem}.txt").write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    stats.frames += 1
    stats.boxes += len(lines)
    stats.empty_labels += not lines


def render_verify(processed: Path, split: str, sample: int, seed: int) -> Path:
    """Draw labels back onto a random sample of converted frames.

    The only check that can catch a corner/centre mix-up or an off-by-one in
    frame numbering: both produce coordinates that pass every numeric test.
    """
    verify_dir = processed / "_verify"
    verify_dir.mkdir(parents=True, exist_ok=True)
    labels = sorted((processed / "labels" / split).glob("*.txt"))

    for label_path in random.Random(seed).sample(labels, min(sample, len(labels))):
        image = cv2.imread(str(processed / "images" / split / f"{label_path.stem}.jpg"))
        if image is None:
            continue
        height, width = image.shape[:2]
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            _, cx, cy, box_w, box_h = (float(v) for v in line.split())
            x1 = int((cx - box_w / 2) * width)
            y1 = int((cy - box_h / 2) * height)
            x2 = int((cx + box_w / 2) * width)
            y2 = int((cy + box_h / 2) * height)
            # Padded, so a tiny target is still visible in the rendered check.
            cv2.rectangle(image, (x1 - 12, y1 - 12), (x2 + 12, y2 + 12), (0, 255, 0), 2)
        cv2.imwrite(str(verify_dir / f"{label_path.stem}.jpg"), image)
    return verify_dir


def _area_histogram(areas: list[float]) -> list[tuple[str, int]]:
    """Counts per target-area bucket, matching `src.eval.metrics.AREA_BUCKETS`."""
    buckets = [("tiny   (<16px)", 0, 16 ** 2), ("small  (16-32)", 16 ** 2, 32 ** 2),
               ("medium (32-96)", 32 ** 2, 96 ** 2), ("large  (>96px)", 96 ** 2, float("inf"))]
    return [(label, sum(lo <= a < hi for a in areas)) for label, lo, hi in buckets]


def report(stats: Stats, videos: list[str]) -> bool:
    """Print the validation battery. Returns False if anything looks wrong."""
    print(f"\n{'=' * 58}")
    print(f"{len(videos)} videos | {stats.frames} frames | {stats.boxes} boxes")
    print(f"{'=' * 58}")
    print(f"  unannotated frames skipped {stats.unannotated_skipped}")
    print(f"  frames with no box        {stats.empty_labels}")
    print(f"  class names in source     {dict(stats.class_names)}")

    print("\n  target size distribution:")
    for label, count in _area_histogram(stats.areas):
        share = 100 * count / max(len(stats.areas), 1)
        print(f"    {label}  n={count:<7} {share:5.1f}%")

    ok = True
    for name, offenders in (("size mismatches", stats.size_mismatch),
                            ("coords out of [0,1]", stats.out_of_range),
                            ("degenerate boxes", stats.degenerate)):
        if offenders:
            ok = False
            print(f"\n  ! {name}: {len(offenders)}")
            for sample in offenders[:5]:
                print(f"      {sample}")
        else:
            print(f"  {name:<25} none")

    unexpected = set(stats.class_names) - {SOURCE_CLASS}
    if unexpected:
        ok = False
        print(f"\n  ! unexpected class names: {sorted(unexpected)}")
    return ok


def write_metadata(processed: Path, split: str, videos: list[str], stats: Stats) -> None:
    """Write data.yaml, conditions.json and MANIFEST.md."""
    (processed / "data.yaml").write_text(
        f"path: {processed.as_posix()}\n"
        f"{split}: images/{split}\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {list(CLASS_NAMES)}\n",
        encoding="utf-8",
    )

    category_of = {v: cat for cat, members in SCENE_CATEGORIES.items() for v in members}
    (processed / "conditions.json").write_text(json.dumps(
        {"scene_category": category_of,
         "categories": {k: list(v) for k, v in SCENE_CATEGORIES.items()}}, indent=2),
        encoding="utf-8")
    # Folds in the measured `lighting` and `relative_range` axes alongside the
    # published `scene_category`. Written second, from the stats gathered during
    # extraction, so no frame is decoded twice.
    update_conditions(processed, stats.scene)

    histogram = "\n".join(f"| {label.strip()} | {count} |"
                          for label, count in _area_histogram(stats.areas))
    (processed / "MANIFEST.md").write_text(f"""# ARD-MAV — processed

Generated by `src.data.prepare_ardmav`. Re-derive by deleting this tree and
re-running; `data/raw/` is never modified.

## Provenance

- **Source:** ARD-MAV, 60 videos / 107,497 frames, released with GLAD.
- **License:** MIT.
- **Raw layout:** `data/raw/ARD-MAV/{{videos,Annotations}}`, Pascal VOC XML.

## Split rule

**The official split published with GLAD, used verbatim** so results are
comparable to the paper's. 45 training videos, 15 test videos.

Test videos: {', '.join(videos)}

Scene categories (GLAD reports separately for each, see `conditions.json`):

| Category | Videos |
| --- | --- |
| ordinary | {', '.join(SCENE_CATEGORIES['ordinary'])} |
| complex | {', '.join(SCENE_CATEGORIES['complex'])} |
| small_mav | {', '.join(SCENE_CATEGORIES['small_mav'])} |

## Counts ({split})

| | |
| --- | --- |
| Frames | {stats.frames} |
| Boxes | {stats.boxes} |
| Frames with no box | {stats.empty_labels} |
| Unannotated frames skipped | {stats.unannotated_skipped} |

### Target size

| Bucket | Count |
| --- | --- |
{histogram}

## Known issues

- **`CAP_PROP_FRAME_COUNT` overstates.** The container header reports more
  frames than actually decode (28,644 vs 28,337 on the test 15). Every
  decodable frame does have an annotation. Any frame-count reconciliation
  should use the decoder, not the header.
- **Unlabelled frames are skipped, not emitted as negatives.** A frame with no
  XML was never annotated rather than confirmed empty, so scoring detections
  against it would understate precision. Frames whose XML contains zero objects
  *are* genuine negatives and are kept.
- Frame numbering is one-based and four-digit (`phantom05_0001`), matching the
  XML stems exactly. Verified visually via `_verify/`.
""", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for dataset preparation."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("data/raw/ARD-MAV"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/ARD-MAV"))
    ap.add_argument("--split", default="test", help="Split name to write (default: test).")
    ap.add_argument("--videos", nargs="*", default=None,
                    help="Override the video list. Defaults to the official test 15.")
    ap.add_argument("--verify-sample", type=int, default=20,
                    help="Frames to render with boxes for visual checking.")
    ap.add_argument("--seed", type=int, default=42)
    return ap


def main() -> None:
    """Convert, validate, and write metadata."""
    args = build_parser().parse_args()
    videos = list(args.videos or OFFICIAL_TEST_VIDEOS)

    images_dir = args.out / "images" / args.split
    labels_dir = args.out / "labels" / args.split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    total = Stats()
    for n, video in enumerate(videos, 1):
        print(f"[{n}/{len(videos)}] {video}", flush=True)
        total.merge(convert_video(args.raw, video, images_dir, labels_dir))

    ok = report(total, videos)
    write_metadata(args.out, args.split, videos, total)
    verify_dir = render_verify(args.out, args.split, args.verify_sample, args.seed)

    print(f"\nWrote {args.out}")
    print(f"Inspect {verify_dir} before trusting these labels.")
    if not ok:
        sys.exit("Validation found problems -- see above.")


if __name__ == "__main__":
    main()
