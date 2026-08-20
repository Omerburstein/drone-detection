"""Convert a raw ARD-MAV-shaped download into the canonical YOLO layout.

Reads `data/raw/<dataset>/{videos,Annotations}` and writes
`data/processed/<dataset>/{images,labels}/<split>/` plus `data.yaml`,
`conditions.json` and `MANIFEST.md`. Re-runnable: raw is never modified, so any
step can be re-derived by deleting the processed tree and running again.

Two datasets ship in this shape and `--dataset` selects between them -- ARD-MAV
and ARD100, same lab, same VOC XML, same 1920x1080 30 fps `.mp4`. The
conversion code is identical across the two by construction; everything that
differs lives in `src.data.datasets`. That matters for M4b, where the question
is how much GLAD loses on unseen video: if the extractor differed at all, part
of the answer would be about the extractor.

Two conventions in the source data drive the whole design, and both fail
silently if assumed wrong:

**Frame numbering.** Annotations are named `<video>_0001.xml`, one-based and
four-digit. Decoded frame *i* (zero-based) therefore pairs with `i + 1`. An
off-by-one here still produces boxes that land near the drone -- adjacent
frames are near-identical -- so it cannot be caught numerically. That is what
the `_verify/` renders exist for.

**Unlabelled frames are not negatives.** Each video's XMLs run contiguously
from 1 to N. A frame with no XML was never annotated, so emitting it with an
empty label would count any detection there as a false positive and understate
precision; extraction skips it instead.

On ARD-MAV's test 15 this guard never fires: every decodable frame has an
annotation. `CAP_PROP_FRAME_COUNT` claims 28,644 frames against 28,337 XMLs,
but that is container metadata and it overstates -- decoding yields exactly
28,337. Trust the decoder, not the header. The guard stays because the
training videos have not been checked and it costs nothing.

`--no-images` writes labels and metadata but no JPEGs. `src.glad_detect` reads
the source `.mp4` and never opens an extracted frame, and `src.evaluate
--frame-size` skips the image-header read, so the whole GLAD path runs off
labels alone -- at roughly 900 KB a frame that is the difference between 25 MB
and 30 GB. A stills detector (`src.baseline_detect`) does need the images.

Example
-------
    py -3.13 -m src.data.prepare_ardmav --dataset ARD-MAV --split test
    py -3.13 -m src.data.prepare_ardmav --dataset ARD100 --split test --no-images
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

from .datasets import SPECS, DatasetSpec, spec_for
from .scene_stats import FrameStats, measure_frame, update_conditions
from .voc import parse_voc, to_yolo

CLASS_NAMES = ("drone",)
SOURCE_CLASS = "Drone"  # the single `<name>` used throughout both datasets' XML
JPEG_QUALITY = 95
PROGRESS_EVERY = 250
VERIFY_PAD = 12  # px of slack around a drawn box, so a tiny target stays visible


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


class VerifySampler:
    """A uniform sample of converted frames, drawn with the labels just written.

    The only check that can catch a corner/centre mix-up or an off-by-one in
    frame numbering: both produce coordinates that pass every numeric test.
    Boxes are drawn back from the emitted YOLO *text*, so the round trip
    through normalisation is part of what gets eyeballed.

    Frames are offered as they decode and held as encoded JPEG bytes, which is
    what lets the check survive `--no-images` -- there is no extracted tree to
    re-read afterwards. Reservoir sampling because the frame count is not known
    until decoding ends, and listing a 28k-file annotation tree up front to
    learn it costs minutes.
    """

    def __init__(self, size: int, seed: int) -> None:
        self.size = max(size, 0)
        self._rng = random.Random(seed)
        self._seen = 0
        self._kept: list[tuple[str, bytes]] = []

    def offer(self, frame: np.ndarray, label_lines: list[str], stem: str) -> None:
        """Consider one frame for the sample."""
        if not self.size:
            return
        self._seen += 1
        if len(self._kept) < self.size:
            self._kept.append((stem, self._render(frame, label_lines)))
            return
        slot = self._rng.randrange(self._seen)
        if slot < self.size:
            self._kept[slot] = (stem, self._render(frame, label_lines))

    @staticmethod
    def _render(frame: np.ndarray, label_lines: list[str]) -> bytes:
        """Draw the labels onto a copy of the frame and encode it."""
        image = frame.copy()
        height, width = image.shape[:2]
        for line in label_lines:
            _, cx, cy, box_w, box_h = (float(v) for v in line.split())
            x1 = int((cx - box_w / 2) * width) - VERIFY_PAD
            y1 = int((cy - box_h / 2) * height) - VERIFY_PAD
            x2 = int((cx + box_w / 2) * width) + VERIFY_PAD
            y2 = int((cy + box_h / 2) * height) + VERIFY_PAD
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1]
        return encoded.tobytes()

    def flush(self, verify_dir: Path) -> Path:
        """Write the sample out."""
        verify_dir.mkdir(parents=True, exist_ok=True)
        for stem, encoded in self._kept:
            (verify_dir / f"{stem}.jpg").write_bytes(encoded)
        return verify_dir


@dataclass(frozen=True)
class Extraction:
    """Where converted frames go, and whether the images go at all."""

    images_dir: Path
    labels_dir: Path
    write_images: bool
    verify: VerifySampler


def annotation_path(raw: Path, video: str, frame_number: int) -> Path:
    """Path of the XML for a one-based frame number."""
    return raw / "Annotations" / video / f"{video}_{frame_number:04d}.xml"


def convert_video(raw: Path, video: str, extraction: Extraction) -> Stats:
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
            _write_frame(frame, xml, stem, extraction, stats)

            if stats.frames % PROGRESS_EVERY == 0:
                print(f"    {video} {stats.frames} frames", flush=True)
    finally:
        capture.release()
    return stats


def _write_frame(frame, xml: Path, stem: str, extraction: Extraction,
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
    extraction.verify.offer(frame, lines, stem)

    if extraction.write_images:
        cv2.imwrite(str(extraction.images_dir / f"{stem}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    # Explicit encoding on every write: Path.write_text defaults to the system
    # ANSI codepage on Windows, which mangles non-ASCII (it corrupted the em
    # dashes in a generated MANIFEST before this was pinned).
    (extraction.labels_dir / f"{stem}.txt").write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    stats.frames += 1
    stats.boxes += len(lines)
    stats.empty_labels += not lines


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


def _category_table(spec: DatasetSpec) -> str:
    """The published scene grouping as a markdown table, or why there is none."""
    if not spec.scene_categories:
        return ("No scene grouping is published for this dataset, so "
                "`conditions.json` carries only the measured `lighting` and "
                "`relative_range` axes.")
    rows = "\n".join(f"| {category} | {', '.join(members)} |"
                     for category, members in spec.scene_categories.items())
    return ("Scene categories (GLAD reports separately for each, see "
            f"`conditions.json`):\n\n| Category | Videos |\n| --- | --- |\n{rows}")


def write_metadata(spec: DatasetSpec, out: Path, split: str, videos: list[str],
                   stats: Stats, wrote_images: bool) -> None:
    """Write data.yaml, conditions.json and MANIFEST.md."""
    if wrote_images:
        # Only meaningful with an images tree to point at; a data.yaml naming a
        # directory that does not exist is a trap, not a convenience.
        (out / "data.yaml").write_text(
            f"path: {out.as_posix()}\n"
            f"{split}: images/{split}\n"
            f"nc: {len(CLASS_NAMES)}\n"
            f"names: {list(CLASS_NAMES)}\n",
            encoding="utf-8",
        )

    published = {"scene_category": spec.category_of(),
                 "categories": {k: list(v) for k, v in spec.scene_categories.items()}
                 } if spec.scene_categories else {}
    (out / "conditions.json").write_text(json.dumps(published, indent=2),
                                         encoding="utf-8")
    # Folds in the measured `lighting` and `relative_range` axes alongside any
    # published `scene_category`. Written second, from the stats gathered during
    # extraction, so no frame is decoded twice.
    update_conditions(out, stats.scene)

    histogram = "\n".join(f"| {label.strip()} | {count} |"
                          for label, count in _area_histogram(stats.areas))
    images_note = ("" if wrote_images else
                   "\n> **Labels only.** Built with `--no-images`: no JPEGs and no\n"
                   "> `data.yaml`. `src.glad_detect` reads the source `.mp4` and\n"
                   "> `src.evaluate --frame-size 1920 1080` needs no pixels, so the GLAD\n"
                   "> path runs off this tree as-is. Re-run without the flag to add the\n"
                   "> images; `data/raw/` is untouched either way.\n")
    (out / "MANIFEST.md").write_text(f"""# {spec.name} — processed

Generated by `src.data.prepare_ardmav --dataset {spec.name}`. Re-derive by deleting
this tree and re-running; `data/raw/` is never modified.
{images_note}
## Provenance

- **Source:** {spec.source}
- **License:** {spec.license}
- **Raw layout:** `{spec.raw.as_posix()}/{{videos,Annotations}}`, Pascal VOC XML.

## Split rule

{spec.split_rule}

Videos ({split}): {', '.join(videos)}

{_category_table(spec)}

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

{spec.known_issues}
- **Unlabelled frames are skipped, not emitted as negatives.** A frame with no
  XML was never annotated rather than confirmed empty, so scoring detections
  against it would understate precision. Frames whose XML contains zero objects
  *are* genuine negatives and are kept.
- Frame numbering is one-based and four-digit (`{videos[0]}_0001`), matching the
  XML stems exactly. Verified visually via `_verify/`.
""", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for dataset preparation."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(SPECS), default="ARD-MAV",
                    help="Which download to convert (default: ARD-MAV). Sets the "
                         "defaults for --raw, --out and --videos.")
    ap.add_argument("--raw", type=Path, default=None,
                    help="Source tree. Defaults to the dataset's own.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Destination tree. Defaults to the dataset's own.")
    ap.add_argument("--split", default="test", help="Split name to write (default: test).")
    ap.add_argument("--videos", nargs="*", default=None,
                    help="Override the video list. Defaults to the dataset's test split.")
    ap.add_argument("--no-images", dest="images", action="store_false",
                    help="Write labels and metadata but no JPEGs. The GLAD path needs "
                         "no extracted pixels and this saves ~900 KB per frame; a "
                         "stills run (src.baseline_detect) does need them.")
    ap.add_argument("--verify-sample", type=int, default=20,
                    help="Frames to render with boxes for visual checking.")
    ap.add_argument("--seed", type=int, default=42)
    return ap


def main() -> None:
    """Convert, validate, and write metadata."""
    args = build_parser().parse_args()
    spec = spec_for(args.dataset)
    raw = args.raw or spec.raw
    out = args.out or spec.out
    videos = list(args.videos or spec.videos)

    images_dir = out / "images" / args.split
    labels_dir = out / "labels" / args.split
    labels_dir.mkdir(parents=True, exist_ok=True)
    if args.images:
        images_dir.mkdir(parents=True, exist_ok=True)

    extraction = Extraction(images_dir=images_dir, labels_dir=labels_dir,
                            write_images=args.images,
                            verify=VerifySampler(args.verify_sample, args.seed))

    mode = "images + labels" if args.images else "labels only"
    print(f"{spec.name}: {raw} -> {out} ({mode})")
    total = Stats()
    for n, video in enumerate(videos, 1):
        print(f"[{n}/{len(videos)}] {video}", flush=True)
        total.merge(convert_video(raw, video, extraction))

    ok = report(total, videos)
    write_metadata(spec, out, args.split, videos, total, args.images)
    verify_dir = extraction.verify.flush(out / "_verify")

    print(f"\nWrote {out}")
    print(f"Inspect {verify_dir} before trusting these labels.")
    if not ok:
        sys.exit("Validation found problems -- see above.")


if __name__ == "__main__":
    main()
