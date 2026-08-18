"""Deriving `lighting` and `relative_range` condition axes from the imagery.

The published ARD-MAV grouping (`ordinary` / `complex` / `small_mav`) is one
axis, and it conflates things that need different fixes. Two more are
measurable directly from the frames and their labels:

**lighting** - the luminance gap between the target and the background ring
around it. This, not global brightness, is the signal an appearance-only
detector actually has: a drone at 132 mean brightness against clutter at 130 is
invisible however well exposed the frame is. Measured across the ARD-MAV test
split, 16.6% of targets sit below 10 grey levels of separation and 7.0% below 5.

**relative_range** - apparent size, expressed as range. Every ARD-MAV target is
the same airframe (a DJI Phantom), so real size is constant and the pinhole
relation `d = f_px * W_real / w_px` collapses to `d` proportional to `1/size`.
Without camera intrinsics -- the raw download ships none -- that fixes range
only up to a scale factor, so buckets are stated as multiples of the split's own
closest approach rather than in metres. Anything absolute needs a dataset that
labels telemetry range; see `docs/datasets.md`.

Both are *frame*-level, unlike `scene_category`, because both genuinely vary
within a sequence: one ARD-MAV video spans a 5.7x range change, and another
blows out its highlights only for part of its run. Averaging either to a
per-video label would throw away the variation that makes them worth measuring.

Example
-------
    py -3.13 -m src.data.scene_stats --processed data/processed/ARD-MAV --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# The background ring extends to this multiple of the box, so "background" means
# the clutter immediately around the target rather than the whole frame. A drone
# against a bright sky and the same drone against a roofline are different
# detection problems even at identical global exposure.
RING = 3.0

CLIP_HI, CLIP_LO = 250, 8  # grey levels counted as blown / crushed
# Backlighting is a distinct optical regime rather than a point on the contrast
# scale, so it overrides the bucket. 2% of pixels blown is well clear of the
# 0.0-0.2% the cleanly exposed ARD-MAV videos sit at.
BACKLIT_CLIP_FRACTION = 0.02
BACKLIT = "backlit"
NO_TARGET = "no_target"

# Grey levels of target-vs-background separation.
LIGHTING_BUCKETS = (
    ("invisible (<5)", 0.0, 5.0),
    ("low (5-15)", 5.0, 15.0),
    ("moderate (15-30)", 15.0, 30.0),
    ("strong (>=30)", 30.0, float("inf")),
)

# Range as a multiple of the split's closest approach. Open at the bottom
# because the reference is a percentile, so the nearest few targets sit under 1x.
RANGE_BUCKETS = (
    ("near (<2x)", 0.0, 2.0),
    ("mid (2-3x)", 2.0, 3.0),
    ("far (3-5x)", 3.0, 5.0),
    ("very far (>5x)", 5.0, float("inf")),
)
# Closest approach is taken as a high percentile, not the max: a single
# mislabelled oversized box would otherwise rescale the entire axis.
REFERENCE_PERCENTILE = 95.0


@dataclass(frozen=True)
class FrameStats:
    """What one frame contributes to the derived axes.

    `contrast` is the *worst* target in the frame and `size` the *largest*,
    because a frame is only as easy as its hardest target and only as near as
    its nearest one. On ARD-MAV this is nearly moot -- 28,160 boxes across
    28,337 frames -- but the rule has to be stated for datasets where it is not.
    """

    key: str
    n_targets: int
    contrast: float
    size: float
    brightness: float
    clipped_hi: float
    clipped_lo: float


def _crop(gray: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Pixels inside a box, clipped to the frame."""
    x0, y0, x1, y1 = box
    return gray[max(int(y0), 0):int(np.ceil(y1)), max(int(x0), 0):int(np.ceil(x1))]


def _ring_mean(gray: np.ndarray, box: np.ndarray) -> float | None:
    """Mean luminance of the annulus around a box, excluding the box itself.

    None when the ring cannot be formed -- a zero-area box, or one whose
    surroundings fall entirely outside the frame.
    """
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw, bh = x1 - x0, y1 - y0

    inner = _crop(gray, box)
    outer = _crop(gray, np.array([cx - RING * bw / 2, cy - RING * bh / 2,
                                  cx + RING * bw / 2, cy + RING * bh / 2]))
    if inner.size == 0 or outer.size <= inner.size:
        return None
    return float((outer.sum() - inner.sum()) / (outer.size - inner.size))


def measure_frame(image: np.ndarray, boxes: np.ndarray, key: str) -> FrameStats:
    """Measure one frame's exposure and its targets' separation from background.

    `image` may be BGR or already greyscale; `boxes` are absolute xyxy.
    """
    gray = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3
            else image).astype(np.float32)

    contrasts, sizes = [], []
    for box in np.asarray(boxes, dtype=float).reshape(-1, 4):
        ring = _ring_mean(gray, box)
        if ring is None:
            continue
        x0, y0, x1, y1 = box
        contrasts.append(abs(float(_crop(gray, box).mean()) - ring))
        sizes.append(float(np.sqrt(max(x1 - x0, 0) * max(y1 - y0, 0))))

    return FrameStats(
        key=key,
        n_targets=len(contrasts),
        contrast=min(contrasts) if contrasts else float("nan"),
        size=max(sizes) if sizes else float("nan"),
        brightness=float(gray.mean()),
        clipped_hi=float((gray > CLIP_HI).mean()),
        clipped_lo=float((gray < CLIP_LO).mean()),
    )


def _bucket(value: float, buckets: tuple[tuple[str, float, float], ...]) -> str:
    """Label of the bucket `value` falls in."""
    for label, lo, hi in buckets:
        if lo <= value < hi:
            return label
    return buckets[-1][0]


def lighting_label(stats: FrameStats) -> str:
    """Lighting bucket for one frame, backlighting taking precedence."""
    if stats.clipped_hi > BACKLIT_CLIP_FRACTION:
        return BACKLIT
    if not stats.n_targets:
        return NO_TARGET
    return _bucket(stats.contrast, LIGHTING_BUCKETS)


def reference_size(stats: list[FrameStats]) -> float:
    """Apparent size at closest approach, the unit `relative_range` counts in."""
    sizes = np.array([s.size for s in stats if s.n_targets], dtype=float)
    if sizes.size == 0:
        return float("nan")
    return float(np.percentile(sizes, REFERENCE_PERCENTILE))


def range_label(stats: FrameStats, reference: float) -> str:
    """Relative-range bucket for one frame."""
    if not stats.n_targets or not stats.size > 0 or not reference > 0:
        return NO_TARGET
    return _bucket(reference / stats.size, RANGE_BUCKETS)


def build_axes(stats: list[FrameStats]) -> dict[str, dict]:
    """The derived `axes` entries, keyed as `src.eval.conditions` reads them."""
    reference = reference_size(stats)
    return {
        "lighting": {
            "level": "frame",
            "description": ("Target-vs-background luminance separation in grey "
                            f"levels, over a {RING:g}x background ring. "
                            f"{BACKLIT!r} overrides when more than "
                            f"{BACKLIT_CLIP_FRACTION:.0%} of pixels are blown."),
            # Buckets are ordered worst-to-best, not alphabetically: the report
            # reads as a trend, and "far / mid / near / very far" would not.
            "order": [label for label, _, _ in LIGHTING_BUCKETS] + [BACKLIT, NO_TARGET],
            "labels": {s.key: lighting_label(s) for s in stats},
        },
        "relative_range": {
            "level": "frame",
            "description": ("Apparent size as range, in multiples of this split's "
                            f"closest approach (p{REFERENCE_PERCENTILE:g} size = "
                            f"{reference:.1f} px). Relative only -- no intrinsics."),
            "reference_size_px": reference,
            "order": [label for label, _, _ in RANGE_BUCKETS] + [NO_TARGET],
            "labels": {s.key: range_label(s, reference) for s in stats},
        },
    }


def summarise(stats: list[FrameStats]) -> dict[str, dict]:
    """Per-video exposure and contrast summary, for the manifest and for eyeballing."""
    by_video: dict[str, list[FrameStats]] = {}
    for stat in stats:
        by_video.setdefault(stat.key.rsplit("_", 1)[0], []).append(stat)

    summary = {}
    for video, group in sorted(by_video.items()):
        contrasts = np.array([g.contrast for g in group if g.n_targets], dtype=float)
        summary[video] = {
            "frames": len(group),
            "brightness": round(float(np.mean([g.brightness for g in group])), 1),
            "pct_blown": round(float(np.mean([g.clipped_hi for g in group]) * 100), 2),
            "pct_crushed": round(float(np.mean([g.clipped_lo for g in group]) * 100), 2),
            "contrast_p50": (round(float(np.median(contrasts)), 1)
                             if contrasts.size else None),
            "frac_below_5": (round(float(np.mean(contrasts < 5)), 4)
                             if contrasts.size else None),
        }
    return summary


def _boxes_from_label(path: Path, width: int, height: int) -> np.ndarray:
    """Absolute xyxy boxes from one YOLO label file."""
    if not path.exists():
        return np.zeros((0, 4))
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, box_w, box_h = (float(v) for v in parts[1:5])
        cx, cy = cx * width, cy * height
        box_w, box_h = box_w * width, box_h * height
        rows.append([cx - box_w / 2, cy - box_h / 2, cx + box_w / 2, cy + box_h / 2])
    return np.array(rows, dtype=float).reshape(-1, 4)


def measure_split(processed: Path, split: str,
                  progress_every: int = 2000) -> list[FrameStats]:
    """Measure every frame of an already-extracted split.

    Lets the axes be regenerated for a processed tree without re-decoding the
    source videos; `prepare_ardmav` measures inline instead, from the frame it
    has already decoded.
    """
    images, labels = processed / "images" / split, processed / "labels" / split
    if not images.is_dir():
        sys.exit(f"No extracted images at {images}")

    stats = []
    paths = sorted(images.glob("*.jpg"))
    for i, image_path in enumerate(paths, 1):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        boxes = _boxes_from_label(labels / f"{image_path.stem}.txt", width, height)
        stats.append(measure_frame(image, boxes, image_path.stem))
        if i % progress_every == 0:
            print(f"    {i}/{len(paths)} frames", flush=True)
    return stats


def update_conditions(processed: Path, stats: list[FrameStats]) -> Path:
    """Merge the derived axes into `conditions.json`, keeping what is there.

    `scene_category` is published by GLAD and is not ours to recompute, so it is
    read back out of the existing file and re-listed as an axis. The legacy
    top-level `scene_category` key stays where it is, so a reader written
    against the old shape keeps working.
    """
    path = processed / "conditions.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    axes = {}
    if existing.get("scene_category"):
        axes["scene_category"] = {
            "level": "video",
            "description": "The grouping published with GLAD; the paper's "
                           "per-category results use these.",
            "labels": existing["scene_category"],
        }
    axes.update(build_axes(stats))

    existing["axes"] = axes
    existing["lighting_summary"] = summarise(stats)
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return path


def print_axis_counts(stats: list[FrameStats]) -> None:
    """Show how the frames landed across each derived axis."""
    for axis, spec in build_axes(stats).items():
        counts: dict[str, int] = {}
        for label in spec["labels"].values():
            counts[label] = counts.get(label, 0) + 1
        print(f"\n  {axis}:")
        for label, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"    {label:<20} {count:>7}  ({count / len(stats):.1%})")


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for regenerating the derived axes."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", required=True, type=Path,
                    help="Processed dataset root, e.g. data/processed/ARD-MAV")
    ap.add_argument("--split", default="test", help="Split to measure (default test).")
    return ap


def main() -> None:
    """Measure a split and write its derived axes into conditions.json."""
    args = build_parser().parse_args()
    print(f"Measuring {args.processed} [{args.split}] ...")
    stats = measure_split(args.processed, args.split)
    if not stats:
        sys.exit("No frames measured.")

    path = update_conditions(args.processed, stats)
    print(f"\n{len(stats)} frames measured; closest approach "
          f"p{REFERENCE_PERCENTILE:g} = {reference_size(stats):.1f} px")
    print_axis_counts(stats)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
