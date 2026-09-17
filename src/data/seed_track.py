"""Turn one hand-placed box into a labelled segment, for footage nobody annotated.

EXP-011 could not be scored: our own clips have no ground truth, so "the model
found one drone in six videos" is an impression rather than a measurement. This
produces the missing labels semi-automatically -- a human places a box on one
frame, correlation tracking carries it across neighbouring frames, and a contact
sheet makes every proposed box reviewable at a glance.

**A tracker, not a detector.** The detector is the thing under test; using it to
label its own test set would measure nothing. Normalised cross-correlation knows
only what the seed box looked like.

Why not `cv2.TrackerCSRT`: it lives in `opencv-contrib`, which shares the `cv2`
namespace and would shadow the pinned `opencv-python` build -- the failure
recorded in `requirements.txt` on 2026-08-24. The built-in trackers either need
downloaded weights or are black boxes. Correlation matching needs neither and
reports a **score per frame**, which is what makes a review pass efficient: sort
ascending and the drift is at the top.

**Expect a seed to cover tens of frames, not hundreds.** An intercept target
grows and rotates fast, so one appearance stops matching quickly. That is the
honest answer -- place another seed rather than loosening the threshold, which
buys frames by admitting rock and scrub.

Output is two things, and the split is the point:

* `labels/test/<stem>_<frame>.txt` -- YOLO boxes for frames holding a target.
* `verified.jsonl` -- every frame a human has actually adjudicated, target or
  not. `src.evaluate --keys-from` restricts scoring to exactly these, so frames
  nobody looked at are never silently counted as negatives.

Example
-------
    py -3.13 -m src.data.seed_track
        --video data/processed/SOFA-O4/videos/first_catch.avi
        --seed 962 690 236 46 26
        --labels-out data/processed/SOFA-O4/labels/test
        --verified-out data/processed/SOFA-O4/verified.jsonl
        --review-out runs/labels_review
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DEFAULT_SEARCH = 4.0  # search window, in target sizes
# Measured on first_catch: 0.45 tracks the drone and stops when it is gone;
# 0.35 walks onto trees and rock and keeps going.
DEFAULT_MIN_SCORE = 0.45
DEFAULT_SCALES = (0.92, 1.0, 1.09)  # per-frame scale search, for a closing target
# Template blending is OFF by default, and that is a finding rather than a
# preference. With blending on, a track that slips off the drone onto terrain
# adopts the terrain as its template and then matches it at **0.99** -- the
# highest scores on the review sheet were the worst proposals, which is the
# opposite of what the sheet is for. Pinned to the seed appearance, a lost track
# dies within a few frames and says so. Raise it only for a target whose
# appearance changes slowly.
DEFAULT_UPDATE = 0.0


@dataclass(frozen=True)
class TrackPoint:
    """One frame's proposed box and how well it matched.

    `score` is the peak normalised correlation, 0-1. It is carried all the way
    to the contact sheet because it is the review order: the lowest scores are
    where the tracker let go.
    """

    frame: int
    box: tuple[float, float, float, float]  # x, y, w, h in frame pixels
    score: float


def parse_seed(values: list[str]) -> tuple[int, tuple[float, float, float, float]]:
    """`FRAME X Y W H` -> `(frame, box)`, rejecting a degenerate box."""
    if len(values) != 5:
        raise ValueError(f"--seed takes FRAME X Y W H, got {len(values)} values")
    try:
        frame = int(values[0])
        x, y, w, h = (float(v) for v in values[1:])
    except ValueError:
        raise ValueError(f"--seed values must be numbers, got {values!r}") from None
    if w <= 0 or h <= 0:
        raise ValueError(f"--seed box must have positive size, got {w}x{h}")
    if frame < 1:
        raise ValueError(f"--seed frame is 1-based, got {frame}")
    return frame, (x, y, w, h)


def parse_range(text: str) -> range:
    """`100-240` -> the inclusive 1-based frame range it names."""
    parts = text.split("-")
    if len(parts) != 2:
        raise ValueError(f"expected START-END, got {text!r}")
    try:
        start, end = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(f"range bounds must be integers, got {text!r}") from None
    if start < 1 or end < start:
        raise ValueError(f"range must be 1-based and non-empty, got {text!r}")
    return range(start, end + 1)


def to_yolo(box: tuple[float, float, float, float], width: int,
            height: int) -> tuple[float, float, float, float]:
    """Frame-pixel `(x, y, w, h)` -> normalised YOLO `(cx, cy, w, h)`.

    Clamped to the frame: a tracker that drifted off the edge would otherwise
    write a label outside [0, 1] that every later reader has to second-guess.
    """
    x, y, w, h = box
    cx = min(max((x + w / 2) / width, 0.0), 1.0)
    cy = min(max((y + h / 2) / height, 0.0), 1.0)
    return cx, cy, min(w / width, 1.0), min(h / height, 1.0)


def search_window(box: tuple[float, float, float, float], search: float,
                  shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Search window around a box: `(x, y, w, h)`, clamped to the frame."""
    height, width = shape
    x, y, w, h = box
    pad_x, pad_y = w * search, h * search
    x0 = int(max(x - pad_x, 0))
    y0 = int(max(y - pad_y, 0))
    x1 = int(min(x + w + pad_x, width))
    y1 = int(min(y + h + pad_y, height))
    return x0, y0, max(x1 - x0, 1), max(y1 - y0, 1)


def _grey(frame: np.ndarray) -> np.ndarray:
    """Greyscale view of a frame that may already be single channel."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


class TemplateTracker:
    """Follows one patch by normalised cross-correlation in a local window.

    Deliberately small and inspectable. Three behaviours earn their place:

    * **a local search window**, so a repeated texture elsewhere in the frame
      cannot capture the track;
    * **a coarse scale search**, because a target being closed on grows, and a
      fixed-size template loses it within a second or two;
    * **an optional template blend, off by default**. Blending sounds like the
      obvious way to tolerate a changing target, and on this footage it is a
      trap: a track that slips onto terrain adopts the terrain and then matches
      it at 0.99, so the drift arrives wearing the highest confidence in the
      run. Pinned, the score is a real signal.
    """

    def __init__(self, frame: np.ndarray, box: tuple[float, float, float, float],
                 search: float = DEFAULT_SEARCH,
                 scales: tuple[float, ...] = DEFAULT_SCALES,
                 min_score: float = DEFAULT_MIN_SCORE,
                 update: float = DEFAULT_UPDATE) -> None:
        self.box = box
        self.search = search
        self.scales = scales
        self.min_score = min_score
        self.update = update
        template = self._patch(frame, box)
        if template is None:
            raise ValueError(f"seed box {box} does not lie inside the frame")
        self._template = template

    @staticmethod
    def _patch(frame: np.ndarray,
               box: tuple[float, float, float, float]) -> np.ndarray | None:
        """The greyscale patch under a box, or None if it is degenerate."""
        x, y, w, h = (int(round(v)) for v in box)
        if w < 2 or h < 2:
            return None
        patch = frame[max(y, 0):y + h, max(x, 0):x + w]
        if patch.shape[0] < 2 or patch.shape[1] < 2:
            return None
        return _grey(patch)

    def step(self, frame: np.ndarray) -> tuple[tuple[float, float, float, float],
                                               float] | None:
        """Match the template in the next frame; None once it is lost."""
        grey = _grey(frame)
        x0, y0, win_w, win_h = search_window(self.box, self.search, grey.shape[:2])
        window = grey[y0:y0 + win_h, x0:x0 + win_w]

        best: tuple[float, tuple[int, int, int, int]] | None = None
        for scale in self.scales:
            tw = max(int(round(self._template.shape[1] * scale)), 2)
            th = max(int(round(self._template.shape[0] * scale)), 2)
            if tw > window.shape[1] or th > window.shape[0]:
                continue
            template = cv2.resize(self._template, (tw, th),
                                  interpolation=cv2.INTER_LINEAR)
            result = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(result)
            if best is None or score > best[0]:
                best = (float(score), (x0 + loc[0], y0 + loc[1], tw, th))

        if best is None or best[0] < self.min_score:
            return None

        score, found = best
        self.box = (float(found[0]), float(found[1]), float(found[2]), float(found[3]))
        patch = self._patch(frame, self.box)
        if patch is not None and self.update > 0:
            resized = cv2.resize(patch, (self._template.shape[1],
                                         self._template.shape[0]))
            self._template = cv2.addWeighted(self._template, 1 - self.update,
                                             resized, self.update, 0)
        return self.box, score


# --- CLI -------------------------------------------------------------------

SHEET_COLUMNS = 8
SHEET_TILE = 132


def open_video(path: Path) -> tuple[cv2.VideoCapture, int, int]:
    """Open a video and read back the geometry labels are normalised by."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        sys.exit(f"Could not open {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return capture, width, height


def read_frame(capture: cv2.VideoCapture, index: int) -> np.ndarray | None:
    """Decode one 1-based frame by index, seeking to it.

    Random access is exact here because the processed clips are FFV1, which is
    all-intra: every frame is a keyframe. On a long-GOP source this would land
    on the nearest keyframe instead and every label would be off by frames.

    **Seeking costs about 40x what reading the next frame does** -- measured on
    `first_catch.avi`, 1,109 ms against 26 ms. Use it to enter a segment, never
    to walk one; `scan` is the walking version.
    """
    capture.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
    ok, frame = capture.read()
    return frame if ok else None


def scan(capture: cv2.VideoCapture, first: int, last: int):
    """Yield `(index, frame)` for a 1-based inclusive range, reading in order.

    One seek to the start and then sequential decoding, which is the whole
    difference between a labelling pass that takes seconds and one that takes
    minutes.
    """
    capture.set(cv2.CAP_PROP_POS_FRAMES, first - 1)
    for index in range(first, last + 1):
        ok, frame = capture.read()
        if not ok:
            return
        yield index, frame


def track_segment(capture: cv2.VideoCapture, seed_frame: int,
                  seed_box: tuple[float, float, float, float],
                  first: int, last: int, **options) -> list[TrackPoint]:
    """Carry the seed box outward in both directions until the match is lost.

    Both directions start from a tracker freshly initialised on the seed, so
    drift going forward cannot corrupt the backward half.

    The backward half buffers the frames it has to revisit, because a video
    decoder only goes one way and seeking per frame is 40x slower. The buffer
    is **greyscale** -- the only thing the tracker looks at -- which is a third
    of the memory: roughly 1.5 MB per frame at 1440x1080, so `--backward 200`
    costs about 300 MB.
    """
    seed_image = read_frame(capture, seed_frame)
    if seed_image is None:
        sys.exit(f"Could not read seed frame {seed_frame}")
    points = [TrackPoint(seed_frame, seed_box, 1.0)]

    tracker = TemplateTracker(seed_image, seed_box, **options)
    for index, frame in scan(capture, seed_frame + 1, last):
        found = tracker.step(frame)
        if found is None:
            break
        points.append(TrackPoint(index, found[0], found[1]))

    if first < seed_frame:
        earlier = [(index, _grey(frame))
                   for index, frame in scan(capture, first, seed_frame - 1)]
        tracker = TemplateTracker(seed_image, seed_box, **options)
        for index, frame in reversed(earlier):
            found = tracker.step(frame)
            if found is None:
                break
            points.append(TrackPoint(index, found[0], found[1]))

    return sorted(points, key=lambda p: p.frame)


def write_labels(points: list[TrackPoint], out_dir: Path, stem: str,
                 width: int, height: int) -> None:
    """One YOLO label file per proposed box, class 0."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for point in points:
        cx, cy, w, h = to_yolo(point.box, width, height)
        path = out_dir / f"{stem}_{point.frame:04d}.txt"
        path.write_text(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n", encoding="utf-8")


def write_verified(frames: set[int], path: Path, images_dir: Path,
                   stem: str) -> int:
    """Record which frames a human has adjudicated, merging with earlier runs.

    Keyed exactly as the run's `detections.jsonl` is, so `src.evaluate
    --keys-from` accepts this file directly. Frames absent from it were never
    looked at and must not be scored -- counting them as negatives would invent
    precision nobody measured.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                keys[Path(record["image"]).stem] = record
    for frame in frames:
        name = f"{stem}_{frame:04d}"
        keys[name] = {"image": str(images_dir / f"{name}.jpg"), "detections": []}
    with path.open("w", encoding="utf-8") as handle:
        for name in sorted(keys):
            handle.write(json.dumps(keys[name]) + "\n")
    return len(keys)


def contact_sheet(capture: cv2.VideoCapture, points: list[TrackPoint],
                  out_dir: Path, stem: str) -> Path | None:
    """Every proposed box as one reviewable image, worst match first.

    Ordered by score rather than by frame on purpose: tracking fails at the
    ends of a segment, so the boxes worth a human's attention are the low
    scores, and they are all in the first row.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if not points:
        return None
    # One sequential pass over the covered span, keeping only the crops: seeking
    # per point would cost 40x more, and the full frames would not fit anyway.
    wanted = {point.frame: point for point in points}
    crops: dict[int, np.ndarray] = {}
    span = (min(wanted), max(wanted))
    for index, frame in scan(capture, span[0], span[1]):
        point = wanted.get(index)
        if point is None:
            continue
        x, y, w, h = point.box
        cx, cy = int(x + w / 2), int(y + h / 2)
        half = int(max(w, h) * 2.2) + 8
        x0 = max(0, min(cx - half, frame.shape[1] - 2 * half))
        y0 = max(0, min(cy - half, frame.shape[0] - 2 * half))
        crops[index] = (frame[y0:y0 + 2 * half, x0:x0 + 2 * half].copy(), x0, y0)

    tiles = []
    for point in sorted(points, key=lambda p: p.score):
        held = crops.get(point.frame)
        if held is None:
            continue
        crop, x0, y0 = held
        x, y, w, h = point.box
        if crop.size == 0:
            continue
        scale = SHEET_TILE / crop.shape[1]
        tile = cv2.resize(crop, (SHEET_TILE, SHEET_TILE))
        cv2.rectangle(tile, (int((x - x0) * scale), int((y - y0) * scale)),
                      (int((x + w - x0) * scale), int((y + h - y0) * scale)),
                      (0, 230, 230), 1)
        cv2.putText(tile, f"{point.frame} {point.score:.2f}", (3, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1)
        tiles.append(tile)

    if not tiles:
        return None
    rows = []
    for start in range(0, len(tiles), SHEET_COLUMNS):
        row = tiles[start:start + SHEET_COLUMNS]
        while len(row) < SHEET_COLUMNS:
            row.append(np.zeros((SHEET_TILE, SHEET_TILE, 3), dtype=np.uint8))
        rows.append(np.hstack(row))
    out_path = out_dir / f"{stem}_proposals.png"
    cv2.imwrite(str(out_path), np.vstack(rows))
    return out_path


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for the seed-and-track labeller."""
    ap = argparse.ArgumentParser(prog="src.data.seed_track", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path,
                    help="Source clip. Use the processed one the run was made on.")
    ap.add_argument("--stem", default=None,
                    help="Frame-key prefix. Defaults to the video's stem.")
    ap.add_argument("--seed", action="append", nargs=5, required=True,
                    metavar=("FRAME", "X", "Y", "W", "H"),
                    help="FRAME X Y W H -- a box placed by eye on one frame, "
                         "1-based. Repeatable: give one per appearance of the "
                         "target, since a track does not survive an occlusion.")
    ap.add_argument("--negatives", action="append", default=[], metavar="START-END",
                    help="Frame ranges confirmed by eye to hold NO target. "
                         "Repeatable. These are what make precision real -- "
                         "without them only frames with a drone are scored.")
    ap.add_argument("--labels-out", required=True, type=Path,
                    help="Label directory, e.g. data/processed/SOFA-O4/labels/test.")
    ap.add_argument("--verified-out", required=True, type=Path,
                    help="JSONL listing every adjudicated frame, for "
                         "src.evaluate --keys-from. Merged across runs.")
    ap.add_argument("--review-out", type=Path, default=None,
                    help="Directory for the proposal contact sheet. Skipped if "
                         "omitted, but review it before trusting a number.")
    ap.add_argument("--images-dir", type=Path, default=None,
                    help="Path the verified rows are keyed by. Defaults to the "
                         "labels tree's sibling images/<split>.")
    ap.add_argument("--forward", type=int, default=400, metavar="N",
                    help="Frames to track after each seed (default 400).")
    ap.add_argument("--backward", type=int, default=400, metavar="N",
                    help="Frames to track before each seed (default 400).")
    ap.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE,
                    help="End a segment below this correlation (default "
                         f"{DEFAULT_MIN_SCORE}). Raise it if the sheet shows drift.")
    ap.add_argument("--search", type=float, default=DEFAULT_SEARCH, metavar="SIZES",
                    help=f"Search window in target sizes (default {DEFAULT_SEARCH}).")
    ap.add_argument("--update", type=float, default=DEFAULT_UPDATE, metavar="RATE",
                    help=f"Template blend rate (default {DEFAULT_UPDATE}). 0 pins "
                         "the seed appearance.")
    return ap


def main() -> None:
    """Track every seed, write the labels, and report what to review."""
    args = build_parser().parse_args()
    stem = args.stem or args.video.stem
    images_dir = args.images_dir or (args.labels_out.parent.parent / "images"
                                     / args.labels_out.name)
    try:
        seeds = [parse_seed(values) for values in args.seed]
        negatives = [parse_range(text) for text in args.negatives]
    except ValueError as error:
        sys.exit(f"error: {error}")

    capture, width, height = open_video(args.video)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{args.video.name}: {width}x{height}, {total} frames")

    points: dict[int, TrackPoint] = {}
    sheet = None
    try:
        for seed_frame, seed_box in seeds:
            found = track_segment(
                capture, seed_frame, seed_box,
                first=max(1, seed_frame - args.backward),
                last=min(total, seed_frame + args.forward),
                search=args.search, min_score=args.min_score, update=args.update)
            print(f"  seed {seed_frame}: tracked {len(found)} frames "
                  f"({found[0].frame}-{found[-1].frame})")
            for point in found:
                # A later seed is a human decision and outranks a drift into
                # the same frame from an earlier one.
                points[point.frame] = point

        ordered = sorted(points.values(), key=lambda p: p.frame)
        write_labels(ordered, args.labels_out, stem, width, height)

        negative_frames = {f for r in negatives for f in r if f <= total}
        overlap = negative_frames & points.keys()
        if overlap:
            sys.exit(f"error: {len(overlap)} frames are both tracked and declared "
                     f"target-free, e.g. {sorted(overlap)[:5]}")
        kept = write_verified(set(points) | negative_frames, args.verified_out,
                              images_dir, stem)
        if args.review_out:
            sheet = contact_sheet(capture, ordered, args.review_out, stem)
    finally:
        capture.release()

    print(f"\n{len(ordered)} labels written to {args.labels_out}")
    print(f"{len(negative_frames)} frames declared target-free")
    print(f"{kept} verified frames in {args.verified_out}")
    if sheet is not None:
        print(f"Review {sheet} before scoring -- worst match first.")


if __name__ == "__main__":
    main()
