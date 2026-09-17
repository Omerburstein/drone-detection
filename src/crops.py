"""Contact sheet of what a run actually fired on, cropped from its source video.

`src.render_video` answers *what happened* over time; this answers *what is it
finding*. On footage nobody has labelled the second question is the only one
available, and it is settled by looking rather than by a number -- there is no
ground truth to score against, so "866 detections" means nothing until somebody
has seen what they are.

Three ways to ask, all the same code path:

    # everything, grouped and coloured by the branch that found it
    py -3.13 -m src.crops --pred runs/exp012/detections.jsonl
        --video data/raw/FIELD/videos/capture.mp4 --out runs/exp012/all_hits.png

    # a seeded sample, for a precision-conditional-on-firing estimate
    py -3.13 -m src.crops --pred ... --video ... --out sample.png --sample 24

    # one span, big, to settle whether a thing is a drone
    py -3.13 -m src.crops --pred ... --video ... --out zoom.png
        --frames 1990-2120 --cell 230 --cols 6 --window 2.6

The video is decoded **once, in order**, keeping only the frames wanted. Seeking
to 800 scattered positions in a 388 MB mp4 costs minutes; one pass costs seconds.

Boxes are read in the coordinates the run recorded, so a `--scale` run needs
nothing special here. A `--crop` run does: pass the same `--crop`, exactly as
`src.render_video` requires, or every box lands offset by the crop origin.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .data.crop import Crop
from .eval.labels import load_frames
from .output.contact import Layout, cell, group_sort_key, sheet
from .output.video import open_video

UNGROUPED = "all"
NO_GROUPING = "none"


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for the contact sheet."""
    ap = argparse.ArgumentParser(prog="src.crops", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, type=Path,
                    help="detections.jsonl from a run.")
    ap.add_argument("--video", required=True, type=Path,
                    help="Source video the run was made over.")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output .png. Give every sheet its own name.")
    ap.add_argument("--key-prefix", default=None,
                    help="Frame-key prefix to keep (default: the video's stem). "
                         "A multi-video run holds keys for all of them.")
    ap.add_argument("--crop", type=Crop.parse, default=None, metavar="X,Y,W,H",
                    help="The same --crop the run used. Boxes were recorded in "
                         "cropped coordinates, so without it every one is offset.")
    ap.add_argument("--frames", action="append", default=None, metavar="LO-HI",
                    help="Keep only detections in this frame span. Repeatable. "
                         "Pair with a larger --cell to settle what an object is.")
    ap.add_argument("--sample", type=int, default=None, metavar="N",
                    help="Seeded random sample of N detections instead of all of "
                         "them. Quote the seed with any number taken from it.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for --sample (default 0).")
    ap.add_argument("--group-by", default="branch", metavar="KEY",
                    help="Per-frame field to group and colour by -- any key the "
                         "run recorded, `branch` for GLAD. 'none' disables "
                         "grouping and keeps frame order.")
    ap.add_argument("--title", default=None,
                    help="Title for the legend band (default: the run and video).")
    ap.add_argument("--cell", type=int, default=Layout.cell, metavar="PX",
                    help=f"Rendered cell edge (default {Layout.cell}).")
    ap.add_argument("--cols", type=int, default=Layout.cols, metavar="N",
                    help=f"Cells per row (default {Layout.cols}).")
    ap.add_argument("--window", type=float, default=Layout.window, metavar="X",
                    help=f"Crop side as a multiple of the box's longest side "
                         f"(default {Layout.window}). Lower is tighter.")
    ap.add_argument("--min-window", type=int, default=Layout.min_window,
                    metavar="PX",
                    help=f"Never crop tighter than this, in source pixels "
                         f"(default {Layout.min_window}). What stops a 6 px blob "
                         f"filling a cell with no context around it.")
    return ap


def parse_spans(values: list[str] | None) -> list[tuple[int, int]]:
    """`LO-HI` strings into inclusive integer spans."""
    spans = []
    for value in values or []:
        try:
            lo, hi = (int(v) for v in value.replace(" ", "").split("-", 1))
        except ValueError:
            sys.exit(f"--frames expects LO-HI, got {value!r}")
        if hi < lo:
            sys.exit(f"--frames {value!r} ends before it starts")
        spans.append((lo, hi))
    return spans


def frame_index(key: str) -> int:
    """The one-based frame number a key ends with."""
    return int(key.rsplit("_", 1)[-1])


def wanted(frames, spans: list[tuple[int, int]], group_by: str) -> list[tuple]:
    """Every detection worth a cell: `(index, box, group)`, in frame order."""
    picks = []
    for frame in frames:
        index = frame_index(frame.key)
        if spans and not any(lo <= index <= hi for lo, hi in spans):
            continue
        group = (UNGROUPED if group_by == NO_GROUPING
                 else str(frame.extras.get(group_by, UNGROUPED)))
        for box in frame.preds.boxes:
            picks.append((index, box, group))
    return picks


def order(picks: list[tuple], group_by: str) -> list[tuple]:
    """Grouped into blocks when grouping is on, frame order within each.

    Grouping is the point of the sheet rather than a convenience: GLAD's rare
    branches are the informative ones, and `global yolo` firing six times in
    3,600 frames is invisible in frame order and obvious as a block.
    """
    if group_by == NO_GROUPING:
        return sorted(picks, key=lambda pick: pick[0])
    return sorted(picks, key=lambda pick: (group_sort_key(pick[2]), pick[0]))


def render(picks: list[tuple], video: Path, crop: Crop | None,
           layout: Layout) -> dict[int, list[np.ndarray]]:
    """Decode the video once in order, cropping a cell per wanted detection.

    Returns cells keyed by frame index, in the order they were requested within
    that frame, so the caller can re-assemble them into its own ordering.
    """
    by_index: dict[int, list] = {}
    for index, box, group in picks:
        by_index.setdefault(index, []).append((box, group))
    last = max(by_index)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        sys.exit(f"Could not open {video}")
    cells: dict[int, list] = {}
    index = 0
    try:
        while index < last:
            ok, image = capture.read()
            if not ok:
                break
            index += 1
            if index not in by_index:
                continue
            if crop is not None:
                image = crop.apply(image)
            for box, group in by_index[index]:
                tile = cell(image, box, f"{index}  {group}", group, layout)
                if tile is not None:
                    cells.setdefault(index, []).append(tile)
    finally:
        capture.release()
    return cells


def main() -> None:
    """Build one contact sheet."""
    args = build_parser().parse_args()
    prefix = args.key_prefix or args.video.stem

    # The geometry comes from the video rather than from the processed stills,
    # which a field run keys at a directory that need not exist.
    capture, width, height, _ = open_video(args.video)
    capture.release()
    if args.crop is not None:
        if not args.crop.fits((height, width)):
            sys.exit(f"--crop {args.crop.label} does not fit {args.video.name} "
                     f"({width}x{height})")
        width, height = args.crop.width, args.crop.height

    # Labels are never read: this renders predictions only, which is the whole
    # point on footage nobody has annotated. An empty path makes every lookup
    # miss, exactly as `src.render_video --no-labels` relies on.
    frames = load_frames(args.pred, Path(""), (width, height),
                         key_filter=lambda key: key.startswith(prefix))
    if not frames:
        sys.exit(f"No frames keyed '{prefix}_*' in {args.pred}")

    picks = wanted(frames, parse_spans(args.frames), args.group_by)
    if not picks:
        sys.exit("No detections in the selected frames -- check --frames.")
    if args.sample is not None and args.sample < len(picks):
        picks = random.Random(args.seed).sample(picks, args.sample)
        print(f"Sampled {len(picks)}, seed {args.seed} -- quote the seed with "
              f"any number taken from this sheet.")

    picks = order(picks, args.group_by)
    counts = Counter(group for _, _, group in picks)
    print(f"{len(picks)} detections: " +
          ", ".join(f"{g} {counts[g]}" for g in sorted(counts, key=group_sort_key)))

    layout = Layout(cell=args.cell, cols=args.cols, window=args.window,
                    min_window=args.min_window)
    rendered = render(picks, args.video, args.crop, layout)
    if not rendered:
        sys.exit("Decoded no frames -- does the video match this run?")

    # Re-assemble into the requested order. `render` returns per-frame lists in
    # request order, so popping the front of each frame's list keeps a frame
    # carrying several boxes matched to the right cells.
    pending = {index: list(tiles) for index, tiles in rendered.items()}
    tiles = [pending[index].pop(0)
             for index, _, _ in picks if pending.get(index)]

    title = args.title or f"{args.pred.parent.name}  {args.video.stem}"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    image = sheet(tiles, dict(counts), title, layout)
    cv2.imwrite(str(args.out), image)
    print(f"Wrote {args.out}  {image.shape[1]}x{image.shape[0]}  "
          f"({len(tiles)} cells)")


if __name__ == "__main__":
    main()
