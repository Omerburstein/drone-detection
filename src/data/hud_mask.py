"""Finding the HUD in footage that has one painted on.

Our O4 clips are recordings of the goggles *display*, so a pitch ladder, centre
arrows and a telemetry strip are burned into the picture. There is no clean feed
to fall back on -- see `data/raw/SOFA-O4/PROVENANCE.md`.

The HUD gives itself away by being **white and in the same place while the world
underneath changes**. Sample frames spread across every clip, count how often
each pixel is near-saturated in all three channels, and the glyphs stand out:
0.43% of the frame at a threshold of 0.35, rising to 1.8% after dilation.

Three details decide whether this works:

* **All three channels, not brightness.** Sky is bright but blue; requiring the
  *minimum* channel to be saturated keeps sky out and glyphs in.
* **Sample across all six clips.** The scenery differs completely between them
  while the HUD does not, so a pixel white in a third of samples drawn from six
  different scenes is HUD rather than a cloud that happened to sit still.
* **Dilate.** Glyph edges are anti-aliased and the digits change -- `3` becomes
  `4` in the same cell -- so the mask has to cover the cell, not one glyph.

**What this does not catch: HUD that moves.** The centre horizon bar sweeps
vertically with pitch, so its pixels are rarely white at any one position and it
survives the threshold. It was not among the offenders in EXP-011's sample;
lower `--min-fraction` if it starts producing detections, at the cost of masking
more of the frame.

Example
-------
    py -3.13 -m src.data.hud_mask --videos data/processed/SOFA-O4/videos
        --out data/processed/SOFA-O4/hud_mask.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

WHITE_LEVEL = 225  # minimum channel value for a pixel to count as HUD-white
MIN_FRACTION = 0.35  # share of sampled frames a pixel must be white in
DILATE = 9  # square structuring element, in pixels
SAMPLES_PER_VIDEO = 12
BLOCK_JOIN = 25  # characters closer than this belong to one OSD text block


def white_frequency(videos: list[Path], samples_per_video: int = SAMPLES_PER_VIDEO,
                    white_level: int = WHITE_LEVEL) -> np.ndarray:
    """Per-pixel share of sampled frames in which the pixel is HUD-white.

    Frames are sampled by walking each clip once rather than seeking to them:
    a seek costs about 40x a sequential read, and this reads several gigabytes.
    """
    total: np.ndarray | None = None
    count = 0
    for path in videos:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            sys.exit(f"Could not open {path}")
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        wanted = set(np.linspace(1, max(frames, 1), samples_per_video,
                                 dtype=int).tolist())
        index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                index += 1
                if index not in wanted:
                    continue
                white = (frame.min(axis=2) >= white_level).astype(np.float32)
                total = white if total is None else total + white
                count += 1
        finally:
            capture.release()
    if total is None or count == 0:
        sys.exit("No frames sampled -- check --videos")
    return total / count


def build_mask(videos: list[Path], samples_per_video: int = SAMPLES_PER_VIDEO,
               white_level: int = WHITE_LEVEL,
               min_fraction: float = MIN_FRACTION,
               dilate: int = DILATE,
               block_fraction: float | None = None,
               picture_rows: tuple[int, int] | None = None) -> np.ndarray:
    """Boolean mask of where the HUD is painted.

    With `block_fraction` and `picture_rows`, the OSD text blocks outside the
    picture rows are added as whole rectangles -- see `osd_blocks`.
    """
    frequency = white_frequency(videos, samples_per_video, white_level)
    mask = dilate_mask(frequency >= min_fraction, dilate)
    if block_fraction is not None and picture_rows is not None:
        mask |= osd_blocks(frequency, block_fraction, picture_rows, dilate)
    return mask


def osd_blocks(frequency: np.ndarray, fraction: float,
               picture_rows: tuple[int, int], dilate: int = DILATE,
               join: int = BLOCK_JOIN) -> np.ndarray:
    """Whole rectangles over the OSD's text blocks, outside `picture_rows`.

    The analog OSD's telemetry and compass tape sit in fixed rows, but their
    *contents* change -- digits tick, compass letters scroll -- so no single
    pixel is white often, and the default threshold leaves most of each block
    unmasked (EXP-013: 31 `local mod` hits on telemetry the 0.35 mask did not
    hold). A looser threshold finds them, but applied everywhere it also paints
    the middle of the sky, where the moving horizon bar smears and where a
    target being flown at appears. So the loose threshold is confined to rows
    outside `picture_rows`, nearby characters are joined into blocks, and each
    block is filled to its bounding rectangle -- a line of telemetry is a
    rectangle of character cells, not a scatter of glyph pixels.
    """
    top, bottom = picture_rows
    loose = frequency >= fraction
    loose[top:bottom] = False
    loose = dilate_mask(loose, dilate)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        dilate_mask(loose, join).astype(np.uint8))
    blocks = np.zeros_like(loose)
    for label in range(1, count):
        x, y, w, h = stats[label, :4]
        rows, cols = np.nonzero(loose[y:y + h, x:x + w])
        if rows.size:
            blocks[y + rows.min():y + rows.max() + 1,
                   x + cols.min():x + cols.max() + 1] = True
    return blocks


def dilate_mask(mask: np.ndarray, dilate: int = DILATE) -> np.ndarray:
    """Grow a mask to cover anti-aliased edges and a changing digit's whole cell."""
    if dilate <= 1:
        return mask
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate, dilate))
    return cv2.dilate(mask.astype(np.uint8), element) > 0


def save_mask(mask: np.ndarray, path: Path) -> None:
    """Write the mask as a black-and-white PNG, readable by eye and by cv2."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


def load_mask(path: Path) -> np.ndarray:
    """Read a mask PNG back as booleans, failing loudly on a missing file."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        sys.exit(f"Could not read mask {path}")
    return image > 127


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for building a HUD mask."""
    ap = argparse.ArgumentParser(prog="src.data.hud_mask", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True, type=Path,
                    help="Directory of clips sharing one HUD layout. Sampling "
                         "across several different scenes is what separates the "
                         "overlay from the world.")
    ap.add_argument("--out", required=True, type=Path, help="Mask PNG to write.")
    ap.add_argument("--preview", type=Path, default=None,
                    help="Also write the mask painted onto one frame, so it can "
                         "be checked by eye before a run depends on it.")
    ap.add_argument("--samples-per-video", type=int, default=SAMPLES_PER_VIDEO,
                    help=f"Frames sampled per clip (default {SAMPLES_PER_VIDEO}).")
    ap.add_argument("--white-level", type=int, default=WHITE_LEVEL,
                    help=f"Minimum channel value counting as HUD-white (default "
                         f"{WHITE_LEVEL}).")
    ap.add_argument("--min-fraction", type=float, default=MIN_FRACTION,
                    help=f"Share of samples a pixel must be white in (default "
                         f"{MIN_FRACTION}). Lower it to catch HUD that moves.")
    ap.add_argument("--dilate", type=int, default=DILATE,
                    help=f"Dilation in pixels (default {DILATE}).")
    ap.add_argument("--block-fraction", type=float, default=None,
                    help="Looser threshold for the OSD text blocks, applied only "
                         "outside --picture-rows; each block found is filled to "
                         "its rectangle. Off by default.")
    ap.add_argument("--picture-rows", type=parse_rows, default=None,
                    metavar="TOP:BOTTOM",
                    help="Rows the loose --block-fraction must never touch -- "
                         "the band where the scene, the moving horizon bar and "
                         "the target are.")
    return ap


def parse_rows(text: str) -> tuple[int, int]:
    """`TOP:BOTTOM` as a pair of row indices, top above bottom."""
    try:
        top, bottom = (int(v) for v in text.split(":"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected TOP:BOTTOM, got {text!r}") from None
    if not 0 <= top < bottom:
        raise argparse.ArgumentTypeError(f"need 0 <= TOP < BOTTOM, got {text!r}")
    return top, bottom


def main() -> None:
    """Build the mask, report its coverage, and optionally show it on a frame."""
    args = build_parser().parse_args()
    videos = sorted(p for p in args.videos.iterdir()
                    if p.suffix.lower() in {".avi", ".mp4", ".mkv", ".mov"})
    if not videos:
        sys.exit(f"No videos in {args.videos}")
    print(f"Sampling {args.samples_per_video} frames from each of "
          f"{len(videos)} clips ...")

    mask = build_mask(videos, args.samples_per_video, args.white_level,
                      args.min_fraction, args.dilate,
                      args.block_fraction, args.picture_rows)
    save_mask(mask, args.out)
    print(f"Wrote {args.out} -- {int(mask.sum())} px, "
          f"{100 * mask.mean():.2f}% of the frame")

    if args.preview:
        capture = cv2.VideoCapture(str(videos[0]))
        ok, frame = capture.read()
        capture.release()
        if ok:
            frame[mask] = (0, 0, 255)
            args.preview.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.preview), frame)
            print(f"Wrote {args.preview} -- check the glyphs are covered and the "
                  f"sky is not")


if __name__ == "__main__":
    main()
