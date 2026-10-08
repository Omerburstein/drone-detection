"""EXP-026: EXP-025d's sky/ground split, only each frame's #1 drawn.

    PYTHONPATH="experiments/exp023_sky_branch/analog_catch_2;experiments/exp026_top1;experiments/exp025_top3;experiments/exp023_sky_branch;experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
        py -3.13 -m overlay_top1

`overlay_split.py` unchanged, with its latest settings as defaults: the sky branch (c >= 6)
on the sky, the 2-of-4 motion window on the ground, blobs within 20 px merged, circles drawn
at least 20 px across, and `--top 1`. Any `overlay_split` flag given here overrides them
(argparse keeps the last value). Outputs go to `runs/sofa_analog/exp026_top1/`.
"""
from __future__ import annotations

import sys

import overlay_split
from clipcfg import CLIP

DEFAULTS = ["--top", "1", "--merge", "20", "--min-draw", "20", "--ref-width", "960"]


def main() -> None:
    args = sys.argv[1:]
    if "--out" not in args:
        start = args[args.index("--start") + 1] if "--start" in args else "441"
        end = args[args.index("--end") + 1] if "--end" in args else "800"
        args += ["--out", f"runs/sofa_analog/exp026_top1/split_sky_window2of4_top1_merge20"
                          f"_{CLIP['name']}_{start}_{end}.mp4"]
    sys.argv = [sys.argv[0], *DEFAULTS, *args]
    overlay_split.main()


if __name__ == "__main__":
    main()
