"""EXP-027: EXP-026's top-1 sky/ground split with the OSD horizon dashes dropped.

    PYTHONPATH="experiments/exp023_sky_branch/analog_catch_2;experiments/exp027_osd_grid;experiments/exp025_top3;experiments/exp023_sky_branch;experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
        py -3.13 -m overlay_grid

`overlay_split.py` with EXP-026's defaults (`--top 1 --merge 20 --min-draw 20`) plus
`--osd-grid`: before merging, every candidate that sits in a row on the analog OSD's
30-column character grid is dropped (`src.algo.masking.on_osd_grid`). Any `overlay_split`
flag given here overrides them. Outputs go to `runs/sofa_analog/exp027_osd_grid/`.
"""
from __future__ import annotations

import sys

import overlay_split
from clipcfg import CLIP

DEFAULTS = ["--top", "1", "--merge", "20", "--min-draw", "20", "--osd-grid"]


def main() -> None:
    args = sys.argv[1:]
    if "--out" not in args:
        start = args[args.index("--start") + 1] if "--start" in args else "441"
        end = args[args.index("--end") + 1] if "--end" in args else "800"
        top = args[args.index("--top") + 1] if "--top" in args else "1"
        args += ["--out", f"runs/sofa_analog/exp027_osd_grid/split_sky_window2of4_top{top}"
                          f"_merge20_osdgrid_{CLIP['name']}_{start}_{end}.mp4"]
    sys.argv = [sys.argv[0], *DEFAULTS, *args]
    overlay_split.main()


if __name__ == "__main__":
    main()
