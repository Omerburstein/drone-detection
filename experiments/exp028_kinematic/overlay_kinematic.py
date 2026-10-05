"""EXP-028: EXP-027's top-3 sky/ground split, with answers held to a drone's top speed.

    PYTHONPATH="experiments/exp023_sky_branch/analog_catch_2;experiments/exp028_kinematic;experiments/exp025_top3;experiments/exp023_sky_branch;experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
        py -3.13 -m overlay_kinematic

`overlay_split.py` with EXP-027's top-3 defaults (`--top 3 --merge 20 --min-draw 20
--osd-grid`) plus `--kinematic`: every answer must sit on a track that moves no faster
than the speed limit allows (`src.algo.kinematics`). Any `overlay_split` flag given here
overrides them, including the gate's own (`--v-max`, `--min-range`, `--hfov`, `--ceiling`,
`--c-keep`, `--confirm`, `--max-coast`). Outputs go to `runs/sofa_analog/exp028_kinematic/`.
"""
from __future__ import annotations

import sys

import overlay_split
from clipcfg import CLIP

DEFAULTS = ["--top", "3", "--merge", "20", "--min-draw", "20", "--osd-grid", "--kinematic"]


def main() -> None:
    args = sys.argv[1:]
    if "--out" not in args:
        start = args[args.index("--start") + 1] if "--start" in args else "441"
        end = args[args.index("--end") + 1] if "--end" in args else "800"
        top = args[args.index("--top") + 1] if "--top" in args else "3"
        args += ["--out", f"runs/sofa_analog/exp028_kinematic/split_top{top}_merge20_osdgrid"
                          f"_kinematic_{CLIP['name']}_{start}_{end}.mp4"]
    sys.argv = [sys.argv[0], *DEFAULTS, *args]
    overlay_split.main()


if __name__ == "__main__":
    main()
