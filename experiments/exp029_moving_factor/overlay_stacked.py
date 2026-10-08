"""EXP-029 stacked: EXP-028's kinematic gate with the moving factor on top of it.

    PYTHONPATH="experiments/exp023_sky_branch/analog_catch_2;experiments/exp029_moving_factor;experiments/exp028_kinematic;experiments/exp025_top3;experiments/exp023_sky_branch;experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
        py -3.13 -m overlay_stacked

The stages stack rather than compete. `overlay_moving.py` in this folder rebuilt the sky
branch from scratch to measure the moving factor in isolation, which cost it everything
EXP-025 through EXP-028 had added -- the merge, the OSD grid veto, the sky/ground split,
c-keep, and above all **evidence ranking**, without which it reproduced EXP-027's weaker
#1 column and then lost on it.

This instead runs the shipping pipeline and adds one flag. EXP-028's defaults are
`--top 3 --merge 20 --min-draw 20 --osd-grid --kinematic`, which is EXP-027 plus the speed
limit; `--min-move 20` adds the **limit from below** inside the same `KinematicTracker`:

  * the **speed limit** overrules a candidate that moves faster than a drone can
    (EXP-028);
  * the **moving factor** overrules a confirmed track that has not moved far enough
    against the static scene to be flying at all (EXP-029).

One tracker, two bounds, and the second is the gap the first one's docstring named: "a
piece of clutter that stays put moves plausibly by definition, and this gate passes it."

Any `overlay_split` flag given here overrides these, including `--min-move 0` to recover
EXP-028 exactly. Outputs go to `runs/sofa_analog/exp029_moving_factor/`.
"""
from __future__ import annotations

import sys

import overlay_split
from clipcfg import CLIP

DEFAULTS = ["--top", "3", "--merge", "20", "--min-draw", "20", "--osd-grid", "--kinematic",
            "--min-move", "20", "--ref-width", "960"]


def main() -> None:
    args = sys.argv[1:]
    if "--out" not in args:
        start = args[args.index("--start") + 1] if "--start" in args else "441"
        end = args[args.index("--end") + 1] if "--end" in args else "800"
        top = args[args.index("--top") + 1] if "--top" in args else "3"
        mm = args[args.index("--min-move") + 1] if "--min-move" in args else "20"
        args += ["--out",
                 f"runs/sofa_analog/exp029_moving_factor/stacked_top{top}_merge20_osdgrid"
                 f"_kinematic_m{mm}_{CLIP['name']}_{start}_{end}.mp4"]
    sys.argv = [sys.argv[0], *DEFAULTS, *args]
    overlay_split.main()


if __name__ == "__main__":
    main()
