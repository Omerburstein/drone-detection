# EXP-028 — answers held to a drone's top speed

EXP-027 ranked every frame on its own, so the top-3 circles hopped around the picture: #1
moved more than 29 px between consecutive frames in 103 of 156 pairs. A real drone cannot
do that. `--kinematic` (in `exp025_top3/overlay_split.py`, logic in `src/algo/kinematics.py`)
overrules any answer that does not move like one.

- **Speed limit.** 40 m/s, at an assumed minimum range of 10 m, through the R1 Mini's 130°
  lens, comes to 29.8 px/frame at 960 px and 30 fps. The 25 px/frame ceiling caps that, and
  4 px of slack is added. The reach grows with the frames since a track was last seen.
- **Hysteresis.** A strong candidate (one that passes the sky c >= 6 or ground 2-of-4
  test) starts a track. The track is shown from its second hit, and is continued by any
  candidate at c >= `--c-keep` inside its reach. It survives 5 missed frames. A pop far
  from every track is overruled.
- **Camera.** A step passes if its raw or camera-compensated displacement is within reach,
  so a host whip-turn does not kill the track. A camera that follows the target does not
  either.
- **Ranking.** By default (`--rank-by track`) shown answers are ranked by the track's
  evidence, the sum of c decayed by 0.8 per frame, not by this frame's c alone.

`overlay_kinematic.py` is a wrapper: EXP-027's top-3 settings plus `--kinematic`. Any
`overlay_split` flag overrides them: `--v-max`, `--min-range`, `--hfov`, `--ceiling`,
`--c-keep`, `--confirm`, `--max-coast`, `--rank-by` and `--decay`.

## Running

```bash
PYTHONPATH="experiments/exp023_sky_branch/analog_catch_2;experiments/exp028_kinematic;experiments/exp025_top3;experiments/exp023_sky_branch;experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
    py -3.13 -m overlay_kinematic
```

Outputs go to `runs/sofa_analog/exp028_kinematic/split_top3_merge20_osdgrid_kinematic_catch_2_441_800`:
the `.mp4`, a `.csv` of every shown blob (with `track`, `reason` and `evidence`), a
`_drone.csv`, and a `_frames.csv` (with `born`, `overruled`, `continued_weak` and
`coasting`).

## Result

All runs are on catch_2 441–800, top 3, merge 20, `--osd-grid`. The drone counts are out
of 224 labelled frames. A #1 jump is a move of more than 29 px between consecutive frames
that both have a #1.

| run | shown/frame | FA shown | drone in top 3 | drone #1 | #1 jumps |
| --- | ---: | ---: | ---: | ---: | ---: |
| EXP-027 (no gate) | 1.08 | 318 | 67 | 57 | 103 / 156 (66%) |
| gate, c-keep 3, rank by c | 2.47 | 796 | 89 | 72 | 165 / 318 (52%) |
| gate, c-keep 3, rank by track | 2.47 | 791 | 95 | 78 | 82 / 318 (26%) |
| gate, no weak continuation | 0.44 | 107 | 49 | 48 | 22 / 71 (31%) |
| gate, c-keep 6, rank by c | 0.93 | 264 | 69 | 58 | 65 / 165 (39%) |
| **gate, c-keep 6, rank by track (default)** | **0.93** | **264** | **69** | **60** | **43 / 165 (26%)** |

Without `--kinematic`, `overlay_split` reproduces EXP-027's CSVs byte for byte (checked).
See EXP-028 in `docs/experiments.md` for the reading.
