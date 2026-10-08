# EXP-032 — which confirmed track is the drone

The user asked for every candidate to be tracked and the drone chosen as a person would
choose it: by colour like the previous #1, closeness to where #1 recently was (decaying
with age), consistency, and speed. EXP-031 ranks the kinematic gate's survivors by one
number, the track's decayed sum of contrast. On the FIELD `.raw` that let a ridge-top tree
hold #1 for 71 frames.

`src/algo/selection.py`'s `DroneSelector` ranks the same survivors by a weighted mean of
six cues, each in [0, 1]:

| cue | what it measures | weight |
| --- | --- | ---: |
| `evidence` | the track's decayed contrast sum, relative to this frame's strongest (the old ranking) | 1 |
| `appearance` | colour (Lab of the core, and core-minus-ring lightness) against the drone's, learned from stable #1s | 1 |
| `proximity` | closeness to past #1s, each weighted 0.9^age, spread 30 px at 1440 x sqrt(age) | 1 |
| `consistency` | hits / age since first shown, x (1 - exp(-age / 10)) | 0.5 |
| `smoothness` | constant-velocity prediction error, Gaussian with 12 px at 1440 | 0.5 |
| `motion` | speed against the scene, 1 - exp(-v / 3 px/frame at 1440) | 0.5 |

An unmeasurable cue reads 0.5 for every candidate, so it moves no ranking. The colour is
compared only on clips whose config sets `use_colour`, so analog CVBS compares lightness
alone. It learns only from a #1 that held two frames running, so one wrong frame cannot
rewrite it. It is `overlay_split --rank-by drone`, weights by `--cues evidence=1,motion=0,...`.

## Running

```bash
py -3.13 experiments/exp032_drone_selector/run_clips.py           # catch_2, 4, 5 at merge 30
py -3.13 experiments/exp031_combined/run_field_raw.py --full --rank-by drone   # FIELD .raw
```

`run_clips.py` reads EXP-031's `all` and `exp030_ck10` runs as baselines and runs the same
two with `--rank-by drone`. It also runs `all_drone_no<cue>` with each cue's weight at 0 in
turn. Outputs go to `runs/sofa_analog/exp032_drone_selector/merge30/<clip>/`, and the table
to `run_clips.log` one level up.

## Also in this change: pixel numbers scale with resolution

Every pixel threshold the split pipeline used as an absolute number is now quoted at 1440
wide and scaled to the clip (`src/algo/scale.py`):

- `overlay_split`'s `--radius`, `--merge`, `--min-move`, `--ceiling` and `--min-draw`;
- the speed limit's slack;
- the OSD test's box and the grid slacks in `src.algo.masking`, which are now fractions of
  an OSD column;
- the tag font and strokes;
- and the window radius the drivers pass.

The analog drivers declare `--ref-width 960`, and EXP-031's catch_2 `all` CSVs reproduce
byte for byte. A defaults-only run matches the pre-change code exactly as well.

## Result

**On the labelled analog clips the selector does not beat EXP-031's ranking.** Pooled over
catch_2, 4 and 5 at merge 30, the drone is #1 in 149 frames against 153 (`all`) and 116
against 123 (`exp030_ck10`). #1 jumps more than 29 px 9% less often, 183 of 448 frame
pairs against 202. Every cue but proximity costs #1 frames when removed, but no mix beats
evidence alone, so `--rank-by track` stays the default everywhere. The colour cue ran as
lightness only (CVBS), so the user's colour hypothesis is untested until colour footage is
labelled. Full entry: EXP-032 in `docs/experiments.md`.
