# EXP-026 — the sky/ground split, #1 only

EXP-025d's detector at its latest settings, drawing only each frame's highest-c blob:

* the sky branch (c >= 6) ranks blobs on stage 1's sky
* EXP-025c's 2-of-4 motion window ranks blobs on the ground
* blobs within 20 px are merged first (`--merge 20`)
* circles are drawn at least 20 px across (`--min-draw 20`)
* one pooled ranking by c, **top 1**

`overlay_top1.py` is a wrapper that passes these defaults to EXP-025's `overlay_split.py`,
which is unchanged. Any `overlay_split` flag overrides them. It needs the same two dumps:
EXP-024's `seeds_k4_catch_2_441_800.csv` and EXP-023's `candidates_catch_2_441_800.csv`.

## Running

```bash
PYTHONPATH="experiments/exp023_sky_branch/analog_catch_2;experiments/exp026_top1;experiments/exp025_top3;experiments/exp023_sky_branch;experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
    py -3.13 -m overlay_top1
```

The outputs go to `runs/sofa_analog/exp026_top1/split_sky_window2of4_top1_merge20_catch_2_441_800`:
`.mp4`, a `.csv` of every shown blob, and a `_drone.csv` with one row per labelled frame.

## Result

| catch_2 441-800 | shown/frame | drone #1 (of 224 labelled) | shown blobs on the drone |
| --- | ---: | ---: | ---: |
| top 3, merge 20 (EXP-025d) | 1.62 | 53 | — |
| **top 1, merge 20** | **0.79** | **53** (36 sky, 17 ground) | **53 of 284** |

The #1 count is EXP-025d's, as expected, because the cap does not change the ranking. 76
of 360 frames show nothing. See EXP-026 in `docs/experiments.md`.
