# EXP-025 — the sky branch, top 3 per frame

EXP-023's detector exactly as `clean_catch_2_441_800.mp4` runs it (whole ring, c >= 6,
uncertain band scored), with one change after the threshold: each frame shows **only its
3 highest-contrast candidates**, fewer if fewer pass. Each is a red circle with a `#rank c`
tag; `#1` is drawn thicker. Nothing is drawn from the labels.

The question it answers: if a downstream stage takes only a frame's top N, how often is the
drone among them, and how often is it #1?

## Files

| file | what it is |
| --- | --- |
| `overlay_top3.py` | renderer and report. `--top N` changes the cap, `--no-video` prints the report only. Writes a CSV of every ranked candidate (`frame, rank, x, y, c, diameter, on_target`) next to the video. |

The candidates come from EXP-023's `silhouette.detect` and the drawing helpers from its
`overlay_sky.py`, both unchanged, so before the cap they are EXP-023's to the digit.

## Running

```bash
# SOFA-ANALOG catch_2 (the default clip)
PYTHONPATH="experiments/exp023_sky_branch/analog_catch_2;experiments/exp025_top3;experiments/exp023_sky_branch;experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
    py -3.13 -m overlay_top3 --start 441 --end 800 \
    --out runs/sofa_analog/exp025_top3/top3_catch_2_441_800.mp4
```

## Result

See EXP-025 in `docs/experiments.md`.
