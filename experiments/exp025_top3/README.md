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
| `overlay_gate.py` | EXP-025b: the same top N, after a persistence gate of `--min-appear` of `--k` frames (default 4 of 5). `--coords scene` (default) chains camera-compensated, `--coords image` in raw picture coordinates. Prints every threshold 1..k from one pass. Writes a CSV of every kept candidate (`frame, x, y, c, diameter, appearances, rank_all, on_target`). |
| `drone_ranks.py` | reads `overlay_gate.py`'s CSV and writes a Markdown table of every frame the drone is in a top N, with its rank with and without the gate. Any threshold, no re-render. |
| `overlay_window_skyc.py` | EXP-025c: EXP-024's 2-of-4 motion window decides what survives, the sky branch's `c` decides the order. Drawn from EXP-024's `seeds_k4_` dump and EXP-023's candidate dump, no detector re-run. A blob is ranked when a survivor lies within 9 px; no c threshold, cloud vetoes dropped (`--keep-cloud` keeps them). Writes a CSV of every shown blob. |
| `overlay_split.py` | EXP-025d: stage 1 splits the frame; the sky branch (c >= 6) ranks on the sky, EXP-025c's 2-of-4 window on the ground, one pooled top 3 by c. `--merge R` folds blobs within R px into one first; `--min-draw` sets the smallest drawn circle. Also writes `_frames.csv` (per-frame sky fraction and counts); `--report-only` reprints the report from it and `_drone.csv`. Writes a CSV of every shown blob and a per-labelled-frame `_drone.csv`. |
| `split_rank_hist.py` | reads `overlay_split.py`'s `_drone.csv` and draws the drone's rank histogram, stacked by section. |
| `run_clips.py`, `clips/clipcfg.py`, `clips_summary.py` | EXP-025e: the split overlay on catch_4, catch_5 and FIELD. `sky_chunks` / `split_chunks` in a clip's config run those steps in parallel frame ranges and join them, exactly (each frame reads only itself and the frame before). One config for both clips, chosen by `EXP025_CLIP`. The driver runs the window and the overlay per clip, skipping any step whose output exists, and reuses EXP-023's sky dumps. The summary prints a row per clip from the logs. |

The candidates come from EXP-023's `silhouette.detect` and the drawing helpers from its
`overlay_sky.py`, both unchanged, so before the cap they are EXP-023's to the digit.

## Running

```bash
# SOFA-ANALOG catch_2 (the default clip)
PYTHONPATH="experiments/exp023_sky_branch/analog_catch_2;experiments/exp025_top3;experiments/exp023_sky_branch;experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
    py -3.13 -m overlay_top3 --start 441 --end 800 \
    --out runs/sofa_analog/exp025_top3/top3_catch_2_441_800.mp4
```

```bash
# EXP-025b: 4 of 5, then top 3 -- chained in picture coordinates (the one that works)
PYTHONPATH="experiments/exp023_sky_branch/analog_catch_2;experiments/exp025_top3;experiments/exp023_sky_branch;experiments/exp017_motion_first;experiments/exp015_normalised_motion;."     py -3.13 -m overlay_gate --start 441 --end 800 --coords image     --out runs/sofa_analog/exp025_top3/gate4of5_image_top3_catch_2_441_800.mp4
py -3.13 experiments/exp025_top3/drone_ranks.py     runs/sofa_analog/exp025_top3/gate4of5_image_top3_catch_2_441_800.csv
```

## The gate (EXP-025b)

Each kept candidate (c >= 6) is chained back through the 4 frames before it: step into the
older frame, take the nearest kept candidate within 9 px (14 px at 1440 wide, EXP-024's
radius), move the chain onto it. The candidate survives at 4 appearances of 5, its own
frame included. Then the top 3 survivors by c are drawn, as `overlay_top3.py` draws them.
`appearances` does not depend on the threshold, so `--min-appear 1` is EXP-025 to the digit
(73 / 61 / 31-18-12), and every threshold is in the report.

**Chain in picture coordinates on intercept footage, not camera-compensated.** The camera
follows the drone, so the drone is steadier in the picture than the background: the
labelled drone's step is a median 4.1 px raw against 6.7 px compensated, over 9 px in 20%
of frames against 40%. Compensation threw the chain off the drone, and kept it in 3 frames.

| catch_2 441-800, 4 of 5, top 3 | shown/frame | drone in top 3 (of 224) | #1 / #2 / #3 |
| --- | ---: | ---: | :---: |
| no gate (EXP-025) | 2.88 | 61 | 31 / 18 / 12 |
| gate, `--coords scene` | 1.03 | 3 | 2 / 1 / 0 |
| **gate, `--coords image`** | **1.71** | **27** | **20 / 6 / 1** |
| ceiling: drone kept in >= 4 of its last 5 frames | | 46 | |

### Every frame the drone is in the gated top 3 (`--coords image`)

| frame | rank, 4 of 5 gate | rank, no gate | c | appearances of 5 |
| ---: | :---: | :---: | ---: | :---: |
| 513 | #2 | #3 | 7.0 | 4 |
| 514 | #2 | #3 | 6.8 | 5 |
| 515 | #2 | #2 | 11.0 | 5 |
| 522 | #1 | #1 | 9.3 | 4 |
| 523 | #1 | #1 | 20.6 | 5 |
| 524 | #2 | — | 7.0 | 5 |
| 525 | #3 | — | 7.7 | 5 |
| 526 | #1 | #2 | 7.7 | 5 |
| 537 | #1 | #1 | 7.7 | 4 |
| 538 | #1 | #1 | 7.7 | 4 |
| 543 | #1 | #1 | 6.9 | 4 |
| 544 | #1 | #1 | 6.9 | 5 |
| 556 | #2 | #2 | 9.7 | 4 |
| 557 | #1 | #2 | 8.7 | 5 |
| 561 | #1 | #1 | 9.6 | 4 |
| 562 | #1 | #1 | 11.3 | 5 |
| 563 | #1 | #3 | 7.8 | 5 |
| 564 | #1 | #2 | 9.8 | 4 |
| 570 | #1 | #2 | 11.1 | 4 |
| 571 | #1 | #1 | 20.5 | 5 |
| 572 | #1 | — | 8.1 | 5 |
| 573 | #1 | #2 | 10.2 | 5 |
| 574 | #2 | #3 | 10.3 | 5 |
| 575 | #1 | #1 | 20.8 | 5 |
| 576 | #1 | #1 | 14.8 | 5 |
| 577 | #1 | #1 | 26.1 | 5 |
| 578 | #1 | #1 | 18.1 | 5 |

"—": the drone was a kept candidate but ranked 4th or lower before the gate. The gate
removed the clutter above it. With `--coords scene` the drone is in the gated top 3 in
frames 513 (#1), 514 (#2) and 515 (#1) only. `drone_ranks.py` writes the full table,
including the 37 frames the gate removed, next to each CSV.

**Duplicate frames help this gate rather than hurt it.** `catch_2` repeats 1 frame in 6
(found in EXP-024's 2026-10-04 addendum). The motion branch gets no candidates on a
duplicate pair, but the sky branch detects on one frame at a time, so a duplicate repeats
the previous frame's candidates exactly (537/538 above). Two windows in three contain a
duplicate pair, and there 4 of 5 can be 3 live frames plus a copy. Dropping duplicates
before the window is built, as the open todo proposes for EXP-024, would make this gate
stricter.

## The motion window, ranked by sky contrast (EXP-025c)

```bash
PYTHONPATH="experiments/exp023_sky_branch/analog_catch_2;experiments/exp025_top3;experiments/exp023_sky_branch;experiments/exp017_motion_first;experiments/exp015_normalised_motion;."     py -3.13 -m overlay_window_skyc
```

Needs `runs/sofa_analog/exp024_window_length/seeds_k4_catch_2_441_800.csv` (EXP-024's
`overlay_window --k 4 --dump`) and EXP-023's `candidates_catch_2_441_800.csv`.

| catch_2 441-800, top 3 | ranked/frame | shown/frame | drone in top 3 (of 224) | #1 / #2 / #3 |
| --- | ---: | ---: | ---: | :---: |
| sky c >= 6, no gate (EXP-025) | 5.48 | 2.88 | 61 | 31 / 18 / 12 |
| sky 4 of 5, picture coordinates (EXP-025b) | 1.94 | 1.71 | 27 | 20 / 6 / 1 |
| **2-of-4 motion window, ranked by sky c** | **1.52** | **1.29** | **49** | **39 / 8 / 2** |

The drone is #1 in more frames than EXP-025's, at 45% of the shown load. It gives up 12
top-3 frames, mostly at #2 and #3. Video: `window2of4_skyc_top3_catch_2_441_800.mp4`.

## Sky and ground split, one detector each (EXP-025d)

```bash
PYTHONPATH="experiments/exp023_sky_branch/analog_catch_2;experiments/exp025_top3;experiments/exp023_sky_branch;experiments/exp017_motion_first;experiments/exp015_normalised_motion;."     py -3.13 -m overlay_split
py -3.13 experiments/exp025_top3/split_rank_hist.py
```

`overlay_split.py` recomputes stage 1 (`skyline.split`) on every frame. A blob on a sky
pixel is the sky branch's to rank (kept, c >= 6); any other blob is the ground's, ranked
when a 2-of-4 window survivor lies within 9 px (EXP-025c's rule). Both sections are
pooled into **one** ranking by `c`, and the top 3 are drawn tagged `#rank c sky|gnd`, with
stage 1's horizon as a thin line (`--no-split-line` drops it). It needs the same two dumps
as EXP-025c. It writes the video, a CSV of every shown blob, and a `_drone.csv` with one
row per labelled frame. `split_rank_hist.py` reads that CSV and draws the drone's rank
histogram, stacked by section.

| catch_2 441-800, top 3 | ranked/frame | shown/frame | drone in top 3 (of 224) | #1 / #2 / #3 |
| --- | ---: | ---: | ---: | :---: |
| sky c >= 6 everywhere (EXP-025) | 5.48 | 2.88 | 61 | 31 / 18 / 12 |
| 2-of-4 window everywhere (EXP-025c) | 1.52 | 1.29 | 49 | 39 / 8 / 2 |
| **sky branch on sky, window on ground** | **2.00** | **1.66** | **66** | **52 / 10 / 4** |

The sky section ranks the drone in 36 frames, all at #1. The ground section ranks it in
33 (16 / 10 / 4 in the top 3) and drops it in 145.

`--merge 10` folds every candidate within 10 px of a stronger one into it before the
sections' tests, so a 20 px-wide cluster is one blob: anchor's centre, c and section,
grown to cover its members, passing a test if any member does. On catch_2 it folds 231
candidates (0.64/frame) and leaves the drone's numbers unchanged: 66 in the top 3, 52 / 10
/ 4. The outputs carry `_merge10` in their names, and the CSV has a `members` column.

`--merge 20 --min-draw 20` (40 px across; circles drawn at least 20 px, appearance only)
folds 1893 candidates and gets 68 in the top 3, 53 / 10 / 5. Two of the gained frames are a
clutter anchor absorbing the drone, and one (585, #2 to #1) is the drone's own blob.

## Other labelled clips (EXP-025e)

```bash
py -3.13 experiments/exp025_top3/run_clips.py          # catch_4 and catch_5, 2 at a time
py -3.13 experiments/exp025_top3/clips_summary.py
```

Outputs in `runs/sofa_analog/exp025_top3/<clip>/`. The drone is in the top 3 in 14 of 82
frames on catch_4 and 31 of 188 on catch_5, against 33 and 30 for the sky branch alone.
Stage 1 finds little sky on either (median 0% and 7%), so the window ranks nearly
everything. The split pays off only where the sky mask works.

FIELD (`run_clips.py field`, output in `runs/field/exp025_top3/`) has no labels. Its sky is
darker than its ground, so its config turns stage 1's brightness test off. The sky branch
shows 0.64–0.77/frame inside the drone episodes and 0.17 outside them; the ground section
shows 0.01.

## Result

See EXP-025 in `docs/experiments.md`, and its 2026-10-04 addendum for the gate.
