# EXP-027 — the OSD horizon dashes dropped from the sky/ground split

The white dots the operator sees across the middle of the analog picture are the OSD's
**artificial horizon**: a row of identical dashes on the chip's 30-column character grid
(32 px apart at 960 wide), sliding with pitch and roll. The static HUD mask cannot hold
them, and in EXP-026 they were the largest single source of false alarms: 120 of the 231
circles drawn off the drone sat in the dash band.

The sky branch fires on the dash's **black outline**, so a dash is a dark blob just like
the drone, and polarity does not separate them. What does is the grid. A dash has copies
exactly whole columns away; the drone does not. `src.algo.masking.on_osd_grid` asks for
copies (normalised correlation >= 0.70) at two of the four positions +-1 and +-2 columns
away, within 3 px horizontally and 8 px per column vertically, for roll.

The plain `has_twin` test, which looks anywhere 0.5–2.5 columns away, does **not** work
here. On the sky branch's 8–14 px blobs a dark spot on plain sky matches any other dark
spot. It "twinned" the drone in 32 of the 53 frames EXP-026 showed it at #1.

`overlay_grid.py` is a wrapper that runs `overlay_split.py` with EXP-026's defaults
(`--top 1 --merge 20 --min-draw 20`) plus its new `--osd-grid`. That flag drops grid blobs
before merging and before either section's test, so it clears the sky and the window
results together. Any `overlay_split` flag overrides the defaults.

## Running

```bash
PYTHONPATH="experiments/exp023_sky_branch/analog_catch_2;experiments/exp027_osd_grid;experiments/exp025_top3;experiments/exp023_sky_branch;experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
    py -3.13 -m overlay_grid            # top 1
    py -3.13 -m overlay_grid --top 3    # top 3
```

The outputs go to `runs/sofa_analog/exp027_osd_grid/split_sky_window2of4_top{1,3}_merge20_osdgrid_catch_2_441_800`:
the `.mp4`, a `.csv` of every shown blob, and a `_drone.csv`. The top-3 run also has a
`_rank_hist.png` (`exp025_top3/split_rank_hist.py` on its `_drone.csv`).

## Result

| catch_2 441-800, top 1, merge 20 | shown/frame | drone #1 (of 224) | false alarms shown | in the dash band |
| --- | ---: | ---: | ---: | ---: |
| EXP-026 | 0.79 | 53 | 231 | 120 |
| **EXP-027 `--osd-grid`** | **0.64** | **57** (36 sky, 21 ground) | **175** | **49** |

| catch_2 441-800, top 3, merge 20 | shown/frame | drone in top 3 (of 224) | drone #1 | false alarms shown | in the dash band |
| --- | ---: | ---: | ---: | ---: | ---: |
| EXP-025d | 1.62 | 68 | 53 | 513 | 254 |
| **EXP-027 `--osd-grid --top 3`** | **1.08** | **67** (36 sky, 31 ground) | **57** | **318** | **92** |

Without `--osd-grid`, `overlay_split` reproduces EXP-026's CSVs byte for byte. See EXP-027
in `docs/experiments.md` for what is left in the band and the one frame lost (585).
