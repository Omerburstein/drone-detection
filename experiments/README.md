# experiments/ — the scripts behind EXP-011 to EXP-024

One folder per experiment, holding the code that produced its ledger entry in
`docs/experiments.md`. Until 2026-10-01 these lived inside the gitignored run folders
under `runs/`, so they existed nowhere but this laptop. They were moved here, and
**`runs/` now holds outputs only — no scripts.** When the ledger names a script under
`runs/<clip>/expNNN_*/`, it is now `experiments/expNNN_*/`; a clip config that sat in a
non-O4 run folder is now a subfolder named after its clip:

| was | now |
| --- | --- |
| `runs/sofa_analog/exp023_sky_branch/{,catch_4/,catch_5/}clipcfg.py` | `exp023_sky_branch/analog_catch_{2,4,5}/clipcfg.py` |
| `runs/sofa_analog/exp017_motion_first/clipcfg.py` | `exp017_motion_first/analog_catch_2/clipcfg.py` |
| `runs/sofa_analog/exp016_multiframe/clipcfg.py`, `interlace_check.py` | `exp016_multiframe/analog_catch_2/clipcfg.py`, `exp016_multiframe/interlace_check.py` |
| `runs/field/exp017_multiframe/clipcfg.py` | `exp016_multiframe/field/clipcfg.py` (EXP-016's machinery on the field clip) |
| `runs/field/exp011_field_glad_scaled/*.py`, `runs/field/exp012_field_glad_inverted/*.py` | `exp011_field_glad_scaled/`, `exp012_field_glad_inverted/` |

Clip configs still write their outputs to `runs/<clip>/expNNN_*/`. The O4 run folders'
outputs (videos, CSVs, stills) were deleted; re-run a script to regenerate them.

These are experiment scripts, not the `src/` package: no tests, no CLI contract, and an
experiment's code is frozen once its numbers are in the ledger. A later experiment that
changes a module gets its own folder (e.g. EXP-023's `silhouette.py` sits beside
EXP-022's rather than replacing it).

## Running

Modules import each other by bare name, so a run is a `PYTHONPATH` stack in which
**the first folder wins**: clip config first, then the newest experiment, then whatever it
builds on. Run from the repo root (`.` on the path is what makes `src` importable).

```bash
# the default overlay -- SOFA-ANALOG catch_2, clean look (no stage-1 tint, no labels)
PYTHONPATH="experiments/exp023_sky_branch/analog_catch_2;experiments/exp023_sky_branch;experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
    py -3.13 -m overlay_sky --start 441 --end 800 --no-sky --no-truth \
    --out runs/sofa_analog/exp023_sky_branch/clean_catch_2_441_800.mp4

# the same on O4 first_catch
PYTHONPATH="experiments/exp023_sky_branch;experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
    py -3.13 -m overlay_sky --start 650 --end 964 --no-sky --no-truth \
    --out runs/sofa_o4/clean_first_catch_650_964.mp4
```

The O4 `first_catch` config is the bare `clipcfg.py` in each experiment folder; every other
clip's is in a subfolder. Kept circles are drawn red, at least 10 px across
(`overlay_sky.MIN_DRAW_DIAMETER`, drawing only).

## Calibrated inputs

The O4 structural masks (`prop_mask.npy`, `ladder_mask.npy`, from EXP-017's
`calibrate_masks`) and EXP-015's `screen_fixed_frac.npy` now sit in
`data/processed/SOFA-O4/` beside `hud_mask.png`. They are gitignored like the rest of
`data/` and can be regenerated: `calibrate_masks` writes to whatever path the clip config
names. **A missing mask is not an error:** `masks._load_or_empty` returns an empty mask, so
a wrong path silently renders without masking. Check the `[stage 0]` line of a run's log:
`prop` and `ladder` should be non-zero.
