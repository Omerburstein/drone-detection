# experiments/ — the scripts behind EXP-012b to EXP-024

One folder per experiment, holding the code that produced its ledger entry in
`docs/experiments.md`. Until 2026-10-01 these lived inside the gitignored run folders
under `runs/sofa_o4/`, so they existed nowhere but this laptop. They were moved here
before those run folders were deleted. **When the ledger names a script under
`runs/sofa_o4/expNNN_*/`, it is now `experiments/expNNN_*/`.** Outputs (videos, CSVs,
stills) were not kept; re-run the script to regenerate them.

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
PYTHONPATH="runs/sofa_analog/exp023_sky_branch;experiments/exp023_sky_branch;runs/sofa_analog/exp017_motion_first;experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
    py -3.13 -m overlay_sky --start 441 --end 800 --no-sky --no-truth \
    --out runs/sofa_analog/exp023_sky_branch/clean_catch_2_441_800.mp4

# the same on O4 first_catch
PYTHONPATH="experiments/exp023_sky_branch;experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
    py -3.13 -m overlay_sky --start 650 --end 964 --no-sky --no-truth \
    --out runs/sofa_o4/clean_first_catch_650_964.mp4
```

The O4 clip configs (`clipcfg.py` here) still write outputs to `runs/sofa_o4/<experiment>/`.
The analog clip configs were never in `runs/sofa_o4/` and still live in
`runs/sofa_analog/*/clipcfg.py`.

## Calibrated inputs

The O4 structural masks (`prop_mask.npy`, `ladder_mask.npy`, from EXP-017's
`calibrate_masks`) and EXP-015's `screen_fixed_frac.npy` now sit in
`data/processed/SOFA-O4/` beside `hud_mask.png`. They are gitignored like the rest of
`data/` and can be regenerated: `calibrate_masks` writes to whatever path the clip config
names. **A missing mask is not an error:** `masks._load_or_empty` returns an empty mask, so
a wrong path silently renders without masking. Check the `[stage 0]` line of a run's log:
`prop` and `ladder` should be non-zero.
