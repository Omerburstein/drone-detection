# EXP-023 — the sky branch with depth-aware rings, and size as evidence

A new folder rather than more files in `exp017_motion_first/`, because EXP-022 is cited in
the ledger and its `silhouette.py` / `overlay_stage2.py` must keep returning EXP-022's
numbers. Shared machinery (`masks`, `skyline`, `common`, `overlay_video` helpers) still
comes from those directories via `PYTHONPATH`; only what changed lives here.

## Files

| file | what it is |
| --- | --- |
| `silhouette.py` | stage 2a, second cut: per-candidate depth-aware ring, the `snr` area-aware statistic, a `min_diameter` floor, and the uncertain-band refusal. `py -3.13 -m silhouette` runs 10 synthetic checks. |
| `overlay_sky.py` | the renderer and report, with an internal A/B (every candidate carries both the depth-aware and whole-ring contrast) and a per-candidate CSV dump. |
| `clipcfg.py` | O4 `first_catch`, output to `runs/sofa_o4/exp023_sky_branch/`. The analog twin is `analog_catch_2/clipcfg.py` (with `analog_catch_4/`, `analog_catch_5/`); the O4 prop/ladder masks are read from `data/processed/SOFA-O4/`, the analog ones from `runs/sofa_analog/exp017_motion_first/`. |

## Running it

```bash
# O4
PYTHONPATH="experiments/exp023_sky_branch;experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
    py -3.13 -m overlay_sky --start 650 --end 964

# analog -- its clipcfg goes first, everything else falls through
PYTHONPATH="experiments/exp023_sky_branch/analog_catch_2;experiments/exp023_sky_branch;experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
    py -3.13 -m overlay_sky --start 441 --end 800
```

`--plain-ring` reproduces EXP-022's whole ring, `--keep-uncertain` scores the band instead
of refusing it, `--min-diameter D` applies a size floor, `--no-video` skips rendering.
Those four flags are the attribution passes; their logs are `ab_*.log` here.

`--no-sky` drops the stage-1 drawing (sky / uncertain tint, horizon dots, caption line) and
`--no-truth` drops everything label-derived (green box, FOUND/missed text, zoom inset).
Both are drawing-only; the report is unchanged. Clean renders made with both (2026-10-01):
`runs/sofa_o4/clean_first_catch_650_964.mp4` and
`runs/sofa_analog/exp023_sky_branch/clean_catch_2_441_800.mp4`.

## The headline

**The depth-aware ring is a recall win. The uncertain-band refusal is a false-alarm win
that costs recall, and which of the two is right depends on the clip.**

| | O4 FA/frame | O4 recall | analog FA/frame | analog recall |
| --- | ---: | ---: | ---: | ---: |
| A whole ring (= EXP-022) | 14.65 | 48 (18.7%) | 5.16 | 73 (32.6%) |
| B depth ring, band scored | 17.56 | **57 (22.2%)** | 4.79 | 71 (31.7%) |
| C depth ring, band refused | **9.35** | 40 (15.6%) | **3.30** | 49 (21.9%) |

At **matched** false-alarm rate, C wins on O4 (40 vs 30 frames) and B wins on analog
(60 vs 49). Neither dominates; it is a per-clip operating point. Condition A reproduces
EXP-022 to the digit, which is what makes the rest of the table trustworthy.

## Two mistakes worth not repeating

1. **I changed two things and measured them together.** Turning on the ring and refusing
   the band at once, on a 17-frame probe, read as "87% fewer false alarms at zero recall
   cost". Over the full spans it is nothing like that, and the two changes pull in opposite
   directions on recall. The four `--plain-ring` / `--keep-uncertain` passes exist because
   of this, and the conclusion only became readable once they ran.
2. **The first depth-aware render made things worse.** Scoring a candidate that stands in
   the uncertain band against *other uncertain-band pixels* compares it to a thin, fairly
   uniform ribbon, which deflates `sigma_ring` and manufactures horizon detections — 85% of
   all kept candidates landed in a band covering 7% of the frame. That is what the refusal
   is for, and it is now a named self-check rather than a surprise.

## The dump is the point

`candidates_*.csv` carries one row per kept candidate with `c`, `c_plain`, `snr`,
`ring_sigma`, `ring_median`, `core_min`, `core_mean`, `n_core`, `ring_kept`, `diameter`,
`edge_grad`, `label`, `floored` and `on_target`. Every number in the ledger entry after the
two renders — the matched-false-alarm comparison, the size-floor tables, the separation
figures — is a re-cut of these two files taking seconds. EXP-022 had no dump, so every
threshold question cost a 14-minute span.


# Corrected 2026-09-30 — read this before any number above

The user asked how the target could be under 8 px when it is plainly ~20, and was right.

**"Measured size" above is the DoG's DETECTED SCALE, not the target.** The labelled boxes
are a median **71.6 px on O4 (17-142)** and **37.0 px on analog (18-80)**, while the ladder
tops out at 40 and 27 px. The detector fires at **0.048x** and **0.163x** the target's true
size: it matches a small dark sub-feature inside a box `on_drone` has grown by
`max(10 px, 25%)`, not the airframe. Every recall figure here means *a candidate landed in
the grown box*. The A/B comparisons between conditions survive — both arms are credited the
same way — but none of them shows the branch detecting a drone.

The minimum-diameter tables therefore cut on detected scale, which is an artefact of the
ladder, and `MAX_SCALE_FOR_TARGET` at ~56 px would reject the real O4 airframe at close
range. Ladder and ceiling must be fixed together, and before any threshold is tuned.

## Also changed

- **Defaults rolled back to EXP-022** (whole ring, uncertain band scored). The two EXP-023
  changes are `--depth-ring` and `--refuse-uncertain`. Verified: 73/224 at 5.16 FA/frame on
  analog and 48/257 at 14.65 on O4, both EXP-022 exactly.
- **Rejections are no longer drawn** unless `--show-rejected`. There were ~190 per frame
  against ~10 kept, which is why the render looked full while the report said 5/frame.
- **The dump records every candidate**, kept and rejected, with `reason` and `kept`.
  `--dump-floor` (default 3.0) always keeps structural rejections and on-target rows;
  analog went from 530,273 rows / 59.6 MB to 44,988 / 4.8 MB, and the dropped rows sat
  below the c = 1.8 ceiling measured on pure Gaussian noise sky.
- **Report sections are gated on the condition they describe**, instead of printing zeros
  for a comparison that was never made.

One wording lag: `sky_*_rolledback.log` print the size line as "labelled target size",
which the code now calls "labelled box of the frames hit" — it is a per-detection median
(112 px on O4) and not the clip's median box (71.6 px). The logs were written before that
label was fixed; the numbers are unchanged.
