# EXP-029 — the moving factor, added to the direction test

The user's question, after EXP-024: why not follow each candidate across the frames and use
**how far it moved**, rather than only which way it pointed?

Most of that machinery already existed and the measurement was being thrown away.
`window.Track.total_mu` is the ego-compensated displacement accumulated over the window,
and `criteria.epipolar_direction` reduces its magnitude to a 2 px veto
(`min_displacement`) before testing only its *direction*. This experiment promotes the
magnitude to a statistic in its own right and lets the direction test keep only as much
authority as it has earned.

## Why it runs on the sky branch and not the motion window

EXP-024 measured two things that together rule the window out on analog:

- the motion front-end makes the target a candidate in **174 of 224** labelled frames
  (77.7%), against the sky branch's **223 of 224** (99.6%), and the sky branch wins at
  every matched load by ~1.7x;
- **Lucas-Kanade dies at a median of 0 usable steps** on analog, so the window's
  accumulated displacement does not exist to be measured for most seeds, and a longer
  window does not lengthen the tracks.

So association is done by **chaining candidate peaks**, not by tracking pixels:
`src/algo/kinematics.KinematicTracker` from EXP-028. It needs no texture, and a duplicated
frame — 1 in 6 of every analog clip — contributes no candidate rather than killing a
tracker.

## The two halves

**The moving factor** lives in `src/algo/kinematics.py` (unit-tested, numpy only) because
it is a general gate and needs nothing but the tracker. It is the **limit from below**,
complementing the speed limit EXP-028 added: a confirmed track must have travelled at least
`min_move` px against the static scene over the last `move_window` frames, or it is
overruled as `too-still`. Each track keeps its past sightings as `anchors` and carries every
one forward through each frame's homography, so differencing the newest sighting against the
oldest anchor removes ego-motion.

Why a threshold between them exists at all, from EXP-024 (2026-10-04):

| ego-compensated travel over 5 frames | p10 | median | p90 |
| :--- | ---: | ---: | ---: |
| the target, from the labels | **13.85** | 28.38 | 97.97 px |
| the background residual field | 2.38 | 3.18 | **5.60 px** |

The tails do not overlap. The default `--min-move 8` sits between them.

**The direction test** is added in `moving.py`, not in `src/`, because it needs an epipole
and therefore the residual field — and `src/` must not depend on `experiments/`.
`--mode` sets how much authority it gets:

| mode | rule | why it exists |
| :--- | :--- | :--- |
| `off` | magnitude alone | the control |
| `veto` *(default)* | moving, unless the direction test is usable **and** says the travel is along the epipolar line | direction may only take away, and only where it has an answer |
| `require` | moving only if the direction test is usable **and** says off-epipolar | EXP-021's failure mode, kept to measure the cost |

**Magnitude leads and direction may only veto**, because EXP-021 measured the direction test
rejecting at chance (30.7% agreement against a 22.8% floor) while costing 33 points of
recall, and `criteria.epipolar_direction` names the reason itself: an intercept target sits
near the focus of expansion, where every direction is nearly radial. A test that rejects at
chance must never be able to reject *by default*, which is exactly what `require` allows.

`still`, `parallax`, `unjudged` and `moving` are kept as four separate outcomes. Collapsing
`unjudged` into `rejected` is how the direction test's cost stayed hidden in EXP-017's
headline numbers.

### Two deliberate choices worth knowing

- **The tracker's own `min_move` is passed 0 here.** The report has to count `still`
  separately from `parallax` and `unjudged`, so it must see every confirmed track rather
  than a pre-filtered list. `moving.judge` is the single decision point; nothing is applied
  twice. The `src` gate is the shipping form of the magnitude-only test.
- **The epipole is one pair's, not a window's.** EXP-019 measured the per-pair epipole
  hopping 99–127 px between frames, which is why EXP-017's window pools k pairs' votes.
  This renders frame by frame, so this is the weakest part of the direction half, and the
  report prints `agreement` for that reason. Read it before reading any `parallax` count.

## Running

```bash
PYTHONPATH="experiments/exp029_moving_factor/analog_catch_2;experiments/exp029_moving_factor;experiments/exp023_sky_branch;experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
    py -3.13 -m overlay_moving --start 441 --end 800 --mode veto
```

`--mode off --no-video` for the control, `--mode require --no-video` for the cost.
`--dump <csv>` writes one row per judged track per frame (`moved`, `span`, `sin`, `outcome`,
`on_target`), so any threshold can be recut without re-running. `--show-rejected` draws
what the factor removed in grey.

Self-check for the combination, no video needed:

```bash
PYTHONPATH="experiments/exp029_moving_factor/analog_catch_2;experiments/exp029_moving_factor;experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
    py -3.13 -m moving
```

Outputs go to `runs/sofa_analog/exp029_moving_factor/`. The video is drawn in the clean
EXP-023 look: stage-0 tints, kept tracks as red circles at least 10 px across, a two-line
caption.

## Result — catch_2 441-800, 224 labelled frames

Recall is quoted on the **224 labelled frames** of the span; EXP-024's 174 is a motion
front-end ceiling, not a denominator.

### The direction half adds nothing

| mode | kept/frame | false alarms | drone kept |
| :--- | ---: | ---: | ---: |
| `off` (magnitude alone) | 16.14 | 5626 | **105 (46.9%)** |
| `veto` (+ direction) | 14.59 | 5071 | 104 (46.4%) |
| `require` | 12.93 | 4494 | 96 (42.9%) |

The veto drops 555 false alarms for a single drone frame, which looks like a win until the
load is matched: **magnitude alone at `--min-move 15` reaches 14.57/frame and the identical
104 drone frames.** The epipole buys exactly zero, consistent with its measured agreement
of 28.9% against a ~23% chance floor. `require` is worse on both axes, as EXP-021 predicted.

**So ship `--mode off`.** The direction test is kept only because `require` is the evidence
for not using it.

### Against plain EXP-023 thresholding, at matched load

| load/frame | EXP-029 chain + factor | EXP-023 plain `c` | winner |
| ---: | ---: | ---: | :--- |
| 21.01 | **108 (48.2%)** | 102 (45.5%) | EXP-029 |
| 16.14 | **105 (46.9%)** | 97 (43.3%) | EXP-029 |
| 14.57 | **104 (46.4%)** | 96 (42.9%) | EXP-029 |
| 12.05 | **96 (42.9%)** | 94 (42.0%) | EXP-029 |
| 10.06 | 71 (31.7%) | **89 (39.7%)** | EXP-023 |
| 8.30 | 65 (29.0%) | **83 (37.1%)** | EXP-023 |
| 6.69 | 60 (26.8%) | **78 (34.8%)** | EXP-023 |

**The gain is real but lives in the wrong place.** Above ~12 candidates/frame chaining plus
the moving factor beats plain thresholding by 2-8 drone frames. Below it the curve collapses
and plain EXP-023 wins outright. EXP-023 ships at 5.59/frame and EXP-028 at 0.93 shown/frame
— both below the crossover, so **as built this is not an improvement at any load the project
wants.**

Two reasons, both measured here rather than assumed:

- **A floor of 5.42 kept/frame.** 1950 of 7564 confirmed-track rows (25.8%) have no span
  yet, and `off`/`veto` keep what they cannot judge. Raising `--min-move` cannot push below
  that floor, and those rows carry 42 of the drone frames.
- **The separation is far weaker than the prediction.** Measured travel over 5 frames:

  | | p10 | median | p90 |
  | :--- | ---: | ---: | ---: |
  | on the drone | 15.63 | 63.33 | 131.25 px |
  | everything else | 1.98 | 19.97 | 76.06 px |

  EXP-024 predicted target p10 13.9 against a **background-field** p90 of 5.60 — tails that
  do not overlap. The clutter's real median is 19.97 px, so they overlap heavily. That is
  the caveat EXP-024 filed coming true: the background grid is not the false alarms.
  Chained clutter genuinely moves, from parallax and from association error — the 25 px/frame
  speed limit permits 125 px of travel across a 5-frame window, so a chain that hops between
  two objects banks the hop as travel.

### What this says

The moving factor is sound and cheap, and its own prediction was over-optimistic by the
exact amount EXP-024 warned. The honest next step is not a better statistic but a tighter
**chain**: the speed limit that makes association permissive is what lets clutter accumulate
travel. A cap (`--top`) is also untried here and is how EXP-028 reaches 0.93/frame; without
it these loads are not comparable to the shipping pipeline.
