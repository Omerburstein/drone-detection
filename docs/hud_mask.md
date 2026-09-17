# `src.data.hud_mask` — reference

Finds the overlay burned into footage that was recorded off a goggles display, so the
detector can be told to ignore it.

```
py -3.13 -m src.data.hud_mask --videos <dir> --out <mask.png> [--preview <png>]
```

Our O4 clips are recordings of the goggles *screen*: a pitch ladder, centre arrows and a
telemetry strip are painted into the picture, and there is no clean feed to fall back on.
EXP-011 measured what that costs — of 24 randomly sampled detections, **21 were HUD glyphs
and none were drones**, and the tracker held a battery digit for 300 frames.

---

## How the HUD is found

It gives itself away by being **white, and in the same place while the world underneath
changes**. Sample frames across every clip, count how often each pixel is near-saturated
in all three channels, keep the pixels above a frequency threshold, dilate.

Three details decide whether it works:

- **All three channels, not brightness.** Sky is bright but blue. Requiring the *minimum*
  channel to be saturated keeps sky out and glyphs in.
- **Sample across all the clips.** The scenery differs completely between them and the HUD
  does not, so a pixel white in a third of samples drawn from six scenes is overlay rather
  than a cloud that sat still.
- **Dilate.** Glyph edges are anti-aliased and digits change — `3` becomes `4` in the same
  cell — so the mask must cover the cell, not one glyph.

Measured on `data/processed/SOFA-O4/videos`: 0.43% of the frame before dilation, **1.80%
after** — 28,002 px of 1440×1080.

## Parameters

| Parameter | Default | What it does |
| --- | --- | --- |
| `--videos` | required | Directory of clips sharing one HUD layout. |
| `--out` | required | Mask PNG to write. |
| `--preview` | none | Also write the mask painted red onto one frame. **Look at it.** |
| `--samples-per-video` | 12 | Frames sampled per clip. |
| `--white-level` | 225 | Minimum channel value counting as HUD-white. |
| `--min-fraction` | 0.35 | Share of samples a pixel must be white in. |
| `--dilate` | 9 | Dilation, in pixels. |

## Using it

```
py -3.13 -m src.glad_detect --videos data/processed/SOFA-O4/videos
    --hud-mask data/processed/SOFA-O4/hud_mask.png --record-all ...
```

`src.glad_detect --hud-mask` rejects a box that lies mostly on the mask, at **every point
a box can be emitted or locked onto**. The lock is the half that matters: one rejected
detection costs a frame, a lock on a digit costs every frame until the 30-miss fallback
fires, and throughout it the detector is searching the wrong part of the picture.

**Nothing is inpainted.** The veto changes the decision, not the pixels — no invented
texture enters a frame that a number is taken from.

### Why a fraction and not the centre point

The reticle sits at the middle of the frame, which is exactly where a target being flown at
appears. A centre test would delete a real drone crossing a ladder dash. Measured on
EXP-011's 1,347 detections, the fraction separates the populations cleanly:

| Overlap threshold | Detections vetoed |
| --- | ---: |
| 0.3 | 90.4% |
| **0.5** (`HUD_VETO_FRACTION`) | **83.1%** |
| 0.6 | 82.1% |
| 0.8 | 49.8% |

The curve is flat from 0.4 to 0.6, so the default is not balanced on a knife edge. And the
**one confirmed drone** in that run — `first_catch` frame 962 — has an overlap of
**0.000**, so the veto cannot reach it.

## What it does not catch

**HUD that moves.** The centre horizon bar sweeps vertically with pitch, so its pixels are
rarely white at any single position and it survives the threshold. It was not among the
offenders in EXP-011's sample. Lower `--min-fraction` if it starts producing detections,
at the cost of masking more of the frame.

## Do not use it on clean footage

On a real camera feed there is no overlay, so the mask can only cost detections. Leaving
`--hud-mask` off is also what keeps EXP-004, EXP-005 and EXP-010 reproducing byte-for-byte.
