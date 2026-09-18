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
| `--block-fraction` | none | A looser threshold for OSD **text blocks**, applied only outside `--picture-rows`. Each block it finds is filled to its bounding rectangle. Off by default. |
| `--picture-rows` | none | `TOP:BOTTOM` — the rows `--block-fraction` must never touch: the band where the scene, the moving horizon bar and the target are. |

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

## Analog OSD

Measured on the Sofa Base **analog** goggles recordings (960×720, EXP-013 and EXP-014).
Three things differ from O4.

**Lower `--white-level`.** At the default 225, the mask caught only the date strip. Analog
OSD glyphs are soft grey after the analog link and the capture re-encode, so they are never
saturated in all three channels. 180 finds them without masking any sky.

**Text blocks, not glyph pixels.** The telemetry digits tick and the compass letters
scroll, so no single pixel in those blocks is white often enough. At 0.35 the mask held the
glyph cores, but the motion branch still made 31 `local mod` hits on the edges of the
telemetry in EXP-013. Lowering the threshold everywhere does not work either: at 0.08 it
also paints the middle of the sky, where the horizon bar smears across samples and where a
target being flown at appears. So `--block-fraction` applies the loose threshold **only
outside `--picture-rows`**, joins nearby characters, and fills each block to its rectangle:

```
py -3.13 -m src.data.hud_mask --videos data/raw/SOFA-ANALOG/videos
    --out data/processed/SOFA-ANALOG/hud_mask.png
    --white-level 180 --block-fraction 0.08 --picture-rows 150:530
```

The result is 90,883 px (13.15%): the clock, compass tape, link block, date and battery,
with the sky from row 150 to row 530 left clear. Against EXP-013's 82 detections it vetoes
**64**, compared with 0 for the O4-style mask.

**Every clip under `--videos` must share one resolution.** The analog source folder mixes
in a 2520×1080 file and a 1280×720 file.

## The HUD that moves

The analog **artificial horizon** is a row of identical dashes that slides with pitch and
roll, so no static mask can hold it. It is also the most convincing false alarm this
project has recorded. At 10 px, a dash (a bright line over a black outline) looks like a
sunlit quadcopter, and in EXP-013 one held a `local yolo` lock for 43 frames.

What gives a dash away is that it has **twins**. The OSD chip draws on a 30-column
character grid, so identical dashes sit one column (`width / 30`, 32 px at 960 wide) to
either side. A drone has no twin. `src.glad_detect --osd-twins` looks 0.5–2.5 columns left
and right, with ±16 px of vertical slack for roll, and vetoes a box whose best normalised
match is **≥ 0.70**. Featureless boxes (std < 8) never twin, because correlation stretches
flat sky to full contrast.

Measured on EXP-013's detections, the 15 horizon-dash boxes score 0.58–0.95, and 13 of
them clear 0.70. Together with the block mask, **77 of 82** are vetoed. The survivors are
3 frames of ground clutter and 2 weaker dashes.

**What it would cost:** a real drone flying in formation with an identical drone one
column apart. It would also veto clutter that repeats at that spacing, which is not a loss.

## Do not use it on clean footage

On a real camera feed there is no overlay, so the mask can only cost detections. Leaving
`--hud-mask` off is also what keeps EXP-004, EXP-005 and EXP-010 reproducing byte-for-byte.
