# SOFA-O4 — processed

The picture region of six goggles screen recordings, cropped out **losslessly**.
`data/raw/` is never modified; delete this tree and re-run to re-derive.

## Provenance

- **Source:** our own **EXP Sofa Base** intercept trials, 2026-08-24, recorded off the
  **DJI O4** digital FPV link. Six clips, **7,386 frames, 4.1 minutes**.
- **Raw layout:** `data/raw/SOFA-O4/videos/*.mp4`, 2520×1080 @ 30 fps. Per-file MD5s in
  `data/raw/SOFA-O4/PROVENANCE.md`.
- **Licence:** ours.

## The transform: crop 540,0,1440,1080

Each raw file is a capture of the **goggles display**, not a camera feed. Only a
**1440×1080 window at x=540** is picture; the 540 px either side is pillarbox carrying
HUD glyphs. Verified identical across all six clips at three sample frames each.

```
frame[0:1080, 540:1980]
```

**Why it is not optional.** The global detector letterboxes the frame's *longest side* to
640. Uncropped, 2520 wide scales by 0.254 and a 20 px target arrives as **5 px**. Cropped,
1440 scales by 0.444 and it arrives as **9 px**.

**Why no rescale.** The crop is already **1080 lines** — the dimension GLAD's absolute-pixel
motion constants are tuned against, and the same height as ARD-MAV and ARD100. Widening
1440→1920 to "match 1080p" would stretch 4:3 into 16:9 and turn a round drone into an
ellipse, which is worse for an appearance detector than the 0.817× diagonal it would fix.
So: cropped, not resized.

## Codec: FFV1 in `.avi`, and why not mp4

Measured on a 30-frame sample of `first_catch`, encoding the same cropped frames:

| Codec | Size (full set) | Fidelity |
| --- | ---: | --- |
| `mp4v` | 415 MB | 37.5 dB |
| `avc1` | 1.4 GB | 37.8 dB |
| `MJPG` | 1.2 GB | 41.2 dB |
| **`FFV1`** | **5.7 GB** | **lossless** |
| `HFYU` | 12.8 GB | lossless |

`FFV1` was taken. The targets here are 10–30 px, and a second generation of lossy
compression between `data/raw/` and every number taken from it would smear exactly what
is being measured — for a saving of disk, which this machine has.

**MP4 cannot carry FFV1**, so these are `.avi`. `src.data.sources.resolve_video` finds a
stem in any known container and still prefers `.mp4`, so ARD-MAV and ARD100 resolve
exactly as they did for EXP-004.

**Verified bit-exact:** `catch_5.avi` was decoded frame by frame against
`raw/catch_5.mp4[0:1080, 540:1980]` — 932 of 932 frames identical, zero mismatches.

## What is *not* removed

The HUD burned **into** the picture: a dashed pitch ladder and centre arrows, and a bottom
telemetry strip (`4.04v`, `AIR`, `ALT 13.0M`, `24.3V`, `19 Mbps`, `69%`). There is no clean
feed to fall back on. It is mostly static, so the motion branches difference it away, but
the appearance branch has no such protection — check where detections land before trusting
a count.

## No labels

There is no `labels/` tree here and no annotations anywhere. **No precision, recall, AP or
mAP is computable from this footage.** Runs use `src.glad_detect --record-all` and are
keyed at `data/processed/SOFA-O4/images/test/`, so labels dropped there later score the
existing JSONL with no second inference pass.

`catch` and `miss` in the filenames are **trial outcomes** — whether the interceptor
reached the target — not detection labels. A clip named `catch` still contains long
stretches with no target in view.
