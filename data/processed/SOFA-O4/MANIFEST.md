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

**Verified bit-exact, all six.** Every processed file was decoded frame by frame against
`raw/<stem>.mp4[0:1080, 540:1980]`, and both streams were checked to end together:

| File | Frames | Mismatched | Size |
| --- | ---: | ---: | ---: |
| `catch_5.avi` | 932 | **0** | 0.73 GB |
| `first_catch.avi` | 964 | **0** | 0.69 GB |
| `forth_catch.avi` | 1,060 | **0** | 0.68 GB |
| `miss_1.avi` | 1,724 | **0** | 1.19 GB |
| `second_catch.avi` | 1,551 | **0** | 1.12 GB |
| `third_catch.avi` | 1,155 | **0** | 0.86 GB |
| **total** | **7,386** | **0** | **5.28 GB** |

Decoded frame counts match the headers exactly, unlike ARD-MAV and ARD100 where the header
overstates.

**FFV1 decode is not free.** GLAD sustains ~1.9 fps reading these against ~2.5 fps on the
raw mp4 with a decode-time crop — the lossless files are 5–8× the bytes. That is a cost
paid at read time, not a property of the detector, and it must not be quoted as a
throughput number for the pipeline.

## What is *not* removed

The HUD burned **into** the picture: a dashed pitch ladder and centre arrows, and a bottom
telemetry strip (`4.04v`, `AIR`, `ALT 13.0M`, `24.3V`, `19 Mbps`, `69%`). There is no clean
feed to fall back on. It is mostly static, so the motion branches difference it away, but
the appearance branch has no such protection — check where detections land before trusting
a count.

## The HUD mask

`hud_mask.png` -- 28,002 px, **1.80% of the frame** -- marks where the overlay is painted.
Built on 2026-09-17 by sampling 12 frames from each of the six clips and keeping pixels
saturated in all three channels in at least 35% of them, dilated by 9 px:

```
py -3.13 -m src.data.hud_mask --videos data/processed/SOFA-O4/videos
    --out data/processed/SOFA-O4/hud_mask.png
```

Runs pass it as `src.glad_detect --hud-mask`, which rejects a box lying mostly on it rather
than altering any pixel. On EXP-011's detections it would veto **83.1%**, while the one
confirmed drone scores **0.000** overlap. See [hud_mask.md](../../../docs/hud_mask.md).

It does not cover the centre horizon bar, which sweeps with pitch.

## No labels

There is no `labels/` tree here and no annotations anywhere. **No precision, recall, AP or
mAP is computable from this footage.** Runs use `src.glad_detect --record-all` and are
keyed at `data/processed/SOFA-O4/images/test/`, so labels dropped there later score the
existing JSONL with no second inference pass.

`catch` and `miss` in the filenames are **trial outcomes** — whether the interceptor
reached the target — not detection labels. A clip named `catch` still contains long
stretches with no target in view.
