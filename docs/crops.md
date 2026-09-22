# `src.crops` — seeing what a run actually fired on

`src.render_video` answers *what happened over time*. This answers *what is it finding*,
by cropping every detection out of the source video and laying them on one sheet.

On footage nobody has labelled that second question is the only one available. There is no
ground truth to score against, so "866 detections" means nothing until somebody has looked
at them — and 866 is far too many to open one at a time and far too few to summarise as a
number nobody can check.

```bash
py -3.13 -m src.crops --pred runs/field/exp012_field_glad_inverted/detections.jsonl \
    --video data/raw/FIELD/videos/captured_raw_20260616_040253_004.mp4 \
    --out runs/field/exp012_field_glad_inverted/all_hits.png
```

## Parameters

| Flag | Default | Effect |
| --- | --- | --- |
| `--pred` | **required** | `detections.jsonl` from a run. |
| `--video` | **required** | The source video the run was made over. |
| `--out` | **required** | Output `.png`. Give every sheet its own name. |
| `--key-prefix` | the video's stem | Frame-key prefix to keep. A multi-video run holds keys for all fifteen. |
| `--crop` | none | **The same `--crop` the run used.** Boxes were recorded in cropped coordinates, so without it every one is offset by the crop origin. |
| `--frames` | all | `LO-HI`, inclusive, repeatable. Restrict to one span. |
| `--sample` | all | Seeded random sample of N detections. |
| `--seed` | `0` | Seed for `--sample`. **Quote it beside any number taken from the sheet.** |
| `--group-by` | `branch` | Per-frame field to group and colour by — any key the run recorded. `none` keeps frame order. |
| `--title` | run and video | Title for the legend band. |
| `--cell` | `104` | Rendered cell edge, in pixels. |
| `--cols` | `26` | Cells per row. |
| `--window` | `4.0` | Crop side as a multiple of the box's longest side. Lower is tighter. |
| `--min-window` | `40` | Never crop tighter than this, in source pixels. |

## Three ways to ask

**Everything, grouped by branch** — the default. One cell per detection, sorted into blocks
by the branch that produced it and bordered in that branch's colour.

```bash
py -3.13 -m src.crops --pred runs/field/exp010_field_glad/detections.jsonl \
    --video data/raw/FIELD/videos/captured_raw_20260616_040253_004.mp4 \
    --out runs/field/exp010_field_glad/all_hits.png
```

Grouping is the point, not a convenience. GLAD's `global yolo` fires **six times in 3,600
frames**, and in frame order those six cells are invisible among 850 `local yolo`. As a
block at the top of the sheet they are the first thing you see. It is also how the shape of
a run reads at a glance: EXP-010's sheet is mostly clean sky with three short blocks of
hillside in it, and EXP-011's carries an unbroken block of hillside across a fifth of the
sheet — which was the finding, visible without reading a table.

**A seeded sample**, for precision conditional on firing:

```bash
py -3.13 -m src.crops --pred ... --video ... --out sample24.png \
    --sample 24 --cell 192 --cols 6
```

This is the only precision estimate available without labels, and it measures **one
direction only** — it says nothing about recall, because a detection that never happened
has no cell. EXP-010's 23-of-24 came from exactly this.

**One span, big**, to settle what an object actually is:

```bash
py -3.13 -m src.crops --pred ... --video ... --out zoom.png \
    --frames 1990-2120 --cell 230 --cols 6 --window 2.6 --min-window 34
```

This is what distinguished a 22 px drone carrying a payload from a 5 px white shed. At the
default `--cell 104` both are a smudge; at 230 with a tight window one has rotor arms and
the other does not.

## Reading a sheet honestly

**A sheet is a judgement, not a measurement.** Every conclusion drawn from one is somebody
looking at crops and deciding. That is legitimate where no labels exist, and it is not a
score — do not write a precision or a recall into [experiments.md](experiments.md) on the
strength of one without saying it was eyeballed and how many were looked at.

**Size decides how much the judgement is worth.** Above ~15 px a multirotor's arms and
payload resolve and the call is safe. At 4–8 px it is a blob, and a confident verdict either
way is not supportable. [datasets.md](datasets.md) records the case that proves it: a 14×11
px drone at frame 1350 of the field capture was scanned by eye, recorded as empty, and the
detector had boxed it correctly.

**Two tells that cost nothing to check.** A second, identical, *unboxed* object in the same
crop means the detector is picking scene features, not a target — that is how EXP-011's
"clutter" was confirmed. And an object that holds still relative to terrain across a span is
part of the terrain, whatever it looks like.

**The frame number is on every cell.** Use it: pull the span into `--frames` at a larger
`--cell`, or into `src.render_video`, rather than arguing from a thumbnail.

## Notes

- **The video is decoded once, in order.** Seeking to 800 scattered positions in a 388 MB
  mp4 costs minutes; one forward pass costs seconds. `tests/integration/test_crops.py` pins
  that it never seeks and stops at the last wanted frame.
- **Boxes are used in the coordinates the run recorded them in**, so a `--scale` run needs
  nothing special — `src.algo.glad.scaling` maps back before anything is written. A `--crop`
  run does need the matching flag, exactly as `src.render_video` does.
- **Labels are never read.** This draws predictions only. For ground truth coloured by match
  outcome, use `src.render_video` with `--labels`.
- Geometry comes from the **video**, not from the processed stills the run is keyed at —
  a field run keys at a directory that need not exist.

## See also

- [render_video.md](render_video.md) — the same run as video, over time
- [evaluate.md](evaluate.md) — once labels exist, the actual score
- [glad-model.md](glad-model.md) §5b — what the branch colours mean and why the rare ones matter
