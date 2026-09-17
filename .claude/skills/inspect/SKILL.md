---
name: inspect
description: Look at what a detection run actually fired on — render the run back onto its video, and crop every detection onto one contact sheet grouped by the branch that found it. Use when the user asks to "make a video", "render the run", "show me the detections", "what did it catch", "get the pictures", "make a collage", "which ones are drones", or wants to check whether detections on unlabelled footage are real targets or clutter. This is the unlabelled-footage path; once labels exist use /eval for a score.
---

# inspect

Two tools, one question: **what is this run actually finding?**

```
src.render_video   the run over time, drawn back onto its source video
src.crops          every detection cropped onto one sheet, grouped by branch
```

References: [crops.md](../../../docs/crops.md), [render_video.md](../../../docs/render_video.md).

Use this when the footage has **no labels**. Once labels exist, `/eval` gives a score and
this becomes the thing you use to explain it, not to establish it.

## The rule that governs everything here

**A sheet or a video is a judgement, not a measurement.** Nothing produced by this skill is
a score. When a finding from it reaches `docs/experiments.md`, say so explicitly: how many
crops were looked at, what seed, and that it was eyeballed. EXP-010's "23 of 24" is the
correct form; "precision 0.96" is not.

Never write a precision or recall number into the ledger on the strength of a sheet.
Precision conditional on firing is the *only* direction a sheet can speak to — a detection
that never happened has no cell, so recall is invisible by construction.

## Making the video

```bash
py -3.13 -m src.render_video \
    --video data/raw/FIELD/videos/<capture>.mp4 \
    --pred runs/<exp>/detections.jsonl \
    --no-labels --zoom 3 --zoom-span 100 \
    --out runs/<exp>/overlay.mp4
```

- **`--no-labels` on unlabelled footage, always.** The scored renderer colours every box by
  match outcome, so without ground truth it paints all 866 detections red and captions each
  a false alarm — a precision claim of zero against a run nobody measured. The unscored mode
  draws one neutral colour and says on the caption strip that nothing here is known to be
  right.
- **Keep `--zoom`/`--zoom-span` fixed across runs you intend to compare.** The field captures
  use `--zoom 3 --zoom-span 100`. A 10–30 px target is unreadable without the inset.
- Pass the same `--crop` the run used, if it used one.
- ~15 min and ~390 MB for 3,600 frames. Background it.
- **Skip it for a negative result.** A sheet carries the finding at 1% of the size. EXP-011
  deliberately has no video.

## Making the sheet

Start with everything, grouped by branch:

```bash
py -3.13 -m src.crops --pred runs/<exp>/detections.jsonl \
    --video data/raw/FIELD/videos/<capture>.mp4 \
    --out runs/<exp>/all_hits.png
```

Then narrow, depending on the question:

| Question | How |
| --- | --- |
| How often is it right when it fires? | `--sample 24 --cell 192 --cols 6`, and **quote the seed** |
| Is *this thing* a drone? | `--frames LO-HI --cell 230 --cols 6 --window 2.6 --min-window 34` |
| What did this run find that another did not? | Filter the JSONL to the differing spans first, then sheet that |
| Which branch produces the false alarms? | The default sheet; read the blocks |

## Judging a crop

**Size decides whether a verdict is supportable at all.**

- **Above ~15 px** — rotor arms, the body, a slung payload resolve. A call either way is safe.
- **4–8 px** — a blob. Do not claim it is clutter *or* a drone from appearance; say it is
  undetermined and move on. Writing "ambiguous" is a real answer and a cheap one.

**Two tells that cost nothing:**

- **A second identical unboxed object in the same crop** means the detector is picking scene
  features, not a target. This is what confirmed EXP-011's white blobs were buildings.
- **An object that holds still relative to terrain** across a span is part of the terrain,
  whatever it looks like. Pull a span and flip through it.

**Context matters more than the box.** Against sky a drone is an unmistakable silhouette.
Against terrain it is a dark smudge among dark smudges, and this is where mistakes get made
in both directions.

**The standing warning, from [datasets.md](../../../docs/datasets.md): do not trust a
negative from eyeballing.** A 14×11 px drone at frame 1350 of the field capture was scanned
by eye, recorded as empty, and the detector had boxed it correctly. "I looked and there was
nothing there" is not evidence at these sizes.

## What the branch colours mean

Only four branches can produce a box. The other three — `global miss`, `local miss`,
`first frame` — emit nothing and never appear.

| Branch | Colour | Meaning |
| --- | --- | --- |
| `global yolo` | yellow | GAD found it on the full frame, from cold, no prior lock |
| `local yolo` | green | LAD tracking inside the 320×320 search region |
| `global mod` | magenta | Motion candidate, LAD confirmed within 10 px → acquisition |
| `local mod` | blue | Locked on, LAD found nothing, motion carried the frame |

Read the **acquisition** branches first. `global yolo` and `global mod` are how a track
starts, they are rare — single digits in 3,600 frames — and they are where a run's behaviour
is decided. Everything green after them is consequence. On the field captures, a sustained
block of green says only that a lock held, not that it held on the right thing.

## After looking

- Save the sheet **into the run directory** beside `detections.jsonl`, so the evidence sits
  with what it describes. `runs/` is gitignored, so record in `docs/experiments.md` what the
  sheets are and where.
- Report counts as `N of M inspected`, with the seed. Never as a rate.
- If a sheet changes what a ledger entry claims, fix the entry — including an earlier one of
  your own. A wrong call left standing costs more than the correction does.
