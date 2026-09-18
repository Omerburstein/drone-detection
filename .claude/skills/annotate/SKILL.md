---
name: annotate
description: Open the interactive video annotator so the user can label drones by hand — play/pause/step, drag a box, move and resize it, and let it follow the target between corrections. Writes YOLO labels, verified.jsonl and a resumable session. Use when the user asks to "annotate", "label the video", "mark the drone", "draw boxes", "make ground truth", "continue annotating", or wants footage labelled so a run on it can be scored. Not for scoring (use /eval) or for looking at unlabelled detections (use /inspect).
---

# annotate

One tool, [`src.data.annotate`](../../../src/data/annotate.py), reference in
[annotate.md](../../../docs/annotate.md). **The user does the labelling; you set it up,
launch it, and read back what came out.** You cannot see or click the window.

## 1. Resolve the paths before launching

| What | Rule |
| --- | --- |
| `--video` | The **processed** clip runs were made on, e.g. `data/processed/SOFA-O4/videos/<stem>.avi`. Its boxes must share that run's coordinate system. Resolve a bare stem with `ls data/processed/*/videos/`. |
| `--labels-out` | `data/processed/<SET>/labels/test`, the tree `src.evaluate` reads. |
| `--verified-out` | `data/processed/<SET>/verified.jsonl`. |
| session | Defaults to `data/processed/<SET>/annotations/<stem>.json`. **If it exists this is a resume**: say so, with the frame count from it. |

Refuse or warn, don't guess:

- **Never write under `data/raw/`.** It is immutable. A raw-only clip (the FIELD mp4)
  still gets `--labels-out data/processed/FIELD/labels/test`.
- **Long-GOP mp4** (the FIELD capture): seeks may land off-frame. Playing and stepping are
  exact, far jumps may not be. Tell the user to label by playing through, not by clicking
  around the timeline.
- **The session owns its stem.** If `seed_track` labels already exist for this stem
  (`ls <labels-out>/<stem>_*.txt` with no session file), the annotator's export will delete
  them. Ask before launching.

## 2. Launch in the background

It blocks until the window closes, so always `run_in_background`:

```bash
py -3.13 -m src.data.annotate \
    --video data/processed/SOFA-O4/videos/first_catch.avi \
    --labels-out data/processed/SOFA-O4/labels/test \
    --verified-out data/processed/SOFA-O4/verified.jsonl
```

Add `--start N` when the user named a passage. Then give them the short version of the
keys, not the whole table:

> Wheel to zoom in, then drag a box around the drone. `space` plays, and the box follows.
> When it slips, pause and drag it back. When the drone leaves, press `x` and keep playing,
> which marks the empty frames. `d`/`a` step, `h` shows every key, `q` saves and quits.
> Rerunning resumes.

Don't wait on it or poll it. You'll be notified when the window closes.

## 3. When it exits

The process prints the counts. Then report, from the session file:

- boxes and confirmed-empty frames, and **how many boxes a human placed vs the follower
  proposed** (`source` per frame);
- the **lowest-scoring tracked frames** (sort by `score` ascending, show ~5). Those are
  where the follower was least sure, so they're what to check if anything is off;
- the frame ranges still unjudged within the passage they were working on.

**Don't call the labels verified on the follower's word.** The user watched every tracked
frame play, which is a review, but at 30 fps it's a fast one. If tracked boxes dominate and
the scores go low, suggest a slower pass over those frames before scoring.

Then hand off: scoring a run against these labels is `/eval`, with
`--keys-from <verified.jsonl>` so only judged frames count.

## Things the user should hear once, not every time

- **Zoom before drawing.** A 12 px drone fitted to a laptop window is 9 px across, and a
  box edge placed at that scale is off by a large fraction of the target.
- **Label the empty frames too** (`x`, then play). Without negatives only frames with a
  drone are scored, and a detector that fires constantly looks perfect.
- **Extend past where the detector fired.** Annotating only the spans a detector found
  scores a set chosen by the thing being scored.
- **"Featureless — is the drone inside it?"** means the box was drawn on flat sky. The
  follower refuses those, because correlation matches flat sky everywhere at ~0.8.

## Recording

Labels are data, so `data/` is gitignored and nothing here is committed. Once a clip is
labelled, note it in [datasets.md](../../../docs/datasets.md): which clip, which frames,
how many boxes, and how many were hand-placed. A label set that exists only on this laptop
is not a record.
