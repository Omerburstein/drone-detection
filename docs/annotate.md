# `src.data.annotate` — reference

Annotate a video by hand in a window: play, pause and step; drag a box around the drone;
move or resize it. From then on **the box follows the target** as the video advances, and
you only step in when it is wrong. Your correction becomes the new template. When the
follower loses the target, playback stops and tells you.

```
py -3.13 -m src.data.annotate --video <avi|mp4> --labels-out <dir> --verified-out <jsonl>
```

Writes exactly what [`src.data.seed_track`](seed_track.md) writes (YOLO labels plus
`verified.jsonl` for `src.evaluate --keys-from`), plus a session file so you can stop and
resume. Use this one for real annotation work. `seed_track` is for the case where you
already know a seed box's coordinates and want no window.

---

## Parameters

| Parameter | Default | What it does |
| --- | --- | --- |
| `--video` | required | Clip to annotate. Use the **processed** one runs are made on (e.g. the cropped FFV1 O4 clips), so the boxes share its coordinate system. |
| `--labels-out` | required | Label directory, e.g. `data/processed/SOFA-O4/labels/test`. |
| `--verified-out` | required | JSONL of every judged frame, for `src.evaluate --keys-from`. |
| `--stem` | the video's stem | Frame-key prefix: labels are `<stem>_<frame:04d>.txt`, 1-based. |
| `--session` | `<dataset>/annotations/<stem>.json` | Stop/resume file. `<dataset>` is two levels above `--labels-out`. |
| `--images-dir` | labels' sibling `images/<split>` | Path the verified rows are keyed by. Nothing is read from it. |
| `--start` | last judged frame, else 1 | 1-based frame to open on. |
| `--fps` | 10 | Initial playback speed. `+`/`-` change it live (1, 2, 5, 10, 15, 30, 60). |
| `--max-window W H` | 1400 850 | Largest window, status bar included. The frame is scaled down to fit, never up. |
| `--buffer` | 90 | Decoded frames held for instant stepping back. About 4.7 MB each at 1440×1080. |
| `--min-score` | 0.45 | The follower stops below this correlation. Same measured default as `seed_track`. |
| `--search` | 4.0 | Follower search window, in target sizes, around where motion predicts the target. |

---

## Keys and mouse

| Input | Does |
| --- | --- |
| `space` | Play / pause |
| `d` or `→` | Next frame (the box follows) |
| `a` or `←` | Previous frame (fills only frames with no verdict yet) |
| `D` / `A` | Jump ±30 frames. **Nothing is labelled** in the skipped frames. |
| click the timeline | Jump there. Also labels nothing. |
| drag on empty picture | New box |
| drag the box body / edge / corner | Move / resize it. This is a correction and re-seeds the follower. |
| `x` or `Del` | No target in this frame. It **carries forward** while you play. |
| `c` | Clear this frame's verdict (back to unjudged) |
| mouse wheel | Zoom at the cursor (up to 16×) |
| right-drag | Pan |
| `f` / `0` | Zoom to the box / back to the whole frame |
| `s` | Save and export |
| `q` / `Esc` / close the window | Save, export, quit |
| `h` | Key help on screen |

Colours: **green** is a box you placed, **yellow** a box the follower proposed (with its
score), **cyan** a box being dragged, and a **red banner** means no target. The strip above
the status line is the whole video: green where boxes are, red where frames are confirmed
empty, dark where nothing has been judged. The dark stretches are what's left to do.

---

## The workflow

1. Open the clip and find the first frame where the drone is visible. Zoom in (wheel)
   before drawing. A 12 px drone fitted to a laptop window is 9 px across.
2. Drag a tight box. Press `space`.
3. Watch. When the yellow box slips, pause, drag it back onto the drone, and play on.
   When the follower loses the target itself, it pauses for you.
4. When the drone leaves the frame, press `x` and keep playing. The empty span labels
   itself as you watch it.
5. `q` when done. Rerunning the same command resumes where you stopped.

While you're zoomed in, the view re-centres on the box whenever it leaves the middle of the
window, so the target doesn't walk out of view during playback.

---

## How the box follows the target, and why not by frame differencing

**Correlation against the box you drew, plus a constant-velocity motion prior.** Each step
predicts where the target has gone from its last displacement, then searches a window of
`--search` target sizes around that prediction for the best match to your box's
appearance. The tracker core is `seed_track.TemplateTracker`, with its pinned-template
default (`seed_track.md` records why blending is a trap).

The follower does not use frame differencing. A difference image tells you *something*
moved, but on this footage it also holds ~100 other blobs from the terrain per frame
(EXP-012b), and it cannot say which of them is the box you drew. Correlation against your
box can. Motion is still used, as a **prior on where to look**. `tests/unit/test_annotation.py` pins what it
buys: at 60 px/frame, correlation alone loses the target on the first step and the prior
holds it every frame.

**The template is always the last box *you* placed**, never a tracked one. Every correction
re-seeds it, so the appearance stays current exactly as fast as you correct. After a
correction, the next step does not extrapolate velocity from it: the jump from a wrong
proposal to your fix measures the mistake, not the target's speed.

### Featureless boxes are refused (measured)

Normalised correlation is contrast-invariant: it stretches a flat patch to full contrast
before comparing. On `first_catch`, a box drawn on empty sky (grey std **1.1**) "followed"
across open sky for 20 frames at scores of **~0.8**, the same range a real drone scores.
The 80×34 drone at frame 962 has a std of **46.9**. So:

- a box whose grey std is under **4** is recorded but not followed. The status bar asks
  whether the drone is inside it, and the next step reports the target lost;
- a match with less than **0.35×** the template's contrast is rejected as sky.

### What to expect on real footage

Backward from the 80×34 drone at frame 962 of `first_catch`, the follower held it through
frame 955, with scores falling 0.94 → 0.47 as the drone recedes, and stopped at 954 rather
than guess. **Expect a correction every few to few tens of frames** on intercept footage,
where the target grows and rotates fast. That's the design: you draw a box where the
follower gives up, rather than on every frame.

---

## What gets written

- **`labels/<split>/<stem>_<frame>.txt`**: one YOLO box, class 0, per frame holding one.
- **`verified.jsonl`**: every frame with a verdict, box *or* confirmed-empty. Pass it to
  `src.evaluate --keys-from`; frames you never judged are not scored, so they are never
  silently counted as negatives.
- **`annotations/<stem>.json`**: the session. Each frame carries `source` (`human`,
  `tracked` or `carried`) and the follower's `score`, so a later review can pull out
  exactly the frames no human placed.

Saved on `s`, on quit, and every 30 s while there are unsaved changes. The write is atomic.

> **The session owns its stem.** On export, label files and `verified.jsonl` rows for this
> stem that the session doesn't hold are **deleted**. That's what makes deleting a box
> actually delete it, but it also means you shouldn't mix `seed_track` and `annotate` on
> the same clip: the annotator's export would drop `seed_track`'s rows for it. Other stems
> are never touched.

A session made on a video of different dimensions is refused. That would be the raw
2520×1080 goggles recording against its 1440×1080 crop, and every box would be misplaced.

---

## Performance and exactness

Playback decodes sequentially. A seek costs ~40× a sequential read on `first_catch`
(1,109 ms against 26 ms), so the frame cache avoids seeking wherever it can:

- stepping back within `--buffer` frames is instant;
- a jump forward of up to 60 frames decodes through instead of seeking;
- stepping back past the cache seeks **half a cache earlier** and decodes forward, so the
  following steps are already held. Measured walking back from frame 962: 192 ms/step
  including the refill, against 454 ms when every step sought.

Decoding through is also what keeps **long-GOP mp4** exact. A seek there can land a
keyframe away from the frame asked for, and a box on the wrong frame is a wrong label. Only
far jumps and long backward walks seek, so prefer annotating the processed FFV1 copy when
one exists. The O4 clips are FFV1 (all-intra), where every seek is exact.

Tracking, rendering and decoding together run at ~75 fps on 1440×1080, well above any
playback speed you'd use.
