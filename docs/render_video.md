# `src.render_video` — reference

Re-renders one source video with **both boxes on it**: the ground-truth drone position
and what the detector claimed, coloured by whether the two matched.

```
py -3.13 -m src.render_video --video <mp4> --pred <detections.jsonl> \
    --labels <labels dir> --out <mp4> [options]
```

Nothing is re-inferred. Boxes come from a run's persisted `detections.jsonl` and the
match outcome from `src.eval.metrics.match_frame` — the same function `src.eval.records`
and `src.eval.curves` call — so **the video cannot disagree with the numbers in the
ledger**, and rendering costs a video decode rather than an inference pass. Re-render the
same run under a different `--match` and you are looking at the same detections judged by
a different rule, not at a different detector.

## Parameters

| Parameter | Default | What it does |
| --- | --- | --- |
| `--video` | required | Source `.mp4`. Its stem is the frame-key prefix unless `--key-prefix` says otherwise. |
| `--pred` | required | `detections.jsonl` from a recorded run (`src.glad_detect`, `src.baseline_detect`). |
| `--labels` | required | Directory of YOLO-format `.txt` labels for the split. |
| `--out` | required | Output `.mp4`. Parent directories are created. |
| `--key-prefix` | the video's stem | Which frames of `--pred` belong to this video, e.g. `phantom19`. |
| `--match` | `center` | How a prediction claims a target, exactly as in `src.evaluate`. Decides only the **colour** of a box, never whether it is drawn. |
| `--iou` | `0.5` | IoU threshold for `--match iou`. |
| `--match-tol` | `1.0` | Centre tolerance for `--match center`, in target sizes. |
| `--zoom` | `5` | Magnification of the inset. `0` draws no inset. |
| `--zoom-span` | `110` | Side of the source window the inset magnifies, in original pixels. |
| `--no-caption` | off | Drop the bottom caption strip and colour legend. |
| `--fps` | the source's | Output frame rate. Lower it to slow the footage down. |
| `--max-frames` | none | Stop after N rendered frames — a contiguous prefix, for checking the overlay before committing to the whole video. |

## What you are looking at

| Colour | Meaning |
| --- | --- |
| **green** | ground truth, claimed by some prediction |
| **orange** | ground truth that nothing claimed — a miss |
| **blue** | a prediction that claimed a target |
| **red** | a prediction that claimed nothing — a false alarm |

Ground-truth boxes are drawn 5 px outside the true box and predictions 2 px outside,
because on a correct detection of a 10 px drone the two boxes coincide and a single
rectangle would be all you could see. Neither rectangle is anti-aliased, for the same
reason: at this scale a blended edge is indistinguishable from the target.

The **inset**, top right, is a nearest-neighbour magnification of a `--zoom-span` square
centred on the ground truth (on the prediction when there is no ground truth, since a
false alarm on an empty frame is the case worth looking at). The white square on the full
frame is where it came from. Nearest-neighbour rather than smooth interpolation is
deliberate — the inset is meant to show what is actually in those pixels, not to invent
plausible detail on a target nine of them across. Magnification is reduced automatically
if the inset would not fit the frame, and the corner reads back the factor actually used.

The **caption strip** carries the frame key, the box counts, this frame's TP/FP/missed
tally, the matching rule in force, and every extra field the run recorded — for
`src.glad_detect` that is `branch`, the state-machine path that produced the box, so a
run can be watched switching between acquisition and tracking.

## Frame keys

Frames are keyed `<video stem>_<1-based frame number, 4 digits>` — what
`src.data.prepare_ardmav` named them — so the `.mp4` and the run's JSONL line up with no
manifest. Decoding is sequential from the first frame for that reason; seeking would be
faster but is inexact on some codecs, which would silently shift every label by a frame
or two.

Frames the run never recorded are **skipped, not drawn blank**: the output is exactly the
frames that were measured. ARD-MAV videos carry an unannotated tail (28 frames on
phantom19), and drawing them would show boxless footage that no metric ever saw. The
count is printed at the end.

## Cost

Two decodes' worth of work and no inference, but two things dominate:

- **Reading the labels.** ~200 s for phantom19's 2,158 label files on this machine, which
  is most of the run. `--labels` points at a directory of 28,337 small files and Windows
  charges roughly 90 ms each; nothing in the renderer can fix that.
- **Encoding.** `mp4v` at 1080p writes ~120 KB per frame — phantom19's 2,158 frames come
  out at **251 MB** for 72 seconds. `runs/` is gitignored, so these stay local; re-render
  rather than archiving them.

Rendering itself runs faster than real time: 2,158 frames of 1080p in about a minute.

## Example — EXP-004 on phantom19

`phantom19` is `small_mav`, the category GLAD scores worst on (P 0.642 / R 0.522 at
IoU 0.50) and the worst-lit video in the split — 18.8% of its targets sit within 5 grey
levels of their own background. It is the video to watch to see what those numbers mean.

```
py -3.13 -m src.render_video \
    --video data/raw/ARD-MAV/videos/phantom19.mp4 \
    --pred runs/exp004_glad/detections.jsonl \
    --labels data/processed/ARD-MAV/labels/test \
    --out runs/exp004_glad/examples/phantom19_overlay.mp4
```

Add `--match iou --iou 0.5` to watch the same detections under COCO's rule instead: the
boxes do not move, but blue-on-green turns red-on-orange wherever the centre is a couple
of pixels off — which is the EXP-004 finding, visible frame by frame rather than inferred
from a table.
