# `src.live_detect` — reference

Runs GLAD against the drone's HDMI feed and shows what it finds on screen, choosing
a duty-cycle policy from measured throughput.

```
py -3.13 -m src.live_detect [--source 0|PATH] [options]
```

> **This is the only path in the project where accuracy is not the whole story.**
> `src.glad_detect` reads a file, which waits. A feed does not. GLAD sustains
> 2.67–3.60 fps on this host against a 30 fps downlink, so most frames cannot be
> looked at, and *which* frames get dropped decides what is detected as much as the
> detector does. See [edge-budget.md](edge-budget.md) §3 and the duty-cycle
> experiments EXP-006 to EXP-009 in [experiments.md](experiments.md).

## Parameters

| Parameter | Default | What it does |
| --- | --- | --- |
| `--source` | `0` | A capture-device index for the live HDMI feed, or a path to saved footage to replay. See "Where the video comes from" below. |
| `--no-realtime` | off | Replay only: decode as fast as the pipeline asks. **Stops it being a real-time simulation** — nothing is dropped. |
| `--loop` | off | Replay only: restart the recording when it ends. |
| `--policy` | `auto` | `auto`, `full`, `half`, or `burst`. `auto` measures and picks; the rest force a choice and print what it costs. |
| `--source-fps` | reported | Override the feed's frame rate. Capture cards routinely report 0 or nonsense, and the policy choice divides by this. |
| `--width` / `--height` | `1920` / `1080` | Requested capture geometry. GLAD's motion constants are absolute pixels tuned for 1080p. |
| `--max-width` | `1280` | Scale the *window* down to fit a screen. The detector always sees the full-resolution frame. |
| `--pad` | `released` | Letterbox fill. Defaults to `released`, matching EXP-004→EXP-009, so the on-screen recall caveat describes this code path. |
| `--glad-repo` | `third_party/GLAD` | Clone holding `weights/`. |
| `--benchmark-only` | off | Measure, report the policy and its cost, exit without a window. Use it to qualify a machine before flying. |

## Where the video comes from

The drone's downlink arrives over **HDMI**. To be processed it has to enter this
machine through an **HDMI-to-USB capture card**, which enumerates as an ordinary UVC
camera — so `--source 0` opens it exactly like a webcam.

**No capture hardware exists yet**, so the path that runs today is the other one:

```
py -3.13 -m src.live_detect --source data/raw/ARD-MAV/videos/phantom05.mp4
```

`--source <path>` replays saved footage **at the recording's own frame rate**, and
that pacing is the entire point. Frames the pipeline was too slow to collect are
dropped exactly as a capture card would drop them, so a replay is a *simulation of
the deployed system* rather than an offline run with a window on it — same policy,
same losses, same picture. Measured on `phantom05.mp4` with a 0.35 s/frame stand-in:

```
source: replay of phantom05.mp4  1920x1080 @ 29.97 fps
  pass 0: frame    2  jumped   2   age 351.2 ms
  pass 1: frame   16  jumped  14   age 387.9 ms
  pass 2: frame   27  jumped  11   age 372.0 ms
  pass 4: frame   48  jumped  10   age 377.6 ms
produced by feed: 59, consumed: 5
```

**The age column is the check that matters.** It stays flat at ~370 ms instead of
growing, which means the feed is discarding what nobody collected rather than
queueing it. A naive `read()` loop shows an age that climbs without bound while
appearing to work — see [`src/data/live.py`](../src/data/live.py) for the two ways
that fails silently.

`--no-realtime` disables the pacing. Nothing is then dropped, which makes the run
equivalent to `src.glad_detect` and the on-screen duty cycle meaningless. It is for
checking that the window draws, not for results.

## Choosing the policy

`--policy auto` warms the pipeline up, times it over six pairs, and takes the most
accurate policy the machine can actually sustain:

| Sustained | Policy | Retains | Why |
| --- | --- | --- | --- |
| ≥ source fps | `full rate` | 100% | Nothing needs dropping. |
| ≥ half source fps | `half rate` | **91.5%** | Holds the tracking lock. EXP-008: 0.6290 against 0.6876 on ARD100. |
| below that | `burst pairs` | **36.0%** | EXP-009: 0.2496 against 0.6934. The only policy that stays coherent — see below. |

**Why burst pairs is the right fallback and not just the lossy one.** When the
pipeline is slower than the feed, frames must be dropped and the only question is
how. Taking whatever frame is newest whenever the pipeline comes free leaves gaps
that are large, variable and unknown — which is plain striding, ruled out in
[edge-budget.md](edge-budget.md) §3 item 7 because both of GLAD's motion branches
difference against the previous frame. Grabbing two *genuinely adjacent* frames
keeps that difference meaningful however long the sleep between pairs.

The thresholds are measured against the **source** rate, not a fixed target: the
same machine deserves a different policy on a 30 fps feed than on a 60 fps one.

Retention figures are quoted from ARD100, the split GLAD never trained on, because
the duty-cycle penalty is consistently worse on unseen video (36% against ARD-MAV's
48%) and quoting the friendlier number would flatter the fallback.

### The latency the fallback costs

Burst pairs detects on one frame per burst at a measured 25%, so the delay to a
first detection is what disqualifies it, not the recall:

```
Policy: burst pairs
  Burst period 0.70s; at 36% per burst the mean delay to a first detection is 2.8s.
    a target closing at 20 m/s covers 56 m in that time
    a target closing at 40 m/s covers 112 m in that time
```

That is printed before the window opens, because a recall number without its delay
is only half the deployment question.

## What is on screen

- The frame, scaled to fit, with the detection outlined and **pushed outward** so
  the box does not cover a 10–30 px target.
- A **magnified inset** of the detection, top right. Without it a drone at this
  scale is two or three screen pixels and the window cannot be judged.
- A HUD carrying `TRACKING`/`SEARCHING`, the branch that produced the frame, the
  policy in force, sustained fps, duty cycle, and the lag between capture and
  display.
- **A caveat line naming what the policy cost when it was measured.** A box drawn
  without its policy invites comparison against a full-rate number it cannot be
  compared to.

Press `q` or `Esc` to stop. The branch summary prints on exit, in the same format
`src.glad_detect` uses.

## Requirements

**`opencv-python-headless` must not be installed.** It shares the `cv2` namespace
with `opencv-python`, and whichever wins the import decides whether a window can
open at all. The headless build still *exposes* `imshow` and raises when it is
called, so the failure looks like a bug in this code and is not one:

```
py -3.13 -m pip uninstall -y opencv-python-headless
py -3.13 -m pip install --force-reinstall opencv-python
```

`src.live_detect` checks for this **before loading the checkpoints** and prints the
fix, rather than failing after a 40-second model load. `--benchmark-only` needs no
window and runs either way.

## Limitations

- **Never run against live hardware.** No HDMI capture card exists; everything here
  is verified against replayed recordings and unit tests. The `HdmiCapture` path is
  written and tested against a stubbed device, not a real one.
- **A capture card reports the link, not the sensor.** It will happily say
  1920×1080 while the drone upstream sends an upscaled 720p picture, and nothing in
  software can tell. The resolution check is necessary, not sufficient.
- **Retention figures are ARD-MAV/ARD100 numbers**, not numbers from this drone's
  camera. [edge-budget.md](edge-budget.md) §4.3 projects that a low-resolution
  wide-FOV feed drops recall to near zero on optics alone, so a policy that retains
  91.5% of recall retains 91.5% of *whatever this camera supports*, which is
  unmeasured.
- **GLAD reports at most one target.** `MOD2_global` breaks on its first accepted
  candidate; the window can never show two drones.
