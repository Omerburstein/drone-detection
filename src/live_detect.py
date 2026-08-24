"""Run GLAD against the drone's HDMI feed and show what it finds, on screen.

This is the deployment path the duty-cycle experiments were for. It differs from
`src.glad_detect` in the one way that matters: a file waits, and a feed does
not. GLAD sustains 2.67-3.60 fps on this host against a 30 fps downlink, so on
this hardware **most frames cannot be looked at**, and the only real question is
which ones to drop.

`--policy auto` answers that from measurement rather than preference. It warms
the pipeline up, times it against the feed's actual rate, and picks:

    full rate     if the machine keeps up. Nothing is dropped.
    half rate     if it keeps up with half the feed. Holds the tracking lock;
                  retained 91.5% of recall in EXP-008.
    burst pairs   otherwise. Two adjacent frames, then a sleep. The only policy
                  that keeps the motion branches coherent when the pipeline is
                  slower than the feed, at 36% of recall (EXP-009).

See `src.algo.deployment` for why that ordering is measured rather than guessed,
and `src.data.live` for why a feed needs a drain thread to deliver two genuinely
adjacent frames at all.

**`--source` takes either a capture device or a recording, and the difference is
not what it looks like.** The drone's HDMI downlink reaches this machine through
an HDMI-to-USB capture card, which enumerates as an ordinary UVC camera --
`--source 0`. Until that hardware exists, `--source flight.mp4` replays saved
footage **at the recording's own frame rate**, dropping every frame the pipeline
was too slow to collect exactly as a capture card would. That makes a replay a
simulation of the deployed system rather than an offline run: same policy, same
losses, same picture. `--no-realtime` turns the pacing off and is only useful
for checking that the window draws.

**What is on screen is a duty-cycled result and the window says so**, because a
box drawn without its policy invites comparison against a full-rate number it
cannot be compared to.

Example
-------
    py -3.13 -m src.live_detect --source 0                    # live HDMI capture
    py -3.13 -m src.live_detect --source data/raw/ARD-MAV/videos/phantom05.mp4
    py -3.13 -m src.live_detect --source 0 --benchmark-only   # measure, then exit
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

from .algo.deployment import (BURST_PAIRS, FULL_RATE, HALF_RATE, PolicyChoice,
                              choose_policy)
from .algo.glad.pipeline import GladPipeline
from .algo.glad.vendor import GLAD_DIR
from .algo.glad.yolo import PAD_STYLES
from .data.live import NATIVE_HEIGHT, NATIVE_WIDTH, SourceError, open_source
from .data.sampling import Bursts, EveryNth, Schedule
from .output import live_view
from .output.live_view import LiveStatus

GLAD_CONFIDENCE = 1.0  # GLAD emits no score; see StepResult.as_detections
WARMUP_PAIRS = 3  # discarded: the first torch call is far slower than the rest
BENCHMARK_PAIRS = 6  # timed, after the warm-up


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for the live detection path."""
    ap = argparse.ArgumentParser(prog="src.live_detect", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0",
                    help="A capture-device index (0, 1, ...) for the live HDMI "
                         "feed, or a path to saved drone footage to replay at its "
                         "own frame rate. Default 0.")
    ap.add_argument("--no-realtime", action="store_true",
                    help="Replay only: decode as fast as the pipeline asks rather "
                         "than at the recording's frame rate. **This stops it "
                         "being a real-time simulation** -- nothing is dropped, so "
                         "it becomes src.glad_detect with a window on it.")
    ap.add_argument("--loop", action="store_true",
                    help="Replay only: start the recording over when it ends.")
    ap.add_argument("--policy", choices=("auto", "full", "half", "burst"),
                    default="auto",
                    help="Duty-cycle policy. 'auto' (default) measures the "
                         "pipeline against the feed's rate and picks the most "
                         "accurate policy that is actually sustainable. The others "
                         "force a choice and print what it costs.")
    ap.add_argument("--source-fps", type=float, default=None,
                    help="Override the feed's reported frame rate. Capture cards "
                         "routinely report 0 or a nonsense value, and the policy "
                         "choice divides by this number.")
    ap.add_argument("--width", type=int, default=NATIVE_WIDTH,
                    help="Requested capture width (default 1920). GLAD's motion "
                         "constants are absolute pixels tuned for 1920x1080.")
    ap.add_argument("--height", type=int, default=NATIVE_HEIGHT,
                    help="Requested capture height (default 1080).")
    ap.add_argument("--max-width", type=int, default=1280,
                    help="Scale the window down to at most this wide (default "
                         "1280). Display only -- the detector always sees the "
                         "full-resolution frame.")
    ap.add_argument("--glad-repo", type=Path, default=GLAD_DIR,
                    help="Clone of the GLAD release holding weights/.")
    ap.add_argument("--pad", choices=sorted(PAD_STYLES), default="released",
                    help="Letterbox fill for the global detector. Defaults to "
                         "'released', matching EXP-004 to EXP-009, so the "
                         "on-screen recall caveat describes this code path.")
    ap.add_argument("--benchmark-only", action="store_true",
                    help="Measure throughput, report the policy that would be "
                         "chosen and what it costs, then exit without opening a "
                         "window. Use this to qualify a machine before flying.")
    return ap


def measure_throughput(pipeline: GladPipeline, source) -> float:
    """Sustained frames per second, after discarding the warm-up passes.

    Timed over feed pairs rather than a file read on demand, so it includes the
    capture and colour-conversion cost a deployment actually pays. The warm-up
    matters more than it looks: the first forward pass through a freshly loaded
    checkpoint takes several times the steady-state cost, and letting that into
    the average picks a worse policy than the machine deserves.
    """
    seen = 0
    for _ in range(WARMUP_PAIRS):
        pair = source.pair(after=seen)
        seen = pair.index
        pipeline.reset()
        pipeline.step(pair.previous)
        pipeline.step(pair.current)

    frames = 0
    start = time.perf_counter()
    for _ in range(BENCHMARK_PAIRS):
        pair = source.pair(after=seen)
        seen = pair.index
        pipeline.reset()
        pipeline.step(pair.previous)
        pipeline.step(pair.current)
        frames += 2
    elapsed = time.perf_counter() - start
    return frames / elapsed if elapsed > 0 else 0.0


def forced_policy(name: str, measured_fps: float, source_fps: float) -> PolicyChoice:
    """The policy the operator asked for, described in the same terms as `auto`."""
    period = max(3, round(source_fps * 2 / max(measured_fps, 1e-9)))
    schedules: dict[str, tuple[str, Schedule]] = {
        "full": (FULL_RATE, Schedule()),
        "half": (HALF_RATE, EveryNth(2)),
        "burst": (BURST_PAIRS, Bursts(2, period)),
    }
    policy_name, schedule = schedules[name]
    required = {FULL_RATE: source_fps, HALF_RATE: source_fps / 2,
                BURST_PAIRS: 0.0}[policy_name]
    note = ("" if measured_fps >= required else
            f" This machine sustains {measured_fps:.2f} fps and cannot hold it, "
            f"so the feed will fall behind and the picture will drift into the "
            f"past.")
    return PolicyChoice(policy_name, schedule,
                        f"Forced by --policy {name}.{note}",
                        measured_fps, source_fps, required)


def report_policy(policy: PolicyChoice, source, label: str) -> None:
    """Print the choice and its consequences before anything is displayed."""
    width, height = source.frame_size
    print(f"\nSource: {label} -- {width}x{height} at {policy.source_fps:.0f} fps")
    if (width, height) != (NATIVE_WIDTH, NATIVE_HEIGHT):
        print("  !! Not 1920x1080. GLAD's motion constants are absolute pixels "
              "tuned for 1080p, so the motion branches are running at the wrong "
              "scale and recall is not comparable to any recorded experiment.")
    print(f"Sustained: {policy.measured_fps:.2f} fps")
    print(f"\nPolicy: {policy.name}")
    print(f"  {policy.reason}")
    print(f"  Retains {policy.retention:.1%} of full-rate recall (measured, ARD100).")

    period = policy.burst_period_seconds
    if period is not None:
        delay = policy.expected_detection_delay
        print(f"  Burst period {period:.2f}s; at {policy.retention:.0%} per burst "
              f"the mean delay to a first detection is {delay:.1f}s.")
        for speed in (20.0, 40.0):
            print(f"    a target closing at {speed:.0f} m/s covers "
                  f"{policy.closing_distance(speed):.0f} m in that time")


def run_loop(pipeline: GladPipeline, source, policy: PolicyChoice,
             max_width: int) -> Counter:
    """Drive the pipeline from the feed and draw every processed frame."""
    branches: Counter = Counter()
    schedule = policy.schedule
    seen = processed = dropped = 0
    started = time.perf_counter()

    while True:
        pair = source.pair(after=seen)
        # Frames that arrived while the last pass was thinking were never looked
        # at. Counting them is what makes the displayed duty cycle honest.
        dropped += max(0, pair.index - seen - 1)
        seen = pair.index

        if policy.name == BURST_PAIRS:
            # Exactly EXP-007/EXP-009: cold start, then one usable frame. The
            # reset is what stops the motion branches differencing across the
            # sleep between bursts.
            pipeline.reset()
            pipeline.step(pair.previous)
            result = pipeline.step(pair.current)
            processed += 2
        else:
            # Keyed on the feed's own frame number, not a local counter: the
            # schedule is a statement about the *source*, and a counter that
            # advanced once per loop would drift into meaning something else the
            # moment a pass ran long.
            if not schedule.wants(pair.index):
                dropped += 1
                continue
            result = pipeline.step(pair.current)
            processed += 1

        branches[result.branch] += 1
        elapsed = time.perf_counter() - started
        status = LiveStatus(
            policy=policy,
            branch=result.branch,
            detections=result.as_detections(GLAD_CONFIDENCE),
            processed=processed,
            dropped=dropped,
            sustained_fps=processed / elapsed if elapsed > 0 else 0.0,
            latency=pair.age,
        )
        if not live_view.show(live_view.render(pair.current, status, max_width)):
            return branches


def print_branches(branches: Counter) -> None:
    """Which branch produced each processed frame, as `src.glad_detect` reports."""
    total = max(sum(branches.values()), 1)
    print("\nBranch that handled each frame:")
    for branch, count in branches.most_common():
        print(f"  {branch:<14} {count:>7}  {100 * count / total:5.1f}%")


def main() -> None:
    """Open the feed, choose a policy, and show what the detector finds."""
    args = build_parser().parse_args()

    # Checked before the checkpoints load, so a headless OpenCV is reported in a
    # second rather than after a 40-second model load.
    if not args.benchmark_only:
        try:
            live_view.require_display()
        except live_view.DisplayUnavailable as exc:
            sys.exit(f"\n{exc}\n\nOr pass --benchmark-only to measure without a "
                     f"window.")

    print(f"Loading GLAD from {args.glad_repo} ...")
    print(f"Letterbox fill: {args.pad} ({PAD_STYLES[args.pad]})")
    pipeline = GladPipeline.from_release(args.glad_repo, PAD_STYLES[args.pad])

    branches: Counter = Counter()
    try:
        with open_source(args.source, realtime=not args.no_realtime,
                         loop=args.loop, width=args.width,
                         height=args.height) as source:
            source_fps = args.source_fps or source.source_fps
            print(f"Measuring sustained throughput over {BENCHMARK_PAIRS} pairs ...")
            measured = measure_throughput(pipeline, source)

            policy = (choose_policy(measured, source_fps) if args.policy == "auto"
                      else forced_policy(args.policy, measured, source_fps))
            report_policy(policy, source, source.label)

            if args.benchmark_only:
                return

            print("\nRunning. Press q or Esc in the window to stop.")
            pipeline.reset()
            branches = run_loop(pipeline, source, policy, args.max_width)
    except SourceError as exc:
        sys.exit(f"\n{exc}")
    finally:
        live_view.close()

    print_branches(branches)


if __name__ == "__main__":
    main()
