"""Which frames a duty-cycled run actually processes, and where its state resets.

A camera delivers 30 fps; GLAD sustains 2.67 on this host. The obvious response
is to process fewer frames — but *which* fewer is not a free choice, because
both of GLAD's motion branches difference the current frame against the previous
one. Plain striding (frame `n`, then frame `n+10`) hands the motion module two
frames a third of a second apart and is rejected outright in
`docs/edge-budget.md` §3 item 7. That objection is about the *gap*, not about
processing fewer frames, and two schemes dodge it:

    nth      run the whole pipeline at a lower rate. Adjacent processed frames
             are still adjacent to each other, merely further apart in time.
    burst    process a short contiguous run of frames, then sleep. Differencing
             happens *inside* the burst, where the frames are a true 1/30 s
             apart, and the sleep costs opportunity rather than coherence.

Both trade detection opportunity for duty cycle, and each breaks differently:
`nth` doubles apparent inter-frame motion against thresholds tuned in absolute
pixels at 30 fps, while `burst` can never enter the local regime because a lock
does not survive the sleep.

A schedule is a pure function of the frame index — no decoding, no model, no
I/O — so the decode loop stays a loop and the policy is testable on its own.
Frame indices are **one-based**, matching the label numbering that
`prepare_ardmav` wrote and that `src.glad_detect` counts with.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

EVERY = "every"
NTH = "nth"
BURST = "burst"


@dataclass(frozen=True)
class Schedule:
    """Base policy: process every frame, reset only at the start of a video.

    `wants` decides whether a frame reaches the detector; `resets` decides
    whether the pipeline's carried state — crucially `_prev`, the frame the
    motion branches difference against — is cleared before it does.
    """

    def wants(self, index: int) -> bool:  # noqa: ARG002 -- subclasses use it
        """Whether frame `index` (one-based) is processed at all."""
        return True

    def resets(self, index: int) -> bool:  # noqa: ARG002 -- subclasses use it
        """Whether pipeline state must be cleared before processing `index`."""
        return False

    @property
    def label(self) -> str:
        """Short description of the policy, for the run log and the ledger."""
        return "every frame"

    def position(self, index: int) -> dict[str, int]:  # noqa: ARG002 -- as above
        """Where `index` sits in the schedule, recorded on the frame's JSONL row.

        Empty for the contiguous policies, which have no structure to record.
        `src.eval.labels` carries this through to the dump, so a cut like
        "second frame of each burst only" is a filter on a CSV rather than a
        second run.
        """
        return {}


@dataclass(frozen=True)
class EveryNth(Schedule):
    """Run the whole pipeline at 1/n of the source rate.

    The stream stays contiguous in the only sense the motion module cares
    about — every processed frame's predecessor is the previously processed
    frame — so `resets` is never true. What changes is the interval between
    them, and with it the apparent displacement of both the target and the
    background. At n=2 that displacement doubles against `dist_ref=200`,
    `TrackingDetector.MAX_DISTANCE=50` and a blob-area window tuned for 30 fps.
    """

    n: int

    def wants(self, index: int) -> bool:
        """True on every nth frame, counting the first frame of the video."""
        return (index - 1) % self.n == 0

    @property
    def label(self) -> str:
        """Names the rate rather than the stride, which is what deploys."""
        return f"1 frame in {self.n} ({100 / self.n:.1f}% duty, 1/{self.n} rate)"


@dataclass(frozen=True)
class Bursts(Schedule):
    """Process `length` consecutive frames once every `period` frames.

    The sleep between bursts is the whole point and also the hazard: `_prev`
    would otherwise hold a frame from the previous burst, seconds of ego-motion
    ago, and `MOD2_global` would difference two unrelated scenes into a field of
    motion blobs. Those blobs are an artefact of the harness, not a property of
    the scheme, so **`resets` fires on the first frame of every burst** and the
    pipeline starts each one cold.

    Starting cold has a consequence worth stating plainly rather than
    discovering in the numbers: `GladPipeline.step` returns an empty result for
    the first frame after a reset, because there is nothing to difference. A
    burst of `length` frames therefore offers `length - 1` chances to detect,
    and at the default length of 2 that is exactly one — spent in the global
    regime, since a lock needs frames the burst does not have.
    """

    length: int
    period: int

    def __post_init__(self) -> None:
        """Reject a schedule that is not actually duty-cycled."""
        if self.length < 2:
            sys.exit("--burst-length must be at least 2: the first frame after a "
                     "reset has nothing to difference against and can never detect.")
        if self.period <= self.length:
            sys.exit(f"--burst-period ({self.period}) must exceed --burst-length "
                     f"({self.length}), otherwise the bursts run together and the "
                     f"schedule is just every frame.")

    def wants(self, index: int) -> bool:
        """True for the first `length` frames of each `period`-frame cycle."""
        return (index - 1) % self.period < self.length

    def resets(self, index: int) -> bool:
        """True on the opening frame of every burst, including the first."""
        return (index - 1) % self.period == 0

    @property
    def label(self) -> str:
        """Burst geometry plus the duty cycle it implies."""
        return (f"{self.length}-frame bursts every {self.period} frames "
                f"({100 * self.length / self.period:.2f}% duty)")

    def position(self, index: int) -> dict[str, int]:
        """Which burst this frame belongs to, and where it sits inside it."""
        return {"burst": (index - 1) // self.period,
                "in_burst": (index - 1) % self.period}


def build_schedule(mode: str, nth: int, burst_length: int, burst_period: int) -> Schedule:
    """The schedule named by the CLI flags.

    `every` returns the base policy, so the default path is the one EXP-004 ran
    and nothing about a duty-cycled option changes it.
    """
    if mode == EVERY:
        return Schedule()
    if mode == NTH:
        if nth < 1:
            sys.exit("--sample-n must be at least 1.")
        return EveryNth(nth)
    if mode == BURST:
        return Bursts(burst_length, burst_period)
    sys.exit(f"Unknown --sample {mode!r}. Known: {EVERY}, {NTH}, {BURST}")
