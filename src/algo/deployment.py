"""Choosing a duty-cycle policy from measured throughput and measured accuracy.

This is where the experiments become a deployment decision. EXP-006 to EXP-009
scored two ways of running GLAD on fewer frames than the camera delivers, each
against a full-rate control over the *same* frames, so the cost of each policy
is known rather than assumed:

    half rate    process every 2nd frame. Keeps the lock, so the local regime
                 still carries the run. **Retains 91.5% of recall** (0.6290
                 against 0.6876 on ARD100, EXP-008) for roughly half the frames.
    burst pairs  process two consecutive frames, then sleep. Cannot hold a lock
                 across the sleep, so every burst is a cold acquisition.
                 **Retains 36% of recall** on unseen video (0.2496 against
                 0.6934, EXP-009) and 48% on training-adjacent video (EXP-007).

Half rate is better on accuracy by a wide margin and is the policy to run
wherever the hardware can sustain it. What decides the matter is not preference
but arithmetic: **half rate is only coherent if the machine can actually keep up
with half the source rate.** A pipeline slower than that falls behind the camera
without bound, and the frame on screen drifts further into the past every
second — which for a closing target is worse than not detecting at all.

Burst pairs is the fallback, and the reason it is the *right* fallback is
subtle. When the pipeline is slower than the camera, frames must be dropped, and
the only question is how. Taking whatever frame is newest whenever the pipeline
comes free produces gaps that are large, variable and unknown — which is plain
striding, the thing `docs/edge-budget.md` §3 item 7 rules out, because both of
GLAD's motion branches difference against the previous frame. Grabbing two
genuinely *adjacent* frames keeps that difference meaningful no matter how long
the sleep between pairs. Burst pairs is therefore the only policy that stays
coherent on hardware that cannot keep up.

Nothing here touches a camera, a model or the filesystem: throughput in, policy
out, so the choice is testable and can be explained on screen.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..data.sampling import Bursts, EveryNth, Schedule

FULL_RATE = "full rate"
HALF_RATE = "half rate"
BURST_PAIRS = "burst pairs"

# Recall retained against a full-rate control over the *same frames*, so these
# are ratios of like to like rather than of one run to another. Measured at
# centre@1x on ARD100, the split GLAD never trained on, because that is the
# conservative side of the pair -- the duty-cycle penalty is consistently worse
# on unseen video than on training-adjacent video (EXP-009 36% against EXP-007
# 48%), so quoting ARD-MAV here would flatter the fallback.
RETENTION = {
    FULL_RATE: 1.0,
    HALF_RATE: 0.915,   # EXP-008: 0.6290 / 0.6876
    BURST_PAIRS: 0.360,  # EXP-009: 0.2496 / 0.6934
}

# Probability that one burst detects the target at all, from EXP-009's
# second-of-burst cut. Drives the detection-latency estimate, which is the
# number that actually disqualifies a long burst period.
BURST_DETECTION_RATE = 0.2496

BURST_LENGTH = 2  # the first frame after a reset can never detect; see sampling.py


@dataclass(frozen=True)
class PolicyChoice:
    """The policy selected for a live feed, and the evidence for selecting it.

    `reason` is written to be read aloud on screen. A live view that shows a
    detection without showing which policy produced it invites the viewer to
    compare it against a full-rate number it cannot be compared to.
    """

    name: str
    schedule: Schedule
    reason: str
    measured_fps: float
    source_fps: float
    required_fps: float

    @property
    def retention(self) -> float:
        """Fraction of full-rate recall this policy retained when measured."""
        return RETENTION[self.name]

    @property
    def burst_period_seconds(self) -> float | None:
        """Seconds between burst starts, running bursts back to back.

        `None` for the contiguous policies. The period is emergent rather than
        configured on a live feed: a pair costs `BURST_LENGTH / measured_fps`
        seconds to process, and the next pair is grabbed as soon as that ends.
        """
        if self.name != BURST_PAIRS or self.measured_fps <= 0:
            return None
        return BURST_LENGTH / self.measured_fps

    @property
    def expected_detection_delay(self) -> float | None:
        """Mean seconds to the first detection of a target already in frame.

        Only burst pairs has a delay worth quoting. Each burst is an independent
        attempt at `BURST_DETECTION_RATE`, so the expected number of attempts is
        its reciprocal, and the delay is that many periods. The contiguous
        policies detect within a frame or two and return `None`.
        """
        period = self.burst_period_seconds
        if period is None or BURST_DETECTION_RATE <= 0:
            return None
        return period / BURST_DETECTION_RATE

    def closing_distance(self, closing_speed: float) -> float | None:
        """Metres a target closes before it is expected to be detected.

        The quantity `docs/edge-budget.md` §1 actually budgets. Returns `None`
        where there is no meaningful delay to convert.
        """
        delay = self.expected_detection_delay
        return None if delay is None else delay * closing_speed


def choose_policy(measured_fps: float, source_fps: float) -> PolicyChoice:
    """The best policy the measured throughput can actually sustain.

    Ordered by accuracy, taking the first that fits — full rate loses nothing,
    half rate loses 8.5% of recall, burst pairs loses 64%. The comparison is
    against the *source* rate rather than a fixed target: a 30 fps camera and a
    60 fps camera demand different things of the same machine.
    """
    if source_fps <= 0:
        raise ValueError(f"source_fps must be positive, got {source_fps}")

    if measured_fps >= source_fps:
        return PolicyChoice(
            FULL_RATE, Schedule(),
            f"{measured_fps:.2f} fps sustained meets the {source_fps:.0f} fps source, "
            f"so no frames need to be dropped.",
            measured_fps, source_fps, source_fps)

    half = source_fps / 2
    if measured_fps >= half:
        return PolicyChoice(
            HALF_RATE, EveryNth(2),
            f"{measured_fps:.2f} fps sustained cannot meet the {source_fps:.0f} fps "
            f"source but does meet half of it ({half:.1f} fps). Half rate keeps the "
            f"tracking lock and retained {RETENTION[HALF_RATE]:.1%} of recall in "
            f"EXP-008.",
            measured_fps, source_fps, half)

    # Below half rate nothing contiguous is sustainable, so the choice is
    # between dropping frames coherently and dropping them arbitrarily.
    period = max(BURST_LENGTH + 1, round(source_fps * BURST_LENGTH / max(measured_fps, 1e-9)))
    return PolicyChoice(
        BURST_PAIRS, Bursts(BURST_LENGTH, period),
        f"{measured_fps:.2f} fps sustained is below half the {source_fps:.0f} fps "
        f"source ({half:.1f} fps), so half rate would fall behind without bound. "
        f"Burst pairs is the only policy that keeps the motion branches coherent "
        f"on hardware this slow, at {RETENTION[BURST_PAIRS]:.1%} of recall "
        f"(EXP-009).",
        measured_fps, source_fps, half)
