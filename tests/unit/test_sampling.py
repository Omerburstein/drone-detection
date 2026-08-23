"""The duty-cycle schedules: which frames run, and where the pipeline resets.

Every assertion here is stated in one-based frame indices, because that is what
`src.glad_detect` counts with and what the label filenames use. An off-by-one in
`wants` silently shifts a whole run onto different frames than the ones the
schedule claims; an off-by-one in `resets` is worse, because it leaves the motion
branches differencing across a gap and the run measures an artefact.
"""

from __future__ import annotations

import pytest

from src.data.sampling import BURST, EVERY, NTH, Bursts, EveryNth, Schedule, build_schedule


def processed(schedule: Schedule, last: int) -> list[int]:
    """The one-based frame indices `schedule` would process out of 1..last."""
    return [i for i in range(1, last + 1) if schedule.wants(i)]


def reset_at(schedule: Schedule, last: int) -> list[int]:
    """The frames at which `schedule` demands a pipeline reset."""
    return [i for i in range(1, last + 1) if schedule.resets(i)]


class TestEveryFrame:
    """The default policy has to stay exactly what EXP-004 ran."""

    def test_processes_every_frame(self):
        assert processed(Schedule(), 10) == list(range(1, 11))

    def test_never_resets(self):
        # `run_video` resets once per video on its own; a schedule-driven reset
        # mid-video would break the differencing the full-rate run depends on.
        assert reset_at(Schedule(), 10) == []

    def test_records_no_position(self):
        assert Schedule().position(7) == {}


class TestEveryNth:
    """Lower rate, same contiguity: frame 1 runs, then every nth after it."""

    def test_half_rate_takes_the_odd_frames(self):
        assert processed(EveryNth(2), 10) == [1, 3, 5, 7, 9]

    def test_third_rate(self):
        assert processed(EveryNth(3), 10) == [1, 4, 7, 10]

    def test_n_of_one_is_every_frame(self):
        assert processed(EveryNth(1), 5) == [1, 2, 3, 4, 5]

    def test_never_resets(self):
        # The stream stays contiguous in the sense the motion module cares
        # about, so state must survive from one processed frame to the next.
        assert reset_at(EveryNth(2), 20) == []

    def test_label_names_the_rate(self):
        assert "1 frame in 2" in EveryNth(2).label


class TestBursts:
    """Short contiguous runs, and a cold start at the head of each one."""

    def test_takes_pairs_at_the_period(self):
        assert processed(Bursts(2, 5), 12) == [1, 2, 6, 7, 11, 12]

    def test_longer_bursts(self):
        assert processed(Bursts(3, 10), 21) == [1, 2, 3, 11, 12, 13, 21]

    def test_resets_only_at_the_head_of_each_burst(self):
        # This is the load-bearing one. A reset anywhere else throws away a
        # usable previous frame; a missing reset differences across the sleep.
        assert reset_at(Bursts(2, 5), 12) == [1, 6, 11]

    def test_every_reset_is_a_processed_frame(self):
        schedule = Bursts(2, 7)
        assert set(reset_at(schedule, 50)) <= set(processed(schedule, 50))

    def test_position_identifies_the_burst_and_the_slot(self):
        schedule = Bursts(2, 5)
        assert schedule.position(1) == {"burst": 0, "in_burst": 0}
        assert schedule.position(2) == {"burst": 0, "in_burst": 1}
        assert schedule.position(6) == {"burst": 1, "in_burst": 0}

    def test_duty_cycle_appears_in_the_label(self):
        assert "3.33%" in Bursts(2, 60).label

    def test_rejects_a_burst_too_short_to_detect(self):
        # A one-frame burst is all cold starts: `GladPipeline.step` returns an
        # empty result for the frame after a reset, so it could never detect.
        with pytest.raises(SystemExit):
            Bursts(1, 30)

    def test_rejects_bursts_that_run_together(self):
        with pytest.raises(SystemExit):
            Bursts(5, 5)


class TestBuildSchedule:
    """The CLI's three modes, and the arguments they ignore."""

    def test_every_is_the_plain_policy(self):
        schedule = build_schedule(EVERY, 2, 2, 60)
        assert type(schedule) is Schedule
        assert processed(schedule, 4) == [1, 2, 3, 4]

    def test_nth_uses_only_sample_n(self):
        assert build_schedule(NTH, 3, 2, 60) == EveryNth(3)

    def test_burst_uses_only_the_burst_arguments(self):
        assert build_schedule(BURST, 3, 2, 60) == Bursts(2, 60)

    def test_rejects_an_unknown_mode(self):
        with pytest.raises(SystemExit):
            build_schedule("stride", 2, 2, 60)

    def test_rejects_a_nonsense_rate(self):
        with pytest.raises(SystemExit):
            build_schedule(NTH, 0, 2, 60)
