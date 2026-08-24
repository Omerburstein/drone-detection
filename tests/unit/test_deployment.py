"""Choosing a duty-cycle policy from measured throughput.

The thresholds here are the whole deployment decision, and getting one wrong is
not visible on screen: picking half rate on a machine that cannot sustain it
produces a picture that looks fine and drifts further into the past every
second. So the boundaries are pinned exactly, including at equality.
"""

from __future__ import annotations

import pytest

from src.algo.deployment import (BURST_DETECTION_RATE, BURST_PAIRS, FULL_RATE,
                                 HALF_RATE, RETENTION, choose_policy)
from src.data.sampling import Bursts, EveryNth, Schedule

SOURCE = 30.0


class TestChoosePolicy:
    """Ordered by accuracy, taking the first the hardware can actually hold."""

    def test_keeping_up_drops_nothing(self):
        policy = choose_policy(30.0, SOURCE)
        assert policy.name == FULL_RATE
        assert type(policy.schedule) is Schedule

    def test_faster_than_the_source_is_still_full_rate(self):
        assert choose_policy(60.0, SOURCE).name == FULL_RATE

    def test_half_the_source_is_half_rate(self):
        policy = choose_policy(20.0, SOURCE)
        assert policy.name == HALF_RATE
        assert policy.schedule == EveryNth(2)

    def test_exactly_half_still_qualifies(self):
        # The boundary is inclusive: a machine that holds 15.0 fps against a
        # 30 fps source holds half rate by definition.
        assert choose_policy(15.0, SOURCE).name == HALF_RATE

    def test_a_hair_under_half_falls_back(self):
        assert choose_policy(14.99, SOURCE).name == BURST_PAIRS

    def test_this_laptop_gets_burst_pairs(self):
        """2.79 fps is EXP-008's measured half-rate throughput on this host."""
        policy = choose_policy(2.79, SOURCE)
        assert policy.name == BURST_PAIRS
        assert isinstance(policy.schedule, Bursts)
        assert policy.schedule.length == 2

    def test_an_orin_class_machine_gets_half_rate(self):
        """~9x this host, the speedup edge-budget.md projects for an Orin Nano."""
        assert choose_policy(2.79 * 9, SOURCE).name == HALF_RATE

    def test_the_source_rate_matters_not_just_the_machine(self):
        """The same machine deserves different policies on different cameras."""
        assert choose_policy(20.0, 30.0).name == HALF_RATE
        assert choose_policy(20.0, 60.0).name == BURST_PAIRS

    def test_a_nonsense_source_rate_is_rejected(self):
        # A UVC driver reporting 0 fps would otherwise divide by zero and pick a
        # policy from noise.
        with pytest.raises(ValueError):
            choose_policy(5.0, 0.0)

    def test_every_choice_explains_itself(self):
        for measured in (30.0, 20.0, 2.79):
            policy = choose_policy(measured, SOURCE)
            assert f"{measured:.2f}" in policy.reason
            assert policy.reason.endswith(".")


class TestRetention:
    """The measured cost of each policy, quoted on screen."""

    def test_full_rate_loses_nothing(self):
        assert choose_policy(30.0, SOURCE).retention == 1.0

    def test_half_rate_matches_exp008(self):
        assert choose_policy(15.0, SOURCE).retention == pytest.approx(0.915)

    def test_burst_matches_exp009(self):
        assert choose_policy(2.0, SOURCE).retention == pytest.approx(0.360)

    def test_retention_is_ordered_by_policy_quality(self):
        assert RETENTION[FULL_RATE] > RETENTION[HALF_RATE] > RETENTION[BURST_PAIRS]


class TestBurstLatency:
    """The arithmetic that actually disqualifies a long burst period."""

    def test_contiguous_policies_have_no_period_or_delay(self):
        for measured in (30.0, 15.0):
            policy = choose_policy(measured, SOURCE)
            assert policy.burst_period_seconds is None
            assert policy.expected_detection_delay is None
            assert policy.closing_distance(40.0) is None

    def test_period_is_the_time_to_process_one_pair(self):
        policy = choose_policy(2.0, SOURCE)
        assert policy.burst_period_seconds == pytest.approx(1.0)  # 2 frames / 2 fps

    def test_delay_is_the_period_over_the_per_burst_rate(self):
        policy = choose_policy(2.0, SOURCE)
        assert policy.expected_detection_delay == pytest.approx(
            1.0 / BURST_DETECTION_RATE)

    def test_closing_distance_is_delay_times_speed(self):
        policy = choose_policy(2.0, SOURCE)
        delay = policy.expected_detection_delay
        assert policy.closing_distance(40.0) == pytest.approx(delay * 40.0)

    def test_a_slower_machine_costs_more_ground(self):
        """The consequence that decides deployability, and it must be monotonic."""
        fast = choose_policy(4.0, SOURCE).closing_distance(40.0)
        slow = choose_policy(1.0, SOURCE).closing_distance(40.0)
        assert slow > fast
