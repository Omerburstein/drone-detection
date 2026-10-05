"""Unit tests for the kinematic gate.

Two failures matter, and they pull in opposite directions. A gate that is too loose lets
the answer teleport, so #1 hops from the drone to a cloud edge across the frame. That is
the complaint the gate exists to fix. A gate that is too tight is quieter and worse: it
drops the drone whenever the host turns hard or the detector misses a frame. That shows up
only as unexplained recall loss, which is what EXP-025b's hard persistence gate cost
(61 -> 27 of 224 frames).

So the reach is tested from both sides of its boundary, and the two escape hatches that
keep the drone are tested on their own: coasting through misses, and the camera motion
excusing a whole-picture jump.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.algo.kinematics import (NEW, OK, UNCONFIRMED, WEAK_ORPHAN, KinematicTracker,
                                 SpeedLimit)

ANALOG = SpeedLimit(width_px=960, fps=30.0)


def tracker(**kw) -> KinematicTracker:
    """A gate with a round 10 px/frame limit and no slack, so reaches are easy to read."""
    limit = SpeedLimit(width_px=960, fps=30.0, ceiling_px=10.0, slack_px=0.0)
    return KinematicTracker(limit, **kw)


def step(t: KinematicTracker, frame: int, *points, strong=None, hmat=None):
    """Feed one frame of (x, y) points, all strong unless `strong` says otherwise."""
    xy = np.array(points, float).reshape(-1, 2)
    s = np.ones(len(xy), bool) if strong is None else np.array(strong, bool)
    return t.step(frame, xy, np.ones(len(xy)), s, hmat)


def shift(dx: float, dy: float) -> np.ndarray:
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], float)


class TestSpeedLimit:

    def test_analog_defaults_convert_to_about_30_px(self):
        """40 m/s at 10 m through a 130 degree lens on 960 px at 30 fps."""
        assert ANALOG.focal_px == pytest.approx(223.8, abs=0.1)
        assert ANALOG.physical_px == pytest.approx(29.8, abs=0.1)

    def test_the_ceiling_binds_at_the_defaults(self):
        assert ANALOG.px_per_frame == 25.0

    def test_a_far_minimum_range_lets_physics_bind(self):
        """At 50 m the physical limit (~6 px) is the tighter one."""
        far = SpeedLimit(width_px=960, fps=30.0, min_range_m=50.0)
        assert far.px_per_frame == pytest.approx(far.physical_px)
        assert far.px_per_frame < far.ceiling_px

    def test_reach_grows_linearly_with_the_gap(self):
        """A frame missed is a frame the drone kept flying, not a jump."""
        assert ANALOG.reach(1) == 4.0 + 25.0
        assert ANALOG.reach(3) == 4.0 + 75.0


class TestConfirmation:

    def test_a_steady_mover_is_shown_from_its_second_hit(self):
        t = tracker()
        first = step(t, 0, (100, 100))
        second = step(t, 1, (105, 100))
        assert first.reason.tolist() == [NEW] and not first.shown.any()
        assert second.reason.tolist() == [OK] and second.shown.all()
        assert second.track_id[0] == first.track_id[0]

    def test_confirm_one_shows_at_first_sight(self):
        assert step(tracker(confirm=1), 0, (100, 100)).shown.all()

    def test_confirm_three_waits_one_frame_longer(self):
        t = tracker(confirm=3)
        step(t, 0, (100, 100))
        assert step(t, 1, (101, 100)).reason.tolist() == [UNCONFIRMED]
        assert step(t, 2, (102, 100)).shown.all()


class TestReach:

    def confirmed(self) -> KinematicTracker:
        t = tracker()
        step(t, 0, (100, 100))
        step(t, 1, (100, 100))
        return t

    def test_a_step_exactly_at_reach_is_admitted(self):
        """The boundary is inclusive: 10 px at 10 px/frame is a legal step."""
        r = step(self.confirmed(), 2, (110, 100))
        assert r.reason.tolist() == [OK]

    def test_a_step_past_reach_is_a_new_object(self):
        r = step(self.confirmed(), 2, (110.5, 100))
        assert r.reason.tolist() == [NEW] and not r.shown.any()

    def test_a_teleport_is_overruled(self):
        """The complaint itself: #1 jumping across the frame."""
        r = step(self.confirmed(), 2, (700, 500))
        assert r.reason.tolist() == [NEW]

    def test_the_drone_survives_a_teleport_elsewhere(self):
        """A pop does not displace the track it is not near."""
        r = step(self.confirmed(), 2, (104, 100), (700, 500))
        assert r.reason.tolist() == [OK, NEW]


class TestHysteresis:

    def test_a_weak_candidate_continues_a_confirmed_track(self):
        t = tracker()
        step(t, 0, (100, 100))
        step(t, 1, (100, 100))
        r = step(t, 2, (103, 100), strong=[False])
        assert r.reason.tolist() == [OK] and r.continued_weak == 1

    def test_a_weak_candidate_never_starts_a_track(self):
        t = tracker()
        assert step(t, 0, (100, 100), strong=[False]).reason.tolist() == [WEAK_ORPHAN]
        assert step(t, 1, (100, 100), strong=[False]).reason.tolist() == [WEAK_ORPHAN]

    def test_a_weak_candidate_cannot_confirm_a_tentative_track(self):
        """Otherwise one strong flash plus any noise nearby would be shown."""
        t = tracker()
        step(t, 0, (100, 100))
        r = step(t, 1, (100, 100), strong=[False])
        assert r.reason.tolist() == [WEAK_ORPHAN]

    def test_a_tentative_track_dies_on_its_first_miss(self):
        t = tracker()
        step(t, 0, (100, 100))
        assert step(t, 1).died == 1
        assert step(t, 2, (100, 100)).reason.tolist() == [NEW]


class TestCoasting:

    def test_a_confirmed_track_coasts_through_misses_with_growing_reach(self):
        """Three missed frames at 10 px/frame: 40 px from the last sighting is legal."""
        t = tracker(max_coast=5)
        step(t, 0, (100, 100))
        step(t, 1, (100, 100))
        for f in (2, 3, 4):
            assert step(t, f).coasting == 1
        assert step(t, 5, (140, 100)).reason.tolist() == [OK]

    def test_a_track_dies_after_max_coast_misses(self):
        t = tracker(max_coast=2)
        step(t, 0, (100, 100))
        step(t, 1, (100, 100))
        assert step(t, 2).died == 0
        assert step(t, 3).died == 0
        assert step(t, 4).died == 1
        assert step(t, 5, (100, 100)).reason.tolist() == [NEW]

    def test_a_skipped_frame_index_is_a_longer_gap(self):
        """Striding or a dropped duplicate: frame 1 -> 3 allows two frames of travel."""
        t = tracker()
        step(t, 0, (100, 100))
        step(t, 1, (100, 100))
        assert step(t, 3, (120, 100)).reason.tolist() == [OK]


class TestCameraMotion:

    def confirmed(self) -> KinematicTracker:
        t = tracker()
        step(t, 0, (100, 100))
        step(t, 1, (100, 100))
        return t

    def test_a_pan_the_homography_explains_passes(self):
        """The host whips 60 px right; everything in the picture moves with it."""
        r = step(self.confirmed(), 2, (160, 100), hmat=shift(60, 0))
        assert r.reason.tolist() == [OK]

    def test_the_same_jump_without_camera_motion_is_overruled(self):
        r = step(self.confirmed(), 2, (160, 100))
        assert r.reason.tolist() == [NEW]

    def test_the_camera_never_costs_a_steady_target(self):
        """The camera follows the drone, so it sits still in the picture while the
        homography says the scene moved. Raw coordinates still admit it."""
        r = step(self.confirmed(), 2, (102, 100), hmat=shift(60, 0))
        assert r.reason.tolist() == [OK]

    def test_camera_motion_accumulates_while_coasting(self):
        t = self.confirmed()
        step(t, 2, hmat=shift(60, 0))
        r = step(t, 3, (220, 100), hmat=shift(60, 0))
        assert r.reason.tolist() == [OK]


class TestAssociation:

    def test_two_tracks_never_claim_one_candidate(self):
        t = tracker()
        step(t, 0, (100, 100), (112, 100))
        step(t, 1, (100, 100), (112, 100))
        r = step(t, 2, (106, 100))
        assert (r.track_id >= 0).sum() == 1
        assert r.coasting == 1

    def test_the_nearer_candidate_wins_the_track(self):
        t = tracker()
        step(t, 0, (100, 100))
        step(t, 1, (100, 100))
        r = step(t, 2, (108, 100), (102, 100))
        assert r.reason.tolist() == [NEW, OK]

    def test_a_duplicate_frame_is_harmless(self):
        """Analog repeats 1 frame in 6: the same candidates, a zero step."""
        t = tracker()
        for f in range(4):
            r = step(t, f, (100, 100))
        assert r.reason.tolist() == [OK]


class TestState:

    def test_frames_must_increase(self):
        t = tracker()
        step(t, 5, (100, 100))
        with pytest.raises(ValueError):
            step(t, 5, (100, 100))

    def test_reset_forgets_every_track(self):
        t = tracker()
        step(t, 0, (100, 100))
        step(t, 1, (100, 100))
        t.reset()
        assert step(t, 0, (100, 100)).reason.tolist() == [NEW]

    def test_an_empty_frame_is_not_an_error(self):
        r = step(tracker(), 0)
        assert len(r.track_id) == 0 and r.born == 0


class TestEvidence:

    def test_evidence_accumulates_scores_with_decay(self):
        """Ranking by evidence is what keeps #1 from swapping between two plausible
        tracks whose per-frame scores cross."""
        t = tracker(decay=0.5)
        t.step(0, np.array([[100, 100]]), np.array([4.0]), np.array([True]))
        r = t.step(1, np.array([[100, 100]]), np.array([2.0]), np.array([True]))
        assert r.evidence[0] == pytest.approx(4.0 * 0.5 + 2.0)

    def test_a_missed_frame_costs_a_second_decay(self):
        t = tracker(decay=0.5)
        t.step(0, np.array([[100, 100]]), np.array([4.0]), np.array([True]))
        t.step(1, np.array([[100, 100]]), np.array([4.0]), np.array([True]))
        r = t.step(3, np.array([[100, 100]]), np.array([1.0]), np.array([True]))
        assert r.evidence[0] == pytest.approx((4.0 * 0.5 + 4.0) * 0.25 + 1.0)

    def test_a_steady_track_outranks_a_single_strong_flash(self):
        t = tracker(confirm=1)
        for f in range(5):
            r = t.step(f, np.array([[100, 100]]), np.array([5.0]), np.array([True]))
        r = t.step(5, np.array([[100, 100], [600, 400]]), np.array([5.0, 8.0]),
                   np.array([True, True]))
        assert r.shown.all() and r.evidence[0] > r.evidence[1]

    def test_a_candidate_on_no_track_has_no_evidence(self):
        r = step(tracker(), 0, (100, 100), strong=[False])
        assert r.evidence.tolist() == [0.0]
