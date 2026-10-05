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


# --- the moving factor: the limit from below -------------------------------------------
#
# The speed limit passes anything that stays put, so these cover the other side. The two
# failures that matter mirror the ones above: a factor that is too loose keeps static
# clutter forever, and one that is too tight drops the drone -- here by being applied
# before the track has enough history to measure, or by charging the target for the
# camera's own motion.


def test_min_move_zero_changes_nothing():
    """The default must be inert: EXP-028's measured behaviour has to survive untouched."""
    a, b = tracker(), tracker(min_move=50.0, move_window=3)
    for f in range(1, 8):
        ra = step(a, f, (100.0, 100.0))
        rb = step(b, f, (100.0, 100.0))
        assert ra.reason[0] == OK if f >= 2 else True
        if f < 1 + 3:                      # before the window spans, both agree
            assert ra.reason[0] == rb.reason[0]


def test_static_clutter_is_overruled_once_the_window_spans():
    """A point that never moves is shown until its history spans the window, then dropped."""
    t = tracker(min_move=5.0, move_window=3)
    reasons = [step(t, f, (100.0, 100.0)).reason[0] for f in range(1, 7)]
    assert reasons[0] == NEW                     # born, not yet confirmed
    assert reasons[1] == OK                      # confirmed, no span yet
    assert reasons[2] == OK                      # span 2 < window 3
    assert reasons[3:] == ["too-still"] * 3      # span reaches 3: overruled from here on


def test_a_mover_passes_the_same_factor():
    """4 px/frame over a 3 frame window is 12 px of travel, above a 5 px factor."""
    t = tracker(min_move=5.0, move_window=3)
    out = [step(t, f, (100.0 + 4.0 * f, 100.0)).reason[0] for f in range(1, 7)]
    assert out[1:] == [OK] * 5
    assert "too-still" not in out


def test_the_factor_is_measured_against_the_scene_not_the_picture():
    """A still object under a panning camera must not be credited with the camera's motion.

    The anchors are carried by the homography, so a point that tracks the pan exactly has
    zero scene displacement even though its picture position moved 24 px.
    """
    t = tracker(min_move=5.0, move_window=3)
    reason = None
    for f in range(1, 7):
        x = 100.0 + 8.0 * (f - 1)            # the point moves with the pan
        r = step(t, f, (x, 100.0), hmat=shift(8.0, 0.0) if f > 1 else None)
        reason, moved = r.reason[0], r.moved[0]
    assert reason == "too-still"
    assert moved == pytest.approx(0.0, abs=1e-6)


def test_a_real_mover_under_a_panning_camera_still_passes():
    """The mirror of the test above: motion on top of the pan is kept."""
    t = tracker(min_move=5.0, move_window=3)
    for f in range(1, 7):
        x = 100.0 + 8.0 * (f - 1) + 3.0 * (f - 1)   # pan plus 3 px/frame of its own
        r = step(t, f, (x, 100.0), hmat=shift(8.0, 0.0) if f > 1 else None)
    assert r.reason[0] == OK
    assert r.moved[0] == pytest.approx(9.0, abs=1e-6)   # 3 px/frame across 3 frames


def test_moved_and_span_are_reported_and_counted():
    """The report needs the measurement itself, not just the verdict."""
    t = tracker(min_move=5.0, move_window=2)
    step(t, 1, (100.0, 100.0))
    step(t, 2, (100.0, 100.0))
    r = step(t, 3, (100.0, 100.0))
    assert r.move_span[0] == 2
    assert r.moved[0] == pytest.approx(0.0)
    assert r.too_still == 1
    assert not r.shown[0]


def test_move_vector_points_the_way_the_track_travelled():
    """EXP-029's direction test needs the vector, so it has to come out intact."""
    t = tracker(min_move=1.0, move_window=2)
    step(t, 1, (100.0, 100.0))
    step(t, 2, (103.0, 104.0))
    r = step(t, 3, (106.0, 108.0))
    assert r.move_vec[0] == pytest.approx([6.0, 8.0])
    assert r.moved[0] == pytest.approx(10.0)


def test_a_coasting_track_does_not_measure_across_a_stale_anchor():
    """Anchors older than the window are dropped, so a gap cannot inflate the travel."""
    t = tracker(min_move=5.0, move_window=3)
    step(t, 1, (100.0, 100.0))
    step(t, 2, (100.0, 100.0))
    step(t, 6, (100.0, 100.0))          # seen again after 4 missed frames
    r = step(t, 7, (100.0, 100.0))
    assert r.move_span[0] == 1           # only the frame-6 anchor is inside the window
    assert r.reason[0] == OK             # and with no span, it is not yet judged
