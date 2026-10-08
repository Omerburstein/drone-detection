"""Unit tests for the drone selector.

The selector ranks candidates the kinematic gate already allowed, so the failures that
matter are about *which* one comes first. Each cue is therefore tested alone, with every
other weight at zero, so a test shows the cue pulling in the right direction. Then two
properties of the whole are pinned. The score does not depend on the picture's resolution
once lengths are quoted. And one wrong frame at #1 cannot rewrite the drone's colour.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.algo.scale import PixelScale
from src.algo.selection import (CUES, LOOK_SIZE, NEUTRAL, CueWeights, DroneSelector,
                                colour_signature, signature_similarity)

DARK = np.array([60.0, 128.0, 128.0, -80.0])     # a dark airframe against bright sky
GREEN = np.array([90.0, 100.0, 150.0, -40.0])    # a tree: darker than sky, and green
BRIGHT = np.array([220.0, 128.0, 128.0, 60.0])   # a white structure


def only(cue: str) -> CueWeights:
    """Weights with every cue off but one."""
    return CueWeights(**{c: float(c == cue) for c in CUES})


def selector(cue: str | None = None, width: float = 1440, **kw) -> DroneSelector:
    weights = only(cue) if cue else CueWeights()
    return DroneSelector(PixelScale(width), weights, **kw)


def step(sel: DroneSelector, frame: int, *cands, speed=None, look=None):
    """Feed (track id, x, y, evidence) tuples; returns the Selection."""
    rows = np.array(cands, float).reshape(-1, 4)
    return sel.step(frame, rows[:, 0].astype(int), rows[:, 1:3], rows[:, 3],
                    speed=speed, look=look)


def winner(sel_result, track_ids) -> int:
    return int(np.asarray(track_ids)[sel_result.order[0]])


class TestCueWeights:

    def test_parse_changes_only_what_it_names(self):
        w = CueWeights.parse("appearance=2, motion=0")
        assert w.appearance == 2.0 and w.motion == 0.0
        assert w.evidence == CueWeights().evidence

    def test_parse_of_nothing_is_the_defaults(self):
        assert CueWeights.parse("") == CueWeights()

    @pytest.mark.parametrize("text", ["colour=1", "appearance", "appearance:1"])
    def test_parse_refuses_what_it_cannot_read(self, text):
        with pytest.raises(ValueError):
            CueWeights.parse(text)

    def test_negative_weights_are_refused(self):
        with pytest.raises(ValueError):
            CueWeights(motion=-1.0)

    def test_all_zero_is_refused(self):
        with pytest.raises(ValueError):
            CueWeights(**{c: 0.0 for c in CUES})


class TestEachCue:

    def test_without_history_the_strongest_evidence_wins(self):
        """First frame: no colour, no #1 history, so the old ranking decides."""
        sel = selector()
        res = step(sel, 0, (1, 100, 100, 5.0), (2, 500, 500, 9.0))
        assert winner(res, [1, 2]) == 2
        assert res.cues["appearance"].tolist() == [NEUTRAL, NEUTRAL]
        assert res.cues["proximity"].tolist() == [0.0, 0.0]

    def test_proximity_prefers_where_number_one_just_was(self):
        sel = selector("proximity")
        step(sel, 0, (1, 100, 100, 9.0))
        res = step(sel, 1, (2, 108, 100, 1.0), (3, 600, 400, 9.0))
        assert winner(res, [2, 3]) == 2
        assert res.cues["proximity"][1] < 1e-6

    def test_an_older_sighting_supports_less(self):
        """Decay over time: the same distance counts less the longer ago #1 was there."""
        def support(gap: int) -> float:
            sel = selector("proximity")
            step(sel, 0, (1, 100, 100, 1.0))
            return float(step(sel, gap, (2, 120, 100, 1.0)).cues["proximity"][0])
        assert support(1) > support(5) > support(30) > 0.0

    def test_appearance_prefers_the_colour_of_the_previous_number_one(self):
        sel = selector("appearance")
        step(sel, 0, (1, 100, 100, 1.0), look=DARK[None])
        res = step(sel, 1, (2, 400, 100, 1.0), (3, 100, 400, 1.0),
                   look=np.stack([GREEN, DARK]))
        assert winner(res, [2, 3]) == 3

    def test_an_unreadable_look_is_neutral(self):
        sel = selector("appearance")
        step(sel, 0, (1, 100, 100, 1.0), look=DARK[None])
        res = step(sel, 1, (2, 100, 100, 1.0), look=np.full((1, LOOK_SIZE), np.nan))
        assert res.cues["appearance"].tolist() == [NEUTRAL]

    def test_a_stale_colour_fades_to_neutral(self):
        sel = selector("appearance")
        step(sel, 0, (1, 100, 100, 1.0), look=DARK[None])
        res = step(sel, 200, (2, 100, 100, 1.0), look=GREEN[None])
        assert res.cues["appearance"][0] == pytest.approx(NEUTRAL, abs=1e-6)

    def test_consistency_prefers_a_track_seen_every_frame(self):
        sel = selector("consistency")
        for f in range(10):
            cands = [(1, 100, 100, 1.0)] + ([(2, 400, 400, 1.0)] if f % 3 == 0 else [])
            res = step(sel, f, *cands)
        # Frame 9: track 2 is seen (9 % 3 == 0) after 4 sightings in 10 frames.
        assert winner(res, [1, 2]) == 1

    def test_a_newborn_track_has_earned_little_consistency(self):
        sel = selector("consistency")
        for f in range(20):
            res = step(sel, f, (1, 100, 100, 1.0), *([(2, 400, 400, 1.0)] if f == 19 else []))
        assert res.cues["consistency"][0] > 0.8 > 0.2 > res.cues["consistency"][1]

    def test_smoothness_prefers_constant_velocity(self):
        sel = selector("smoothness")
        for f in range(3):
            zig = 30 if f % 2 else -30
            res = step(sel, f, (1, 100 + 10 * f, 100, 1.0), (2, 400 + 10 * f, 400 + zig, 1.0))
        assert res.cues["smoothness"][0] == pytest.approx(1.0)
        assert res.cues["smoothness"][1] < 0.01

    def test_smoothness_waits_for_three_sightings(self):
        sel = selector("smoothness")
        step(sel, 0, (1, 100, 100, 1.0))
        assert step(sel, 1, (1, 140, 100, 1.0)).cues["smoothness"].tolist() == [NEUTRAL]

    def test_a_skipped_frame_is_a_longer_step_not_a_jerk(self):
        sel = selector("smoothness")
        step(sel, 0, (1, 100, 100, 1.0))
        step(sel, 1, (1, 110, 100, 1.0))
        res = step(sel, 3, (1, 130, 100, 1.0))
        assert res.cues["smoothness"][0] == pytest.approx(1.0)

    def test_motion_prefers_a_flyer_over_a_still_object(self):
        sel = selector("motion")
        res = step(sel, 0, (1, 100, 100, 1.0), (2, 400, 400, 1.0), (3, 50, 50, 1.0),
                   speed=np.array([0.5, 8.0, np.nan]))
        assert winner(res, [1, 2, 3]) == 2
        assert res.cues["motion"][2] == NEUTRAL


class TestTheWhole:

    def test_scores_are_the_weighted_mean_of_the_cues(self):
        sel = selector()
        res = step(sel, 0, (1, 100, 100, 5.0), (2, 500, 500, 9.0),
                   speed=np.array([1.0, 2.0]))
        w = CueWeights().as_array()
        expected = sum(w[k] * res.cues[c] for k, c in enumerate(CUES)) / w.sum()
        np.testing.assert_allclose(res.score, expected)
        assert ((0.0 <= res.score) & (res.score <= 1.0)).all()

    def test_the_answer_does_not_depend_on_resolution(self):
        """The same scene at 960 and at 4128 wide, lengths quoted at 1440: same scores."""
        def run(width: float) -> np.ndarray:
            k = width / 960
            sel = selector(width=width)
            for f in range(6):
                res = step(sel, f, (1, k * (100 + 7 * f), k * 100, 5.0),
                           (2, k * (300 + 3 * f), k * (200 + (9 if f % 2 else 0)), 6.0),
                           speed=k * np.array([7.0, 2.0]), look=np.stack([DARK, GREEN]))
            return res.score
        np.testing.assert_allclose(run(960), run(4128))

    def test_one_wrong_frame_does_not_rewrite_the_colour(self):
        """The drone's colour learns only from a #1 that held for two frames running."""
        sel = selector("evidence")
        step(sel, 0, (1, 100, 100, 1.0), look=DARK[None])
        step(sel, 1, (2, 400, 100, 1.0), look=BRIGHT[None])   # a one-frame hijack
        np.testing.assert_allclose(sel.reference_look, DARK)
        step(sel, 2, (2, 400, 100, 1.0), look=BRIGHT[None])   # now it held: it starts to learn
        assert sel.reference_look[0] > DARK[0]

    def test_the_colour_learns_slowly(self):
        sel = selector("evidence", look_rate=0.2)
        step(sel, 0, (1, 100, 100, 1.0), look=DARK[None])
        step(sel, 1, (1, 100, 100, 1.0), look=BRIGHT[None])
        assert sel.reference_look[0] == pytest.approx(DARK[0] + 0.2 * (BRIGHT[0] - DARK[0]))

    def test_proximity_follows_the_camera(self):
        """A #1 carried by the camera's motion still supports the candidate at its new place."""
        sel = selector("proximity")
        step(sel, 0, (1, 100, 100, 1.0))
        shift = np.array([[1, 0, 300], [0, 1, 0], [0, 0, 1]], float)
        res = sel.step(1, np.array([2, 3]), np.array([[400.0, 100.0], [100.0, 300.0]]),
                       np.ones(2), prev_to_cur=shift)
        assert res.cues["proximity"][0] > 0.8

    def test_no_candidates_is_not_an_error(self):
        res = step(selector(), 0)
        assert len(res.score) == 0 and len(res.order) == 0

    def test_frames_must_increase(self):
        sel = selector()
        step(sel, 5, (1, 100, 100, 1.0))
        with pytest.raises(ValueError):
            step(sel, 5, (1, 100, 100, 1.0))

    def test_reset_forgets_everything(self):
        sel = selector("proximity")
        step(sel, 0, (1, 100, 100, 1.0), look=DARK[None])
        sel.reset()
        res = step(sel, 0, (2, 100, 100, 1.0))
        assert res.cues["proximity"].tolist() == [0.0] and sel.reference_look is None

    def test_old_tracks_and_sightings_are_forgotten(self):
        sel = selector(memory_frames=10)
        step(sel, 0, (1, 100, 100, 1.0))
        res = step(sel, 50, (1, 100, 100, 1.0))
        assert res.cues["proximity"].tolist() == [0.0]
        assert res.cues["consistency"][0] == pytest.approx(1.0 - np.exp(-1 / 10))


class TestColourSignature:

    @staticmethod
    def scene(colour=(40, 40, 40)) -> np.ndarray:
        img = np.full((200, 300, 3), 220, np.uint8)
        yy, xx = np.mgrid[:200, :300]
        img[np.hypot(xx - 150, yy - 100) <= 6] = colour
        return img

    def test_a_dark_object_on_bright_sky_has_negative_contrast(self):
        sig = colour_signature(self.scene(), 150, 100, 6)
        assert sig[0] < 80 and sig[3] < -100

    def test_the_same_object_matches_itself_and_not_a_green_one(self):
        a = colour_signature(self.scene(), 150, 100, 6)
        b = colour_signature(self.scene((40, 160, 40)), 150, 100, 6)
        assert signature_similarity(a, a) == 1.0
        assert signature_similarity(a, b) < 0.1

    def test_without_colour_only_lightness_counts(self):
        a = np.array([100.0, 128.0, 128.0, -50.0])
        b = np.array([100.0, 160.0, 90.0, -50.0])
        assert signature_similarity(a, b, use_colour=False) == 1.0
        assert signature_similarity(a, b) < 0.01

    def test_off_the_picture_is_nan(self):
        assert np.isnan(colour_signature(self.scene(), 5000, 5000, 6)).all()

    def test_at_the_edge_it_reads_what_is_there(self):
        assert np.isfinite(colour_signature(self.scene(), 0, 0, 6)).all()

    def test_a_grey_frame_is_read(self):
        grey = self.scene()[..., 0]
        assert np.isfinite(colour_signature(grey, 150, 100, 6)).all()

    def test_look_clamps_the_radius_to_the_quoted_range(self):
        """A merged object 300 px wide is read from a disc no wider than the cap."""
        sel = selector(width=1440, look_min_radius=3, look_max_radius=6)
        img = self.scene()
        np.testing.assert_allclose(sel.look(img, [[150, 100]], [300]),
                                   colour_signature(img, 150, 100, 6)[None])
