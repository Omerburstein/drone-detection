"""Unit tests for the annotation session: how verdicts spread and are written.

The propagation rules decide which frames a label file claims a human looked
at, and the export decides what `src.evaluate` will score, so both are pinned
exactly. The follower is exercised on synthetic frames with a known motion, so
"did it follow" has an exact answer.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.data.annotation import (CARRIED, HUMAN, TRACKED, Follower, Session,
                                 Verdict, predict)


def make_session(tmp_path) -> Session:
    return Session(tmp_path / "clip.avi", "clip", 320, 200)


def follow_by(dx: float):
    """A stand-in follower that moves the box by `dx` and records its prior."""
    calls = []

    def follow(box, prior):
        calls.append(prior)
        return (box[0] + dx, box[1], box[2], box[3]), 0.8
    return follow, calls


def lost(_box, _prior):
    return None


class TestPropagate:

    def test_box_is_tracked_forward(self, tmp_path):
        session = make_session(tmp_path)
        session.set_box(1, (10.0, 10.0, 8.0, 8.0))
        follow, _ = follow_by(3)
        verdict, was_lost = session.propagate(1, 2, follow)
        assert not was_lost
        assert verdict == Verdict((13.0, 10.0, 8.0, 8.0), TRACKED, 0.8)

    def test_negative_carries_forward(self, tmp_path):
        session = make_session(tmp_path)
        session.mark_absent(1)
        verdict, _ = session.propagate(1, 2, lost)
        assert verdict.absent and verdict.source == CARRIED

    def test_nothing_spreads_from_an_unjudged_frame(self, tmp_path):
        session = make_session(tmp_path)
        assert session.propagate(1, 2, lost) == (None, False)
        assert session.get(2) is None

    def test_human_verdict_is_never_overwritten(self, tmp_path):
        session = make_session(tmp_path)
        session.set_box(1, (10.0, 10.0, 8.0, 8.0))
        session.mark_absent(2)
        session.propagate(1, 2, follow_by(3)[0])
        assert session.get(2) == Verdict(None, HUMAN)

    def test_stale_proposal_is_replaced_going_forward(self, tmp_path):
        """A correction upstream makes the old downstream proposal stale."""
        session = make_session(tmp_path)
        session.verdicts[2] = Verdict((90.0, 90.0, 8.0, 8.0), TRACKED, 0.5)
        session.set_box(1, (10.0, 10.0, 8.0, 8.0))
        verdict, _ = session.propagate(1, 2, follow_by(1)[0])
        assert verdict.box == (11.0, 10.0, 8.0, 8.0)

    def test_stepping_back_never_rewrites(self, tmp_path):
        session = make_session(tmp_path)
        session.verdicts[1] = Verdict((90.0, 90.0, 8.0, 8.0), TRACKED, 0.5)
        session.set_box(2, (10.0, 10.0, 8.0, 8.0))
        verdict, _ = session.propagate(2, 1, follow_by(1)[0])
        assert verdict.box == (90.0, 90.0, 8.0, 8.0)

    def test_stepping_back_fills_an_empty_frame(self, tmp_path):
        session = make_session(tmp_path)
        session.set_box(5, (10.0, 10.0, 8.0, 8.0))
        verdict, _ = session.propagate(5, 4, follow_by(-2)[0])
        assert verdict.box == (8.0, 10.0, 8.0, 8.0)

    def test_losing_the_target_drops_the_stale_proposal(self, tmp_path):
        session = make_session(tmp_path)
        session.set_box(1, (10.0, 10.0, 8.0, 8.0))
        session.verdicts[2] = Verdict((12.0, 10.0, 8.0, 8.0), TRACKED, 0.9)
        assert session.propagate(1, 2, lost) == (None, True)
        assert session.get(2) is None


class TestVelocityPrior:

    def test_prior_is_the_frame_behind(self, tmp_path):
        session = make_session(tmp_path)
        session.set_box(1, (10.0, 10.0, 8.0, 8.0))
        session.verdicts[2] = Verdict((14.0, 10.0, 8.0, 8.0), TRACKED, 0.9)
        follow, calls = follow_by(4)
        session.propagate(2, 3, follow)
        assert calls == [(10.0, 10.0, 8.0, 8.0)]

    def test_prior_withheld_after_a_correction(self, tmp_path):
        """Proposal -> correction is the size of the mistake, not a velocity."""
        session = make_session(tmp_path)
        session.verdicts[1] = Verdict((90.0, 90.0, 8.0, 8.0), TRACKED, 0.5)
        session.set_box(2, (10.0, 10.0, 8.0, 8.0))
        follow, calls = follow_by(1)
        session.propagate(2, 3, follow)
        assert calls == [None]

    def test_predict_extrapolates_centre_displacement(self):
        assert predict((20.0, 10.0, 8.0, 8.0), (15.0, 12.0, 8.0, 8.0)) == (
            25.0, 8.0, 8.0, 8.0)
        assert predict((20.0, 10.0, 8.0, 8.0), None) == (20.0, 10.0, 8.0, 8.0)


def textured_frame(x: int, y: int, shape=(240, 480)) -> np.ndarray:
    """Low-contrast noise background with a structured 12 px target at `(x, y)`."""
    rng = np.random.default_rng(7)
    image = rng.integers(20, 60, size=(*shape, 3), dtype=np.int16).astype(np.uint8)
    patch = np.random.default_rng(20260917).integers(40, 250, size=(12, 12),
                                                     dtype=np.int16)
    image[y:y + 12, x:x + 12] = np.repeat(patch[:, :, None], 3, axis=2)
    return image


class TestFollower:

    @staticmethod
    def run(speed: int, use_prior: bool, frames: int = 5) -> list[float]:
        """Track a target moving `speed` px/frame, feeding back the previous box
        as the session does. The target is already moving when seeded, so the
        first step has a prior too. Returns the x positions found."""
        follower = Follower()
        follower.seed(textured_frame(20, 100), (20.0, 100.0, 12.0, 12.0), 1)
        box, prior = (20.0, 100.0, 12.0, 12.0), (20.0 - speed, 100.0, 12.0, 12.0)
        found = []
        for step in range(1, frames + 1):
            result = follower.follow(textured_frame(20 + speed * step, 100), box,
                                     prior if use_prior else None)
            if result is None:
                break
            prior, box = box, result[0]
            found.append(box[0])
        return found

    def test_follows_a_moving_target(self):
        assert self.run(10, use_prior=True) == [30, 40, 50, 60, 70]

    def test_motion_prior_holds_a_target_faster_than_the_window(self):
        """The search window reaches 4 target sizes (48 px) past the box. At
        60 px/frame the target is outside it every frame: correlation alone
        loses it at once, and the velocity prior is what keeps it."""
        assert self.run(60, use_prior=False) == []
        assert self.run(60, use_prior=True) == [80, 140, 200, 260, 320]

    def test_unseeded_follower_finds_nothing(self):
        assert Follower().follow(textured_frame(20, 100), (20.0, 100.0, 12.0, 12.0),
                                 None) is None

    def test_featureless_seed_is_refused(self):
        """Correlation stretches a flat patch to full contrast, so a box on
        empty sky would 'follow' faint gradient anywhere at a high score."""
        follower = Follower()
        sky = np.full((240, 480, 3), 180, dtype=np.uint8)
        sky[:, :, 0] += (np.arange(480) // 160).astype(np.uint8)  # a faint gradient
        assert not follower.seed(sky, (100.0, 100.0, 40.0, 20.0), 1)
        assert not follower.ready

    def test_match_on_a_flat_patch_is_rejected(self):
        """The target vanishes into flat sky: no match, however well a faint
        gradient correlates once normalised."""
        follower = Follower()
        follower.seed(textured_frame(20, 100), (20.0, 100.0, 12.0, 12.0), 1)
        flat = np.full((240, 480, 3), 40, dtype=np.uint8)
        assert follower.follow(flat, (20.0, 100.0, 12.0, 12.0), None) is None

    def test_degenerate_seed_is_refused(self):
        follower = Follower()
        assert not follower.seed(textured_frame(20, 100), (20.0, 100.0, 1.0, 1.0), 1)
        assert not follower.ready


class TestPersistence:

    def test_round_trip(self, tmp_path):
        session = make_session(tmp_path)
        session.set_box(3, (10.5, 20.25, 8.0, 6.0))
        session.mark_absent(4)
        session.verdicts[5] = Verdict((11.0, 20.0, 8.0, 6.0), TRACKED, 0.77)
        path = tmp_path / "ann" / "clip.json"
        session.save(path)
        assert not session.dirty
        loaded = Session.load(path, session.video, "clip", 320, 200)
        assert loaded.verdicts == session.verdicts

    def test_refuses_a_session_from_different_geometry(self, tmp_path):
        session = make_session(tmp_path)
        session.set_box(1, (1.0, 1.0, 4.0, 4.0))
        path = tmp_path / "clip.json"
        session.save(path)
        with pytest.raises(ValueError, match="320x200"):
            Session.load(path, session.video, "clip", 1440, 1080)

    def test_missing_file_is_a_fresh_session(self, tmp_path):
        loaded = Session.load(tmp_path / "none.json", tmp_path / "v.avi", "v", 10, 10)
        assert loaded.verdicts == {}


class TestExport:

    def test_writes_labels_and_verified(self, tmp_path):
        session = make_session(tmp_path)
        session.set_box(7, (150.0, 90.0, 20.0, 20.0))
        session.mark_absent(8)
        labels, verified = tmp_path / "labels", tmp_path / "verified.jsonl"
        session.export(labels, verified, tmp_path / "images")

        assert (labels / "clip_0007.txt").read_text() == (
            "0 0.500000 0.500000 0.062500 0.100000\n")
        assert not (labels / "clip_0008.txt").exists()
        keys = [json.loads(line)["image"] for line in verified.read_text().splitlines()]
        assert [k.replace("\\", "/").rsplit("/", 1)[1] for k in keys] == [
            "clip_0007.jpg", "clip_0008.jpg"]

    def test_session_is_authoritative_for_its_stem(self, tmp_path):
        """A box the user deleted must not survive in the export and be scored,
        and another clip's rows must not be touched."""
        labels, verified = tmp_path / "labels", tmp_path / "verified.jsonl"
        labels.mkdir()
        (labels / "clip_0009.txt").write_text("0 0.1 0.1 0.1 0.1\n")
        (labels / "clip_extra_0009.txt").write_text("0 0.1 0.1 0.1 0.1\n")
        verified.write_text(json.dumps({"image": "i/clip_0009.jpg", "detections": []})
                            + "\n" + json.dumps({"image": "i/other_0001.jpg",
                                                 "detections": []}) + "\n")
        session = make_session(tmp_path)
        session.set_box(7, (150.0, 90.0, 20.0, 20.0))
        session.export(labels, verified, tmp_path / "images")

        assert not (labels / "clip_0009.txt").exists()
        assert (labels / "clip_extra_0009.txt").exists()   # a different stem
        names = {json.loads(line)["image"].replace("\\", "/").rsplit("/", 1)[1]
                 for line in verified.read_text().splitlines()}
        assert names == {"clip_0007.jpg", "other_0001.jpg"}


def test_counts(tmp_path):
    session = make_session(tmp_path)
    session.set_box(1, (1.0, 1.0, 4.0, 4.0))
    session.verdicts[2] = Verdict((1.0, 1.0, 4.0, 4.0), TRACKED, 0.9)
    session.mark_absent(3)
    assert session.counts() == {"boxes": 2, "absent": 1, "human": 2}

