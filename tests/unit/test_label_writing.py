"""Unit tests for `seed_track`'s two output writers.

Neither `write_labels` nor `write_verified` had a test, and both are about to be
touched by the split that moves the tracker out of `seed_track.py`. They are also
the point where a labelling session becomes durable, so a fault here costs a
human's afternoon rather than a re-run.

`write_verified` carries the load-bearing rule. It records which frames a person
actually adjudicated, keyed exactly as the run's `detections.jsonl` is, so
`src.evaluate --keys-from` accepts it directly. Frames absent from it were never
looked at, and scoring them as negatives would invent precision nobody measured.
The merge with earlier runs is what makes a campaign resumable across sittings,
so these tests pin that a second pass adds to the file rather than replacing it.
"""

from __future__ import annotations

import json

import pytest

from src.data.labels_io import to_yolo, write_labels, write_verified
from src.data.template_track import TrackPoint


def point(frame, box=(100.0, 100.0, 20.0, 20.0), score=0.9):
    return TrackPoint(frame=frame, box=box, score=score)


def read_jsonl(path):
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TestWriteLabels:

    def test_writes_one_file_per_point(self, tmp_path):
        write_labels([point(1), point(2), point(3)], tmp_path, "catch_2", 640, 480)
        assert sorted(p.name for p in tmp_path.glob("*.txt")) == [
            "catch_2_0001.txt", "catch_2_0002.txt", "catch_2_0003.txt"]

    def test_the_stem_is_zero_padded_to_four(self, tmp_path):
        """Keys must match the frame keys prepare_ardmav wrote, or evaluate
        pairs a label with nothing."""
        write_labels([point(7)], tmp_path, "clip", 640, 480)
        assert (tmp_path / "clip_0007.txt").is_file()

    def test_writes_class_zero_and_normalised_centre_form(self, tmp_path):
        write_labels([point(1, box=(320.0, 240.0, 64.0, 48.0))], tmp_path,
                     "clip", 640, 480)
        cls, cx, cy, w, h = (tmp_path / "clip_0001.txt").read_text(
            encoding="utf-8").split()
        assert cls == "0"
        assert float(cx) == pytest.approx(0.55)    # (320 + 32) / 640
        assert float(cy) == pytest.approx(0.55)    # (240 + 24) / 480
        assert float(w) == pytest.approx(0.1) and float(h) == pytest.approx(0.1)

    def test_creates_the_output_directory(self, tmp_path):
        out = tmp_path / "labels" / "train"
        write_labels([point(1)], out, "clip", 640, 480)
        assert (out / "clip_0001.txt").is_file()

    def test_no_points_writes_nothing_but_still_makes_the_directory(self, tmp_path):
        out = tmp_path / "labels"
        write_labels([], out, "clip", 640, 480)
        assert out.is_dir() and not list(out.glob("*.txt"))


class TestWriteVerified:

    def test_writes_one_record_per_frame(self, tmp_path):
        path = tmp_path / "verified.jsonl"
        assert write_verified({1, 2}, path, tmp_path / "images", "clip") == 2
        assert [r["image"] for r in read_jsonl(path)] == [
            str(tmp_path / "images" / "clip_0001.jpg"),
            str(tmp_path / "images" / "clip_0002.jpg")]

    def test_records_carry_no_detections(self, tmp_path):
        """This file says *what was looked at*, not what was found."""
        path = tmp_path / "verified.jsonl"
        write_verified({1}, path, tmp_path / "images", "clip")
        assert read_jsonl(path)[0]["detections"] == []

    def test_rows_are_sorted_by_key(self, tmp_path):
        path = tmp_path / "verified.jsonl"
        write_verified({9, 1, 4}, path, tmp_path / "images", "clip")
        stems = [r["image"].rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
                 for r in read_jsonl(path)]
        assert stems == ["clip_0001.jpg", "clip_0004.jpg", "clip_0009.jpg"]

    def test_a_second_pass_merges_rather_than_replaces(self, tmp_path):
        """What makes a labelling campaign resumable across sittings. If this
        truncated, the first sitting's frames would silently stop being scored."""
        path = tmp_path / "verified.jsonl"
        write_verified({1, 2}, path, tmp_path / "images", "clip")
        assert write_verified({3}, path, tmp_path / "images", "clip") == 3
        assert len(read_jsonl(path)) == 3

    def test_rewriting_the_same_frame_does_not_duplicate_it(self, tmp_path):
        path = tmp_path / "verified.jsonl"
        write_verified({1, 2}, path, tmp_path / "images", "clip")
        assert write_verified({2}, path, tmp_path / "images", "clip") == 2

    def test_merges_across_different_stems(self, tmp_path):
        """One verified.jsonl can cover several clips of a campaign."""
        path = tmp_path / "verified.jsonl"
        write_verified({1}, path, tmp_path / "images", "catch_2")
        assert write_verified({1}, path, tmp_path / "images", "catch_3") == 2

    def test_creates_the_parent_directory(self, tmp_path):
        path = tmp_path / "runs" / "exp" / "verified.jsonl"
        write_verified({1}, path, tmp_path / "images", "clip")
        assert path.is_file()

    def test_no_frames_leaves_an_existing_file_intact(self, tmp_path):
        path = tmp_path / "verified.jsonl"
        write_verified({1, 2}, path, tmp_path / "images", "clip")
        assert write_verified(set(), path, tmp_path / "images", "clip") == 2


class TestToYolo:

    def test_converts_corner_box_to_normalised_centre(self):
        cx, cy, w, h = to_yolo((100.0, 50.0, 40.0, 20.0), 1440, 1080)
        assert cx == pytest.approx(120 / 1440)
        assert cy == pytest.approx(60 / 1080)
        assert w == pytest.approx(40 / 1440)
        assert h == pytest.approx(20 / 1080)

    def test_clamps_a_box_that_drifted_off_the_edge(self):
        """Out-of-range labels are legal-looking and poison every later reader."""
        cx, cy, _, _ = to_yolo((-40.0, -30.0, 20.0, 10.0), 100, 100)
        assert cx == 0.0
        assert cy == 0.0

    def test_never_exceeds_one(self):
        cx, cy, w, h = to_yolo((90.0, 90.0, 400.0, 400.0), 100, 100)
        assert max(cx, cy, w, h) <= 1.0

