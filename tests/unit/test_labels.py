"""Tests for label reading and prediction pairing in `src.eval.labels`.

The coordinate conversion here is the highest-risk code in the project: a
corner/centre or normalised/absolute mix-up produces labels that pass every
automated sanity check while being silently wrong, which then reads as a bad
model rather than a bad pipeline.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.eval.labels import load_frames, load_label_file, yolo_to_xyxy


class TestYoloToXyxy:
    """Normalised centre/size -> absolute corners."""

    def test_centred_half_size_box(self):
        """cx=cy=0.5, w=h=0.5 in a 100x100 frame spans the middle half."""
        rows = np.array([[0.5, 0.5, 0.5, 0.5]])
        assert yolo_to_xyxy(rows, 100, 100)[0] == pytest.approx([25, 25, 75, 75])

    def test_non_square_frame_scales_axes_independently(self):
        """Catches a width/height swap, which a square frame would hide."""
        rows = np.array([[0.5, 0.5, 1.0, 1.0]])
        assert yolo_to_xyxy(rows, 1920, 1080)[0] == pytest.approx([0, 0, 1920, 1080])

    def test_off_centre_box_keeps_top_left_origin(self):
        """y must grow downward: a low cy stays near the top of the image."""
        rows = np.array([[0.25, 0.1, 0.1, 0.2]])
        x1, y1, x2, y2 = yolo_to_xyxy(rows, 1000, 1000)[0]
        assert (x1, x2) == pytest.approx((200, 300))
        assert (y1, y2) == pytest.approx((0, 200))

    def test_empty_input_keeps_shape(self):
        assert yolo_to_xyxy(np.zeros((0, 4)), 100, 100).shape == (0, 4)


class TestLoadLabelFile:
    """Reading one YOLO .txt."""

    def test_missing_file_is_an_empty_frame_not_an_error(self, tmp_path):
        """ARD-MAV has ~7k frames with no visible drone; those are negatives."""
        rows, classes = load_label_file(tmp_path / "absent.txt")
        assert rows.shape == (0, 4)
        assert len(classes) == 0

    def test_reads_multiple_objects(self, tmp_path):
        path = tmp_path / "two.txt"
        path.write_text("0 0.5 0.5 0.2 0.2\n1 0.25 0.25 0.1 0.1\n")
        rows, classes = load_label_file(path)
        assert rows.shape == (2, 4)
        assert list(classes) == [0, 1]
        assert rows[0] == pytest.approx([0.5, 0.5, 0.2, 0.2])

    def test_blank_lines_are_skipped(self, tmp_path):
        path = tmp_path / "gappy.txt"
        path.write_text("\n0 0.5 0.5 0.2 0.2\n\n")
        rows, _ = load_label_file(path)
        assert rows.shape == (1, 4)

    def test_empty_file_is_an_empty_frame(self, tmp_path):
        path = tmp_path / "empty.txt"
        path.write_text("")
        rows, _ = load_label_file(path)
        assert rows.shape == (0, 4)

    def test_truncated_line_fails_loudly(self, tmp_path):
        """A malformed label must stop the run, not silently score against garbage."""
        path = tmp_path / "bad.txt"
        path.write_text("0 0.5 0.5\n")
        with pytest.raises(SystemExit):
            load_label_file(path)


class TestLoadFrames:
    """Pairing recorded predictions with their labels."""

    @staticmethod
    def _write_run(tmp_path, records):
        pred = tmp_path / "detections.jsonl"
        pred.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        labels = tmp_path / "labels"
        labels.mkdir(exist_ok=True)
        return pred, labels

    def test_image_records_pair_by_filename_stem(self, tmp_path):
        pred, labels = self._write_run(tmp_path, [
            {"image": "data/phantom02_0001.jpg",
             "detections": [{"bbox": [0, 0, 10, 10], "conf": 0.9, "cls": 0}]},
        ])
        (labels / "phantom02_0001.txt").write_text("0 0.5 0.5 0.5 0.5\n")

        frames = load_frames(pred, labels, frame_size=(100, 100))
        assert len(frames) == 1
        assert frames[0].key == "phantom02_0001"
        assert frames[0].gt_boxes[0] == pytest.approx([25, 25, 75, 75])
        assert len(frames[0].preds) == 1

    def test_video_records_pair_by_frame_index(self, tmp_path):
        pred, labels = self._write_run(tmp_path, [{"frame": 42, "detections": []}])
        (labels / "42.txt").write_text("0 0.5 0.5 0.5 0.5\n")

        frames = load_frames(pred, labels, frame_size=(100, 100))
        assert frames[0].key == "42"
        assert len(frames[0].gt_boxes) == 1

    def test_video_records_without_frame_size_fail_loudly(self, tmp_path):
        """The JSONL carries no dimensions, so guessing would corrupt every box."""
        pred, labels = self._write_run(tmp_path, [{"frame": 0, "detections": []}])
        with pytest.raises(SystemExit):
            load_frames(pred, labels, frame_size=None)

    def test_unlabelled_prediction_yields_an_empty_ground_truth(self, tmp_path):
        """A false alarm on a negative frame must survive to be counted."""
        pred, labels = self._write_run(tmp_path, [
            {"image": "a.jpg",
             "detections": [{"bbox": [1, 2, 3, 4], "conf": 0.5, "cls": 0}]},
        ])
        frames = load_frames(pred, labels, frame_size=(100, 100))
        assert len(frames[0].gt_boxes) == 0
        assert len(frames[0].preds) == 1

    def test_blank_trailing_lines_are_ignored(self, tmp_path):
        pred, labels = self._write_run(tmp_path, [{"frame": 0, "detections": []}])
        pred.write_text(pred.read_text() + "\n\n", encoding="utf-8")
        assert len(load_frames(pred, labels, frame_size=(10, 10))) == 1
