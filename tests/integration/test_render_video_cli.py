"""End-to-end tests for the `src.render_video` CLI.

The seams here are the ones no unit test covers: that a video's decode order and
the run's frame keys line up (`<stem>_0001` is the *first* decoded frame, not the
zeroth), that unscored frames are dropped rather than drawn, and that the file
that comes out is a decodable video of the expected length.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIDTH, HEIGHT = 160, 120
SCORED_FRAMES = 6
TOTAL_FRAMES = 8


def frame_count(path: Path) -> int:
    """Decode a video and count the frames that actually come back out."""
    capture = cv2.VideoCapture(str(path))
    try:
        return sum(1 for _ in iter(lambda: capture.read()[0], False))
    finally:
        capture.release()


@pytest.fixture
def clip(tmp_path):
    """A synthetic run: 8-frame video, 6 of them scored with one target each.

    The last two frames stand for the unannotated tail every ARD-MAV video has —
    processed by the detector, never scored, and so not to be rendered.
    """
    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0,
                             (WIDTH, HEIGHT))
    for i in range(TOTAL_FRAMES):
        frame = np.full((HEIGHT, WIDTH, 3), 30 + i, dtype=np.uint8)
        frame[58:64, 78:84] = 255  # a bright speck where the label says it is
        writer.write(frame)
    writer.release()

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    records = []
    for i in range(1, SCORED_FRAMES + 1):
        stem = f"clip_{i:04d}"
        (labels_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.0375 0.05\n")
        records.append({
            "image": str(Path("images") / f"{stem}.jpg"),
            "branch": "local yolo",
            "detections": [{"bbox": [78, 58, 84, 64], "conf": 1.0, "cls": 0}],
        })
    pred = tmp_path / "detections.jsonl"
    pred.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return video, pred, labels_dir


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Invoke `python -m src.render_video` from the repo root."""
    return subprocess.run([sys.executable, "-m", "src.render_video", *args],
                          cwd=REPO_ROOT, capture_output=True, text=True)


class TestRenderVideoCli:

    def test_renders_only_the_scored_frames(self, clip, tmp_path):
        video, pred, labels = clip
        out = tmp_path / "examples" / "overlay.mp4"
        result = run_cli("--video", str(video), "--pred", str(pred),
                         "--labels", str(labels), "--out", str(out))

        assert result.returncode == 0, result.stderr
        assert frame_count(out) == SCORED_FRAMES
        assert "2 unscored frames skipped" in result.stdout

    def test_output_keeps_the_source_frame_size(self, clip, tmp_path):
        video, pred, labels = clip
        out = tmp_path / "overlay.mp4"
        assert run_cli("--video", str(video), "--pred", str(pred),
                       "--labels", str(labels), "--out", str(out)).returncode == 0

        capture = cv2.VideoCapture(str(out))
        try:
            ok, frame = capture.read()
        finally:
            capture.release()
        assert ok and frame.shape[:2] == (HEIGHT, WIDTH)

    def test_max_frames_stops_early(self, clip, tmp_path):
        video, pred, labels = clip
        out = tmp_path / "overlay.mp4"
        assert run_cli("--video", str(video), "--pred", str(pred),
                       "--labels", str(labels), "--out", str(out),
                       "--max-frames", "3").returncode == 0
        assert frame_count(out) == 3

    def test_draws_both_boxes_on_a_true_positive(self, clip, tmp_path):
        """The detection sits exactly on the label, so the frame must carry a
        green ground-truth box and a blue matched-prediction box — and nothing
        the false-alarm or missed-target colours would explain.

        Compared by *which legend colour is nearest*, not by equality: mp4v is
        lossy and shifts a drawn value by tens of levels at this frame size.
        """
        from src.output import overlay

        video, pred, labels = clip
        out = tmp_path / "overlay.mp4"
        assert run_cli("--video", str(video), "--pred", str(pred),
                       "--labels", str(labels), "--out", str(out),
                       "--zoom", "0", "--no-caption").returncode == 0

        capture = cv2.VideoCapture(str(out))
        try:
            _, frame = capture.read()
        finally:
            capture.release()

        def nearest(colour) -> int:
            return int(np.abs(frame.astype(int) - np.array(colour)).sum(axis=2).min())

        assert nearest(overlay.GT_COLOUR) < nearest(overlay.MISS_COLOUR)
        assert nearest(overlay.TP_COLOUR) < nearest(overlay.FP_COLOUR)
        assert max(nearest(overlay.GT_COLOUR), nearest(overlay.TP_COLOUR)) < 120

    def test_unknown_prefix_is_an_error_not_an_empty_video(self, clip, tmp_path):
        video, pred, labels = clip
        out = tmp_path / "overlay.mp4"
        result = run_cli("--video", str(video), "--pred", str(pred),
                         "--labels", str(labels), "--out", str(out),
                         "--key-prefix", "phantom99")

        assert result.returncode != 0
        assert "phantom99" in result.stderr
        assert not out.exists()

    def test_missing_label_directory_is_rejected(self, clip, tmp_path):
        video, pred, _ = clip
        result = run_cli("--video", str(video), "--pred", str(pred),
                         "--labels", str(tmp_path / "nope"),
                         "--out", str(tmp_path / "overlay.mp4"))
        assert result.returncode != 0
        assert "must be a directory" in result.stderr
