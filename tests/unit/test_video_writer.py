"""Unit tests for the lazily-opened mp4 writer shared by both video sinks."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.output.video import LazyVideoWriter


@pytest.fixture
def frame():
    """One small colour frame, big enough for the codec to accept."""
    return np.full((64, 96, 3), 90, dtype=np.uint8)


def frame_count(path) -> int:
    """Decode a written file and count what actually came back out."""
    capture = cv2.VideoCapture(str(path))
    try:
        return sum(1 for _ in iter(lambda: capture.read()[0], False))
    finally:
        capture.release()


class TestLazyVideoWriter:

    def test_writes_nothing_until_a_frame_arrives(self, tmp_path):
        path = tmp_path / "out.mp4"
        with LazyVideoWriter(path, 30.0) as writer:
            assert writer.frames == 0
        assert not path.exists()

    def test_round_trips_every_frame_written(self, tmp_path, frame):
        path = tmp_path / "out.mp4"
        with LazyVideoWriter(path, 30.0) as writer:
            for _ in range(5):
                writer.write(frame)
            assert writer.frames == 5
        assert frame_count(path) == 5

    def test_creates_missing_parent_directories(self, tmp_path, frame):
        path = tmp_path / "examples" / "nested" / "out.mp4"
        with LazyVideoWriter(path, 30.0) as writer:
            writer.write(frame)
        assert path.exists()

    def test_closing_twice_is_harmless(self, tmp_path, frame):
        writer = LazyVideoWriter(tmp_path / "out.mp4", 30.0)
        writer.write(frame)
        writer.close()
        writer.close()
        assert frame_count(tmp_path / "out.mp4") == 1
