"""Live feeds: adjacent pairs, and frames genuinely lost while the detector thinks.

The property under test is the one a live path gets wrong silently. A source
that hands back stale frames, or two frames that were not adjacent, still
produces boxes -- they are just boxes from a difference that means nothing. So
the assertions are about *which* frames come out, not about whether anything
comes out at all.

`cv2.VideoCapture` is stubbed throughout: no device, no file, no decoding.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from src.data import live
from src.data.live import HdmiCapture, Replay, SourceError, open_source

WIDTH, HEIGHT = 64, 48


def frame(marker: int) -> np.ndarray:
    """A frame carrying its own index, so a test can say which one it got."""
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[0, 0, 0] = marker % 256
    return image


def marker_of(image: np.ndarray) -> int:
    """Read back the index a frame was built with."""
    return int(image[0, 0, 0])


class StubCapture:
    """A capture that yields numbered frames, optionally ending or stalling."""

    instances: list["StubCapture"] = []

    def __init__(self, *_args, total: int = 10_000, fps: float = 30.0,
                 opened: bool = True, delay: float = 0.0) -> None:
        self.index = 0
        self.total = total
        self.fps = fps
        self._opened = opened
        self._delay = delay
        self.released = False
        self.props: dict[int, float] = {}
        StubCapture.instances.append(self)

    def isOpened(self) -> bool:  # noqa: N802 -- mirrors cv2
        return self._opened

    def read(self):
        if self.index >= self.total:
            return False, None
        if self._delay:
            time.sleep(self._delay)
        self.index += 1
        return True, frame(self.index)

    def set(self, prop: int, value: float) -> bool:
        self.props[prop] = value
        return True

    def get(self, prop: int) -> float:
        import cv2
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(WIDTH)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(HEIGHT)
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        return 0.0

    def release(self) -> None:
        self.released = True


@pytest.fixture
def stub_capture(monkeypatch):
    """Replace `cv2.VideoCapture` and hand back the factory for assertions."""
    StubCapture.instances.clear()

    def factory(*args, **kwargs):
        return StubCapture(*args, **kwargs)

    monkeypatch.setattr(live.cv2, "VideoCapture", factory)
    monkeypatch.setattr(Path, "is_file", lambda _self: True)
    return StubCapture


@pytest.mark.usefixtures("stub_capture")
class TestPairsAreAdjacent:
    """The invariant everything else rests on."""

    def test_a_pair_is_two_consecutive_frames(self):
        with HdmiCapture(0) as source:
            pair = source.pair()
            assert marker_of(pair.current) == marker_of(pair.previous) + 1

    def test_pairs_stay_adjacent_after_a_long_pause(self):
        """The whole point: thinking for a while must not straddle a gap.

        A naive `read()` loop would return two frames from the *queue* here --
        adjacent to each other but both stale, or spanning dropped frames.
        """
        with HdmiCapture(0) as source:
            source.pair()
            time.sleep(0.05)  # the detector "thinking"
            pair = source.pair()
            assert marker_of(pair.current) == marker_of(pair.previous) + 1

    def test_after_blocks_until_a_genuinely_newer_pair(self):
        with HdmiCapture(0) as source:
            first = source.pair()
            second = source.pair(after=first.index)
            assert second.index > first.index

    def test_frames_arriving_while_busy_are_lost_not_queued(self):
        """A live feed drops what nobody collected; it does not bank it."""
        with HdmiCapture(0) as source:
            first = source.pair()
            time.sleep(0.05)
            second = source.pair(after=first.index)
            # More frames were produced than were consumed, and the consumer
            # jumped straight to the newest rather than working through a queue.
            assert second.index - first.index > 1


@pytest.mark.usefixtures("stub_capture")
class TestGeometry:
    """Resolution is load-bearing: GLAD's motion constants are absolute pixels."""

    def test_reports_what_it_actually_got(self):
        with HdmiCapture(0) as source:
            assert source.frame_size == (WIDTH, HEIGHT)

    def test_a_non_native_feed_is_flagged(self):
        with HdmiCapture(0) as source:
            assert not source.is_native_resolution

    def test_an_implausible_frame_rate_is_discarded(self, monkeypatch):
        """UVC and capture cards routinely report 0 fps, and the policy divides by it."""
        monkeypatch.setattr(live.cv2, "VideoCapture",
                            lambda *_a, **_k: StubCapture(fps=0.0))
        with HdmiCapture(0) as source:
            assert source.source_fps == live.DEFAULT_FPS

    def test_a_plausible_frame_rate_is_kept(self, monkeypatch):
        monkeypatch.setattr(live.cv2, "VideoCapture",
                            lambda *_a, **_k: StubCapture(fps=59.94))
        with HdmiCapture(0) as source:
            assert source.source_fps == pytest.approx(59.94)


@pytest.mark.usefixtures("stub_capture")
class TestFailures:
    """Every failure names the cable or the file, not a stack trace."""

    def test_a_device_that_will_not_open(self, monkeypatch):
        monkeypatch.setattr(live.cv2, "VideoCapture",
                            lambda *_a, **_k: StubCapture(opened=False))
        with pytest.raises(SourceError, match="HDMI"):
            HdmiCapture(0).open()

    def test_a_missing_recording(self, monkeypatch):
        monkeypatch.setattr(Path, "is_file", lambda _self: False)
        with pytest.raises(SourceError, match="No such recording"):
            Replay(Path("nope.mp4")).open()

    def test_a_recording_that_ends(self, monkeypatch):
        monkeypatch.setattr(live.cv2, "VideoCapture",
                            lambda *_a, **_k: StubCapture(total=3))
        source = Replay(Path("short.mp4"), realtime=False)
        source.open()
        try:
            with pytest.raises(SourceError, match="ended"):
                for _ in range(20):
                    source.pair(after=source.produced + 100)
        finally:
            source.close()

    def test_closing_releases_the_device(self):
        source = HdmiCapture(0)
        source.open()
        source.close()
        assert all(c.released for c in StubCapture.instances)


@pytest.mark.usefixtures("stub_capture")
class TestReplayPacing:
    """A replay is a simulation only if it refuses to run faster than real time."""

    def test_realtime_replay_does_not_outrun_the_clock(self):
        source = Replay(Path("f.mp4"), realtime=True)
        source.open()
        try:
            time.sleep(0.2)
            produced = source.produced
        finally:
            source.close()
        # At 30 fps, 0.2 s can produce about 6 frames. A generous ceiling still
        # catches a replay that ignored the clock entirely and produced hundreds.
        assert produced < 40, f"replay outran real time: {produced} frames in 0.2s"

    def test_no_realtime_replay_runs_flat_out(self):
        source = Replay(Path("f.mp4"), realtime=False)
        source.open()
        try:
            time.sleep(0.2)
            produced = source.produced
        finally:
            source.close()
        assert produced > 40, "unpaced replay should be far faster than real time"


class TestOpenSource:
    """One flag serves the capture card that will exist and the file that does."""

    def test_a_digit_is_a_capture_device(self):
        assert isinstance(open_source("0"), HdmiCapture)
        assert open_source("2").index == 2

    def test_anything_else_is_a_recording(self):
        source = open_source("data/raw/ARD-MAV/videos/phantom05.mp4")
        assert isinstance(source, Replay)
        assert source.path.name == "phantom05.mp4"

    def test_realtime_and_loop_reach_the_replay(self):
        source = open_source("f.mp4", realtime=False, loop=True)
        assert source.realtime is False
        assert source.loop is True
