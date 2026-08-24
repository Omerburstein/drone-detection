"""Sources that behave like a live feed: they keep running whether or not you read.

`FrameSource` in `frames.py` reads a file, where the next frame is simply the
next one and nothing is lost by taking time to think. A deployed feed is the
opposite, and that difference is the entire reason this module exists: video
keeps arriving at 30 fps while the detector spends 0.4 s on a frame, and
whatever arrives meanwhile is gone. What a real-time system detects is decided
as much by which frames it managed to look at as by how good the detector is.

Two sources, one behaviour:

    HdmiCapture    an HDMI-to-USB capture device. It enumerates as a UVC camera,
                   so OpenCV opens it like a webcam; what differs from a sensor
                   is that its reported geometry describes the *link* the drone
                   is sending, not an imager.
    Replay         saved drone footage, decoded on a clock instead of on demand.
                   With `realtime=True` it drops frames the consumer was too slow
                   to collect, exactly as the capture device would, so a
                   recording can be used to see what the live system *would have*
                   detected. Without it, nothing is dropped and the result is an
                   offline run.

Both hand out adjacent pairs, because GLAD's two motion branches difference the
current frame against the previous one. A naive `read()` loop cannot supply that
on a live feed, and fails silently in two ways:

* **The queue goes stale.** A `VideoCapture` buffers frames nobody took. After a
  0.7 s inference pass the next `read()` returns something from 0.7 s ago, and
  the loop falls further behind every iteration while appearing to work.
* **Consecutive reads are not consecutive frames.** Reading twice after a long
  gap yields two frames from the *queue* — both stale, or straddling a gap.

So the source is drained continuously on its own thread and keeps only the two
newest frames. A consumer asks for a pair when it is ready and gets two frames
that were genuinely adjacent, however long it spent thinking. That is what makes
burst-pair inference coherent on hardware slower than the feed; see
`src.algo.deployment`.

**Resolution is not cosmetic.** GLAD's motion constants are absolute pixels
tuned for 1920x1080 (`area 30-3000`, search region `a=160`, `dist_ref=200`), so
a feed that quietly delivers 1280x720 runs the motion branches at the wrong
scale. Every source reports what it actually got.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self

import cv2
import numpy as np

NATIVE_WIDTH = 1920  # what GLAD's absolute-pixel constants were tuned for
NATIVE_HEIGHT = 1080
DEFAULT_FPS = 30.0  # assumed when the device or container will not say
OPEN_TIMEOUT = 5.0  # seconds to wait for frames before giving up
POLL = 0.002  # seconds between checks while waiting for a fresh pair


@dataclass(frozen=True)
class FramePair:
    """Two frames that were adjacent in the feed, and when the later arrived.

    `previous` is the earlier of the two. Both are the producer's own buffers,
    so a consumer that draws on one must copy it first.
    """

    previous: np.ndarray
    current: np.ndarray
    captured_at: float
    index: int  # how many frames the feed had produced when `current` arrived

    @property
    def age(self) -> float:
        """Seconds since `current` arrived — the display's true latency."""
        return time.perf_counter() - self.captured_at


class SourceError(RuntimeError):
    """The feed could not be opened, or stopped delivering frames."""


class PairSource(Protocol):
    """What `src.live_detect` needs from a feed, live or replayed."""

    frame_size: tuple[int, int]
    source_fps: float

    def pair(self, after: int = 0) -> FramePair:
        ...


class _PairBuffer:
    """The two newest frames, written by a producer thread and read by anyone.

    Holding exactly two is the point: one frame is not enough to difference, and
    three would let a slow consumer work on frames that are no longer current
    while pretending to be live.
    """

    def __init__(self) -> None:
        self._frames: deque[tuple[np.ndarray, float, int]] = deque(maxlen=2)
        self._lock = threading.Lock()
        self._count = 0
        self.error: str | None = None
        self.finished = False

    def push(self, frame: np.ndarray) -> None:
        """Record one newly arrived frame, discarding anything older than the last."""
        with self._lock:
            self._count += 1
            self._frames.append((frame, time.perf_counter(), self._count))

    def take(self, after: int) -> FramePair | None:
        """The newest pair if it is newer than `after`, else None."""
        with self._lock:
            if len(self._frames) == 2 and self._frames[-1][2] > after:
                (previous, _, _), (current, stamp, index) = self._frames
                return FramePair(previous, current, stamp, index)
        return None

    @property
    def produced(self) -> int:
        """How many frames the producer has delivered so far."""
        with self._lock:
            return self._count


class _ThreadedSource:
    """Shared machinery: a producer thread feeding a `_PairBuffer`.

    Subclasses supply `_open`, which returns the `VideoCapture` and sets the
    geometry, and `_pace`, which decides how fast to read.
    """

    def __init__(self) -> None:
        self._capture: cv2.VideoCapture | None = None
        self._buffer = _PairBuffer()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.frame_size: tuple[int, int] = (0, 0)
        self.source_fps: float = DEFAULT_FPS

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc: BaseException | None, tb: TracebackType | None) -> None:
        self.close()

    def _open(self) -> cv2.VideoCapture:
        raise NotImplementedError

    def _pace(self) -> None:
        """Called after each frame is pushed. The default reads as fast as it can."""

    @property
    def label(self) -> str:
        """How the feed describes itself in the run log."""
        raise NotImplementedError

    def open(self) -> None:
        """Start the producer and block until a first pair exists."""
        self._capture = self._open()
        self._thread = threading.Thread(target=self._produce, daemon=True,
                                        name="live-source")
        self._thread.start()

        deadline = time.perf_counter() + OPEN_TIMEOUT
        while time.perf_counter() < deadline:
            if self._buffer.error:
                raise SourceError(self._buffer.error)
            if self._buffer.take(0) is not None:
                return
            time.sleep(POLL)
        raise SourceError(
            f"{self.label} opened but delivered no frames within "
            f"{OPEN_TIMEOUT:.0f}s.")

    def _produce(self) -> None:
        """Read continuously, keeping only the two newest frames."""
        assert self._capture is not None
        while not self._stop.is_set():
            ok, frame = self._capture.read()
            if not ok:
                self._buffer.finished = True
                return
            self._buffer.push(frame)
            self._pace()

    def pair(self, after: int = 0) -> FramePair:
        """The newest adjacent pair, waiting until it is newer than `after`.

        Passing the `index` of a pair already consumed is what stops a fast
        consumer processing the same two frames twice and reporting a
        motionless difference as a real result.
        """
        deadline = time.perf_counter() + OPEN_TIMEOUT
        while True:
            if self._buffer.error:
                raise SourceError(self._buffer.error)
            found = self._buffer.take(after)
            if found is not None:
                return found
            if self._buffer.finished:
                raise SourceError(f"{self.label} ended.")
            if time.perf_counter() > deadline:
                raise SourceError(
                    f"{self.label} delivered no new frame in {OPEN_TIMEOUT:.0f}s.")
            time.sleep(POLL)

    @property
    def produced(self) -> int:
        """Frames the feed has delivered, whether or not anyone looked at them."""
        return self._buffer.produced

    @property
    def is_native_resolution(self) -> bool:
        """Whether the feed is the 1920x1080 GLAD's constants were tuned for."""
        return self.frame_size == (NATIVE_WIDTH, NATIVE_HEIGHT)

    def close(self) -> None:
        """Stop the producer and release the device or file."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class HdmiCapture(_ThreadedSource):
    """An HDMI-to-USB capture device carrying the drone's downlink.

    Such devices enumerate as ordinary UVC cameras, so this is a webcam open in
    every mechanical respect. The difference worth remembering is semantic: the
    width, height and frame rate describe **the HDMI link the drone is sending**,
    not a sensor. A capture card will happily report 1920x1080 while the drone
    upstream is sending an upscaled 720p picture, and nothing in software can
    tell the difference — so a resolution check here is necessary and not
    sufficient.
    """

    def __init__(self, index: int = 0, width: int = NATIVE_WIDTH,
                 height: int = NATIVE_HEIGHT, backend: int | None = None) -> None:
        super().__init__()
        self.index = index
        self._requested = (width, height)
        # DirectShow honours resolution requests on Windows UVC devices; the
        # default MSMF backend frequently ignores them.
        self._backend = cv2.CAP_DSHOW if backend is None else backend

    @property
    def label(self) -> str:
        """Names the device by index, as the operator selected it."""
        return f"HDMI capture device {self.index}"

    def _open(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self.index, self._backend)
        if not capture.isOpened():
            raise SourceError(
                f"Could not open capture device {self.index}. Check the HDMI "
                f"cable into the capture card, the USB cable out of it, and that "
                f"nothing else holds the device.")

        width, height = self._requested
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # Ask for no backlog. Honoured by some backends, ignored by others,
        # which is why the drain thread exists regardless.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        reported = capture.get(cv2.CAP_PROP_FPS)
        # Capture cards routinely report 0 or something absurd, and the policy
        # choice divides by this number, so an implausible value is discarded.
        self.source_fps = reported if 1.0 <= reported <= 240.0 else DEFAULT_FPS
        return capture


class Replay(_ThreadedSource):
    """Saved drone footage decoded on a clock, as the live feed would arrive.

    This is the source to use before capture hardware exists, and it is more
    than a convenience. Decoding a file on demand measures the detector; pacing
    it at the recording's own frame rate measures **the deployed system**,
    because the frames the pipeline was too slow to collect are dropped here
    exactly as they would be dropped by a capture card that never waited.

    `realtime=False` turns that off and reads as fast as the consumer asks,
    which reproduces `src.glad_detect` and is useful only for checking that the
    display works.
    """

    def __init__(self, path: Path, realtime: bool = True,
                 loop: bool = False) -> None:
        super().__init__()
        self.path = path
        self.realtime = realtime
        self.loop = loop
        self._next_due = 0.0

    @property
    def label(self) -> str:
        """Names the file, which is what a replay's log line should say."""
        return f"replay of {self.path.name}"

    def _open(self) -> cv2.VideoCapture:
        if not self.path.is_file():
            raise SourceError(f"No such recording: {self.path}")
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            raise SourceError(f"Could not decode {self.path}.")

        self.frame_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        reported = capture.get(cv2.CAP_PROP_FPS)
        self.source_fps = reported if 1.0 <= reported <= 240.0 else DEFAULT_FPS
        self._next_due = time.perf_counter()
        return capture

    def _pace(self) -> None:
        """Sleep so frames leave the decoder at the recording's own rate.

        Paced against a running deadline rather than by sleeping a fixed
        interval each time: sleeping `1/fps` after work that already took
        longer would let the replay drift slower than real time, which would
        quietly flatter the duty cycle by giving the pipeline more wall-clock
        per frame than a live feed ever would.
        """
        if not self.realtime:
            return
        self._next_due += 1.0 / self.source_fps
        remaining = self._next_due - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)

    def _produce(self) -> None:
        """Decode to the end, optionally starting over."""
        assert self._capture is not None
        while not self._stop.is_set():
            ok, frame = self._capture.read()
            if not ok:
                if not self.loop:
                    self._buffer.finished = True
                    return
                self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._next_due = time.perf_counter()
                continue
            self._buffer.push(frame)
            self._pace()


def open_source(spec: str, realtime: bool = True, loop: bool = False,
                width: int = NATIVE_WIDTH,
                height: int = NATIVE_HEIGHT) -> _ThreadedSource:
    """Build the feed named by `--source`: a device index, or a path to footage.

    A bare integer is a capture device; anything else is treated as a file, so
    the same command line serves the recording available today and the HDMI
    capture card that replaces it later.
    """
    if spec.isdigit():
        return HdmiCapture(int(spec), width, height)
    return Replay(Path(spec), realtime=realtime, loop=loop)
