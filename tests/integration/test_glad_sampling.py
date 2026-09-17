"""`src.glad_detect.run_video` under a duty cycle: which frames reach the model.

The schedules themselves are pinned in `tests/unit/test_sampling.py`. What is
covered here is the wiring, which is where a duty cycle can go wrong invisibly:
the recorded frame keys have to be the ones the schedule names, and a burst has
to start from a cleared pipeline. Neither shows up in the metric block if it is
wrong -- a stale `_prev` produces boxes, just meaningless ones -- so it is
asserted directly.

No weights and no video file: the pipeline is a stub and the decoder is
monkeypatched, which is what lets this run in the fast suite.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from src import glad_detect
from src.algo.glad.pipeline import GLOBAL_YOLO, StepResult
from src.data.sampling import Bursts, EveryNth, Schedule
from src.output.recording import RunRecorder

TOTAL_FRAMES = 12


class StubPipeline:
    """Records which frames it was stepped over and when it was reset."""

    def __init__(self) -> None:
        self.stepped: list[int] = []
        self.resets: list[int] = []
        self._cold = True

    def reset(self) -> None:
        self._cold = True
        self.resets.append(len(self.stepped))

    def step(self, frame: np.ndarray) -> StepResult:
        # The marker in the corner is the source frame index, so the stub can
        # report exactly which frames the loop chose to hand it.
        self.stepped.append(int(frame[0, 0, 0]))
        self._cold = False
        return StepResult(np.array([10.0, 20.0, 5.0, 5.0]), GLOBAL_YOLO)


class StubCapture:
    """A `cv2.VideoCapture` that yields `TOTAL_FRAMES` numbered frames."""

    def __init__(self, _path: str) -> None:
        self._index = 0

    def isOpened(self) -> bool:  # noqa: N802 -- mirrors cv2
        return True

    def read(self):
        if self._index >= TOTAL_FRAMES:
            return False, None
        self._index += 1
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[0, 0, 0] = self._index
        return True, image

    def release(self) -> None:
        pass


@pytest.fixture
def labelled(tmp_path, monkeypatch):
    """Every frame labelled, so recording is decided by the schedule alone."""
    monkeypatch.setattr(glad_detect.cv2, "VideoCapture", StubCapture)
    labels = tmp_path / "labels"
    labels.mkdir()
    for i in range(1, TOTAL_FRAMES + 1):
        (labels / f"vid_{i:04d}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    return labels


def drive(tmp_path, labels: Path, schedule: Schedule, record_all: bool = False):
    """Run one stubbed video under `schedule`; return the stub and the rows."""
    args = argparse.Namespace(labels=labels, images=tmp_path / "images",
                              max_frames_per_video=None, record_all=record_all)
    pipeline = StubPipeline()
    out = tmp_path / "detections.jsonl"
    with RunRecorder(out) as recorder:
        decoded, processed = glad_detect.run_video(
            pipeline, tmp_path / "vid.mp4", "vid", args, recorder, Counter(), schedule)
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    return pipeline, rows, decoded, processed


def keys(rows: list[dict]) -> list[str]:
    """Recorded frame stems, which is what `src.evaluate` scores against."""
    return [Path(row["image"]).stem for row in rows]


class TestFullRate:
    """The default path must be byte-for-byte what EXP-004 ran."""

    def test_processes_and_records_every_frame(self, tmp_path, labelled):
        pipeline, rows, decoded, processed = drive(tmp_path, labelled, Schedule())
        assert pipeline.stepped == list(range(1, TOTAL_FRAMES + 1))
        assert decoded == processed == TOTAL_FRAMES
        assert len(rows) == TOTAL_FRAMES

    def test_resets_once_per_video_only(self, tmp_path, labelled):
        pipeline, _, _, _ = drive(tmp_path, labelled, Schedule())
        assert pipeline.resets == [0]

    def test_records_no_burst_fields(self, tmp_path, labelled):
        _, rows, _, _ = drive(tmp_path, labelled, Schedule())
        assert "burst" not in rows[0]


class TestHalfRate:

    def test_steps_only_the_scheduled_frames(self, tmp_path, labelled):
        pipeline, rows, decoded, processed = drive(tmp_path, labelled, EveryNth(2))
        assert pipeline.stepped == [1, 3, 5, 7, 9, 11]
        assert (decoded, processed) == (TOTAL_FRAMES, 6)
        assert keys(rows) == ["vid_0001", "vid_0003", "vid_0005",
                             "vid_0007", "vid_0009", "vid_0011"]

    def test_decodes_everything_it_skips(self, tmp_path, labelled):
        """Decode cost is not what a duty cycle saves, and must not be faked."""
        _, _, decoded, processed = drive(tmp_path, labelled, EveryNth(3))
        assert decoded == TOTAL_FRAMES
        assert processed == 4

    def test_never_resets_mid_video(self, tmp_path, labelled):
        pipeline, _, _, _ = drive(tmp_path, labelled, EveryNth(2))
        assert pipeline.resets == [0]


class TestBursts:

    def test_steps_pairs_at_the_period(self, tmp_path, labelled):
        pipeline, rows, _, processed = drive(tmp_path, labelled, Bursts(2, 5))
        assert pipeline.stepped == [1, 2, 6, 7, 11, 12]
        assert processed == 6
        assert keys(rows) == ["vid_0001", "vid_0002", "vid_0006",
                             "vid_0007", "vid_0011", "vid_0012"]

    def test_resets_before_the_first_frame_of_every_burst(self, tmp_path, labelled):
        """The load-bearing assertion.

        Without a reset the motion branches would difference the head of a burst
        against the tail of the previous one -- seconds of ego-motion apart --
        and fill the run with blobs that are an artefact of the harness. The
        reset counts are the number of frames stepped when each reset happened:
        once for the video, then before frames 1, 6 and 11.
        """
        pipeline, _, _, _ = drive(tmp_path, labelled, Bursts(2, 5))
        assert pipeline.resets == [0, 0, 2, 4]

    def test_records_which_burst_each_frame_belongs_to(self, tmp_path, labelled):
        _, rows, _, _ = drive(tmp_path, labelled, Bursts(2, 5))
        assert [(r["burst"], r["in_burst"]) for r in rows] == [
            (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

    def test_unlabelled_frames_are_processed_but_not_recorded(self, tmp_path,
                                                              monkeypatch):
        """Recording follows the labels; processing follows the schedule."""
        monkeypatch.setattr(glad_detect.cv2, "VideoCapture", StubCapture)
        labels = tmp_path / "labels"
        labels.mkdir()
        (labels / "vid_0002.txt").write_text("0 0.5 0.5 0.1 0.1\n")

        pipeline, rows, _, processed = drive(tmp_path, labels, Bursts(2, 5))
        assert processed == 6
        assert pipeline.stepped == [1, 2, 6, 7, 11, 12]
        assert keys(rows) == ["vid_0002"]


class TestRecordAll:
    """Unlabelled footage: `--record-all` decides recording, the schedule still
    decides processing.

    This is the field-capture path (EXP-010). The gate it lifts is the one that
    keeps a *labelled* run honest, so the pair of tests here is really one
    assertion in two directions: with the flag every processed frame is kept
    even though no label file exists anywhere, and without it the same footage
    records nothing at all.
    """

    @pytest.fixture
    def unlabelled(self, tmp_path, monkeypatch):
        """An empty label directory -- footage nobody has annotated."""
        monkeypatch.setattr(glad_detect.cv2, "VideoCapture", StubCapture)
        labels = tmp_path / "labels"
        labels.mkdir()
        return labels

    def test_records_every_processed_frame(self, tmp_path, unlabelled):
        _, rows, decoded, processed = drive(tmp_path, unlabelled, Schedule(),
                                            record_all=True)
        assert decoded == processed == TOTAL_FRAMES
        assert keys(rows) == [f"vid_{i:04d}" for i in range(1, TOTAL_FRAMES + 1)]

    def test_records_nothing_without_the_flag(self, tmp_path, unlabelled):
        """The default gate, which is what stops an unannotated frame being
        scored as if it were a labelled negative."""
        _, rows, _, processed = drive(tmp_path, unlabelled, Schedule())
        assert processed == TOTAL_FRAMES
        assert rows == []

    def test_still_records_only_scheduled_frames(self, tmp_path, unlabelled):
        """`--record-all` lifts the label gate, not the duty cycle."""
        pipeline, rows, _, processed = drive(tmp_path, unlabelled, EveryNth(2),
                                             record_all=True)
        assert pipeline.stepped == [1, 3, 5, 7, 9, 11]
        assert processed == 6
        assert keys(rows) == ["vid_0001", "vid_0003", "vid_0005",
                             "vid_0007", "vid_0009", "vid_0011"]
