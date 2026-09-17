"""`src.crops` selection and ordering, without decoding a video.

The rendering is covered in `tests/unit/test_contact.py`. What matters here is
which detections get a cell and in what order, because both are easy to get
subtly wrong in ways a finished sheet still looks plausible under: a span filter
off by one silently drops the frames being investigated, and a grouping that
does not actually group buries the six-frame branches the sheet exists to show.

The one thing asserted about `render` is that it decodes **in one forward pass**.
Seeking to 800 scattered positions in a 388 MB mp4 costs minutes, and a
regression to per-detection seeking would still produce a correct sheet.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import crops
from src.algo.detections import Detections
from src.eval.labels import EvalFrame
from src.output.contact import Layout


def frame(index: int, branch: str, boxes: int = 1) -> EvalFrame:
    """One recorded frame with `boxes` detections and a branch."""
    return EvalFrame(
        key=f"vid_{index:04d}",
        preds=Detections(
            boxes=np.array([[10.0 + 5 * i, 10.0, 20.0 + 5 * i, 20.0]
                            for i in range(boxes)]),
            scores=np.ones(boxes),
            classes=np.zeros(boxes, dtype=int),
        ) if boxes else Detections.empty(),
        extras={"branch": branch},
    )


class TestParseSpans:

    def test_a_span(self) -> None:
        assert crops.parse_spans(["1990-2120"]) == [(1990, 2120)]

    def test_several(self) -> None:
        assert crops.parse_spans(["1-2", "5-9"]) == [(1, 2), (5, 9)]

    def test_spaces_are_tolerated(self) -> None:
        assert crops.parse_spans([" 10 - 20 "]) == [(10, 20)]

    def test_none_is_empty(self) -> None:
        assert crops.parse_spans(None) == []

    @pytest.mark.parametrize("bad", ["1990", "a-b", "1-2-3"])
    def test_malformed_exits(self, bad: str) -> None:
        with pytest.raises(SystemExit):
            crops.parse_spans([bad])

    def test_a_backwards_span_exits(self) -> None:
        with pytest.raises(SystemExit, match="ends before it starts"):
            crops.parse_spans(["20-10"])


class TestFrameIndex:

    def test_reads_the_trailing_number(self) -> None:
        assert crops.frame_index("captured_raw_20260616_040253_004_2008") == 2008


class TestWanted:
    """Which detections get a cell."""

    def test_every_detection_of_every_frame(self) -> None:
        frames = [frame(1, "local yolo"), frame(2, "local yolo", boxes=2)]
        assert len(crops.wanted(frames, [], "branch")) == 3

    def test_frames_with_no_detection_contribute_nothing(self) -> None:
        """A `global miss` row is recorded but has no box to crop."""
        frames = [frame(1, "global miss", boxes=0), frame(2, "local yolo")]
        picks = crops.wanted(frames, [], "branch")
        assert [p[0] for p in picks] == [2]

    def test_a_span_is_inclusive_at_both_ends(self) -> None:
        frames = [frame(i, "local yolo") for i in (9, 10, 15, 20, 21)]
        picks = crops.wanted(frames, [(10, 20)], "branch")
        assert [p[0] for p in picks] == [10, 15, 20]

    def test_several_spans_union(self) -> None:
        frames = [frame(i, "local yolo") for i in range(1, 11)]
        picks = crops.wanted(frames, [(1, 2), (9, 10)], "branch")
        assert [p[0] for p in picks] == [1, 2, 9, 10]

    def test_the_group_comes_from_the_recorded_field(self) -> None:
        picks = crops.wanted([frame(1, "global mod")], [], "branch")
        assert picks[0][2] == "global mod"

    def test_a_missing_field_falls_back_rather_than_raising(self) -> None:
        picks = crops.wanted([frame(1, "local yolo")], [], "nosuchfield")
        assert picks[0][2] == crops.UNGROUPED

    def test_grouping_off_puts_everything_in_one_bucket(self) -> None:
        frames = [frame(1, "local yolo"), frame(2, "global mod")]
        picks = crops.wanted(frames, [], crops.NO_GROUPING)
        assert {p[2] for p in picks} == {crops.UNGROUPED}


class TestOrder:
    """Cell order, which is what makes a rare branch visible."""

    def test_groups_into_blocks_with_the_rare_branches_first(self) -> None:
        frames = ([frame(i, "local yolo") for i in range(1, 6)]
                  + [frame(99, "global yolo")])
        picks = crops.order(crops.wanted(frames, [], "branch"), "branch")
        assert [p[2] for p in picks][0] == "global yolo"

    def test_frame_order_is_kept_inside_a_group(self) -> None:
        frames = [frame(i, "local yolo") for i in (7, 3, 5)]
        picks = crops.order(crops.wanted(frames, [], "branch"), "branch")
        assert [p[0] for p in picks] == [3, 5, 7]

    def test_grouping_off_is_plain_frame_order(self) -> None:
        frames = [frame(9, "global yolo"), frame(2, "local yolo")]
        picks = crops.order(crops.wanted(frames, [], crops.NO_GROUPING),
                            crops.NO_GROUPING)
        assert [p[0] for p in picks] == [2, 9]


class TestRenderDecodesOnce:
    """The decode strategy, which is a performance contract worth pinning."""

    @pytest.fixture
    def capture(self, monkeypatch):
        """A capture that counts reads and refuses to be seeked."""

        class Capture:
            reads = 0
            seeks = 0

            def isOpened(self):  # noqa: N802 -- mirrors cv2
                return True

            def read(self):
                Capture.reads += 1
                if Capture.reads > 50:
                    return False, None
                return True, np.zeros((60, 60, 3), dtype=np.uint8)

            def set(self, *_):
                Capture.seeks += 1
                return True

            def release(self):
                pass

        Capture.reads = Capture.seeks = 0
        monkeypatch.setattr(crops.cv2, "VideoCapture", lambda _p: Capture())
        return Capture

    def test_never_seeks(self, capture, tmp_path) -> None:
        picks = crops.order(
            crops.wanted([frame(i, "local yolo") for i in (2, 20, 40)], [], "branch"),
            "branch")
        crops.render(picks, tmp_path / "v.mp4", None, Layout(cell=20))
        assert capture.seeks == 0

    def test_stops_at_the_last_wanted_frame(self, capture, tmp_path) -> None:
        """Decoding past the last detection is wasted work on a long video."""
        picks = crops.order(
            crops.wanted([frame(i, "local yolo") for i in (2, 5)], [], "branch"),
            "branch")
        crops.render(picks, tmp_path / "v.mp4", None, Layout(cell=20))
        assert capture.reads == 5

    def test_returns_a_cell_per_detection_of_a_frame(self, capture, tmp_path) -> None:
        picks = crops.order(crops.wanted([frame(3, "local yolo", boxes=2)], [],
                                         "branch"), "branch")
        cells = crops.render(picks, tmp_path / "v.mp4", None, Layout(cell=20))
        assert len(cells[3]) == 2
        assert capture.reads == 3  # one pass to frame 3, both boxes off that frame
