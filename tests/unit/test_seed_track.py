"""Unit tests for the seed-and-track CLI's argument parsing.

The parsing is pinned because it decides what lands in a label file, and a label
file is ground truth for every number taken afterwards -- a silently clamped or
transposed box would not fail, it would just make the metrics wrong.

The tracker moved to test_template_track.py and the label formats to
test_label_writing.py, mirroring the split of seed_track.py itself.
"""

from __future__ import annotations

import pytest

from src.data.seed_track import parse_range, parse_seed


class TestParseSeed:

    def test_reads_frame_and_box(self):
        assert parse_seed(["962", "738", "382", "80", "34"]) == (
            962, (738.0, 382.0, 80.0, 34.0))

    def test_rejects_wrong_arity(self):
        with pytest.raises(ValueError, match="FRAME X Y W H"):
            parse_seed(["962", "738", "382"])

    def test_rejects_non_numbers(self):
        with pytest.raises(ValueError, match="numbers"):
            parse_seed(["962", "738", "382", "eighty", "34"])

    @pytest.mark.parametrize("box", [["1", "0", "0", "0", "10"],
                                     ["1", "0", "0", "10", "-2"]])
    def test_rejects_empty_box(self, box):
        with pytest.raises(ValueError, match="positive size"):
            parse_seed(box)

    def test_rejects_zero_frame_because_keys_are_one_based(self):
        """Frame keys are `<stem>_0001` for the *first* decoded frame; a 0 here
        would write labels one frame out for the whole segment."""
        with pytest.raises(ValueError, match="1-based"):
            parse_seed(["0", "1", "1", "10", "10"])


class TestParseRange:

    def test_inclusive_of_both_ends(self):
        assert list(parse_range("10-13")) == [10, 11, 12, 13]

    def test_single_frame_range(self):
        assert list(parse_range("7-7")) == [7]

    @pytest.mark.parametrize("text", ["10", "10-9", "0-5", "a-b", "1-2-3"])
    def test_rejects_malformed(self, text):
        with pytest.raises(ValueError):
            parse_range(text)

