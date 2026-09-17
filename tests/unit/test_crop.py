"""Unit tests for `src.data.crop.Crop`.

The crop exists because a goggles screen recording is mostly not picture, and
the bars cost the target resolution in the detector's letterbox. Two things have
to hold or a run is silently wrong rather than loudly broken: the parser must
reject a degenerate rectangle instead of producing an empty frame, and `fits`
must catch a rectangle that runs off the source before 7,000 frames are spent on
it.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.crop import Crop

O4 = "540,0,1440,1080"  # the real one: a 2520x1080 goggles capture


class TestParse:

    def test_parses_the_o4_rectangle(self):
        assert Crop.parse(O4) == Crop(540, 0, 1440, 1080)

    def test_tolerates_spaces(self):
        assert Crop.parse("540, 0, 1440, 1080") == Crop(540, 0, 1440, 1080)

    @pytest.mark.parametrize("text", ["540,0,1440", "540,0,1440,1080,7", ""])
    def test_rejects_the_wrong_number_of_fields(self, text):
        with pytest.raises(ValueError, match="x,y,width,height"):
            Crop.parse(text)

    def test_rejects_non_integers(self):
        with pytest.raises(ValueError, match="integers"):
            Crop.parse("540,0,1440.5,1080")

    @pytest.mark.parametrize("text", ["540,0,0,1080", "540,0,1440,-4"])
    def test_rejects_an_empty_or_inverted_rectangle(self, text):
        """Without this the detector would be handed a zero-size frame and the
        failure would surface somewhere far from the flag that caused it."""
        with pytest.raises(ValueError, match="positive"):
            Crop.parse(text)

    def test_rejects_a_negative_origin(self):
        with pytest.raises(ValueError, match="negative"):
            Crop.parse("-1,0,1440,1080")


class TestFits:

    def test_accepts_a_rectangle_inside_the_frame(self):
        assert Crop.parse(O4).fits((1080, 2520))

    def test_rejects_one_that_runs_off_the_right_edge(self):
        assert not Crop.parse(O4).fits((1080, 1920))

    def test_rejects_one_that_runs_off_the_bottom(self):
        assert not Crop.parse("0,0,100,200").fits((199, 100))

    def test_exact_fit_is_a_fit(self):
        assert Crop.parse("0,0,100,200").fits((200, 100))


class TestApply:

    def test_returns_the_named_region(self):
        frame = np.arange(1080 * 2520 * 3, dtype=np.uint8).reshape(1080, 2520, 3)
        out = Crop.parse(O4).apply(frame)
        assert out.shape == (1080, 1440, 3)
        assert np.array_equal(out, frame[0:1080, 540:1980])

    def test_is_a_view_not_a_copy(self):
        """The caller hands this straight to the detector and drops the source."""
        frame = np.zeros((10, 20, 3), dtype=np.uint8)
        out = Crop.parse("5,2,10,5").apply(frame)
        out[0, 0] = 255
        assert frame[2, 5, 0] == 255


class TestLabel:

    def test_reads_back_as_geometry_and_origin(self):
        assert Crop.parse(O4).label == "1440x1080 at (540, 0)"


class TestResolveVideo:
    """`resolve_video` finds a stem's video whatever container it is in.

    `.mp4` was hardcoded while every dataset was an mp4 release. Our processed
    footage is FFV1 in `.avi`, because MP4 cannot carry a lossless codec — so
    the resolver has to look wider while still preferring `.mp4`, or ARD-MAV
    and ARD100 would start resolving differently than they did for EXP-004.
    """

    def test_prefers_mp4(self, tmp_path):
        from src.data.sources import resolve_video
        (tmp_path / "phantom19.mp4").write_bytes(b"")
        (tmp_path / "phantom19.avi").write_bytes(b"")
        assert resolve_video(tmp_path, "phantom19").name == "phantom19.mp4"

    def test_finds_avi_when_there_is_no_mp4(self, tmp_path):
        from src.data.sources import resolve_video
        (tmp_path / "first_catch.avi").write_bytes(b"")
        assert resolve_video(tmp_path, "first_catch").name == "first_catch.avi"

    def test_a_missing_stem_exits_with_what_is_there(self, tmp_path):
        from src.data.sources import resolve_video
        (tmp_path / "first_catch.avi").write_bytes(b"")
        with pytest.raises(SystemExit) as excinfo:
            resolve_video(tmp_path, "second_catch")
        assert "second_catch" in str(excinfo.value)

    def test_does_not_match_a_non_video_suffix(self, tmp_path):
        from src.data.sources import resolve_video
        (tmp_path / "clip.txt").write_bytes(b"")
        with pytest.raises(SystemExit):
            resolve_video(tmp_path, "clip")
