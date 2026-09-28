"""Unit tests for `src.data.sources`.

This module decides what the user pointed `--source` at, and every detection CLI
starts by calling it. Two things are pinned here.

**The `.mp4` preference in `resolve_video`.** Datasets arrive as `.mp4`, but our
own processed footage cannot be: a lossless crop has to be FFV1, which MP4 cannot
carry, so `data/processed/SOFA-O4/videos/` holds `.avi`. A stem that exists in
both containers must keep resolving to the `.mp4`, or ARD-MAV and ARD100 silently
start reading a different file than the one every recorded run used.

**The exact exit messages.** These are characterization tests, written ahead of
the change that turns these `sys.exit` calls into a raised `UsageError` so the
module is callable from a notebook. They assert today's behaviour -- a
`SystemExit` carrying a particular string -- so that change has to prove it
preserved what the user sees rather than merely still passing.
"""

from __future__ import annotations

import pytest

from src.data.sources import IMAGES, VIDEO, resolve_sources, resolve_video


def touch(directory, *names):
    """Create empty files, so only the name and suffix matter to the test."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"")
    return directory


class TestResolveVideo:

    def test_finds_an_mp4(self, tmp_path):
        touch(tmp_path, "phantom08.mp4")
        assert resolve_video(tmp_path, "phantom08") == tmp_path / "phantom08.mp4"

    def test_finds_an_avi_when_that_is_all_there_is(self, tmp_path):
        """Our own FFV1 crops: MP4 cannot carry the codec, so they are `.avi`."""
        touch(tmp_path, "catch_2.avi")
        assert resolve_video(tmp_path, "catch_2") == tmp_path / "catch_2.avi"

    def test_prefers_mp4_over_every_other_container(self, tmp_path):
        """The load-bearing case: a stem present twice must resolve as it always
        did, or a re-run silently scores different pixels than the run it is
        being compared against."""
        touch(tmp_path, "clip.avi", "clip.mkv", "clip.mp4", "clip.mov")
        assert resolve_video(tmp_path, "clip") == tmp_path / "clip.mp4"

    @pytest.mark.parametrize("suffix", [".avi", ".mov", ".mkv", ".m4v", ".wmv"])
    def test_accepts_every_declared_container(self, tmp_path, suffix):
        touch(tmp_path, f"clip{suffix}")
        assert resolve_video(tmp_path, "clip") == tmp_path / f"clip{suffix}"

    def test_ignores_a_matching_stem_that_is_not_a_video(self, tmp_path):
        touch(tmp_path, "clip.txt", "clip.json")
        with pytest.raises(SystemExit):
            resolve_video(tmp_path, "clip")

    def test_exits_listing_what_it_did_find(self, tmp_path):
        """The listing is the whole value of the message: the cause is nearly
        always a typo in --video-names, and the fix is visible in the list."""
        touch(tmp_path, "clip.txt")
        with pytest.raises(SystemExit) as caught:
            resolve_video(tmp_path, "clip")
        message = str(caught.value)
        assert "No video named 'clip'" in message
        assert "found: clip.txt" in message

    def test_exits_saying_so_when_the_directory_is_missing(self, tmp_path):
        with pytest.raises(SystemExit) as caught:
            resolve_video(tmp_path / "nope", "clip")
        assert "directory empty or missing" in str(caught.value)

    def test_exits_saying_so_when_the_directory_holds_nothing_matching(self, tmp_path):
        touch(tmp_path, "other.mp4")
        with pytest.raises(SystemExit) as caught:
            resolve_video(tmp_path, "clip")
        assert "directory empty or missing" in str(caught.value)


class TestResolveSources:

    def test_a_video_file_is_one_video(self, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"")
        assert resolve_sources(video) == (VIDEO, [video])

    def test_an_image_file_is_one_image(self, tmp_path):
        image = tmp_path / "frame.png"
        image.write_bytes(b"")
        assert resolve_sources(image) == (IMAGES, [image])

    @pytest.mark.parametrize("name", ["clip.MP4", "clip.Avi", "frame.PNG", "frame.JpEg"])
    def test_the_suffix_check_ignores_case(self, tmp_path, name):
        """Windows hands back whatever case the filesystem stored."""
        path = tmp_path / name
        path.write_bytes(b"")
        kind, found = resolve_sources(path)
        assert kind in (VIDEO, IMAGES) and found == [path]

    def test_a_directory_is_its_images_sorted(self, tmp_path):
        touch(tmp_path, "b.jpg", "a.jpg", "c.png")
        kind, found = resolve_sources(tmp_path)
        assert kind == IMAGES
        assert [p.name for p in found] == ["a.jpg", "b.jpg", "c.png"]

    def test_a_directory_is_searched_recursively(self, tmp_path):
        touch(tmp_path, "top.jpg")
        touch(tmp_path / "nested" / "deeper", "low.jpg")
        _kind, found = resolve_sources(tmp_path)
        assert {p.name for p in found} == {"top.jpg", "low.jpg"}

    def test_a_directory_ignores_non_images(self, tmp_path):
        touch(tmp_path, "frame.jpg", "labels.txt", "clip.mp4")
        _kind, found = resolve_sources(tmp_path)
        assert [p.name for p in found] == ["frame.jpg"]

    def test_exits_on_a_directory_with_no_images(self, tmp_path):
        touch(tmp_path, "labels.txt")
        with pytest.raises(SystemExit) as caught:
            resolve_sources(tmp_path)
        assert "No images found under" in str(caught.value)

    def test_exits_on_an_unrecognised_suffix(self, tmp_path):
        path = tmp_path / "notes.pdf"
        path.write_bytes(b"")
        with pytest.raises(SystemExit) as caught:
            resolve_sources(path)
        assert "Unrecognised source type" in str(caught.value)
