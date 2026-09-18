"""Integration tests for the interactive annotator, driven without a window.

`Annotator` takes mouse events and key codes as plain method calls, so the
whole gesture -> session -> export path runs here against a real video file:
draw a box by dragging, let it follow the target, correct it, declare frames
empty, and check what lands on disk.

The clip is written as **FFV1**, the codec the processed O4 clips use, so
frames decode bit-exact and "is this the right frame" has an exact answer.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from src.data.annotate import (KEY_LEFT, KEY_RIGHT, Annotator, FrameCache,
                               display_size)
from src.data.annotation import CARRIED, HUMAN, TRACKED, Follower, Session

WIDTH, HEIGHT, FRAMES, SPEED = 320, 200, 40, 8  # target exits at frame 38
START_X, Y, SIZE = 20, 90, 12


def target_x(frame: int) -> int:
    """Where the synthetic drone sits on a 1-based frame."""
    return START_X + SPEED * (frame - 1)


def render_frame(frame: int) -> np.ndarray:
    rng = np.random.default_rng(7)
    image = rng.integers(20, 60, size=(HEIGHT, WIDTH, 3), dtype=np.int16).astype(np.uint8)
    patch = np.random.default_rng(20260917).integers(40, 250, size=(SIZE, SIZE),
                                                     dtype=np.int16)
    x = target_x(frame)
    if x + SIZE <= WIDTH:
        image[Y:Y + SIZE, x:x + SIZE] = np.repeat(patch[:, :, None], 3, axis=2)
    return image


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("clip") / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 30,
                             (WIDTH, HEIGHT))
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot write FFV1")
    for frame in range(1, FRAMES + 1):
        writer.write(render_frame(frame))
    writer.release()
    return path


@pytest.fixture
def capture(clip):
    capture = cv2.VideoCapture(str(clip))
    yield capture
    capture.release()


class TestFrameCache:

    def test_sequential_reads_never_seek(self, capture):
        cache = FrameCache(capture, size=10)
        for frame in range(1, 15):
            assert np.array_equal(cache.get(frame), render_frame(frame))
        assert cache.seeks == 0

    def test_stepping_back_inside_the_buffer_is_free(self, capture):
        cache = FrameCache(capture, size=10)
        cache.get(12)
        assert np.array_equal(cache.get(8), render_frame(8))
        assert cache.seeks == 0

    def test_short_jump_forward_reads_through(self, capture):
        """Reading through is exact on any container; a seek may not be."""
        cache = FrameCache(capture, size=10, read_through=30)
        cache.get(1)
        assert np.array_equal(cache.get(25), render_frame(25))
        assert cache.seeks == 0

    def test_far_jumps_seek_and_land_on_the_right_frame(self, capture):
        cache = FrameCache(capture, size=5, read_through=3)
        cache.get(30)
        assert np.array_equal(cache.get(4), render_frame(4))
        assert cache.seeks == 2  # 1 -> 30 is past read_through, and so is 30 -> 4

    def test_walking_back_past_the_cache_seeks_once_per_half_cache(self, capture):
        """One seek per keypress cost 454 ms a step on real footage."""
        cache = FrameCache(capture, size=10, read_through=3)
        cache.get(35)
        seeks = cache.seeks
        for frame in range(34, 14, -1):
            assert np.array_equal(cache.get(frame), render_frame(frame))
        assert cache.seeks - seeks <= 20 // 5 + 1

    def test_past_the_end_is_none(self, capture):
        assert FrameCache(capture).get(FRAMES + 1) is None


def make_annotator(capture, tmp_path, start=1):
    session = Session(tmp_path / "clip.avi", "clip", WIDTH, HEIGHT)
    saved = []
    annotator = Annotator(FrameCache(capture), session, Follower(), FRAMES,
                          (WIDTH, HEIGHT), lambda: saved.append(True), start=start)
    return annotator, session, saved


def drag_mouse(annotator, start, end):
    """A left-button drag in window pixels; the display is 1:1 with the frame."""
    annotator.on_mouse(cv2.EVENT_LBUTTONDOWN, *start, 0)
    annotator.on_mouse(cv2.EVENT_MOUSEMOVE, *end, 0)
    annotator.on_mouse(cv2.EVENT_LBUTTONUP, *end, 0)


class TestAnnotator:

    def test_drawn_box_follows_the_target(self, capture, tmp_path):
        annotator, session, _ = make_annotator(capture, tmp_path)
        drag_mouse(annotator, (START_X, Y), (START_X + SIZE, Y + SIZE))
        assert session.get(1).source == HUMAN

        for _ in range(8):
            annotator.on_key(ord("d"))
        assert annotator.index == 9
        for frame in range(2, 10):
            verdict = session.get(frame)
            assert verdict.source == TRACKED
            assert verdict.box[0] == pytest.approx(target_x(frame), abs=1)

    def test_correction_becomes_the_template(self, capture, tmp_path):
        annotator, session, _ = make_annotator(capture, tmp_path, start=5)
        drag_mouse(annotator, (target_x(5) + 30, Y), (target_x(5) + 30 + SIZE, Y + SIZE))
        # Moving the box by its body onto the real target is a correction.
        drag_mouse(annotator, (target_x(5) + 36, Y + 6), (target_x(5) + 6, Y + 6))
        assert session.get(5).box == (target_x(5), Y, SIZE, SIZE)
        assert annotator.follower.seeded_from == 5
        annotator.on_key(KEY_RIGHT)
        assert session.get(6).box[0] == pytest.approx(target_x(6), abs=1)

    def test_playing_until_the_target_leaves_stops_on_loss(self, capture, tmp_path):
        annotator, session, _ = make_annotator(capture, tmp_path)
        drag_mouse(annotator, (START_X, Y), (START_X + SIZE, Y + SIZE))
        annotator.on_key(ord(" "))
        for _ in range(FRAMES):
            annotator.tick()
        assert not annotator.playing
        assert "LOST" in annotator.message
        # The target exits once it no longer fits the frame.
        last_visible = max(f for f in range(1, FRAMES + 1) if target_x(f) + SIZE <= WIDTH)
        assert annotator.index <= last_visible + 1
        assert session.get(annotator.index) is None

    def test_negative_carries_forward_and_back_step_does_not_rewrite(self, capture,
                                                                    tmp_path):
        annotator, session, _ = make_annotator(capture, tmp_path, start=38)
        annotator.on_key(ord("x"))
        annotator.on_key(ord("d"))
        annotator.on_key(ord("d"))
        assert session.get(40).source == CARRIED and session.get(40).absent
        annotator.on_key(KEY_LEFT)
        assert session.get(39).source == CARRIED

    def test_featureless_box_is_kept_but_not_followed(self, capture, tmp_path):
        annotator, session, _ = make_annotator(capture, tmp_path)
        synthetic_sky = render_frame(1)[:, :, :].copy()
        synthetic_sky[:] = 120
        annotator.image = synthetic_sky
        drag_mouse(annotator, (200, 20), (240, 40))
        assert session.get(1).source == HUMAN
        assert "featureless" in annotator.message
        annotator.on_key(ord("d"))
        assert "LOST" in annotator.message and session.get(2) is None

    def test_clear_and_save(self, capture, tmp_path):
        annotator, session, saved = make_annotator(capture, tmp_path)
        annotator.on_key(ord("x"))
        annotator.on_key(ord("c"))
        assert session.get(1) is None
        annotator.on_key(ord("s"))
        assert saved == [True]
        annotator.on_key(ord("q"))
        assert annotator.quit

    def test_clicking_the_timeline_jumps_without_labelling(self, capture, tmp_path):
        annotator, session, _ = make_annotator(capture, tmp_path)
        drag_mouse(annotator, (START_X, Y), (START_X + SIZE, Y + SIZE))
        annotator.on_mouse(cv2.EVENT_LBUTTONDOWN, WIDTH // 2, HEIGHT + 3, 0)
        assert annotator.index == 1 + int(0.5 * FRAMES)
        assert set(session.verdicts) == {1}

    def test_render_has_the_status_bar(self, capture, tmp_path):
        annotator, _, _ = make_annotator(capture, tmp_path)
        annotator.on_key(ord("h"))
        canvas = annotator.render()
        assert canvas.shape[1] == WIDTH and canvas.shape[0] > HEIGHT

    def test_zoomed_drag_lands_in_image_pixels(self, capture, tmp_path):
        """At 4x, a 48-window-pixel drag is a 12-image-pixel box."""
        annotator, session, _ = make_annotator(capture, tmp_path)
        annotator.view.focus((START_X, Y, SIZE, SIZE), fraction=0.24)
        assert annotator.view.zoom == pytest.approx(4.0)
        x0, y0 = annotator.view.to_display(START_X, Y)
        drag_mouse(annotator, (int(x0), int(y0)), (int(x0) + 48, int(y0) + 48))
        assert session.get(1).box == pytest.approx((START_X, Y, SIZE, SIZE), abs=0.3)


def test_end_to_end_export_is_what_evaluate_reads(capture, tmp_path):
    annotator, session, _ = make_annotator(capture, tmp_path)
    drag_mouse(annotator, (START_X, Y), (START_X + SIZE, Y + SIZE))
    for _ in range(3):
        annotator.on_key(ord("d"))
    session.export(tmp_path / "labels" / "test", tmp_path / "verified.jsonl",
                   tmp_path / "images" / "test")
    labels = sorted(p.name for p in (tmp_path / "labels" / "test").iterdir())
    assert labels == ["clip_0001.txt", "clip_0002.txt", "clip_0003.txt", "clip_0004.txt"]
    rows = (tmp_path / "verified.jsonl").read_text().splitlines()
    assert [json.loads(r)["detections"] for r in rows] == [[]] * 4


def test_display_never_upscales_and_fits_the_limit():
    assert display_size(320, 200, (1400, 850)) == (320, 200)
    width, height = display_size(1440, 1080, (1400, 850))
    assert width <= 1400 and height <= 850 - 50
