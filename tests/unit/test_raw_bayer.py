"""Unit tests for `src.data.raw_bayer`: the headerless Bayer reader behind the FIELD `.raw`."""
import cv2
import numpy as np
import pytest

from src.data.raw_bayer import RawBayerCapture, RawLayout, decode, open_capture

LAYOUT = RawLayout(width=8, height=6)


def _write(path, n_frames: int, layout: RawLayout = LAYOUT, tail: int = 0) -> list[np.ndarray]:
    """`n_frames` distinct mosaics back to back, plus `tail` stray bytes."""
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, (layout.height, layout.width), np.uint8)
              for _ in range(n_frames)]
    path.write_bytes(b"".join(f.tobytes() for f in frames) + b"\x00" * tail)
    return frames


def test_frame_count_ignores_a_trailing_partial_frame(tmp_path):
    p = tmp_path / "x.raw"
    _write(p, 3, tail=5)
    assert LAYOUT.frame_bytes == 48
    assert LAYOUT.n_frames(str(p)) == 3


def test_decode_demosaics_then_rotates_180():
    mosaic = np.arange(48, dtype=np.uint8).reshape(6, 8)
    want = cv2.rotate(cv2.cvtColor(mosaic, cv2.COLOR_BayerBG2BGR), cv2.ROTATE_180)
    assert np.array_equal(decode(mosaic, LAYOUT), want)


def test_decode_without_rotation_is_the_plain_demosaic():
    mosaic = np.arange(48, dtype=np.uint8).reshape(6, 8)
    flat = RawLayout(width=8, height=6, rotate_180=False)
    assert np.array_equal(decode(mosaic, flat), cv2.cvtColor(mosaic, cv2.COLOR_BayerBG2BGR))


def test_decode_shrinks_to_the_requested_size():
    mosaic = np.full((6, 8), 100, np.uint8)
    img = decode(mosaic, LAYOUT, (4, 3))
    assert img.shape == (3, 4, 3)


def test_capture_reads_frames_in_order_then_stops(tmp_path):
    p = tmp_path / "x.raw"
    frames = _write(p, 3)
    cap = RawBayerCapture(str(p), layout=LAYOUT)
    for f in frames:
        ok, img = cap.read()
        assert ok and np.array_equal(img, decode(f, LAYOUT))
    assert cap.read() == (False, None)
    cap.release()
    assert not cap.isOpened()


def test_capture_seeks_by_frame_position(tmp_path):
    p = tmp_path / "x.raw"
    frames = _write(p, 4)
    cap = RawBayerCapture(str(p), layout=LAYOUT)
    assert cap.set(cv2.CAP_PROP_POS_FRAMES, 2)
    ok, img = cap.read()
    assert ok and np.array_equal(img, decode(frames[2], LAYOUT))
    assert cap.get(cv2.CAP_PROP_POS_FRAMES) == 3
    cap.release()


def test_capture_reports_count_size_and_fps(tmp_path):
    p = tmp_path / "x.raw"
    _write(p, 2)
    cap = RawBayerCapture(str(p), size=(4, 3), layout=LAYOUT, fps=25.0)
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 2
    assert (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == (4, 3)
    assert cap.get(cv2.CAP_PROP_FPS) == 25.0
    assert cap.read()[1].shape == (3, 4, 3)
    assert not cap.set(cv2.CAP_PROP_FPS, 10)
    cap.release()


@pytest.mark.parametrize("name, raw", [("clip.raw", True), ("clip.RAW", True),
                                       ("clip.mp4", False)])
def test_open_capture_routes_by_extension(tmp_path, name, raw):
    p = tmp_path / name
    p.write_bytes(b"\x00" * (4128 * 3008))
    cap = open_capture(str(p))
    assert isinstance(cap, RawBayerCapture) == raw
    cap.release()
