"""The ported motion branches must agree with the vendored `MOD2` they replace.

This is the gate that makes `src.algo.glad.motion` safe to use. The port exists
so the constants can be moved off 1080p; it is only worth anything if, left at
upstream's constants, it computes what upstream computes. Everything the ledger
says about EXP-004 and EXP-005 rests on that.

Marked `slow`: it needs the vendored GLAD clone, its weights, and real frames.
`py -3.13 -m pytest -m "not slow"` skips it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GLAD_DIR = REPO_ROOT / "third_party" / "GLAD"
# ARD-MAV is the footage GLAD was built for, where the crowding guard rarely
# fires -- so this exercises the path that *accepts* candidates. The O4 clip is
# the opposite case: every frame there saturates. Both have to agree, or the
# equivalence only holds on the half of the algorithm that returns nothing.
ARD_MAV = REPO_ROOT / "data" / "raw" / "ARD-MAV" / "videos" / "phantom05.mp4"
O4 = REPO_ROOT / "data" / "processed" / "SOFA-O4" / "videos" / "first_catch.avi"

pytestmark = pytest.mark.slow


def _require(condition: bool, reason: str) -> None:
    if not condition:
        pytest.skip(reason)


def read_run(path: Path, start: int, count: int) -> list[np.ndarray]:
    """A contiguous run of real frames, which is what the branches difference.

    Synthetic noise would not exercise this: the question is how the candidate
    list behaves on scenes that really do produce hundreds of blobs.
    """
    import cv2

    _require(path.exists(), f"{path} not present")
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    collected = []
    try:
        for _ in range(count):
            ok, frame = capture.read()
            if not ok:
                break
            collected.append(frame)
    finally:
        capture.release()
    _require(len(collected) >= 4, f"could not read enough frames from {path.name}")
    return collected


@pytest.fixture(scope="module")
def frames() -> list[np.ndarray]:
    """Crowded frames: every one of these saturates the guard."""
    return read_run(O4, 900, 8)


@pytest.fixture(scope="module")
def clean_frames() -> list[np.ndarray]:
    """ARD-MAV frames, where the guard does not fire and candidates are kept."""
    return read_run(ARD_MAV, 200, 10)


@pytest.fixture(scope="module")
def modules():
    """The vendored `MOD2` and the port, sharing one CNN gate.

    The gate has to be shared, not merely equivalent: it is a network, and two
    loads could differ in ways that would show up as a false mismatch here.
    """
    from src.algo.glad.classifier import load_gate
    from src.algo.glad.motion import UPSTREAM
    from src.algo.glad.vendor import import_motion, import_motion_port, weights_dir

    _require(GLAD_DIR.exists(), "vendored GLAD clone not present")
    weights = weights_dir(GLAD_DIR)
    _require((weights / "Net_best.pth").exists(), "GLAD weights not present")

    gate = load_gate(weights)
    vendored = import_motion(gate, GLAD_DIR)
    port = import_motion_port(gate, UPSTREAM, None, GLAD_DIR)
    return vendored, port


def as_tuple(result) -> tuple:
    """Both return `[]` for a miss and an `(x, y, w, h)` tuple otherwise."""
    return tuple(result) if len(result) else ()


class TestGlobalBranch:

    def test_agrees_on_crowded_frames(self, modules, frames):
        """The saturation path, where upstream discards everything."""
        vendored, port = modules
        for previous, current in zip(frames, frames[1:]):
            assert as_tuple(port.MOD2_global(previous, current)) == as_tuple(
                vendored.MOD2_global(previous, current))

    def test_agrees_on_the_footage_glad_was_built_for(self, modules, clean_frames):
        """The accept path. Without this the equivalence would only be proven
        for frames on which both return nothing, which is no equivalence at
        all."""
        vendored, port = modules
        for previous, current in zip(clean_frames, clean_frames[1:]):
            assert as_tuple(port.MOD2_global(previous, current)) == as_tuple(
                vendored.MOD2_global(previous, current))

    def test_the_two_fixtures_really_are_different_regimes(self, modules, frames,
                                                           clean_frames):
        """Guards the guard: if ARD-MAV also saturated, the test above would be
        exercising the same path twice and quietly prove nothing."""
        from src.algo.glad.motion import UPSTREAM, find_candidates

        import cv2

        _, port = modules

        def candidate_count(previous, current):
            prev_grey, cur_grey = port._prepare(previous), port._prepare(current)
            compensated, border, _ = port.compensate(prev_grey, cur_grey)
            difference = cv2.absdiff(cur_grey, compensated)
            threshold = UPSTREAM.global_threshold_base + int(np.mean(difference))
            binary = port._binary(difference, threshold, border, median=True)
            return len(find_candidates(binary, UPSTREAM.blob_area,
                                       UPSTREAM.global_blob_ratio))

        crowded = candidate_count(frames[0], frames[1])
        clean = candidate_count(clean_frames[0], clean_frames[1])
        assert crowded > UPSTREAM.max_candidates_global
        assert clean <= UPSTREAM.max_candidates_global


class TestLocalBranch:

    def test_agrees_frame_by_frame(self, modules, frames):
        vendored, port = modules
        region = slice(240, 560), slice(560, 880)  # a 320x320 search region
        for previous, current in zip(frames, frames[1:]):
            crop_prev = previous[region[0], region[1], :]
            crop_cur = current[region[0], region[1], :]
            assert as_tuple(port.MOD2_local(crop_prev, crop_cur, 160, 160)) == (
                as_tuple(vendored.MOD2_local(crop_prev, crop_cur, 160, 160)))


class TestClutterProfileDiffers:
    """The variant must actually change behaviour, or it is not worth a flag.

    Not an equivalence check — the opposite. If `clutter` produced identical
    output on footage that saturates the guard 3,595 times, the profile would be
    doing nothing and the diagnosis behind it would be wrong.
    """

    def test_keeps_candidates_where_upstream_discards_them(self, modules, frames):
        from src.algo.glad.classifier import load_gate
        from src.algo.glad.motion import CLUTTER
        from src.algo.glad.vendor import import_motion_port, weights_dir

        vendored, _ = modules
        clutter = import_motion_port(load_gate(weights_dir(GLAD_DIR)), CLUTTER,
                                     None, GLAD_DIR)
        upstream_hits = sum(bool(len(vendored.MOD2_global(a, b)))
                            for a, b in zip(frames, frames[1:]))
        clutter_hits = sum(bool(len(clutter.MOD2_global(a, b)))
                           for a, b in zip(frames, frames[1:]))
        assert clutter_hits >= upstream_hits
