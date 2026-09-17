"""Tests for running GLAD on footage shot at the wrong resolution.

Two properties carry the weight here, and they pull in opposite directions.

**The no-op must be a genuine no-op.** EXP-004-010 ran the unwrapped pipeline,
and every one of those numbers stays citable only if a scale factor of 1 -- and
`auto` on the 1920x1080 footage all of them used -- reaches the inner pipeline
with the frame untouched. That is asserted on object identity, not on values,
because a resize to the same dimensions would compare equal while still having
run an interpolation kernel over the pixels.

**The box must come back in original coordinates.** A scaled run's
`detections.jsonl` is meant to be directly comparable to an unscaled one's, so
the mapping back is the one thing a caller can never see going wrong -- boxes in
scaled coordinates would look entirely plausible and be silently offset.

Nothing here needs weights, the vendored clone, or OpenCV doing real work on a
real image: the inner pipeline is a stub that reports what it was handed.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.algo.glad.pipeline import GLOBAL_MISS, LOCAL_YOLO, StepResult
from src.algo.glad.scaling import (AUTO, REFERENCE_DIAGONAL, ScaledPipeline,
                                   parse_scale, scale_for_diagonal)

# The capture EXP-010 ran on, and the arithmetic docs/datasets.md records for it.
FIELD_W, FIELD_H = 1032, 752
FIELD_FACTOR = 1.7252
REFERENCE_W, REFERENCE_H = 1920, 1080


def frame(width: int, height: int) -> np.ndarray:
    """A frame of a given size, with no content anything here depends on."""
    return np.zeros((height, width, 3), dtype=np.uint8)


class StubPipeline:
    """Reports the frames it was stepped over and returns scripted results."""

    def __init__(self, *results: StepResult) -> None:
        self._results = list(results)
        self.frames: list[np.ndarray] = []
        self.resets = 0

    def step(self, image: np.ndarray) -> StepResult:
        self.frames.append(image)
        return self._results.pop(0) if self._results else StepResult(None, GLOBAL_MISS)

    def reset(self) -> None:
        self.resets += 1


def box(*values: float) -> StepResult:
    """A scripted hit at `[x, y, w, h]`."""
    return StepResult(np.array(values, dtype=float), LOCAL_YOLO)


class TestScaleForDiagonal:
    """The factor itself, against the arithmetic the ledger already records."""

    def test_reference_resolution_needs_no_scaling(self) -> None:
        assert scale_for_diagonal(REFERENCE_W, REFERENCE_H) == 1.0

    def test_field_capture_matches_the_recorded_arithmetic(self) -> None:
        assert scale_for_diagonal(FIELD_W, FIELD_H) == pytest.approx(FIELD_FACTOR,
                                                                    abs=1e-3)

    def test_it_is_the_diagonal_not_an_axis(self) -> None:
        """A portrait frame and its landscape transpose scale identically."""
        assert scale_for_diagonal(FIELD_W, FIELD_H) == scale_for_diagonal(FIELD_H,
                                                                         FIELD_W)

    def test_reference_diagonal_is_1080p(self) -> None:
        assert REFERENCE_DIAGONAL == pytest.approx(2202.907, abs=1e-3)

    def test_a_frame_with_no_extent_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no extent"):
            scale_for_diagonal(0, 0)


class TestParseScale:
    """`--scale` accepts a positive number or the word `auto`."""

    def test_auto(self) -> None:
        assert parse_scale("auto") == AUTO
        assert parse_scale(" AUTO ") == AUTO

    def test_a_number(self) -> None:
        assert parse_scale("1.725") == 1.725

    @pytest.mark.parametrize("text", ["0", "-2", "1.5x", "", "none"])
    def test_rejects_anything_else(self, text: str) -> None:
        with pytest.raises(ValueError):
            parse_scale(text)


class TestIdentity:
    """A factor that changes nothing must reach the inner pipeline untouched."""

    def test_scale_of_one_passes_the_same_array_object(self) -> None:
        inner = StubPipeline()
        source = frame(FIELD_W, FIELD_H)

        ScaledPipeline(inner, 1.0).step(source)

        assert inner.frames[0] is source

    def test_auto_at_1080p_passes_the_same_array_object(self) -> None:
        """The proof that EXP-004-010's pipeline is unperturbed by this module."""
        inner = StubPipeline()
        source = frame(REFERENCE_W, REFERENCE_H)

        ScaledPipeline(inner, AUTO).step(source)

        assert inner.frames[0] is source

    def test_the_inner_result_is_returned_unchanged(self) -> None:
        scripted = box(10, 20, 30, 40)
        inner = StubPipeline(scripted)

        result = ScaledPipeline(inner, 1.0).step(frame(REFERENCE_W, REFERENCE_H))

        assert result is scripted

    def test_a_factor_rounding_to_the_same_size_is_also_a_no_op(self) -> None:
        """Sub-pixel factors resize to the same dimensions; do not interpolate."""
        inner = StubPipeline()
        source = frame(1000, 1000)

        ScaledPipeline(inner, 1.0004).step(source)

        assert inner.frames[0] is source


class TestResizing:
    """What the inner pipeline is actually handed."""

    def test_the_frame_reaches_the_inner_pipeline_scaled(self) -> None:
        inner = StubPipeline()

        ScaledPipeline(inner, AUTO, announce=False).step(frame(FIELD_W, FIELD_H))

        height, width = inner.frames[0].shape[:2]
        assert (width, height) == (1780, 1297)

    def test_aspect_ratio_survives(self) -> None:
        source_ratio = FIELD_W / FIELD_H
        width, height = ScaledPipeline(StubPipeline(), AUTO).target_size(FIELD_W,
                                                                        FIELD_H)
        assert width / height == pytest.approx(source_ratio, abs=1e-3)

    def test_the_scaled_frame_sits_at_the_reference_diagonal(self) -> None:
        width, height = ScaledPipeline(StubPipeline(), AUTO).target_size(FIELD_W,
                                                                        FIELD_H)
        assert np.hypot(width, height) == pytest.approx(REFERENCE_DIAGONAL, abs=1.0)

    def test_an_explicit_factor_is_used_as_given(self) -> None:
        inner = StubPipeline()

        ScaledPipeline(inner, 2.0, announce=False).step(frame(100, 50))

        height, width = inner.frames[0].shape[:2]
        assert (width, height) == (200, 100)

    def test_each_frame_shape_is_scaled_on_its_own(self) -> None:
        """A run over videos of differing size must not reuse the first factor."""
        inner = StubPipeline()
        pipeline = ScaledPipeline(inner, AUTO, announce=False)

        pipeline.step(frame(FIELD_W, FIELD_H))
        pipeline.step(frame(REFERENCE_W, REFERENCE_H))

        assert inner.frames[0].shape[:2] == (1297, 1780)
        assert inner.frames[1].shape[:2] == (REFERENCE_H, REFERENCE_W)


class TestBoxesComeBackInOriginalCoordinates:
    """The mapping a caller could never see going wrong."""

    def test_a_box_is_divided_by_the_effective_factor(self) -> None:
        inner = StubPipeline(box(200, 100, 40, 20))

        result = ScaledPipeline(inner, 2.0, announce=False).step(frame(100, 50))

        assert result.box == pytest.approx([100, 50, 20, 10])

    def test_the_branch_survives_the_mapping(self) -> None:
        inner = StubPipeline(box(200, 100, 40, 20))

        result = ScaledPipeline(inner, 2.0, announce=False).step(frame(100, 50))

        assert result.branch == LOCAL_YOLO

    def test_a_miss_passes_through_with_its_branch(self) -> None:
        inner = StubPipeline(StepResult(None, GLOBAL_MISS))

        result = ScaledPipeline(inner, AUTO, announce=False).step(frame(FIELD_W,
                                                                       FIELD_H))

        assert result.box is None
        assert result.branch == GLOBAL_MISS

    def test_a_box_at_the_far_corner_lands_inside_the_original_frame(self) -> None:
        """Rounding the resized dimensions must not push boxes out of frame."""
        pipeline = ScaledPipeline(StubPipeline(), AUTO, announce=False)
        width, height = pipeline.target_size(FIELD_W, FIELD_H)
        inner = StubPipeline(box(width - 10, height - 10, 10, 10))
        pipeline = ScaledPipeline(inner, AUTO, announce=False)

        result = pipeline.step(frame(FIELD_W, FIELD_H))

        # Tolerance is float epsilon, not slack: the box maps back to exactly
        # the frame edge, and the sum of the two halves lands a 1e-13 beyond it.
        x, y, box_w, box_h = result.box
        assert 0 <= x and x + box_w == pytest.approx(FIELD_W, abs=1e-9)
        assert 0 <= y and y + box_h == pytest.approx(FIELD_H, abs=1e-9)

    def test_the_effective_factor_comes_from_the_rounded_size(self) -> None:
        """Not from the requested factor, which rounding has already departed from."""
        inner = StubPipeline(box(0, 0, 1780, 1297))

        result = ScaledPipeline(inner, AUTO, announce=False).step(frame(FIELD_W,
                                                                       FIELD_H))

        # The full scaled frame maps back to exactly the full original frame.
        assert result.box == pytest.approx([0, 0, FIELD_W, FIELD_H])


class TestDelegation:
    """The rest of the pipeline protocol."""

    def test_reset_reaches_the_inner_pipeline(self) -> None:
        inner = StubPipeline()

        ScaledPipeline(inner, AUTO).reset()

        assert inner.resets == 1


class TestReporting:
    """What the run log says about what was done to the footage."""

    def test_it_announces_once_per_frame_shape(self, capsys) -> None:
        pipeline = ScaledPipeline(StubPipeline(), AUTO)

        pipeline.step(frame(FIELD_W, FIELD_H))
        pipeline.step(frame(FIELD_W, FIELD_H))

        lines = [line for line in capsys.readouterr().out.splitlines()
                 if line.startswith("Scale:")]
        assert len(lines) == 1
        assert "1032x752 -> 1780x1297" in lines[0]

    def test_downscaling_warns(self, capsys) -> None:
        """Scaling down destroys the targets this project exists to detect."""
        ScaledPipeline(StubPipeline(), 0.5).step(frame(REFERENCE_W, REFERENCE_H))

        assert "WARNING" in capsys.readouterr().out

    def test_upscaling_does_not_warn(self, capsys) -> None:
        ScaledPipeline(StubPipeline(), AUTO).step(frame(FIELD_W, FIELD_H))

        assert "WARNING" not in capsys.readouterr().out
