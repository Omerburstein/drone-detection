"""Running GLAD's pipeline on footage shot at the wrong resolution.

Every constant in GLAD's motion branch is **absolute pixels tuned at 1920x1080**:
blob area 30-3000 and the 50-blob cap in `MOD2_global`, `dist_ref = 200` and the
30-blob cap in `MOD2_local`, the Gaussian kernel 11, the 2 px box pad, the
Shi-Tomasi `minDistance`, `REGION_HALF = 160`, `MAX_DISTANCE` 50 and 10. None of
them is a ratio. Feed the same pipeline a smaller frame and every one of them is
being applied to a target a fraction of the size it was calibrated for -- the
blob-area gate most of all, since area falls with the *square* of the linear
scale.

Our own FIELD capture is 1032x752: diagonal 1277 against 2203, **0.579x linear
and 0.335x in area**. EXP-010 ran it natively and so measured our failure to
rescale alongside the detector. This module is the correction.

Why the frame and not the constants
-----------------------------------
The constants would be the better fix, and `docs/glad-model.md` ranks it as
improvement #4. It is not available cheaply: they are *function-local literals*
inside `third_party/GLAD/MOD2.py`, which `vendor.py` imports verbatim and which
is gitignored. There is no parameter, no module global, and nothing
`import_motion` can monkeypatch -- it can rebind module attributes, not
statements inside a function body. Porting MOD2 into `src/` would break the
port's stated contract ("used verbatim") and put every EXP-004-010 number behind
a code change. Scaling the frame buys the same relationship between target and
threshold without touching the vendored tree.

The coordinate contract
-----------------------
Boxes come back in **original frame pixels**. This wraps the pipeline rather
than transforming the input at decode, and that is the whole reason: `--crop`
can change the recorded coordinate system because a crop is a statement about
which pixels are picture, but a scale factor is an internal detail of how the
detector was run and must not leak into the record. A scaled run's
`detections.jsonl` is directly comparable to an unscaled one's.

Aspect ratio is preserved. Upstream's own attempt at this is a commented-out
`cv2.resize(frame, (1920, 1080))` at `third_party/GLAD/GLAD.py:59`, which would
distort anything that is not 16:9 -- and this capture is 1.372:1. MOD2's
optical-flow coherence tests threshold on the *spread* of flow angles and
distances, so anisotropic scaling perturbs exactly the quantity they gate on.
Matching the reference diagonal keeps the scaling isotropic and still puts the
frame's linear extent where the constants expect it.
"""

from __future__ import annotations

from math import hypot

import cv2
import numpy as np

from .pipeline import GladPipeline, StepResult

# The resolution GLAD's constants were tuned at. ARD-MAV, ARD100 and the paper's
# own figures are all 1920x1080, which is why a run at that size needs no scaling
# and why `auto` resolves to exactly 1.0 there.
REFERENCE_WIDTH = 1920
REFERENCE_HEIGHT = 1080
REFERENCE_DIAGONAL = hypot(REFERENCE_WIDTH, REFERENCE_HEIGHT)

AUTO = "auto"
NATIVE = 1.0


def scale_for_diagonal(width: int, height: int) -> float:
    """Factor putting a `width x height` frame at the reference diagonal.

    The diagonal rather than either axis, so the result does not depend on which
    way round a non-16:9 frame happens to be.
    """
    diagonal = hypot(width, height)
    if diagonal <= 0:
        raise ValueError(f"frame has no extent: {width}x{height}")
    return REFERENCE_DIAGONAL / diagonal


def parse_scale(text: str) -> float | str:
    """Parse `--scale`: a positive float, or `auto` for the reference diagonal."""
    cleaned = text.strip().lower()
    if cleaned == AUTO:
        return AUTO
    try:
        value = float(cleaned)
    except ValueError:
        raise ValueError(f"expected a number or 'auto', got {text!r}") from None
    if value <= 0:
        raise ValueError(f"scale must be positive, got {text!r}")
    return value


class ScaledPipeline:
    """A `GladPipeline` fed resized frames, reporting boxes in original pixels.

    Duck-types the two methods `src.glad_detect.run_video` uses -- `step` and
    `reset` -- so it drops in wherever a `GladPipeline` goes.

    The factor is resolved **per frame from its shape** rather than once at
    construction. That costs one `hypot` per frame, keeps `reset()` stateless,
    and means a run over videos of differing size scales each correctly instead
    of applying the first one's factor to all of them.

    A factor that leaves the frame dimensions unchanged delegates to the inner
    pipeline with the original array, so `--scale 1.0` and `--scale auto` on
    1080p footage are provably the faithful path that EXP-004-010 ran.
    """

    def __init__(self, inner: GladPipeline, scale: float | str = AUTO,
                 announce: bool = True) -> None:
        self.inner = inner
        self.scale = scale
        self._announce = announce
        self._announced: set[tuple[int, int]] = set()

    def factor_for(self, width: int, height: int) -> float:
        """The scale factor this pipeline applies to a `width x height` frame."""
        if self.scale == AUTO:
            return scale_for_diagonal(width, height)
        return float(self.scale)

    def target_size(self, width: int, height: int) -> tuple[int, int]:
        """Dimensions the frame is resized to, rounded to whole pixels."""
        factor = self.factor_for(width, height)
        return max(1, round(width * factor)), max(1, round(height * factor))

    def reset(self) -> None:
        """Clear the inner pipeline's state, as if starting a new video."""
        self.inner.reset()

    def step(self, frame: np.ndarray) -> StepResult:
        """Resize, step the inner pipeline, and map the box back."""
        height, width = frame.shape[:2]
        new_width, new_height = self.target_size(width, height)
        if (new_width, new_height) == (width, height):
            return self.inner.step(frame)

        self._report(width, height, new_width, new_height)
        # INTER_CUBIC upscaling matches the resize upstream commented out;
        # INTER_AREA is the right kernel downwards, though see `_report` for why
        # going down at all is a bad idea here.
        interpolation = cv2.INTER_CUBIC if new_width > width else cv2.INTER_AREA
        scaled = cv2.resize(frame, (new_width, new_height),
                            interpolation=interpolation)
        result = self.inner.step(scaled)
        if result.box is None:
            return result

        # The *effective* factor, from the dimensions actually used rather than
        # the requested one, so rounding to whole pixels cannot drift the mapping.
        x_factor = new_width / width
        y_factor = new_height / height
        x, y, box_width, box_height = result.box
        return StepResult(
            np.array([x / x_factor, y / y_factor,
                      box_width / x_factor, box_height / y_factor], dtype=float),
            result.branch,
        )

    def _report(self, width: int, height: int,
                new_width: int, new_height: int) -> None:
        """Say once per frame shape what is being done to it.

        Once per *shape*, not once per run: a run over several videos of
        different sizes is doing something different to each, and a single line
        at startup would describe only the first.
        """
        shape = (width, height)
        if not self._announce or shape in self._announced:
            return
        self._announced.add(shape)
        factor = new_width / width
        print(f"Scale: {width}x{height} -> {new_width}x{new_height} "
              f"({factor:.4f}x linear, {factor * factor:.3f}x pixels); "
              f"boxes are recorded in original coordinates")
        if new_width < width:
            print("  WARNING: scaling *down* destroys the small targets this "
                  "project exists to detect. A 12 px drone does not survive it.")
