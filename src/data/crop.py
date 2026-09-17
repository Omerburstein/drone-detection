"""A rectangle of a decoded frame, for sources that are not all picture.

Our O4 goggles recordings are 2520x1080 screen captures whose actual video is a
1440x1080 window at x=540; the rest is pillarbox with HUD glyphs on it. Feeding
that to the detector is not merely wasteful. GLAD's global detector letterboxes
the frame's *longest side* to 640, so the bars cost resolution on the target:
2520 wide scales by 0.254 and a 20 px drone arrives as 5 px, where the cropped
1440 scales by 0.444 and it arrives as 9 px.

Cropping at decode rather than writing cropped copies is deliberate. Re-encoding
793 MB through `cv2.VideoWriter`'s mp4v would smear 10-30 px targets -- exactly
what is being measured -- and it would put a lossy generation between
`data/raw/` and every number taken from it.

Nothing here touches the filesystem: a frame goes in, a view of it comes out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CROP_FIELDS = 4


@dataclass(frozen=True)
class Crop:
    """The picture region of a frame, as `x,y,width,height` in source pixels.

    Recorded detections are in **cropped** coordinates, because that is the
    frame the detector saw. Anything drawing those boxes back onto the video
    has to apply the same crop -- which is why `src.render_video` takes this
    option too.
    """

    x: int
    y: int
    width: int
    height: int

    @classmethod
    def parse(cls, text: str) -> Crop:
        """Build one from `x,y,width,height`, rejecting anything degenerate."""
        parts = text.replace(" ", "").split(",")
        if len(parts) != CROP_FIELDS:
            raise ValueError(f"expected 'x,y,width,height', got {text!r}")
        try:
            x, y, width, height = (int(p) for p in parts)
        except ValueError:
            raise ValueError(f"crop values must be integers, got {text!r}") from None
        if width <= 0 or height <= 0:
            raise ValueError(f"crop width and height must be positive, got {text!r}")
        if x < 0 or y < 0:
            raise ValueError(f"crop origin must not be negative, got {text!r}")
        return cls(x, y, width, height)

    @property
    def label(self) -> str:
        """`1440x1080 at (540, 0)` — printed once per run, so a number can be
        traced back to the region it was measured on."""
        return f"{self.width}x{self.height} at ({self.x}, {self.y})"

    def fits(self, shape: tuple[int, int]) -> bool:
        """Does this rectangle lie inside a frame of `(height, width)`?"""
        height, width = shape
        return self.x + self.width <= width and self.y + self.height <= height

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """The cropped region, as a view into `frame`.

        A view, not a copy: the caller is handing it straight to the detector,
        and the frame it came from is discarded on the next decode.
        """
        return frame[self.y:self.y + self.height, self.x:self.x + self.width]
