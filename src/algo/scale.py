"""Pixel constants quoted at one reference width and read at the clip's own.

A threshold measured in pixels is a statement about one resolution. A 9 px radius tuned on
960-wide analog footage is 4x too tight on the FIELD sensor's 4128 pixels, where the same
object is 4x wider. EXP-031 met exactly that: merge 20 and 30 came out byte-identical on
the `.raw`, because both were small next to a 4128-wide blob.

So every pixel constant is *quoted* at `REF_WIDTH` and multiplied by width/`REF_WIDTH`
before use. The detector modules in `experiments/` have done this since EXP-017
(`S = W / 1440.0`). `PixelScale` is the same rule for code under `src/`, with the
reference made explicit so a caller holding numbers tuned at another width can declare it
instead of converting them by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

# SOFA-O4's crop, where the first constants were measured. Every quoted pixel is at this width.
REF_WIDTH = 1440.0


@dataclass(frozen=True)
class PixelScale:
    """Converts pixel constants quoted at `ref_width_px` into this picture's pixels.

    `px` computes `quoted * width / ref` in that order, not `quoted * (width / ref)`, so a
    value quoted at the clip's own width comes back exactly. At 960 wide, 37.5 at 1440
    reads as exactly 25.0, and analog outputs stay byte-identical across the change.
    """

    width_px: float
    ref_width_px: float = REF_WIDTH

    def __post_init__(self) -> None:
        if self.width_px <= 0 or self.ref_width_px <= 0:
            raise ValueError(f"widths must be positive: {self.width_px}, {self.ref_width_px}")

    @property
    def factor(self) -> float:
        """Picture pixels per quoted pixel."""
        return self.width_px / self.ref_width_px

    def px(self, quoted: float) -> float:
        """A length quoted at the reference width, in this picture's pixels."""
        return quoted * self.width_px / self.ref_width_px

    def quoted(self, px: float) -> float:
        """The inverse: a length in this picture's pixels, quoted at the reference width."""
        return px * self.ref_width_px / self.width_px
