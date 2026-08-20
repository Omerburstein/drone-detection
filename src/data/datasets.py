"""The datasets this extractor knows how to read, and what differs between them.

ARD-MAV and ARD100 come out of the same lab in the same shape — 1920x1080
30 fps `.mp4` alongside one Pascal VOC XML per frame, one-based and four-digit,
single class `Drone`. That is why one extractor serves both. What actually
differs is small and entirely declarative: which videos form the test split,
whether the release publishes a scene grouping to report against, and the
provenance its manifest has to record.

Holding that here rather than as `if dataset == ...` branches inside the
extractor is what keeps the conversion itself literally identical across
datasets — which is the whole point of M4b. A generalisation number is only
readable if the pipeline that produced both sides of it is the same pipeline.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    """One dataset's layout, split, and the provenance its manifest records.

    `scene_categories` is `None` where the release publishes no grouping. That
    is not a gap to fill by inventing one: `scene_category` exists so our rows
    line up with GLAD's published table, and a grouping of our own invention
    would look like the same axis while comparing nothing.
    """

    name: str
    raw: Path
    out: Path
    videos: tuple[str, ...]
    source: str
    license: str
    split_rule: str
    known_issues: str
    scene_categories: dict[str, tuple[str, ...]] | None = None

    @property
    def videos_dir(self) -> Path:
        """Directory of source `.mp4` files."""
        return self.raw / "videos"

    @property
    def annotations_dir(self) -> Path:
        """Directory of per-video VOC XML folders."""
        return self.raw / "Annotations"

    def category_of(self) -> dict[str, str]:
        """Video -> published scene category. Empty when none is published."""
        if not self.scene_categories:
            return {}
        return {v: cat for cat, members in self.scene_categories.items() for v in members}


ARD_MAV = DatasetSpec(
    name="ARD-MAV",
    raw=Path("data/raw/ARD-MAV"),
    out=Path("data/processed/ARD-MAV"),
    # The official split published with GLAD. Using it verbatim is what makes
    # our numbers comparable to the paper's; a split of our own invention would
    # not be.
    videos=("phantom05", "phantom08", "phantom09", "phantom10", "phantom19",
            "phantom30", "phantom41", "phantom43", "phantom46", "phantom47",
            "phantom58", "phantom63", "phantom65", "phantom70", "phantom86"),
    # GLAD reports results split three ways, so the same grouping is recorded
    # here. Without it the M4 comparison against their published per-category
    # P/R/F1 cannot be made.
    scene_categories={
        "ordinary": ("phantom09", "phantom10", "phantom30", "phantom47", "phantom70"),
        "complex": ("phantom05", "phantom08", "phantom58", "phantom65", "phantom86"),
        "small_mav": ("phantom19", "phantom41", "phantom43", "phantom46", "phantom63"),
    },
    source="ARD-MAV, 60 videos / 107,497 frames, released with GLAD.",
    license="MIT.",
    split_rule="""**The official split published with GLAD, used verbatim** so results are
comparable to the paper's. 45 training videos, 15 test videos.""",
    known_issues="""- **`CAP_PROP_FRAME_COUNT` overstates.** The container header reports more
  frames than actually decode (28,644 vs 28,337 on the test 15). Every
  decodable frame does have an annotation. Any frame-count reconciliation
  should use the decoder, not the header.""",
)

ARD100 = DatasetSpec(
    name="ARD100",
    raw=Path("data/raw/ARD100"),
    out=Path("data/processed/ARD100"),
    # ARD100's own test split intersected with "not in our local ARD-MAV 60" —
    # exactly 15 videos, matching EXP-004's count, so the M4b comparison is
    # like-for-like in size and the selection is not cherry-picked. The
    # arithmetic and the audit are in data/raw/ARD100/PROVENANCE.md.
    videos=("phantom03", "phantom92", "phantom93", "phantom94", "phantom95",
            "phantom97", "phantom102", "phantom110", "phantom113", "phantom119",
            "phantom133", "phantom135", "phantom136", "phantom141", "phantom144"),
    # ARD100 publishes no ordinary/complex/small_mav grouping. Left None on
    # purpose — see DatasetSpec.
    scene_categories=None,
    source=("ARD100, 100 videos / 202,467 frames, Zenodo DOI 10.5281/zenodo.15870538, "
            "MD5 b52a5c8aae7b7661b57dde5490c56580 verified."),
    license="CC-BY-4.0.",
    split_rule="""**ARD100's own test split, intersected with the videos absent from our local
ARD-MAV 60** — 15 videos, none of them seen by GLAD in training. Same count as
EXP-004's split, so the M4b generalisation comparison is like-for-like in size
and the selection is not a choice we made. ARD100 has 100 videos; 51 overlap our
60, 49 are unseen, and 15 of those 49 are in its test split.""",
    known_issues="""- **`CAP_PROP_FRAME_COUNT` overstates**, same as ARD-MAV: the header exceeds
  the XML count by 0–30 frames per video (phantom03 is the one exact match at
  1799). Trust the decoder, not the header.
- **No published scene grouping.** ARD-MAV's `ordinary` / `complex` /
  `small_mav` axis comes from GLAD's paper and has no ARD100 equivalent, so
  `conditions.json` here carries only the measured `lighting` and
  `relative_range` axes. A per-category comparison against EXP-004 is therefore
  not available; the aggregate and the measured axes are.
- **`relative_range` is scaled to this split's own closest approach**, so its
  buckets are *not* comparable to ARD-MAV's. Compare `gt_size` in pixels from a
  `--dump` CSV instead when comparing across datasets.""",
)

SPECS = {spec.name: spec for spec in (ARD_MAV, ARD100)}


def spec_for(name: str) -> DatasetSpec:
    """Look up a dataset by name, failing with the valid names rather than a KeyError."""
    if name not in SPECS:
        sys.exit(f"Unknown dataset {name!r}. Known: {', '.join(SPECS)}")
    return SPECS[name]
