"""Tests for the dataset registry and the parts of extraction it drives.

Three things here fail silently rather than loudly, which is why they are
pinned:

**The two datasets must not diverge.** ARD-MAV and ARD100 are the two halves of
the M4b comparison, and the only thing that may differ between them is the video
content. Anything that makes the extractor behave differently makes the
generalisation number partly about the extractor.

**A missing scene grouping must stay missing.** ARD100 publishes none, and a
`conditions.json` carrying an empty or invented `scene_category` would report
per-category rows that compare nothing against GLAD's published table.

**`--no-images` must still be a complete tree.** It is the mode the whole GLAD
path runs in, so labels, conditions and the verify renders all have to survive
the absence of JPEGs -- and the `data.yaml` that would point at a directory
which is not there has to not be written.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.data.datasets import ARD100, ARD_MAV, SPECS, spec_for
from src.data.prepare_ardmav import Stats, VerifySampler, write_metadata
from src.data.scene_stats import FrameStats


def frame_stats(n: int) -> list[FrameStats]:
    """A run's worth of plausible per-frame measurements."""
    return [FrameStats(key=f"phantom03_{i:04d}", n_targets=1, contrast=20.0,
                       size=15.0, brightness=120.0, clipped_hi=0.0, clipped_lo=0.0)
            for i in range(1, n + 1)]


class TestRegistry:

    def test_lookup_by_name(self):
        assert spec_for("ARD100") is ARD100
        assert spec_for("ARD-MAV") is ARD_MAV

    def test_unknown_dataset_exits_with_the_valid_names(self):
        with pytest.raises(SystemExit) as excinfo:
            spec_for("ARD-100")
        assert "ARD100" in str(excinfo.value)

    @pytest.mark.parametrize("spec", SPECS.values(), ids=list(SPECS))
    def test_split_is_fifteen_whole_videos(self, spec):
        # Both sides of M4b carry the same statistical weight only if the
        # counts match, and splitting is by video, never by frame.
        assert len(spec.videos) == 15
        assert len(set(spec.videos)) == 15

    def test_the_two_splits_do_not_overlap(self):
        # The whole point of the ARD100 set: GLAD trained on ARD-MAV, so a
        # shared video would put training footage in the held-out measurement.
        assert not set(ARD100.videos) & set(ARD_MAV.videos)

    def test_ard_mav_categories_cover_its_split_exactly(self):
        category_of = ARD_MAV.category_of()
        assert set(category_of) == set(ARD_MAV.videos)

    def test_ard100_publishes_no_grouping(self):
        assert ARD100.scene_categories is None
        assert ARD100.category_of() == {}


class TestVerifySampler:
    """The reservoir behind `_verify/`, the only check on frame numbering."""

    @staticmethod
    def offer_many(sampler: VerifySampler, count: int) -> None:
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        for i in range(1, count + 1):
            sampler.offer(frame, ["0 0.500000 0.500000 0.100000 0.100000"],
                          f"phantom03_{i:04d}")

    def test_keeps_at_most_the_requested_sample(self, tmp_path):
        sampler = VerifySampler(5, seed=42)
        self.offer_many(sampler, 200)
        assert len(list(sampler.flush(tmp_path).glob("*.jpg"))) == 5

    def test_keeps_everything_when_the_split_is_smaller_than_the_sample(self, tmp_path):
        sampler = VerifySampler(20, seed=42)
        self.offer_many(sampler, 3)
        assert len(list(sampler.flush(tmp_path).glob("*.jpg"))) == 3

    def test_same_seed_inspects_the_same_frames(self, tmp_path):
        first, second = VerifySampler(5, seed=42), VerifySampler(5, seed=42)
        self.offer_many(first, 200)
        self.offer_many(second, 200)
        names = [sorted(p.name for p in s.flush(d).glob("*.jpg"))
                 for s, d in ((first, tmp_path / "a"), (second, tmp_path / "b"))]
        assert names[0] == names[1]

    def test_sampling_is_spread_across_the_run(self, tmp_path):
        # A reservoir that only ever kept the first N frames would still pass
        # the count assertions above while checking one corner of one video.
        sampler = VerifySampler(10, seed=7)
        self.offer_many(sampler, 2000)
        kept = [int(p.stem.rsplit("_", 1)[1]) for p in sampler.flush(tmp_path).glob("*.jpg")]
        assert max(kept) > 1000

    def test_a_zero_sample_writes_nothing(self, tmp_path):
        sampler = VerifySampler(0, seed=42)
        self.offer_many(sampler, 50)
        assert not list(sampler.flush(tmp_path).glob("*.jpg"))


class TestMetadata:

    @staticmethod
    def write(spec, out, wrote_images: bool):
        """Write one dataset's metadata over a synthetic run's stats."""
        out.mkdir(parents=True, exist_ok=True)
        stats = Stats(frames=3, boxes=3, areas=[100.0, 400.0, 2000.0],
                      scene=frame_stats(3))
        write_metadata(spec, out, "test", list(spec.videos), stats, wrote_images)
        return json.loads((out / "conditions.json").read_text(encoding="utf-8"))

    def test_labels_only_writes_no_data_yaml(self, tmp_path):
        # A data.yaml naming an images/ directory that was never written is a
        # trap: ultralytics would fail deep inside a dataloader instead of here.
        self.write(ARD100, tmp_path, wrote_images=False)
        assert not (tmp_path / "data.yaml").exists()
        assert (tmp_path / "MANIFEST.md").exists()

    def test_images_run_writes_data_yaml(self, tmp_path):
        self.write(ARD_MAV, tmp_path, wrote_images=True)
        assert "images/test" in (tmp_path / "data.yaml").read_text(encoding="utf-8")

    def test_ard100_conditions_carry_no_scene_category(self, tmp_path):
        conditions = self.write(ARD100, tmp_path, wrote_images=False)
        assert "scene_category" not in conditions
        assert "scene_category" not in conditions["axes"]
        # The measured axes are what remains, and they are the whole breakdown
        # available on this dataset.
        assert set(conditions["axes"]) == {"lighting", "relative_range"}

    def test_ard_mav_conditions_carry_the_published_grouping(self, tmp_path):
        conditions = self.write(ARD_MAV, tmp_path, wrote_images=True)
        assert conditions["axes"]["scene_category"]["level"] == "video"
        assert set(conditions["axes"]) == {"scene_category", "lighting", "relative_range"}

    def test_manifest_records_the_dataset_it_describes(self, tmp_path):
        self.write(ARD100, tmp_path, wrote_images=False)
        manifest = (tmp_path / "MANIFEST.md").read_text(encoding="utf-8")
        assert manifest.startswith("# ARD100 — processed")
        assert "CC-BY-4.0" in manifest
        assert "Labels only" in manifest
        # ARD-MAV is named on purpose -- the split is defined by exclusion from
        # it. What must not leak is the other dataset's own provenance, which
        # is what a hardcoded manifest string would drag in.
        assert "MIT" not in manifest
        assert "107,497" not in manifest
        assert "Scene categories" not in manifest
