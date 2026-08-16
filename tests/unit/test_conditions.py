"""Tests for the per-scene-category breakdown in `src.eval.conditions`.

The per-category numbers are what get compared against published per-category
figures, so a mis-grouped frame would produce a wrong comparison that still
looks entirely plausible.
"""

from __future__ import annotations

import json

import pytest

from src.eval.conditions import UNKNOWN, group_by_condition, load_conditions, video_of
from src.eval.metrics import evaluate, f1_score

CONDITIONS = {"phantom05": "complex", "phantom09": "ordinary", "phantom19": "small_mav"}


class TestVideoOf:

    @pytest.mark.parametrize("key, expected", [
        ("phantom05_0001", "phantom05"),
        ("phantom05_1799", "phantom05"),
        ("42", "42"),                       # video-keyed run: no video to recover
        ("clip_2_00042", "clip_2"),         # only the final underscore splits
    ])
    def test_strips_the_frame_number(self, key, expected):
        assert video_of(key) == expected


class TestLoadConditions:

    def test_reads_scene_category_map(self, tmp_path):
        path = tmp_path / "conditions.json"
        path.write_text(json.dumps({"scene_category": CONDITIONS,
                                    "categories": {"complex": ["phantom05"]}}))
        assert load_conditions(path) == CONDITIONS

    def test_missing_file_fails_loudly(self, tmp_path):
        with pytest.raises(SystemExit):
            load_conditions(tmp_path / "absent.json")

    def test_wrong_shape_fails_loudly(self, tmp_path):
        path = tmp_path / "conditions.json"
        path.write_text(json.dumps({"categories": {}}))
        with pytest.raises(SystemExit):
            load_conditions(path)


class TestGroupByCondition:

    def test_buckets_by_video(self, make_frame):
        frames = [make_frame(key="phantom05_0001"), make_frame(key="phantom05_0002"),
                  make_frame(key="phantom09_0001")]
        grouped = group_by_condition(frames, CONDITIONS)
        assert sorted(grouped) == ["complex", "ordinary"]
        assert len(grouped["complex"]) == 2
        assert len(grouped["ordinary"]) == 1

    def test_unmapped_video_is_kept_not_dropped(self, make_frame):
        """Dropping them would silently change every per-category denominator."""
        grouped = group_by_condition([make_frame(key="phantomXX_0001")], CONDITIONS)
        assert list(grouped) == [UNKNOWN]

    def test_every_frame_lands_somewhere(self, make_frame):
        frames = [make_frame(key=f"phantom{v}_{i:04d}")
                  for v in ("05", "09", "77") for i in range(3)]
        grouped = group_by_condition(frames, CONDITIONS)
        assert sum(len(v) for v in grouped.values()) == len(frames)


class TestEvaluateWithConditions:

    @pytest.fixture
    def frames(self, make_frame):
        """complex: 2 targets, both found. ordinary: 2 targets, neither found."""
        hit = ([0, 0, 100, 100], 0.9, 0)
        return [
            make_frame(key="phantom05_0001", gt=[[0, 0, 100, 100]], preds=[hit]),
            make_frame(key="phantom05_0002", gt=[[0, 0, 100, 100]], preds=[hit]),
            make_frame(key="phantom09_0001", gt=[[0, 0, 100, 100]], preds=None),
            make_frame(key="phantom09_0002", gt=[[0, 0, 100, 100]], preds=None),
        ]

    def test_absent_without_conditions(self, frames):
        assert evaluate(frames, 0.5).by_condition == []

    def test_splits_a_uniform_aggregate_into_opposite_halves(self, frames):
        """Aggregate recall is 0.5; per category it is 1.0 and 0.0.

        This is the entire point of the breakdown -- the aggregate describes
        neither category.
        """
        scored = evaluate(frames, 0.5, conditions=CONDITIONS)
        assert scored.recall == pytest.approx(0.5)

        by_label = {c.label: c for c in scored.by_condition}
        assert by_label["complex"].recall == pytest.approx(1.0)
        assert by_label["complex"].precision == pytest.approx(1.0)
        assert by_label["complex"].f1 == pytest.approx(1.0)
        assert by_label["ordinary"].recall == pytest.approx(0.0)

    def test_counts_partition_the_run(self, frames):
        scored = evaluate(frames, 0.5, conditions=CONDITIONS)
        assert sum(c.n_frames for c in scored.by_condition) == scored.n_frames
        assert sum(c.n_gt for c in scored.by_condition) == scored.n_gt

    def test_categories_are_ordered_stably(self, frames):
        labels = [c.label for c in evaluate(frames, 0.5, conditions=CONDITIONS).by_condition]
        assert labels == sorted(labels)

    def test_category_with_no_ground_truth_is_nan_not_zero(self, make_frame):
        """NaN says 'not measured'; 0.0 would claim the detector failed there."""
        frames = [
            make_frame(key="phantom05_0001", gt=[[0, 0, 100, 100]],
                       preds=[([0, 0, 100, 100], 0.9, 0)]),
            make_frame(key="phantom09_0001", gt=None,
                       preds=[([5, 5, 20, 20], 0.4, 0)]),
        ]
        by_label = {c.label: c
                    for c in evaluate(frames, 0.5, conditions=CONDITIONS).by_condition}
        assert by_label["ordinary"].n_gt == 0
        assert by_label["ordinary"].recall != by_label["ordinary"].recall  # NaN


class TestF1Score:

    @pytest.mark.parametrize("precision, recall, expected", [
        (1.0, 1.0, 1.0),
        (0.0, 0.0, 0.0),      # must not divide by zero
        (1.0, 0.0, 0.0),
        (0.5, 0.5, 0.5),
        (1.0, 0.5, 2 / 3),
    ])
    def test_known_values(self, precision, recall, expected):
        assert f1_score(precision, recall) == pytest.approx(expected)
