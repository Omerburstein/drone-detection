"""Tests for the per-condition breakdowns in `src.eval.conditions`.

The per-bucket numbers are what get compared against published per-category
figures, so a mis-grouped frame would produce a wrong comparison that still
looks entirely plausible. The multi-axis shape adds a second way to get that
wrong: labelling a frame-level axis by video, or a video-level axis by frame,
silently buckets everything into `uncategorised`.
"""

from __future__ import annotations

import json

import pytest

from src.eval.conditions import (UNKNOWN, Axis, as_axes, group_by_axis,
                                 group_by_condition, load_conditions, video_of)
from src.eval.metrics import evaluate, f1_score

CONDITIONS = {"phantom05": "complex", "phantom09": "ordinary", "phantom19": "small_mav"}

AXES_FILE = {
    "scene_category": CONDITIONS,
    "axes": {
        "scene_category": {"level": "video", "labels": CONDITIONS},
        "lighting": {
            "level": "frame",
            "order": ["invisible (<5)", "low (5-15)", "strong (>=30)"],
            "labels": {"phantom05_0001": "strong (>=30)",
                       "phantom05_0002": "invisible (<5)",
                       "phantom09_0001": "low (5-15)"},
        },
    },
}


class TestVideoOf:

    @pytest.mark.parametrize("key, expected", [
        ("phantom05_0001", "phantom05"),
        ("phantom05_1799", "phantom05"),
        ("42", "42"),                       # video-keyed run: no video to recover
        ("clip_2_00042", "clip_2"),         # only the final underscore splits
    ])
    def test_strips_the_frame_number(self, key, expected):
        assert video_of(key) == expected


class TestAxis:

    def test_video_level_labels_by_video(self):
        axis = Axis("scene_category", "video", CONDITIONS)
        assert axis.label_for("phantom05_0001") == "complex"
        assert axis.label_for("phantom05_1799") == "complex"

    def test_frame_level_labels_by_frame(self):
        axis = Axis("lighting", "frame", {"phantom05_0001": "low (5-15)"})
        assert axis.label_for("phantom05_0001") == "low (5-15)"
        # The same video, a different frame -- a frame-level axis must not
        # spread one frame's label across the sequence.
        assert axis.label_for("phantom05_0002") == UNKNOWN

    def test_uncovered_key_is_unknown_not_an_error(self):
        assert Axis("a", "video", CONDITIONS).label_for("phantomXX_0001") == UNKNOWN

    def test_declared_order_beats_alphabetical(self):
        axis = Axis("lighting", "frame", {}, order=("near", "mid", "far"))
        assert sorted(["far", "mid", "near"], key=axis.sort_key) == ["near", "mid", "far"]

    def test_undeclared_labels_sort_after_declared_ones(self):
        axis = Axis("lighting", "frame", {}, order=("near", "far"))
        assert sorted(["zzz", "far", "aaa", "near"],
                      key=axis.sort_key) == ["near", "far", "aaa", "zzz"]


class TestLoadConditions:

    def test_reads_the_legacy_bare_map(self, tmp_path):
        """An older processed tree must stay scoreable without regeneration."""
        path = tmp_path / "conditions.json"
        path.write_text(json.dumps({"scene_category": CONDITIONS,
                                    "categories": {"complex": ["phantom05"]}}))
        axes = load_conditions(path)
        assert [a.name for a in axes] == ["scene_category"]
        assert axes[0].level == "video"
        assert axes[0].labels == CONDITIONS

    def test_reads_every_declared_axis_in_order(self, tmp_path):
        path = tmp_path / "conditions.json"
        path.write_text(json.dumps(AXES_FILE))
        axes = load_conditions(path)
        assert [a.name for a in axes] == ["scene_category", "lighting"]
        assert [a.level for a in axes] == ["video", "frame"]
        assert axes[1].order == ("invisible (<5)", "low (5-15)", "strong (>=30)")

    def test_axes_block_wins_over_the_legacy_key(self, tmp_path):
        """Both are present in a regenerated file; reading both would double-count."""
        path = tmp_path / "conditions.json"
        path.write_text(json.dumps(AXES_FILE))
        assert len(load_conditions(path)) == 2

    def test_missing_file_fails_loudly(self, tmp_path):
        with pytest.raises(SystemExit):
            load_conditions(tmp_path / "absent.json")

    def test_wrong_shape_fails_loudly(self, tmp_path):
        path = tmp_path / "conditions.json"
        path.write_text(json.dumps({"categories": {}}))
        with pytest.raises(SystemExit):
            load_conditions(path)

    def test_axis_without_labels_fails_loudly(self, tmp_path):
        path = tmp_path / "conditions.json"
        path.write_text(json.dumps({"axes": {"lighting": {"level": "frame"}}}))
        with pytest.raises(SystemExit):
            load_conditions(path)

    def test_unknown_level_fails_loudly(self, tmp_path):
        """A typo here would bucket the whole run as uncategorised, silently."""
        path = tmp_path / "conditions.json"
        path.write_text(json.dumps(
            {"axes": {"lighting": {"level": "framewise", "labels": {}}}}))
        with pytest.raises(SystemExit):
            load_conditions(path)


class TestAsAxes:

    def test_none_is_no_axes(self):
        assert as_axes(None) == []
        assert as_axes({}) == []

    def test_bare_map_becomes_one_video_level_axis(self):
        axes = as_axes(CONDITIONS)
        assert len(axes) == 1 and axes[0].level == "video"

    def test_a_list_passes_through(self):
        axes = [Axis("lighting", "frame", {})]
        assert as_axes(axes) == axes


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


class TestGroupByAxis:

    def test_splits_one_video_across_frame_level_buckets(self, make_frame):
        """The whole point of a frame-level axis: lighting varies within a video."""
        axis = Axis("lighting", "frame", {"phantom05_0001": "dark",
                                          "phantom05_0002": "bright"})
        grouped = group_by_axis([make_frame(key="phantom05_0001"),
                                 make_frame(key="phantom05_0002")], axis)
        assert {k: len(v) for k, v in grouped.items()} == {"bright": 1, "dark": 1}

    def test_buckets_come_back_in_declared_order(self, make_frame):
        axis = Axis("range", "frame",
                    {"a_1": "far", "a_2": "near", "a_3": "mid"},
                    order=("near", "mid", "far"))
        frames = [make_frame(key=k) for k in ("a_1", "a_2", "a_3")]
        assert list(group_by_axis(frames, axis)) == ["near", "mid", "far"]


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

    def test_a_bare_map_still_names_its_axis(self, frames):
        scored = evaluate(frames, 0.5, conditions=CONDITIONS)
        assert {c.axis for c in scored.by_condition} == {"scene_category"}

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


class TestEvaluateAcrossAxes:

    @pytest.fixture
    def frames(self, make_frame):
        hit = ([0, 0, 100, 100], 0.9, 0)
        return [
            make_frame(key="phantom05_0001", gt=[[0, 0, 100, 100]], preds=[hit]),
            make_frame(key="phantom05_0002", gt=[[0, 0, 100, 100]], preds=None),
        ]

    @pytest.fixture
    def axes(self):
        return [
            Axis("scene_category", "video", {"phantom05": "complex"}),
            Axis("lighting", "frame",
                 {"phantom05_0001": "strong", "phantom05_0002": "invisible"},
                 order=("invisible", "strong")),
        ]

    def test_each_score_names_its_axis(self, frames, axes):
        scored = evaluate(frames, 0.5, conditions=axes)
        assert [c.axis for c in scored.by_condition] == [
            "scene_category", "lighting", "lighting"]

    def test_each_axis_separately_partitions_the_run(self, frames, axes):
        """Every axis covers all the frames, so counts must partition per axis.

        Summing across axes would double-count -- which is exactly why the
        report splits the table rather than pooling it.
        """
        scored = evaluate(frames, 0.5, conditions=axes)
        for axis in ("scene_category", "lighting"):
            rows = [c for c in scored.by_condition if c.axis == axis]
            assert sum(c.n_frames for c in rows) == scored.n_frames
            assert sum(c.n_gt for c in rows) == scored.n_gt

    def test_a_frame_level_axis_separates_what_the_video_level_one_pools(
            self, frames, axes):
        scored = evaluate(frames, 0.5, conditions=axes)
        by_label = {c.label: c for c in scored.by_condition}
        assert by_label["complex"].recall == pytest.approx(0.5)   # pooled
        assert by_label["strong"].recall == pytest.approx(1.0)    # separated
        assert by_label["invisible"].recall == pytest.approx(0.0)

    def test_declared_bucket_order_survives_into_the_scores(self, frames, axes):
        scored = evaluate(frames, 0.5, conditions=axes)
        lighting = [c.label for c in scored.by_condition if c.axis == "lighting"]
        assert lighting == ["invisible", "strong"]


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
