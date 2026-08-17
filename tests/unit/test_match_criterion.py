"""Tests for the two matching criteria in `src.eval.metrics`.

The criterion decides what counts as a detection at all, so it sits upstream of
every number in the report. The centre rule exists because IoU 0.50 is not a
fair ruler at 10-30 px: a 2 px offset on a 12 px target drops IoU below 0.5
while the detection is plainly correct, and the target is then charged twice —
once as a miss and once as a false alarm.

The design that makes one matching loop serve both is that each criterion
returns an affinity which is **>= 0 exactly when the pair is acceptable**, and
larger when better. These tests pin that contract from both sides.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.metrics import (CENTER, IOU, MatchCriterion, as_criterion, box_size,
                              center_distance, evaluate, match_frame)


def box(cx: float, cy: float, size: float) -> list[float]:
    """A square xyxy box of side `size` centred on (cx, cy)."""
    half = size / 2
    return [cx - half, cy - half, cx + half, cy + half]


class TestBoxSize:
    """`sqrt(w*h)` — the same notion of size `AREA_BUCKETS` bins on."""

    def test_square_box_is_its_own_side(self):
        assert box_size(np.array([box(50, 50, 12)]))[0] == pytest.approx(12)

    def test_oblong_box_is_the_equal_area_square(self):
        assert box_size(np.array([[0, 0, 32, 8]]))[0] == pytest.approx(16)

    def test_degenerate_box_is_zero(self):
        assert box_size(np.array([[10, 10, 10, 10]]))[0] == pytest.approx(0)


class TestCenterDistance:

    def test_identical_boxes_are_zero_apart(self):
        boxes = np.array([box(50, 50, 12)])
        assert center_distance(boxes, boxes)[0, 0] == pytest.approx(0)

    def test_is_euclidean_between_centres(self):
        a, b = np.array([box(0, 0, 4)]), np.array([box(3, 4, 90)])
        assert center_distance(a, b)[0, 0] == pytest.approx(5)

    def test_ignores_box_size(self):
        """Only centres matter — a huge box and a tiny one at the same place."""
        a, b = np.array([box(10, 10, 2)]), np.array([box(10, 10, 400)])
        assert center_distance(a, b)[0, 0] == pytest.approx(0)

    def test_empty_inputs_give_a_correctly_shaped_matrix(self):
        assert center_distance(np.zeros((0, 4)), np.array([box(1, 1, 1)])).shape == (0, 1)


class TestAffinityContract:
    """Both criteria must be >= 0 exactly on acceptable pairs."""

    @pytest.mark.parametrize("criterion", [
        MatchCriterion(IOU, 0.5), MatchCriterion(CENTER, 1.0)])
    def test_a_perfect_prediction_is_accepted(self, criterion):
        boxes = np.array([box(50, 50, 12)])
        assert criterion.affinity(boxes, boxes)[0, 0] >= 0

    @pytest.mark.parametrize("criterion", [
        MatchCriterion(IOU, 0.5), MatchCriterion(CENTER, 1.0)])
    def test_a_distant_prediction_is_rejected(self, criterion):
        pred, gt = np.array([box(500, 500, 12)]), np.array([box(50, 50, 12)])
        assert criterion.affinity(pred, gt)[0, 0] < 0

    def test_iou_affinity_is_iou_shifted_by_the_threshold(self):
        pred, gt = np.array([[0, 0, 10, 10]]), np.array([[0, 0, 20, 10]])
        # intersection 100, union 200 -> IoU 0.5
        assert MatchCriterion(IOU, 0.4).affinity(pred, gt)[0, 0] == pytest.approx(0.1)


class TestCenterCriterion:

    def test_accepts_an_offset_below_one_target_size(self):
        """The whole point: a 2 px offset on a 12 px drone is a hit."""
        pred, gt = np.array([box(52, 50, 12)]), np.array([box(50, 50, 12)])
        assert MatchCriterion(CENTER, 1.0).affinity(pred, gt)[0, 0] >= 0

    def test_rejects_an_offset_beyond_one_target_size(self):
        pred, gt = np.array([box(63, 50, 12)]), np.array([box(50, 50, 12)])
        assert MatchCriterion(CENTER, 1.0).affinity(pred, gt)[0, 0] < 0

    def test_the_boundary_is_exactly_the_target_size(self):
        criterion = MatchCriterion(CENTER, 1.0)
        gt = np.array([box(50, 50, 12)])
        assert criterion.affinity(np.array([box(62, 50, 12)]), gt)[0, 0] == pytest.approx(0)

    def test_is_scale_relative_where_iou_is_not(self):
        """A tenth-of-a-target offset behaves identically at 10 px and at 100 px.

        This is the property IoU lacks and the reason this criterion exists.
        """
        criterion = MatchCriterion(CENTER, 1.0)
        small = criterion.affinity(np.array([box(51, 50, 10)]), np.array([box(50, 50, 10)]))
        large = criterion.affinity(np.array([box(60, 50, 100)]),
                                   np.array([box(50, 50, 100)]))
        assert small[0, 0] == pytest.approx(large[0, 0])

    def test_ignores_box_size_disagreement(self):
        """A correctly centred but badly sized box is still a detection.

        That is the intended trade: size error shows up in mean IoU, not in
        whether the drone was found.
        """
        pred, gt = np.array([box(50, 50, 40)]), np.array([box(50, 50, 12)])
        assert MatchCriterion(CENTER, 1.0).affinity(pred, gt)[0, 0] >= 0

    def test_a_degenerate_target_matches_nothing(self):
        """A zero-size target has no scale to normalise by.

        Nothing matches it, rather than everything — the alternative is a
        divide-by-zero that silently accepts every prediction on the frame.
        """
        pred, gt = np.array([box(50, 50, 12)]), np.array([[50.0, 50.0, 50.0, 50.0]])
        assert MatchCriterion(CENTER, 1.0).affinity(pred, gt)[0, 0] < 0

    def test_tolerance_widens_the_accept_region(self):
        pred, gt = np.array([box(70, 50, 12)]), np.array([box(50, 50, 12)])
        assert MatchCriterion(CENTER, 1.0).affinity(pred, gt)[0, 0] < 0
        assert MatchCriterion(CENTER, 2.0).affinity(pred, gt)[0, 0] >= 0


class TestMatchFrame:

    def test_center_matching_rescues_a_near_miss_iou_rejects(self, make_frame):
        """The EXP-004 case: 3 px off each axis on a 12 px drone -> IoU 0.39.

        Centre-to-centre that is 4.2 px, a third of a target size: obviously the
        same drone, and obviously not a false alarm.
        """
        frame = make_frame(gt=[box(50, 50, 12)], preds=[(box(53, 53, 12), 0.9, 0)])
        assert not match_frame(frame, MatchCriterion(IOU, 0.5))[0][0]
        assert match_frame(frame, MatchCriterion(CENTER, 1.0))[0][0]

    def test_reported_iou_is_always_iou(self, make_frame):
        """Localisation quality must stay a real measurement under either rule.

        Otherwise mean IoU would just restate the matching threshold.
        """
        frame = make_frame(gt=[box(50, 50, 12)], preds=[(box(53, 53, 12), 0.9, 0)])
        _, _, ious = match_frame(frame, MatchCriterion(CENTER, 1.0))
        assert 0.35 < ious[0] < 0.45

    def test_still_one_claim_per_target(self, make_frame):
        """Two predictions on one drone: one hit, one false alarm, under either rule."""
        frame = make_frame(gt=[box(50, 50, 12)],
                           preds=[(box(50, 50, 12), 0.9, 0), (box(51, 51, 12), 0.8, 0)])
        tp, _, _ = match_frame(frame, MatchCriterion(CENTER, 1.0))
        assert tp.tolist() == [True, False]

    def test_still_class_aware(self, make_frame):
        frame = make_frame(gt=[box(50, 50, 12)], gt_classes=[0],
                           preds=[(box(50, 50, 12), 0.9, 1)])
        assert not match_frame(frame, MatchCriterion(CENTER, 1.0))[0][0]

    def test_takes_the_nearest_of_several_targets(self, make_frame):
        frame = make_frame(gt=[box(50, 50, 12), box(56, 50, 12)],
                           preds=[(box(55, 50, 12), 0.9, 0)])
        _, matched, _ = match_frame(frame, MatchCriterion(CENTER, 1.0))
        assert matched[0] == 1


class TestAsCriterion:

    def test_a_bare_float_still_means_iou(self):
        assert as_criterion(0.4) == MatchCriterion(IOU, 0.4)

    def test_a_criterion_passes_through(self):
        criterion = MatchCriterion(CENTER, 2.0)
        assert as_criterion(criterion) is criterion


class TestEvaluateUnderCenterMatching:

    @pytest.fixture
    def frames(self, make_frame):
        """Three tiny targets, each found 3 px off on both axes — IoU 0.39 apiece."""
        return [make_frame(key=f"f{i}", gt=[box(50, 50, 12)],
                           preds=[(box(53, 53, 12), 0.9, 0)]) for i in range(3)]

    def test_iou_scores_them_all_as_misses(self, frames):
        scored = evaluate(frames, MatchCriterion(IOU, 0.5))
        assert (scored.tp, scored.fp, scored.fn) == (0, 3, 3)

    def test_center_scores_them_all_as_hits(self, frames):
        scored = evaluate(frames, MatchCriterion(CENTER, 1.0))
        assert (scored.tp, scored.fp, scored.fn) == (3, 0, 0)

    def test_map_is_nan_because_it_is_an_iou_sweep(self, frames):
        """Reporting a number here would invite comparing it to published mAP."""
        scored = evaluate(frames, MatchCriterion(CENTER, 1.0))
        assert scored.map != scored.map  # NaN

    def test_mean_iou_still_reports_real_localisation(self, frames):
        scored = evaluate(frames, MatchCriterion(CENTER, 1.0))
        assert 0.35 < scored.mean_iou < 0.45

    def test_criterion_is_recorded_in_the_metrics(self, frames):
        assert "centre" in evaluate(frames, MatchCriterion(CENTER, 1.0)).criterion
        assert evaluate(frames, 0.5).criterion == "IoU@0.50"

    def test_recall_by_size_follows_the_criterion(self, frames):
        by_size = dict((label, recall)
                       for label, _, recall in evaluate(frames,
                                                        MatchCriterion(CENTER, 1.0)).by_size)
        assert by_size["tiny   (<16px)"] == pytest.approx(1.0)
