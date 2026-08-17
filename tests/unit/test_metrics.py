"""Tests for the IoU matching and AP math in `src.eval.metrics`.

Every expected value here is hand-computed and shown in the test, because these
functions fail *silently*: a wrong IoU or a mis-sorted match produces a
plausible number rather than an exception, and that number then steers real
modelling decisions.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.metrics import (
    average_precision,
    evaluate,
    iou_matrix,
    match_frame,
)


class TestIouMatrix:
    """Pairwise IoU, against values computed by hand."""

    def test_known_overlaps(self, unit_box):
        # vs (50,50,150,150): inter 50*50=2500, union 10000+10000-2500=17500
        # vs itself:          inter 10000,     union 10000
        # vs (200,...):       disjoint
        others = np.array([[50.0, 50.0, 150.0, 150.0],
                           [0.0, 0.0, 100.0, 100.0],
                           [200.0, 200.0, 300.0, 300.0]])
        got = iou_matrix(unit_box, others)[0]
        assert got == pytest.approx([2500 / 17500, 1.0, 0.0])

    def test_half_overlap_is_exactly_one_half(self, unit_box):
        # inter 50*100=5000, union 10000+5000-5000=10000 -> exactly 0.5.
        # Pins the threshold boundary, which >= vs > would silently move.
        half = np.array([[0.0, 0.0, 50.0, 100.0]])
        assert iou_matrix(unit_box, half)[0, 0] == pytest.approx(0.5)

    def test_shape_and_empty_inputs(self, unit_box):
        assert iou_matrix(unit_box, np.zeros((3, 4))).shape == (1, 3)
        assert iou_matrix(np.zeros((0, 4)), unit_box).shape == (0, 1)
        assert iou_matrix(unit_box, np.zeros((0, 4))).shape == (1, 0)

    def test_degenerate_box_gives_zero_not_nan(self):
        """A zero-area box must not produce NaN through a 0/0 division."""
        zero_area = np.array([[10.0, 10.0, 10.0, 10.0]])
        got = iou_matrix(zero_area, zero_area)
        assert not np.isnan(got).any()
        assert got[0, 0] == 0.0


class TestMatchFrame:
    """Greedy, class-aware, one-claim-per-ground-truth matching."""

    def test_exact_match_is_true_positive(self, make_frame):
        frame = make_frame(gt=[[0, 0, 100, 100]],
                           preds=[([0, 0, 100, 100], 0.9, 0)])
        tp, matched_gt, match_iou = match_frame(frame, 0.5)
        assert list(tp) == [True]
        assert list(matched_gt) == [0]
        assert match_iou[0] == pytest.approx(1.0)

    @pytest.mark.parametrize("threshold, expected", [(0.5, False), (0.1, True)])
    def test_threshold_is_respected(self, make_frame, threshold, expected):
        """IoU here is 2500/17500 = 0.1429: below 0.5, above 0.1."""
        frame = make_frame(gt=[[50, 50, 150, 150]],
                           preds=[([0, 0, 100, 100], 0.9, 0)])
        assert bool(match_frame(frame, threshold)[0][0]) is expected

    def test_boundary_iou_counts_as_a_match(self, make_frame):
        """IoU exactly equal to the threshold must match (>=, not >)."""
        frame = make_frame(gt=[[0, 0, 100, 100]],
                           preds=[([0, 0, 50, 100], 0.9, 0)])  # IoU exactly 0.5
        assert bool(match_frame(frame, 0.5)[0][0]) is True

    def test_duplicate_prediction_becomes_false_positive(self, make_frame):
        """One ground truth can be claimed once; the extra box is punished."""
        frame = make_frame(gt=[[0, 0, 100, 100]],
                           preds=[([0, 0, 100, 100], 0.9, 0),
                                  ([0, 0, 100, 100], 0.8, 0)])
        assert list(match_frame(frame, 0.5)[0]) == [True, False]

    def test_highest_confidence_claims_first(self, make_frame):
        """Greedy order is by confidence, not by IoU.

        Pred A (conf 0.9) overlaps 0.81; pred B (conf 0.5) overlaps 1.0. A is
        more confident, so it takes the target and B is left as a false
        positive. Sorting by IoU instead would flip this.
        """
        frame = make_frame(gt=[[0, 0, 100, 100]],
                           preds=[([10, 10, 100, 100], 0.9, 0),
                                  ([0, 0, 100, 100], 0.5, 0)])
        tp, _, match_iou = match_frame(frame, 0.5)
        assert list(tp) == [True, False]
        assert match_iou[0] == pytest.approx(0.81)

    def test_class_mismatch_cannot_satisfy_ground_truth(self, make_frame):
        """A bird prediction must not be credited for a drone."""
        frame = make_frame(gt=[[0, 0, 100, 100]],
                           preds=[([0, 0, 100, 100], 0.9, 1)])
        assert list(match_frame(frame, 0.5)[0]) == [False]

    def test_two_targets_matched_independently(self, make_frame):
        frame = make_frame(gt=[[0, 0, 100, 100], [500, 500, 600, 600]],
                           preds=[([0, 0, 100, 100], 0.9, 0),
                                  ([500, 500, 600, 600], 0.8, 0)])
        tp, matched_gt, _ = match_frame(frame, 0.5)
        assert list(tp) == [True, True]
        assert sorted(matched_gt) == [0, 1]

    @pytest.mark.parametrize("gt, preds", [
        ([[0, 0, 100, 100]], None),            # nothing predicted
        (None, [([0, 0, 100, 100], 0.9, 0)]),  # nothing to find
        (None, None),
    ])
    def test_empty_sides_do_not_crash(self, make_frame, gt, preds):
        tp, matched_gt, match_iou = match_frame(make_frame(gt=gt, preds=preds), 0.5)
        assert not tp.any()
        assert len(tp) == len(matched_gt) == len(match_iou)


class TestAveragePrecision:
    """101-point interpolated AP."""

    def test_all_correct_scores_one(self):
        assert average_precision(np.array([True, True]),
                                 np.array([0.9, 0.8]), n_gt=2) == pytest.approx(1.0)

    def test_no_predictions_scores_zero(self):
        assert average_precision(np.zeros(0, bool), np.zeros(0), n_gt=5) == 0.0

    def test_all_wrong_scores_zero(self):
        assert average_precision(np.array([False, False]),
                                 np.array([0.9, 0.8]), n_gt=2) == pytest.approx(0.0)

    def test_no_ground_truth_is_undefined(self):
        assert np.isnan(average_precision(np.array([True]), np.array([0.9]), n_gt=0))

    def test_known_mixed_case(self):
        """Two hits and a low-confidence false alarm against three targets.

        Sorted by confidence the run is TP, TP, FP, giving recall
        [1/3, 2/3, 2/3] at precision [1, 1, 2/3]. After the monotonic envelope,
        the 101-point grid sits at 1.0 for the 67 points up to recall 2/3 and at
        0 beyond, so AP = 67/101.
        """
        got = average_precision(np.array([True, True, False]),
                                np.array([0.9, 0.8, 0.3]), n_gt=3)
        assert got == pytest.approx(67 / 101)

    def test_confidence_ordering_changes_the_score(self):
        """The same hit/miss counts score worse when the false alarm is confident.

        Ordered FP, TP, TP the recall points are [0, 1/3, 2/3] at precision
        [0, 1/2, 2/3]; the monotonic envelope flattens all three to 2/3, so the
        same 67 grid points now sit at 2/3 instead of 1.0 -> AP = 134/303.
        """
        confident_fp = average_precision(np.array([False, True, True]),
                                         np.array([0.9, 0.8, 0.7]), n_gt=3)
        assert confident_fp == pytest.approx(134 / 303)
        assert confident_fp < 67 / 101


class TestEvaluate:
    """The assembled Metrics record."""

    @pytest.fixture
    def scored(self, make_frame):
        """Three targets: one clean hit, one hit plus a false alarm, one miss."""
        frames = [
            make_frame(key="a", gt=[[270, 270, 370, 370]],
                       preds=[([270, 270, 370, 370], 0.9, 0)]),
            make_frame(key="b", gt=[[270, 270, 370, 370]],
                       preds=[([270, 270, 370, 370], 0.8, 0),
                              ([10, 10, 60, 60], 0.3, 0)]),
            make_frame(key="c", gt=[[270, 270, 370, 370]], preds=None),
        ]
        return evaluate(frames, 0.5)

    def test_counts(self, scored):
        assert (scored.n_frames, scored.n_gt, scored.n_pred) == (3, 3, 3)
        assert (scored.tp, scored.fp, scored.fn) == (2, 1, 1)

    def test_precision_recall(self, scored):
        assert scored.precision == pytest.approx(2 / 3)
        assert scored.recall == pytest.approx(2 / 3)

    def test_ap_matches_hand_computation(self, scored):
        assert scored.ap50 == pytest.approx(67 / 101)

    def test_map_equals_ap50_when_matches_are_exact(self, scored):
        """Both true positives have IoU 1.0, so they survive every threshold."""
        assert scored.map == pytest.approx(scored.ap50)

    def test_mean_iou_covers_matches_only(self, scored):
        assert scored.mean_iou == pytest.approx(1.0)

    def test_frames_with_miss(self, scored):
        assert scored.frames_with_miss == 1

    def test_recall_by_size_buckets_by_area(self, make_frame):
        """10x10 -> tiny, 20x20 -> small, 100x100 -> large; the small one is missed."""
        frames = [
            make_frame(key="tiny", gt=[[0, 0, 10, 10]],
                       preds=[([0, 0, 10, 10], 0.9, 0)]),
            make_frame(key="small", gt=[[0, 0, 20, 20]], preds=None),
            make_frame(key="large", gt=[[0, 0, 100, 100]],
                       preds=[([0, 0, 100, 100], 0.9, 0)]),
        ]
        by_size = {label.split()[0]: (count, recall)
                   for label, count, recall in evaluate(frames, 0.5).by_size}

        assert by_size["tiny"] == (1, pytest.approx(1.0))
        assert by_size["small"] == (1, pytest.approx(0.0))
        assert by_size["large"] == (1, pytest.approx(1.0))
        assert by_size["medium"][0] == 0
        assert np.isnan(by_size["medium"][1])

    def test_perfect_run_scores_one(self, make_frame):
        frames = [make_frame(gt=[[0, 0, 100, 100]],
                             preds=[([0, 0, 100, 100], 0.9, 0)])]
        scored = evaluate(frames, 0.5)
        assert scored.ap50 == pytest.approx(1.0)
        assert scored.precision == pytest.approx(1.0)
        assert scored.recall == pytest.approx(1.0)
        assert scored.fn == 0
