"""Binning precision and recall by size.

The trap these guard is the one the module docstring names: precision and recall
bin on *different* columns, and silently binning precision on `gt_size` would
drop every false alarm — turning the chart into a restatement of recall while
still being labelled precision.

The localisation error has its own version of the same trap: it exists only for
matched pairs, so a bin whose targets were all missed must come back empty rather
than perfect.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.curves import (MIN_RELIABLE, SIZE_EDGES, Curve, ErrorCurve,
                             curve_rows, loc_error_by_size, precision_by_size,
                             recall_by_size)


def row(outcome, pred_size=None, gt_size=None, offset=None):
    """One dump row, with only the columns the curves read.

    `offset` is `center_dist_rel`, blank on anything that is not a matched pair
    -- which is what the dump itself writes there.
    """
    return {"outcome": outcome,
            "pred_size": "" if pred_size is None else str(pred_size),
            "gt_size": "" if gt_size is None else str(gt_size),
            "center_dist_rel": "" if offset is None else str(offset)}


def test_precision_counts_false_alarms_in_their_own_size_bin():
    """An fp has no target, so it must be binned on the box the detector drew."""
    rows = [row("tp", pred_size=10, gt_size=10),
            row("fp", pred_size=10),
            row("fp", pred_size=40)]
    curve = precision_by_size(rows)

    bin_8_12 = curve.labels.index("8-12")
    assert curve.values[bin_8_12] == pytest.approx(0.5)  # one of two claims right
    assert curve.total[bin_8_12] == 2
    assert curve.values[curve.labels.index(">=48")] != curve.values[bin_8_12]


def test_precision_ignores_missed_targets():
    """A miss is not a wrong claim; counting it as one would double-penalise."""
    rows = [row("tp", pred_size=10, gt_size=10), row("fn", gt_size=10)]
    curve = precision_by_size(rows)
    assert curve.values[curve.labels.index("8-12")] == pytest.approx(1.0)
    assert curve.total[curve.labels.index("8-12")] == 1


def test_recall_bins_on_true_size_and_ignores_false_alarms():
    """Recall's denominator is targets, so only tp and fn rows take part."""
    rows = [row("tp", pred_size=30, gt_size=10),
            row("fn", gt_size=10),
            row("fp", pred_size=10)]
    curve = recall_by_size(rows)

    bin_8_12 = curve.labels.index("8-12")
    assert curve.values[bin_8_12] == pytest.approx(0.5)
    assert curve.total[bin_8_12] == 2  # the false alarm is not a target


def test_a_tp_is_binned_by_prediction_for_precision_and_by_truth_for_recall():
    """The two curves genuinely disagree about where one object belongs."""
    rows = [row("tp", pred_size=40, gt_size=10)]
    assert precision_by_size(rows).total[SIZE_EDGES.index(32.0)] == 1  # 32-48 bin
    assert recall_by_size(rows).total[SIZE_EDGES.index(8.0)] == 1      # 8-12 bin


def test_empty_bins_are_nan_not_zero():
    """0.0 claims the detector was tried at that size and failed; NaN says nothing was."""
    curve = precision_by_size([row("tp", pred_size=10, gt_size=10)])
    empty = curve.labels.index(">=48")
    assert np.isnan(curve.values[empty])
    assert curve.total[empty] == 0


def test_bins_are_half_open_at_the_top_edge():
    """A box exactly on an edge belongs to the bin above it, as np.digitize has it."""
    curve = precision_by_size([row("fp", pred_size=12)])
    assert curve.total[curve.labels.index("12-16")] == 1
    assert curve.total[curve.labels.index("8-12")] == 0


def test_the_open_top_bin_catches_oversized_boxes():
    """An appearance detector's false alarms pile up above any fixed edge."""
    curve = precision_by_size([row("fp", pred_size=5000)])
    assert curve.total[-1] == 1


def test_reliability_flag_follows_the_sample_count():
    """Bins too thin to read are flagged rather than dropped."""
    rows = [row("fp", pred_size=10)] * (MIN_RELIABLE - 1)
    assert not precision_by_size(rows).reliable[SIZE_EDGES.index(8.0)]

    rows.append(row("fp", pred_size=10))
    assert precision_by_size(rows).reliable[SIZE_EDGES.index(8.0)]


def test_labels_read_as_ranges():
    """The x axis has to say what a bin covers without a legend of its own."""
    curve = precision_by_size([])
    assert curve.labels[0] == "<8"
    assert curve.labels[1] == "8-12"
    assert curve.labels[-1] == ">=48"


def test_centres_are_finite_including_the_open_bin():
    """An infinite edge cannot be a plotting coordinate."""
    assert np.isfinite(precision_by_size([]).centres).all()


def test_curve_rows_carry_the_counts_with_the_values():
    """A ratio without its denominator is what makes a tail bin misleadable."""
    curves = {"centre@1x": precision_by_size([row("tp", pred_size=10, gt_size=10),
                                              row("fp", pred_size=10)])}
    rows = [r for r in curve_rows(curves) if r["bin"] == "8-12"]
    assert rows[0]["value"] == pytest.approx(0.5)
    assert (rows[0]["hits"], rows[0]["total"]) == (1, 2)
    assert rows[0]["metric"] == "precision" and rows[0]["binned_on"] == "pred_size"


def test_empty_bin_value_is_none_in_the_written_rows():
    """NaN is not JSON or CSV; the row has to say 'no data' in a readable way."""
    rows = curve_rows({"x": Curve("precision", "pred_size", SIZE_EDGES,
                                  np.array([np.nan] * (len(SIZE_EDGES) - 1)),
                                  np.zeros(len(SIZE_EDGES) - 1),
                                  np.zeros(len(SIZE_EDGES) - 1))})
    assert all(r["value"] is None for r in rows)


def test_loc_error_bins_on_true_size_not_predicted_size():
    """The offset belongs to the target, so an oversized box must not move it."""
    rows = [row("tp", pred_size=40, gt_size=10, offset=0.4)]
    curve = loc_error_by_size(rows)

    assert curve.binned_on == "gt_size"
    assert curve.total[curve.labels.index("8-12")] == 1
    assert curve.total[curve.labels.index(">=48")] == 0


def test_loc_error_covers_matched_pairs_only():
    """A miss has no box and a false alarm has no target: neither has an offset."""
    rows = [row("tp", pred_size=10, gt_size=10, offset=0.2),
            row("fn", gt_size=10),
            row("fp", pred_size=10)]
    curve = loc_error_by_size(rows)

    bin_8_12 = curve.labels.index("8-12")
    assert curve.total[bin_8_12] == 1
    assert curve.mean[bin_8_12] == pytest.approx(0.2)


def test_loc_error_reports_the_tail_not_just_the_mean():
    """Nine tight hits and one near-miss: the median holds, p90 exposes it."""
    rows = [row("tp", gt_size=10, offset=0.1) for _ in range(9)]
    rows.append(row("tp", gt_size=10, offset=0.9))
    curve = loc_error_by_size(rows)

    bin_8_12 = curve.labels.index("8-12")
    assert curve.median[bin_8_12] == pytest.approx(0.1)   # unmoved
    assert curve.mean[bin_8_12] == pytest.approx(0.18)    # dragged
    assert curve.p90[bin_8_12] > curve.median[bin_8_12]   # and the tail shows it


def test_a_bin_whose_targets_were_all_missed_is_empty_not_perfect():
    """The failure this guards: 0.000 offset reads as flawless placement."""
    curve = loc_error_by_size([row("fn", gt_size=10)])

    bin_8_12 = curve.labels.index("8-12")
    assert curve.total[bin_8_12] == 0
    assert np.isnan(curve.mean[bin_8_12])


def test_error_rows_carry_the_spread_where_a_ratio_carries_hits():
    """One row shape for both, with the inapplicable columns blank rather than 0."""
    curve = loc_error_by_size([row("tp", gt_size=10, offset=0.2)])
    written = {r["bin"]: r for r in curve_rows({"offsets": curve})}

    assert written["8-12"]["metric"] == "loc_error"
    assert written["8-12"]["value"] == pytest.approx(0.2)
    assert written["8-12"]["p90"] == pytest.approx(0.2)
    assert written["8-12"]["hits"] is None
    assert written[">=48"]["value"] is None


def test_a_ratio_and_an_error_share_one_x_axis():
    """They are drawn against each other, so the bins must label identically."""
    rows = [row("tp", pred_size=10, gt_size=10, offset=0.2)]
    assert loc_error_by_size(rows).labels == precision_by_size(rows).labels
    assert isinstance(loc_error_by_size(rows), ErrorCurve)
