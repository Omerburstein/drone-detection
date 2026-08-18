"""Binning precision and recall by size.

The trap these guard is the one the module docstring names: precision and recall
bin on *different* columns, and silently binning precision on `gt_size` would
drop every false alarm — turning the chart into a restatement of recall while
still being labelled precision.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.curves import (MIN_RELIABLE, SIZE_EDGES, Curve, curve_rows,
                             precision_by_size, recall_by_size)


def row(outcome, pred_size=None, gt_size=None):
    """One dump row, with only the columns the curves read."""
    return {"outcome": outcome,
            "pred_size": "" if pred_size is None else str(pred_size),
            "gt_size": "" if gt_size is None else str(gt_size)}


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
