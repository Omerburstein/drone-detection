"""Printing the metric block.

Precision and recall are always shown together: they need opposite fixes, and a
single headline number hides which one is failing.
"""

from __future__ import annotations

from .metrics import ConditionScore, Metrics, f1_score

BANNER_WIDTH = 52  # characters, sized to the widest metric line below


def _print_size_table(by_size: list[tuple[str, int, float]]) -> None:
    """Recall per ground-truth size bucket, skipping buckets with no targets."""
    print("\n  recall by target size:")
    for label, count, recall in by_size:
        if count == 0:
            continue
        print(f"    {label}  n={count:<7} recall={recall:.4f}")
    print()


def _print_condition_table(by_condition: list[ConditionScore]) -> None:
    """Scores per scene category, in the shape published papers report them.

    Aggregates hide the failure that matters: a detector can look acceptable
    overall while collapsing on complex backgrounds or the smallest targets.
    """
    print("\n  by scene condition:")
    print(f"    {'category':<16}{'frames':>7}{'gt':>7}{'P':>8}{'R':>8}{'F1':>8}{'AP':>8}")
    for score in by_condition:
        print(f"    {score.label:<16}{score.n_frames:>7}{score.n_gt:>7}"
              f"{score.precision:>8.4f}{score.recall:>8.4f}"
              f"{score.f1:>8.4f}{score.ap50:>8.4f}")


def report(metrics: Metrics) -> None:
    """Print the full metric block for one scored run.

    The matching criterion is named on every line it governs. P/R/F1 under
    centre matching and under IoU matching are different measurements, and a
    report that did not say which would invite comparing them.
    """
    f1 = f1_score(metrics.precision, metrics.recall)
    print(f"\n{'=' * BANNER_WIDTH}")
    print(f"{metrics.n_frames} frames | {metrics.n_gt} ground-truth boxes "
          f"| {metrics.n_pred} predictions")
    print(f"{'=' * BANNER_WIDTH}")
    print(f"  AP@0.50          {metrics.ap50:.4f}   ({metrics.criterion})")
    if metrics.map == metrics.map:  # NaN under any criterion but IoU
        print(f"  mAP@0.50:0.95    {metrics.map:.4f}")
    else:
        print("  mAP@0.50:0.95    n/a      (an IoU sweep by definition -- "
              "undefined here)")
    print(f"\n  matched by {metrics.criterion}:")
    print(f"  precision        {metrics.precision:.4f}   (TP {metrics.tp} / "
          f"FP {metrics.fp})")
    print(f"  recall           {metrics.recall:.4f}   (missed {metrics.fn})")
    print(f"  F1               {f1:.4f}")
    print(f"  mean IoU         {metrics.mean_iou:.4f}   (matched boxes only -- "
          f"localisation quality)")
    print(f"  frames w/ a miss {metrics.frames_with_miss}/{metrics.n_frames}")

    _print_size_table(metrics.by_size)
    if metrics.by_condition:
        _print_condition_table(metrics.by_condition)
        print()
