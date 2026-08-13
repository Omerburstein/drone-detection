"""Printing the metric block.

Precision and recall are always shown together: they need opposite fixes, and a
single headline number hides which one is failing.
"""

from __future__ import annotations

from .metrics import EPS, Metrics

BANNER_WIDTH = 52  # characters, sized to the widest metric line below


def _print_size_table(by_size: list[tuple[str, int, float]]) -> None:
    """Recall per ground-truth size bucket, skipping buckets with no targets."""
    print("\n  recall by target size:")
    for label, count, recall in by_size:
        if count == 0:
            continue
        print(f"    {label}  n={count:<7} recall={recall:.4f}")
    print()


def report(metrics: Metrics, primary_iou: float) -> None:
    """Print the full metric block for one scored run."""
    f1 = (2 * metrics.precision * metrics.recall
          / max(metrics.precision + metrics.recall, EPS))
    print(f"\n{'=' * BANNER_WIDTH}")
    print(f"{metrics.n_frames} frames | {metrics.n_gt} ground-truth boxes "
          f"| {metrics.n_pred} predictions")
    print(f"{'=' * BANNER_WIDTH}")
    print(f"  AP@0.50          {metrics.ap50:.4f}")
    print(f"  mAP@0.50:0.95    {metrics.map:.4f}")
    print(f"\n  at IoU {primary_iou:.2f}:")
    print(f"  precision        {metrics.precision:.4f}   (TP {metrics.tp} / "
          f"FP {metrics.fp})")
    print(f"  recall           {metrics.recall:.4f}   (missed {metrics.fn})")
    print(f"  F1               {f1:.4f}")
    print(f"  mean IoU         {metrics.mean_iou:.4f}   (matched boxes only -- "
          f"localisation quality)")
    print(f"  frames w/ a miss {metrics.frames_with_miss}/{metrics.n_frames}")

    _print_size_table(metrics.by_size)
