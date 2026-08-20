"""Printing the metric block.

Precision and recall are always shown together: they need opposite fixes, and a
single headline number hides which one is failing.
"""

from __future__ import annotations

from .metrics import ConditionScore, LocError, Metrics, f1_score

BANNER_WIDTH = 52  # characters, sized to the widest metric line below
ABSENT = "-"  # column filler where a bucket has no matches to average over


def _size_error_columns(error: LocError | None) -> str:
    """The three localisation columns for one bucket, or dashes if it has none.

    Dashes rather than 0.000: a bucket whose targets were never found has no
    offset at all, and a zero there reads as perfect placement -- the exact
    misreading this table exists to prevent.
    """
    if error is None or error.n == 0:
        return f"{0:>8}{ABSENT:>9}{ABSENT:>9}{ABSENT:>9}"
    return (f"{error.n:>8}{error.mean:>9.3f}{error.median:>9.3f}"
            f"{error.p90:>9.3f}")


def _print_size_table(by_size: list[tuple[str, int, float]],
                      loc_by_size: list[LocError]) -> None:
    """Recall and localisation error per ground-truth size bucket.

    Both in one table because they answer the same question at two removes: how
    often was a drone of this size found, and when it was found, how well was the
    box placed. Split across two tables they get read separately, and the pairing
    -- high recall with a large offset, or the reverse -- is the whole diagnosis.

    `pairs` is not `targets`: the offsets average over matches only, so a bucket
    with 3 matches out of 900 targets shows 3 in that column. Buckets holding no
    targets at all are skipped, since an empty bucket says nothing either way.
    """
    errors = {error.label: error for error in loc_by_size}
    print()
    print("  by target size  (recall is Pd; offsets in multiples of target size):")
    print(f"    {'bucket':<16}{'targets':>8}{'recall':>9}"
          f"{'pairs':>8}{'offset':>9}{'median':>9}{'p90':>9}")
    for label, count, recall in by_size:
        if count == 0:
            continue
        print(f"    {label:<16}{count:>8}{recall:>9.4f}"
              f"{_size_error_columns(errors.get(label))}")
    print()


def _print_condition_table(by_condition: list[ConditionScore]) -> None:
    """Scores per bucket, one table per axis, in the shape published papers use.

    Aggregates hide the failure that matters: a detector can look acceptable
    overall while collapsing on complex backgrounds, on unlit targets, or at
    long range. Splitting the table by axis keeps those separable -- pooling
    them into one list would invite reading a lighting bucket against a scene
    category, which share no denominator.

    Bucket order is whatever `group_by_axis` produced, so contrast and range
    read worst-to-best rather than alphabetically.
    """
    by_axis: dict[str, list[ConditionScore]] = {}
    for score in by_condition:
        by_axis.setdefault(score.axis, []).append(score)

    for axis, scores in by_axis.items():
        print(f"\n  by {axis}:")
        print(f"    {'bucket':<18}{'frames':>7}{'gt':>7}"
              f"{'P':>8}{'R':>8}{'F1':>8}{'AP':>8}{'FA/frm':>9}")
        for score in scores:
            print(f"    {score.label:<18}{score.n_frames:>7}{score.n_gt:>7}"
                  f"{score.precision:>8.4f}{score.recall:>8.4f}"
                  f"{score.f1:>8.4f}{score.ap50:>8.4f}{score.far:>9.4f}")


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
    print(f"  recall (Pd)      {metrics.recall:.4f}   (missed {metrics.fn})")
    print(f"  F1               {f1:.4f}")
    print(f"  false alarms     {metrics.far:.4f}   (per frame: {metrics.fp} FP "
          f"over {metrics.n_frames} frames)")
    print(f"  mean IoU         {metrics.mean_iou:.4f}   (matched boxes only -- "
          f"localisation quality)")
    print(f"  centre offset    {metrics.loc_err:.4f}   (matched boxes only, in "
          f"target sizes; p90 {metrics.loc_err_p90:.4f})")
    print(f"  frames w/ a miss {metrics.frames_with_miss}/{metrics.n_frames}")

    _print_size_table(metrics.by_size, metrics.loc_by_size)
    if metrics.by_condition:
        _print_condition_table(metrics.by_condition)
        print()
