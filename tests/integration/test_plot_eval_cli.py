"""End-to-end tests for the `src.plot_eval` CLI.

The only one of the three dump-consuming CLIs with no integration test, while
`cross_eval` and `alarm_eval` have had one for a while. It is also the strictest
of the three about its `--dump` syntax, and that strictness is deliberate: the
label is what the legend says, and only the caller knows whether a given dump was
the centre-matched scoring or the strict one. Mislabelling those two is the
specific error this project keeps warning about, so `LABEL=` is required here
where the sibling CLIs infer a name.

Pinned ahead of the change that gives all three CLIs one shared dump parser --
these tests are what stops that consolidation quietly relaxing this one to match
the loose siblings.

Rendering is checked only for "a non-empty PNG appeared". What the figure looks
like is not something a test should own; that the CLI runs matplotlib to
completion on a real dump, and that the CSV behind the figure carries the
numbers, is.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

COLUMNS = ["key", "outcome", "gt_size", "pred_size", "center_dist_rel"]


def dump_row(key, outcome, gt_size="", pred_size="", offset=""):
    """One row of a scoring dump, in the columns the size curves read."""
    return {"key": key, "outcome": outcome, "gt_size": gt_size,
            "pred_size": pred_size, "center_dist_rel": offset}


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Invoke `python -m src.plot_eval` from the repo root."""
    return subprocess.run([sys.executable, "-m", "src.plot_eval", *args],
                          cwd=REPO_ROOT, capture_output=True, text=True)


@pytest.fixture
def dump(tmp_path):
    """A dump spanning two size bands, with one false alarm and one miss.

    Shaped so precision and recall differ: 3 of 4 predictions are right, but
    only 3 of 4 targets were found, and they are not the same rows.
    """
    rows = [
        dump_row("a_0001", "tp", gt_size="6.0", pred_size="6.2", offset="0.22"),
        dump_row("a_0002", "fn", gt_size="6.0"),
        dump_row("a_0003", "fp", pred_size="7.1"),
        dump_row("b_0001", "tp", gt_size="30.0", pred_size="29.4", offset="0.05"),
        dump_row("b_0002", "tp", gt_size="34.0", pred_size="33.1", offset="0.08"),
    ]
    path = tmp_path / "matches_center.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


class TestDumpSyntax:

    def test_rejects_a_bare_path(self, dump, tmp_path):
        """The strict form is the point: `cross_eval` and `alarm_eval` infer a
        name from the parent directory, and this CLI deliberately does not."""
        result = run_cli("--dump", str(dump), "--out", str(tmp_path / "f.png"))
        assert result.returncode != 0
        assert "expected LABEL=PATH" in result.stderr

    def test_accepts_label_equals_path(self, dump, tmp_path):
        out = tmp_path / "figure.png"
        result = run_cli("--dump", f"centre@1x={dump}", "--out", str(out))
        assert result.returncode == 0, result.stderr
        assert out.is_file() and out.stat().st_size > 0

    def test_overlays_two_series(self, dump, tmp_path):
        """The use case the CLI exists for: one run under two matching rules."""
        out = tmp_path / "figure.png"
        result = run_cli("--dump", f"centre@1x={dump}",
                         "--dump", f"iou@0.5={dump}", "--out", str(out))
        assert result.returncode == 0, result.stderr
        assert "centre@1x: 5 rows" in result.stdout
        assert "iou@0.5: 5 rows" in result.stdout


class TestOutputs:

    def test_writes_the_csv_beside_the_figure_by_default(self, dump, tmp_path):
        out = tmp_path / "figure.png"
        result = run_cli("--dump", f"centre@1x={dump}", "--out", str(out))
        assert result.returncode == 0, result.stderr
        assert out.with_suffix(".csv").is_file()

    def test_data_out_overrides_where_the_csv_lands(self, dump, tmp_path):
        out, data = tmp_path / "figure.png", tmp_path / "elsewhere" / "numbers.csv"
        result = run_cli("--dump", f"centre@1x={dump}",
                         "--out", str(out), "--data-out", str(data))
        assert result.returncode == 0, result.stderr
        assert data.is_file()
        assert not out.with_suffix(".csv").exists()

    def test_the_csv_carries_precision_recall_and_offset_series(self, dump, tmp_path):
        """All three curves the CLI computes reach the sidecar, not just the
        one that is plotted -- the sidecar is what later analysis reads."""
        out = tmp_path / "figure.png"
        assert run_cli("--dump", f"c={dump}", "--out", str(out)).returncode == 0
        with out.with_suffix(".csv").open(encoding="utf-8", newline="") as handle:
            series = {row["series"] for row in csv.DictReader(handle)}
        assert series == {"c", "c (recall)", "c (loc err)"}

    def test_creates_a_missing_output_directory(self, dump, tmp_path):
        out = tmp_path / "made" / "up" / "figure.png"
        assert run_cli("--dump", f"c={dump}", "--out", str(out)).returncode == 0
        assert out.is_file()


class TestRefusals:

    def test_exits_on_a_dump_that_is_not_there(self, tmp_path):
        result = run_cli("--dump", f"c={tmp_path / 'nope.csv'}",
                         "--out", str(tmp_path / "f.png"))
        assert result.returncode != 0

    def test_requires_a_dump(self, tmp_path):
        result = run_cli("--out", str(tmp_path / "f.png"))
        assert result.returncode != 0
        assert "--dump" in result.stderr
