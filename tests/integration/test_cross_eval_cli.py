"""End-to-end tests for the `src.cross_eval` CLI.

Covers the seams the unit tests cannot: reading a real dump CSV off disk,
the `NAME=PATH` series syntax, the `--band`/`--condition` filters that pick one
cell out of a run, and the `--csv` sidecar the ledger cites. Also pins the two
refusals — a dump scored without the axis being asked for, and a filter that
matches nothing — because both are silent-wrong-answer failures otherwise.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

COLUMNS = ["key", "outcome", "gt_size", "iou", "center_dist_rel", "scene_category"]


def dump_row(key, outcome, gt_size="", iou="", offset="", scene="complex"):
    """One row of a scoring dump, in the columns the cross-cut reads."""
    return {"key": key, "outcome": outcome, "gt_size": gt_size, "iou": iou,
            "center_dist_rel": offset, "scene_category": scene}


def write_dump(path: Path, rows: list[dict], columns: list[str] = COLUMNS) -> Path:
    """Lay a dump CSV down at `path`."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Invoke `python -m src.cross_eval` from the repo root."""
    return subprocess.run([sys.executable, "-m", "src.cross_eval", *args],
                          cwd=REPO_ROOT, capture_output=True, text=True)


@pytest.fixture
def mixed_dump(tmp_path):
    """Two backgrounds x two size bands, with one false alarm on a tiny target.

    Deliberately shaped so a pooled number and the cell number differ: the tiny
    complex cell is 1 of 2 found, while the run as a whole is 3 of 4.
    """
    rows = [
        dump_row("a_0001", "tp", "6.0", "0.55", "0.22", "complex"),
        dump_row("a_0002", "fn", "6.0", scene="complex"),
        dump_row("a_0002", "fp", scene="complex"),
        dump_row("b_0001", "tp", "30.0", "0.85", "0.05", "complex"),
        dump_row("c_0001", "tp", "6.0", "0.60", "0.15", "ordinary"),
    ]
    return write_dump(tmp_path / "matches_center.csv", rows)


class TestCrossEvalCli:

    def test_filters_down_to_one_cell(self, mixed_dump):
        result = run_cli("--dump", str(mixed_dump), "--band", "<8",
                         "--condition", "complex")
        assert result.returncode == 0, result.stderr
        lines = [ln for ln in result.stdout.splitlines() if "complex" in ln]
        assert len(lines) == 1
        assert "0.5000" in lines[0]           # Pd: one of two tiny targets found
        assert "ordinary" not in result.stdout

    def test_false_alarm_is_attributed_to_the_tiny_cell(self, mixed_dump):
        """The alarm fired on a frame whose target was 6 px, so it counts there."""
        result = run_cli("--dump", str(mixed_dump), "--band", "<8",
                         "--condition", "complex")
        row = [ln for ln in result.stdout.splitlines() if "complex" in ln][0]
        assert row.split()[4] == "1"          # FP column
        assert "0.5000" in row                # and FA/frame over 2 frames

    def test_series_are_named_and_kept_apart(self, mixed_dump, tmp_path):
        other = write_dump(tmp_path / "matches_iou50.csv",
                           [dump_row("a_0001", "fn", "6.0", scene="complex")])
        result = run_cli("--dump", f"centre@1x={mixed_dump}",
                         "--dump", f"IoU@0.50={other}", "--band", "<8")
        assert result.returncode == 0, result.stderr
        assert "centre@1x" in result.stdout
        assert "IoU@0.50" in result.stdout

    def test_bare_path_is_named_after_its_run_directory(self, tmp_path):
        run_dir = tmp_path / "exp009_something"
        run_dir.mkdir()
        dump = write_dump(run_dir / "matches_center.csv",
                          [dump_row("a_0001", "tp", "6.0", "0.5", "0.1")])
        result = run_cli("--dump", str(dump))
        assert "exp009_something" in result.stdout

    def test_custom_edges_collapse_the_ladder_to_two_bands(self, mixed_dump):
        result = run_cli("--dump", str(mixed_dump), "--edges", "0,8,inf",
                         "--condition", "complex")
        assert result.returncode == 0, result.stderr
        assert "<8" in result.stdout
        assert ">=8" in result.stdout
        assert "8-12" not in result.stdout

    def test_csv_sidecar_matches_what_was_printed(self, mixed_dump, tmp_path):
        out = tmp_path / "small_complex.csv"
        result = run_cli("--dump", f"centre@1x={mixed_dump}", "--band", "<8",
                         "--condition", "complex", "--csv", str(out))
        assert result.returncode == 0, result.stderr

        rows = list(csv.DictReader(out.open(encoding="utf-8")))
        assert len(rows) == 1
        assert rows[0]["series"] == "centre@1x"
        assert rows[0]["band"] == "<8"
        assert rows[0]["condition"] == "complex"
        assert rows[0]["targets"] == "2"
        assert rows[0]["fp"] == "1"
        assert rows[0]["pd"] == "0.5"
        assert rows[0]["far"] == "0.5"

    def test_pooled_row_appears_only_when_cells_are_pooled(self, mixed_dump):
        one = run_cli("--dump", str(mixed_dump), "--band", "<8",
                      "--condition", "complex")
        many = run_cli("--dump", str(mixed_dump), "--condition", "complex")
        assert "all" not in one.stdout
        assert "all" in many.stdout

    def test_missing_axis_is_refused_not_silently_uncategorised(self, tmp_path):
        """Every cell would read `uncategorised` and the table would say nothing."""
        dump = write_dump(tmp_path / "d.csv",
                          [{"key": "a", "outcome": "tp", "gt_size": "6.0"}],
                          ["key", "outcome", "gt_size"])
        result = run_cli("--dump", str(dump), "--axis", "scene_category")
        assert result.returncode != 0
        assert "scene_category" in result.stderr

    def test_filter_matching_nothing_is_refused(self, mixed_dump):
        """An empty table is indistinguishable from a run that found nothing."""
        result = run_cli("--dump", str(mixed_dump), "--condition", "backlit")
        assert result.returncode != 0
        assert "no cells match" in result.stderr

    def test_missing_dump_is_refused(self, tmp_path):
        result = run_cli("--dump", str(tmp_path / "nope.csv"))
        assert result.returncode != 0
        assert "No dump at" in result.stderr

    def test_unsorted_edges_are_refused(self, mixed_dump):
        result = run_cli("--dump", str(mixed_dump), "--edges", "0,16,8")
        assert result.returncode != 0
        assert "increasing order" in result.stderr
