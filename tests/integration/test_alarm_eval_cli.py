"""End-to-end tests for the `src.alarm_eval` CLI.

Covers what the unit tests cannot: reading a dump off disk, the two unit
ladders, the `--group` split, the CSV sidecar, and the refusals. The dumps
written here deliberately omit the `nearest_gt_*` columns, so these also
exercise the derive-from-the-frame path against real files — which is the path
every dump written before 2026-08-20 takes.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

COLUMNS = ["key", "outcome", "video", "scene_category",
           "pred_x0", "pred_y0", "pred_x1", "pred_y1",
           "gt_x0", "gt_y0", "gt_x1", "gt_y1", "gt_size"]


def square(cx, cy, side):
    """An axis-aligned box of `side` px centred on (cx, cy)."""
    half = side / 2
    return (cx - half, cy - half, cx + half, cy + half)


def dump_row(key, outcome, pred=None, gt=None, video="phantom01", scene="complex"):
    """One dump row in the columns the alarm table reads."""
    record = {"key": key, "outcome": outcome, "video": video,
              "scene_category": scene, "gt_size": ""}
    for prefix, box in (("pred", pred), ("gt", gt)):
        for i, name in enumerate(("x0", "y0", "x1", "y1")):
            record[f"{prefix}_{name}"] = "" if box is None else box[i]
    if gt is not None:
        record["gt_size"] = gt[2] - gt[0]
    return record


def write_dump(path: Path, rows: list[dict], columns: list[str] = COLUMNS) -> Path:
    """Lay a dump CSV down at `path`."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Invoke `python -m src.alarm_eval` from the repo root."""
    return subprocess.run([sys.executable, "-m", "src.alarm_eval", *args],
                          cwd=REPO_ROOT, capture_output=True, text=True)


@pytest.fixture
def spread_dump(tmp_path):
    """Four alarms at deliberately different distances, plus one on an empty frame.

    Target is 10 px, so the offsets below are 0.5x, 3x and 20x its own size —
    one per interesting bin — and the last frame has no target at all.
    """
    rows = [
        dump_row("a_0001", "tp", pred=square(100, 100, 10), gt=square(100, 100, 10)),
        dump_row("a_0001", "fp", pred=square(105, 100, 10)),          # 5 px  = 0.5x
        dump_row("a_0002", "fn", gt=square(100, 100, 10)),
        dump_row("a_0002", "fp", pred=square(130, 100, 10)),          # 30 px = 3x
        dump_row("a_0003", "fn", gt=square(100, 100, 10)),
        dump_row("a_0003", "fp", pred=square(300, 100, 10)),          # 200 px = 20x
        dump_row("b_0001", "fp", pred=square(500, 500, 10),
                 video="phantom02", scene="ordinary"),                # no target
    ]
    return write_dump(tmp_path / "matches_center.csv", rows)


def bin_row(stdout: str, label: str) -> list[str]:
    """The printed row for one bin label, as whitespace-split fields."""
    for line in stdout.splitlines():
        fields = line.split()
        if fields and fields[0] == label:
            return fields
    raise AssertionError(f"no row for bin {label!r} in:\n{stdout}")


class TestAlarmEvalCli:

    def test_alarms_land_in_the_bins_their_distance_names(self, spread_dump):
        result = run_cli("--dump", str(spread_dump))
        assert result.returncode == 0, result.stderr
        assert bin_row(result.stdout, "<1")[1] == "1"
        assert bin_row(result.stdout, "2-4")[1] == "1"
        assert bin_row(result.stdout, "16-32")[1] == "1"

    def test_total_counts_every_alarm_including_the_orphan(self, spread_dump):
        result = run_cli("--dump", str(spread_dump))
        assert "4 false alarms" in result.stdout
        assert "no target in frame" in result.stdout

    def test_pixel_unit_uses_the_other_ladder(self, spread_dump):
        result = run_cli("--dump", str(spread_dump), "--unit", "px")
        assert result.returncode == 0, result.stderr
        assert "pixels from the nearest drone" in result.stdout
        assert bin_row(result.stdout, "5-10")[1] == "1"     # the 5 px alarm
        assert bin_row(result.stdout, "25-50")[1] == "1"    # the 30 px alarm
        assert bin_row(result.stdout, "100-250")[1] == "1"  # the 200 px alarm

    def test_custom_edges_collapse_the_ladder(self, spread_dump):
        result = run_cli("--dump", str(spread_dump), "--edges", "0,1,inf")
        assert result.returncode == 0, result.stderr
        assert bin_row(result.stdout, "<1")[1] == "1"
        assert bin_row(result.stdout, ">=1")[1] == "2"

    def test_group_splits_by_a_dump_column(self, spread_dump):
        result = run_cli("--dump", str(spread_dump), "--group", "video")
        assert result.returncode == 0, result.stderr
        assert "phantom01" in result.stdout
        assert "phantom02" in result.stdout

    def test_series_are_named_and_kept_apart(self, spread_dump, tmp_path):
        other = write_dump(tmp_path / "matches_iou50.csv",
                           [dump_row("a_0001", "fn", gt=square(100, 100, 10)),
                            dump_row("a_0001", "fp", pred=square(101, 100, 10))])
        result = run_cli("--dump", f"centre@1x={spread_dump}",
                         "--dump", f"IoU@0.50={other}")
        assert result.returncode == 0, result.stderr
        assert "centre@1x" in result.stdout
        assert "IoU@0.50" in result.stdout

    def test_csv_sidecar_accounts_for_every_alarm(self, spread_dump, tmp_path):
        out = tmp_path / "alarm_distance.csv"
        result = run_cli("--dump", f"centre@1x={spread_dump}", "--csv", str(out))
        assert result.returncode == 0, result.stderr

        rows = list(csv.DictReader(out.open(encoding="utf-8")))
        assert sum(int(r["alarms"]) for r in rows) == 4
        assert [r["series"] for r in rows] == ["centre@1x"] * len(rows)
        orphan = [r for r in rows if r["bin"] == "no target in frame"]
        assert len(orphan) == 1 and orphan[0]["alarms"] == "1"

    def test_run_with_no_alarms_is_refused(self, tmp_path):
        """An empty ladder is indistinguishable from a run that was never scored."""
        dump = write_dump(tmp_path / "clean.csv",
                          [dump_row("a_0001", "tp", pred=square(100, 100, 10),
                                    gt=square(100, 100, 10))])
        result = run_cli("--dump", str(dump))
        assert result.returncode != 0
        assert "no false alarms" in result.stderr

    def test_missing_group_column_is_refused(self, spread_dump):
        result = run_cli("--dump", str(spread_dump), "--group", "lighting")
        assert result.returncode != 0
        assert "lighting" in result.stderr

    def test_missing_dump_is_refused(self, tmp_path):
        result = run_cli("--dump", str(tmp_path / "nope.csv"))
        assert result.returncode != 0
        assert "No dump at" in result.stderr

    def test_unsorted_edges_are_refused(self, spread_dump):
        result = run_cli("--dump", str(spread_dump), "--edges", "0,8,4")
        assert result.returncode != 0
        assert "increasing order" in result.stderr
