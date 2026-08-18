"""End-to-end tests for the `src.evaluate` CLI.

These run the real command-line entry point over synthetic files, so they cover
the seams the unit tests cannot: argument handling, the JSONL row schema shared
between the recorder and the evaluator, and the `--json-out` contract that the
experiment ledger cites.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.algo.detections import Detections

REPO_ROOT = Path(__file__).resolve().parents[2]


def write_run(tmp_path: Path, records: list[dict], labels: dict[str, str]):
    """Lay out a minimal run directory: predictions plus a label folder."""
    pred = tmp_path / "detections.jsonl"
    pred.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    for stem, content in labels.items():
        (labels_dir / f"{stem}.txt").write_text(content)
    return pred, labels_dir


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Invoke `python -m src.evaluate` from the repo root."""
    return subprocess.run([sys.executable, "-m", "src.evaluate", *args],
                          cwd=REPO_ROOT, capture_output=True, text=True)


@pytest.fixture
def perfect_run(tmp_path):
    """One frame, one target, found exactly. Every metric should be 1.0."""
    records = [{"frame": 0,
                "detections": [{"bbox": [25, 25, 75, 75], "conf": 0.9, "cls": 0}]}]
    return write_run(tmp_path, records, {"0": "0 0.5 0.5 0.5 0.5\n"})


class TestEvaluateCli:

    def test_perfect_run_reports_perfect_scores(self, perfect_run, tmp_path):
        pred, labels = perfect_run
        out = tmp_path / "metrics.json"
        result = run_cli("--pred", str(pred), "--labels", str(labels),
                         "--frame-size", "100", "100", "--match", "iou",
                         "--json-out", str(out))

        assert result.returncode == 0, result.stderr
        metrics = json.loads(out.read_text())
        assert metrics["ap50"] == pytest.approx(1.0)
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["recall"] == pytest.approx(1.0)
        assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (1, 0, 0)

    def test_mixed_run_matches_hand_computed_ap(self, tmp_path):
        """Two hits and a false alarm over three targets -> AP = 67/101.

        Scored with `--match iou` explicitly: the arithmetic is IoU's, so the
        test must not silently start measuring whatever the default becomes.
        """
        records = [
            {"frame": 0, "detections": [{"bbox": [25, 25, 75, 75], "conf": 0.9, "cls": 0}]},
            {"frame": 1, "detections": [{"bbox": [25, 25, 75, 75], "conf": 0.8, "cls": 0},
                                        {"bbox": [0, 0, 5, 5], "conf": 0.3, "cls": 0}]},
            {"frame": 2, "detections": []},
        ]
        label = "0 0.5 0.5 0.5 0.5\n"
        pred, labels = write_run(tmp_path, records, {"0": label, "1": label, "2": label})
        out = tmp_path / "metrics.json"

        result = run_cli("--pred", str(pred), "--labels", str(labels),
                         "--frame-size", "100", "100", "--match", "iou",
                         "--json-out", str(out))
        assert result.returncode == 0, result.stderr

        metrics = json.loads(out.read_text())
        assert metrics["ap50"] == pytest.approx(67 / 101)
        assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (2, 1, 1)
        assert metrics["frames_with_miss"] == 1

    def test_report_reaches_stdout(self, perfect_run):
        pred, labels = perfect_run
        result = run_cli("--pred", str(pred), "--labels", str(labels),
                         "--frame-size", "100", "100")
        assert result.returncode == 0, result.stderr
        assert "AP@0.50" in result.stdout
        assert "recall by target size" in result.stdout.lower()

    def test_default_criterion_is_the_centre_rule(self, perfect_run):
        """The default is centre matching, and the report says so on every line
        it governs -- an unlabelled P/R invites comparison against an IoU one."""
        pred, labels = perfect_run
        result = run_cli("--pred", str(pred), "--labels", str(labels),
                         "--frame-size", "100", "100")
        assert result.returncode == 0, result.stderr
        assert "centre@1x target size" in result.stdout
        assert "mAP@0.50:0.95    n/a" in result.stdout

    def test_save_appends_each_scoring_with_its_settings(self, perfect_run, tmp_path):
        """Two scorings of one run, two lines, each naming its own criterion."""
        pred, labels = perfect_run
        log = tmp_path / "nested" / "results.jsonl"
        common = ("--pred", str(pred), "--labels", str(labels),
                  "--frame-size", "100", "100", "--save", str(log))

        assert run_cli(*common).returncode == 0
        assert run_cli(*common, "--match", "iou", "--iou", "0.75").returncode == 0

        records = [json.loads(line)
                   for line in log.read_text(encoding="utf-8").splitlines()]
        assert [r["criterion"] for r in records] == ["centre@1x target size",
                                                     "IoU@0.75"]
        assert records[0]["settings"]["pred"] == str(pred)
        assert records[1]["metrics"]["ap50"] == pytest.approx(1.0)

    def test_missing_labels_directory_is_rejected(self, perfect_run, tmp_path):
        pred, _ = perfect_run
        result = run_cli("--pred", str(pred),
                         "--labels", str(tmp_path / "nope"), "--frame-size", "10", "10")
        assert result.returncode != 0
        assert "directory" in (result.stderr + result.stdout).lower()

    def test_run_with_no_ground_truth_fails_loudly(self, tmp_path):
        """Scoring against an empty label set would report a meaningless 0.0."""
        records = [{"frame": 0, "detections": []}]
        pred, labels = write_run(tmp_path, records, {})
        result = run_cli("--pred", str(pred), "--labels", str(labels),
                         "--frame-size", "100", "100")
        assert result.returncode != 0
        assert "ground-truth" in (result.stderr + result.stdout).lower()


class TestDetectionsRoundTrip:
    """The JSONL row schema is written by the recorder and read by the evaluator.

    If these two drift, every downstream metric is wrong while both sides look
    individually correct, so the round trip is pinned here.
    """

    def test_records_survive_a_round_trip(self):
        original = Detections(
            boxes=np.array([[1.25, 2.5, 3.75, 4.0], [10.0, 20.0, 30.0, 40.0]]),
            scores=np.array([0.9123, 0.5]),
            classes=np.array([0, 1]),
        )
        restored = Detections.from_records(original.to_records(bbox_decimals=2,
                                                              conf_decimals=4))
        assert restored.boxes == pytest.approx(original.boxes)
        assert restored.scores == pytest.approx(original.scores)
        assert list(restored.classes) == list(original.classes)

    def test_empty_round_trip_keeps_shapes(self):
        restored = Detections.from_records(Detections.empty().to_records(1, 1))
        assert restored.boxes.shape == (0, 4)
        assert len(restored) == 0
