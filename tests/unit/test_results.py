"""Tests for the results log and the CLI defaults that feed it.

The log's whole value is that a metric block is stored *attached to* the
settings that produced it. A line that recorded the numbers but lost the
criterion would be worse than no line at all: P/R under centre matching and
under IoU matching differ by tens of points on this data, so the two would be
read against each other and the comparison would look reasonable.

These tests pin three things: appending never loses an earlier result, every
setting that moves the numbers survives the round trip, and the default
criterion is the centre rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from src.eval.metrics import CENTER, IOU, MatchCriterion, evaluate
from src.eval.results import SCHEMA, EvalSettings, append_result, load_results
from src.evaluate import build_parser


def box(cx: float, cy: float, size: float) -> list[float]:
    """A square xyxy box of side `size` centred on (cx, cy)."""
    half = size / 2
    return [cx - half, cy - half, cx + half, cy + half]


@pytest.fixture
def metrics(make_frame):
    """A scored result from one synthetic frame, matched by the centre rule."""
    frames = [make_frame(gt=[box(50, 50, 12)],
                         preds=[(box(51, 51, 12), 0.9, 0)])]
    return evaluate(frames, MatchCriterion(CENTER, 1.0))


def settings(**overrides) -> EvalSettings:
    """Settings for a run, with only the fields under test spelled out."""
    base = dict(pred="runs/exp/detections.jsonl", labels="labels/test",
                match=CENTER, match_value=1.0)
    return EvalSettings(**{**base, **overrides})


class TestAppend:
    """Appending is what makes re-scoring safe."""

    def test_creates_missing_parent_directory(self, tmp_path, metrics):
        path = tmp_path / "nested" / "results.jsonl"
        append_result(path, settings(), metrics)
        assert path.exists()

    def test_second_result_does_not_replace_the_first(self, tmp_path, metrics):
        # The bug this guards: writing rather than appending, which would leave
        # the log looking healthy while holding only the most recent scoring.
        path = tmp_path / "results.jsonl"
        append_result(path, settings(match=IOU, match_value=0.5), metrics)
        append_result(path, settings(match=IOU, match_value=0.4), metrics)

        recorded = [r["settings"]["match_value"] for r in load_results(path)]
        assert recorded == [0.5, 0.4]

    def test_returns_the_running_total(self, tmp_path, metrics):
        path = tmp_path / "results.jsonl"
        assert append_result(path, settings(), metrics) == 1
        assert append_result(path, settings(), metrics) == 2

    def test_one_json_object_per_line(self, tmp_path, metrics):
        path = tmp_path / "results.jsonl"
        append_result(path, settings(), metrics)
        append_result(path, settings(), metrics)

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert all(json.loads(line)["schema"] == SCHEMA for line in lines)


class TestRecordContents:
    """What a line has to carry to still mean something months later."""

    def test_records_the_criterion_that_scored_it(self, tmp_path, metrics):
        path = tmp_path / "results.jsonl"
        append_result(path, settings(), metrics)
        assert load_results(path)[0]["criterion"] == metrics.criterion

    def test_records_the_full_metric_schema(self, tmp_path, metrics):
        path = tmp_path / "results.jsonl"
        append_result(path, settings(), metrics)

        recorded = load_results(path)[0]["metrics"]
        assert recorded["recall"] == metrics.recall
        assert recorded["ap50"] == metrics.ap50
        assert recorded["by_size"]

    def test_records_every_setting_that_moves_the_numbers(self, tmp_path, metrics):
        path = tmp_path / "results.jsonl"
        append_result(path, settings(conditions="conditions.json",
                                     frame_size=[1920, 1080]), metrics)

        recorded = load_results(path)[0]["settings"]
        assert recorded["pred"] == "runs/exp/detections.jsonl"
        assert recorded["labels"] == "labels/test"
        assert recorded["match"] == CENTER
        assert recorded["match_value"] == 1.0
        assert recorded["conditions"] == "conditions.json"
        assert recorded["frame_size"] == [1920, 1080]

    def test_timestamps_are_written(self, tmp_path, metrics):
        path = tmp_path / "results.jsonl"
        append_result(path, settings(), metrics)
        assert load_results(path)[0]["time"]

    @pytest.mark.parametrize("match, comparable", [(IOU, True), (CENTER, False)])
    def test_only_iou_claims_comparability_to_published_map(self, match, comparable):
        assert settings(match=match).is_comparable_to_published_map is comparable


class TestLoad:
    """Reading the log back."""

    def test_missing_file_is_empty_not_an_error(self, tmp_path):
        assert load_results(tmp_path / "absent.jsonl") == []

    def test_blank_lines_are_skipped(self, tmp_path, metrics):
        path = tmp_path / "results.jsonl"
        append_result(path, settings(), metrics)
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n")
        assert len(load_results(path)) == 1

    def test_a_corrupt_line_is_reported_with_its_number(self, tmp_path, metrics):
        # Silently dropping it would quietly shorten the history the log exists
        # to preserve.
        path = tmp_path / "results.jsonl"
        append_result(path, settings(), metrics)
        with path.open("a", encoding="utf-8") as handle:
            handle.write("{not json}\n")

        with pytest.raises(ValueError, match="results.jsonl:2"):
            load_results(path)


class TestCliDefaults:
    """The command line the user actually types."""

    def parse(self, *extra: str) -> argparse.Namespace:
        """Arguments as the CLI would produce them."""
        return build_parser().parse_args(
            ["--pred", "p.jsonl", "--labels", "labels", *extra])

    def test_centre_matching_is_the_default(self):
        assert self.parse().match == CENTER

    def test_iou_is_still_reachable(self):
        assert self.parse("--match", "iou").match == IOU

    def test_default_tolerance_is_one_target_size(self):
        assert self.parse().match_tol == 1.0

    def test_save_defaults_to_off(self):
        assert self.parse().save is None

    def test_save_takes_a_path(self):
        assert self.parse("--save", "runs/x/results.jsonl").save == Path(
            "runs/x/results.jsonl")

    def test_settings_carry_the_criterion_that_ran(self):
        # Not re-derived from args: --iou keeps its default of 0.5 even when the
        # centre rule is what scored the run, so reading the wrong field would
        # record a threshold nothing used.
        args = self.parse("--match-tol", "2.0")
        recorded = EvalSettings.from_args(args, MatchCriterion(CENTER, 2.0))
        assert (recorded.match, recorded.match_value) == (CENTER, 2.0)

    def test_settings_survive_optional_arguments_being_absent(self):
        recorded = EvalSettings.from_args(self.parse(), MatchCriterion(CENTER, 1.0))
        assert recorded.conditions is None
        assert recorded.frame_size is None
