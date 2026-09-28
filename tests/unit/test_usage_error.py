"""The contract between a library raising `UsageError` and a CLI exiting on it.

Library modules used to call `sys.exit` on bad input. That killed the
interpreter for anyone calling the same function from a notebook or a test --
the use this project's guidance explicitly promises. They now raise
`UsageError`, and each CLI's `main()` turns it back into `sys.exit(str(exc))`.

Both halves are tested together on purpose. Converting one without the other is
the failure mode worth guarding: a library that raises into a CLI with no catch
gives the user a traceback instead of a one-line message, and nothing else in
the suite would notice. So each case below asserts the *same* message comes out
of the function as an exception and out of the CLI as an exit.
"""

from __future__ import annotations

import json

import pytest

from src import alarm_eval, cross_eval
from src.errors import UsageError
from src.algo.sampling import build_schedule
from src.data.sources import resolve_sources
from src.eval.conditions import load_conditions
from src.eval.labels import load_label_file
from src.eval.tables import parse_edges


class TestLibrariesRaiseRatherThanExit:
    """Each of these is callable from a notebook now without ending the session."""

    def test_conditions_on_a_missing_file(self, tmp_path):
        with pytest.raises(UsageError, match="No conditions file at"):
            load_conditions(tmp_path / "nope.json")

    def test_conditions_on_the_wrong_shape(self, tmp_path):
        path = tmp_path / "conditions.json"
        path.write_text(json.dumps({"nonsense": {}}), encoding="utf-8")
        with pytest.raises(UsageError, match="expected an 'axes' object"):
            load_conditions(path)

    def test_labels_on_a_truncated_line(self, tmp_path):
        path = tmp_path / "frame.txt"
        path.write_text("0 0.5 0.5\n", encoding="utf-8")
        with pytest.raises(UsageError, match="expected"):
            load_label_file(path)

    def test_edges_on_a_non_number(self):
        with pytest.raises(UsageError, match="comma-separated numbers"):
            parse_edges("0,eight,16")

    def test_sources_on_an_unrecognised_suffix(self, tmp_path):
        path = tmp_path / "notes.pdf"
        path.write_bytes(b"")
        with pytest.raises(UsageError, match="Unrecognised source type"):
            resolve_sources(path)

    def test_sampling_on_an_unknown_mode(self):
        with pytest.raises(UsageError, match="Unknown --sample"):
            build_schedule("sideways", 2, 2, 30)

    def test_sampling_on_a_burst_too_short_to_difference(self):
        """Two frames is the floor: the motion branches difference the current
        frame against the previous one, so a burst of one has nothing to
        difference against."""
        with pytest.raises(UsageError, match="at least 2"):
            build_schedule("burst", 2, 1, 30)

    def test_the_error_is_not_a_systemexit(self, tmp_path):
        """SystemExit inherits from BaseException, so a bare `except Exception`
        in a caller would not have caught the old behaviour -- the process just
        ended. This is the whole point of the change."""
        with pytest.raises(UsageError) as caught:
            load_conditions(tmp_path / "nope.json")
        assert not isinstance(caught.value, SystemExit)


class TestCLIsStillExitWithTheSameMessage:
    """The user-facing half: a one-line message and a non-zero exit, no traceback."""

    def _dump(self, tmp_path):
        path = tmp_path / "matches.csv"
        path.write_text("key,outcome,gt_size,pred_size,scene_category\n"
                        "a_0001,tp,6.0,6.2,complex\n", encoding="utf-8")
        return path

    def test_cross_eval_reports_a_bad_edge_list(self, tmp_path, capsys):
        """`--edges` is parsed inside `parse_args`, so this only works if the
        catch wraps argument parsing and not just the body after it."""
        with pytest.raises(SystemExit) as caught:
            cross_eval.main(["--dump", str(self._dump(tmp_path)), "--edges", "0,16,8"])
        assert "increasing order" in str(caught.value)
        assert "Traceback" not in capsys.readouterr().err

    def test_alarm_eval_reports_a_bad_edge_list(self, tmp_path):
        with pytest.raises(SystemExit) as caught:
            alarm_eval.main(["--dump", str(self._dump(tmp_path)), "--edges", "5"])
        assert "at least two edges" in str(caught.value)

    def test_cross_eval_reports_a_missing_dump(self, tmp_path):
        with pytest.raises(SystemExit) as caught:
            cross_eval.main(["--dump", str(tmp_path / "nope.csv")])
        assert "No dump at" in str(caught.value)

    @pytest.mark.parametrize("cli", [cross_eval, alarm_eval], ids=["cross", "alarm"])
    def test_the_message_matches_what_the_library_raised(self, cli, tmp_path):
        """Not merely 'it exits' -- it exits saying exactly what the function
        would have raised, so the text the user reads did not change."""
        try:
            parse_edges("0,16,8")
        except UsageError as exc:
            expected = str(exc)

        with pytest.raises(SystemExit) as caught:
            cli.main(["--dump", str(self._dump(tmp_path)), "--edges", "0,16,8"])
        assert str(caught.value) == expected
