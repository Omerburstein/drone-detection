"""Unit tests for `src.eval.tables` and `src.eval.vocabulary`.

These two modules exist to stop three CLIs maintaining their own copies of the
same plumbing. The tests here are mostly about the seams where the copies used
to differ, because that is where a consolidation goes wrong.

The `parse_dump_spec` tests are the important ones. Two conventions are live and
both are documented: `cross_eval` and `alarm_eval` accept a bare path, while
`plot_eval` requires an explicit label because its label ends up in a figure
legend that outlives the command. Collapsing to one function must not quietly
impose either convention on the other -- that would break a documented CLI.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.eval import vocabulary
from src.eval.tables import parse_dump_spec, parse_edges, rounded, write_rows


class TestRounded:

    def test_rounds_to_four_places(self):
        assert rounded(0.123456) == 0.1235

    def test_nan_becomes_none(self):
        """None writes an empty cell. A bin with no objects has no ratio, and a
        0.0 there would read as total failure -- the opposite of unmeasured."""
        assert rounded(float("nan")) is None

    def test_passes_zero_through_as_a_real_number(self):
        """The distinction the empty cell exists for: a measured zero is not a
        missing measurement."""
        assert rounded(0.0) == 0.0

    def test_accepts_numpy_floats(self):
        np = pytest.importorskip("numpy")
        assert rounded(np.float64(0.5)) == 0.5
        assert rounded(np.float64("nan")) is None


class TestWriteRows:

    def test_writes_a_header_and_the_rows(self, tmp_path):
        path = tmp_path / "out.csv"
        assert write_rows(path, [{"a": 1, "b": 2}, {"a": 3, "b": 4}]) == 2
        with path.open(encoding="utf-8", newline="") as handle:
            assert list(csv.DictReader(handle)) == [{"a": "1", "b": "2"},
                                                    {"a": "3", "b": "4"}]

    def test_columns_follow_the_first_rows_key_order(self, tmp_path):
        """Load-bearing and subtle: the caller controls the CSV layout purely by
        the order it builds its dicts in. Three copies of this was three chances
        for one CLI's columns to drift from its siblings."""
        path = tmp_path / "out.csv"
        write_rows(path, [{"z": 1, "a": 2, "m": 3}])
        assert path.read_text(encoding="utf-8").splitlines()[0] == "z,a,m"

    def test_creates_the_parent_directory(self, tmp_path):
        path = tmp_path / "made" / "up" / "out.csv"
        write_rows(path, [{"a": 1}])
        assert path.is_file()

    def test_refuses_to_write_nothing(self, tmp_path):
        """Previously an IndexError from `rows[0]` several frames away."""
        with pytest.raises(ValueError, match="no rows"):
            write_rows(tmp_path / "out.csv", [])


class TestParseDumpSpec:

    def test_splits_label_from_path(self):
        assert parse_dump_spec("centre@1x=runs/x/m.csv") == ("centre@1x",
                                                            Path("runs/x/m.csv"))

    def test_strips_whitespace_around_both_halves(self):
        assert parse_dump_spec(" c = runs/x/m.csv ") == ("c", Path("runs/x/m.csv"))

    def test_keeps_later_equals_signs_in_the_path(self):
        label, path = parse_dump_spec("a=dir=odd/m.csv")
        assert label == "a" and path == Path("dir=odd/m.csv")

    def test_a_bare_path_is_named_after_its_run_directory(self):
        assert parse_dump_spec("runs/exp004/m.csv") == ("exp004",
                                                        Path("runs/exp004/m.csv"))

    def test_a_bare_bare_path_falls_back_to_itself(self):
        assert parse_dump_spec("m.csv") == ("m.csv", Path("m.csv"))

    def test_require_label_rejects_a_bare_path(self):
        """plot_eval's convention: the legend outlives the command."""
        with pytest.raises(ValueError, match="expected LABEL=PATH"):
            parse_dump_spec("runs/exp004/m.csv", require_label=True)

    def test_require_label_still_accepts_the_labelled_form(self):
        assert parse_dump_spec("c=m.csv", require_label=True) == ("c", Path("m.csv"))


class TestParseEdges:

    def test_parses_a_ladder(self):
        assert parse_edges("0,8,16") == (0.0, 8.0, 16.0)

    def test_inf_closes_the_top_bin(self):
        assert parse_edges("0,8,inf")[-1] == float("inf")

    def test_tolerates_spaces_and_a_trailing_comma(self):
        assert parse_edges("0, 8, 16,") == (0.0, 8.0, 16.0)

    def test_rejects_non_numbers(self):
        with pytest.raises(SystemExit, match="comma-separated numbers"):
            parse_edges("0,eight,16")

    def test_rejects_an_unsorted_ladder(self):
        """`0,16,8` is a typo. Sorting it silently would produce a table whose
        bands are not the ones that were asked for."""
        with pytest.raises(SystemExit, match="increasing order"):
            parse_edges("0,16,8")

    def test_rejects_a_single_edge(self):
        with pytest.raises(SystemExit, match="at least two edges"):
            parse_edges("8")


class TestVocabulary:

    def test_the_outcome_labels_are_what_the_dump_writes(self):
        assert (vocabulary.TP, vocabulary.FP, vocabulary.FN) == ("tp", "fp", "fn")

    def test_records_and_crosscut_share_one_definition(self):
        """The reason this module exists: `records` writes the `outcome` column
        and `crosscut` reads it. Separate copies meant a rename in one would not
        fail anywhere -- the cross-cut would just report empty cells, which reads
        as "the detector found nothing" rather than as an error."""
        from src.eval import crosscut, records
        assert records.TP is crosscut.TP is vocabulary.TP
        assert records.FP is crosscut.FP is vocabulary.FP
        assert records.FN is crosscut.FN is vocabulary.FN
