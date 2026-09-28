"""Unit tests for `src.live_detect`'s reporting.

`live_detect` is the only detection CLI with no test at all, and its
`print_branches` is byte-identical to `glad_detect`'s -- its own docstring says
it reports "as `src.glad_detect` reports". That duplication is about to be
collapsed into one shared function, so these tests pin what both currently print.

The branch mix is not decoration. It is the paper's ablation measured on our own
run: how much of the recall is appearance and how much is motion. If the shared
version drifted in rounding or column width, the ledger's tables would stop
lining up with what the CLI prints, and nothing else would catch it.

Importing this module pulls in torch through the GLAD pipeline, which is why the
tests here stay on the reporting surface rather than the run loop -- the loop
needs a feed and a checkpoint, neither of which belongs in a unit test.
"""

from __future__ import annotations

from collections import Counter

import pytest

from src import glad_detect, live_detect


@pytest.fixture(params=["live_detect", "glad_detect"])
def print_branches(request):
    """Both CLIs' copies, so a divergence fails rather than passing on one."""
    return {"live_detect": live_detect.print_branches,
            "glad_detect": glad_detect.print_branches}[request.param]


class TestPrintBranches:

    def test_reports_each_branch_with_its_share(self, print_branches, capsys):
        print_branches(Counter({"global mod": 3, "track": 1}))
        out = capsys.readouterr().out
        assert "Branch that handled each frame:" in out
        assert "global mod" in out and "75.0%" in out
        assert "track" in out and "25.0%" in out

    def test_orders_by_count_descending(self, print_branches, capsys):
        """Most-common first: the branch carrying the run is the headline."""
        print_branches(Counter({"rare": 1, "common": 9}))
        lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
        assert lines[1].strip().startswith("common")
        assert lines[2].strip().startswith("rare")

    def test_shares_sum_to_one_hundred(self, print_branches, capsys):
        print_branches(Counter({"a": 1, "b": 1, "c": 2}))
        out = capsys.readouterr().out
        assert "25.0%" in out and "50.0%" in out

    def test_an_empty_counter_does_not_divide_by_zero(self, print_branches, capsys):
        """A run that processed nothing still has to print a table, not crash --
        `--benchmark-only` and an immediately-closed window both land here."""
        print_branches(Counter())
        assert "Branch that handled each frame:" in capsys.readouterr().out

    def test_the_two_clis_print_identical_tables(self, capsys):
        """The reason the shared version is safe: they already agree exactly."""
        branches = Counter({"global mod": 7, "track": 3, "global miss": 2})

        live_detect.print_branches(branches)
        from_live = capsys.readouterr().out
        glad_detect.print_branches(branches)
        from_glad = capsys.readouterr().out

        assert from_live == from_glad


class TestGladConfidence:

    def test_both_clis_agree_on_the_placeholder_score(self):
        """GLAD emits no score, so both stamp the same constant into the JSONL.
        If these drifted, two runs of the same pipeline would record different
        confidences and `--conf` filtering would mean different things."""
        assert live_detect.GLAD_CONFIDENCE == glad_detect.GLAD_CONFIDENCE == 1.0
