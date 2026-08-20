"""Tests for the definition-of-done hook.

The hook is not part of `src`, but it decides whether a turn is allowed to end,
so a false positive nags on every turn and a false negative silently stops
enforcing anything. Both failure modes are quiet, which is exactly why they need
a test rather than a look.

Nothing here shells out to git: `git()` is stubbed with canned porcelain, so the
parsing is pinned without depending on the state of the working tree the suite
happens to run in.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

HOOK = (Path(__file__).resolve().parents[2] / ".claude" / "hooks"
        / "definition_of_done.py")


def _load():
    """Import the hook by path -- `.claude/hooks/` is not an importable package."""
    spec = importlib.util.spec_from_file_location("definition_of_done", HOOK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def hook(tmp_path, monkeypatch):
    """The hook module, with its state directory redirected into tmp_path."""
    module = _load()
    monkeypatch.setattr(module, "STATE_DIR", tmp_path / "state")
    return module


@pytest.fixture
def repo(tmp_path, monkeypatch, hook):
    """A fake repository root the hook will accept as its own."""
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    monkeypatch.setattr(hook, "repo_root", lambda: root)
    return root


def _canned(hook, monkeypatch, status: str = "", ahead: str = "0"):
    """Stub `git()` so the hook sees the given porcelain and ahead-count."""
    def fake(*args, **_kwargs):
        return status if args and args[0] == "status" else ahead
    monkeypatch.setattr(hook, "git", fake)


# --- porcelain parsing ------------------------------------------------------
# The bug this pins: `git()` strips its own output, so the leading space of
# " M path" is gone by the time it is parsed. A fixed 3-char slice ate the first
# character of every modified filename.

@pytest.mark.parametrize("line, expected", [
    ("M src/eval/metrics.py", "src/eval/metrics.py"),      # leading space stripped
    (" M src/eval/metrics.py", "src/eval/metrics.py"),     # leading space intact
    ("?? docs/new.md", "docs/new.md"),                     # untracked
    ("A  src/added.py", "src/added.py"),                   # staged
    ("MM src/both.py", "src/both.py"),                     # staged and dirty
    ('R  "old name.py" -> "new name.py"', "new name.py"),  # rename: new name wins
    ('?? "docs/has space.md"', "docs/has space.md"),       # quoted path unwrapped
])
def test_dirty_paths_parses_porcelain(hook, repo, monkeypatch, line, expected):
    _canned(hook, monkeypatch, status=line)
    assert hook.dirty_paths(repo, ["ignored"]) == [expected]


def test_dirty_paths_dedupes_and_sorts(hook, repo, monkeypatch):
    _canned(hook, monkeypatch, status="M b.py\n M b.py\n?? a.py")
    assert hook.dirty_paths(repo, ["x"]) == ["a.py", "b.py"]


def test_dirty_paths_empty_when_clean(hook, repo, monkeypatch):
    _canned(hook, monkeypatch, status="")
    assert hook.dirty_paths(repo, ["x"]) == []


# --- unpushed count ---------------------------------------------------------

@pytest.mark.parametrize("output, expected", [
    ("3", 3),
    ("0", 0),
    ("", 0),           # git failed, e.g. no upstream configured
    ("fatal: junk", 0),  # never crash on unexpected output
])
def test_unpushed_count(hook, repo, monkeypatch, output, expected):
    monkeypatch.setattr(hook, "git", lambda *_a, **_k: output)
    assert hook.unpushed_count(repo) == expected


# --- which path an edit reported --------------------------------------------

def test_edited_path_prefers_tool_response(hook):
    payload = {"tool_response": {"filePath": "from/response.py"},
               "tool_input": {"file_path": "from/input.py"}}
    assert hook.edited_path(payload) == "from/response.py"


def test_edited_path_falls_back_to_tool_input(hook):
    assert hook.edited_path({"tool_input": {"file_path": "a.py"}}) == "a.py"


def test_edited_path_none_when_absent(hook):
    assert hook.edited_path({"tool_response": {}}) is None


# --- recording --------------------------------------------------------------

def test_record_stores_repo_relative_path(hook, repo):
    target = repo / "src" / "eval" / "metrics.py"
    target.parent.mkdir(parents=True)
    target.touch()

    hook.record({"session_id": "s1", "tool_input": {"file_path": str(target)}})

    assert hook.state_file({"session_id": "s1"}).read_text(
        encoding="utf-8").splitlines() == ["src/eval/metrics.py"]


@pytest.mark.usefixtures("repo")
def test_record_ignores_paths_outside_the_repo(hook, tmp_path):
    outside = tmp_path / "elsewhere" / "note.txt"
    outside.parent.mkdir(parents=True)
    outside.touch()

    hook.record({"session_id": "s1", "tool_input": {"file_path": str(outside)}})

    assert not hook.state_file({"session_id": "s1"}).exists()


def test_record_accumulates_and_dedupes(hook, repo):
    for name in ("a.py", "b.py", "a.py"):
        hook.record({"session_id": "s1",
                     "tool_input": {"file_path": str(repo / name)}})

    assert hook.state_file({"session_id": "s1"}).read_text(
        encoding="utf-8").splitlines() == ["a.py", "b.py"]


def test_record_keeps_sessions_apart(hook, repo):
    hook.record({"session_id": "one", "tool_input": {"file_path": str(repo / "a.py")}})
    hook.record({"session_id": "two", "tool_input": {"file_path": str(repo / "b.py")}})

    assert hook.state_file({"session_id": "one"}).read_text(encoding="utf-8") == "a.py"
    assert hook.state_file({"session_id": "two"}).read_text(encoding="utf-8") == "b.py"


def test_state_file_survives_a_hostile_session_id(hook):
    """A session id becomes a filename, so path separators must not escape."""
    path = hook.state_file({"session_id": "../../etc/passwd"})
    assert path.parent == hook.STATE_DIR
    assert "/" not in path.name and "\\" not in path.name


# --- the stop check ---------------------------------------------------------

def _record_one(hook, repo, name="src/thing.py", session="s1"):
    hook.record({"session_id": session, "tool_input": {"file_path": str(repo / name)}})


def test_check_blocks_on_an_uncommitted_edit(hook, repo, monkeypatch, capsys):
    _record_one(hook, repo)
    _canned(hook, monkeypatch, status="M src/thing.py", ahead="0")

    assert hook.check({"session_id": "s1"}) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["decision"] == "block"
    assert "src/thing.py" in payload["reason"]


def test_check_blocks_on_unpushed_commits(hook, repo, monkeypatch, capsys):
    _record_one(hook, repo)
    _canned(hook, monkeypatch, status="", ahead="2")

    hook.check({"session_id": "s1"})
    payload = json.loads(capsys.readouterr().out)
    assert payload["decision"] == "block"
    assert "Unpushed commits on this branch: 2" in payload["reason"]


def test_check_passes_when_committed_and_pushed(hook, repo, monkeypatch, capsys):
    _record_one(hook, repo)
    _canned(hook, monkeypatch, status="", ahead="0")

    assert hook.check({"session_id": "s1"}) == 0
    assert capsys.readouterr().out == ""


def test_check_never_blocks_twice_in_a_row(hook, repo, monkeypatch, capsys):
    """`stop_hook_active` is the loop guard: one nudge per turn, then let go."""
    _record_one(hook, repo)
    _canned(hook, monkeypatch, status="M src/thing.py", ahead="9")

    assert hook.check({"session_id": "s1", "stop_hook_active": True}) == 0
    assert capsys.readouterr().out == ""


@pytest.mark.usefixtures("repo")
def test_check_silent_when_the_session_edited_nothing(hook, monkeypatch, capsys):
    """A question-answering turn must not be blocked by another session's work."""
    _canned(hook, monkeypatch, status="M someone/elses/file.py", ahead="4")

    assert hook.check({"session_id": "never-edited"}) == 0
    assert capsys.readouterr().out == ""


def test_check_message_carries_the_full_checklist(hook, repo, monkeypatch, capsys):
    _record_one(hook, repo)
    _canned(hook, monkeypatch, status="M src/thing.py")

    hook.check({"session_id": "s1"})
    reason = json.loads(capsys.readouterr().out)["reason"]
    for expected in ("pytest", "docs/", "docs/todo.md", "push"):
        assert expected in reason


# --- entry point ------------------------------------------------------------

@pytest.mark.parametrize("stdin", [
    "﻿{}",       # a UTF-8 BOM survives some Windows shells' pipes
    "not json",
    "",
    "[1, 2, 3]",      # valid JSON, wrong shape
])
def test_main_survives_bad_stdin(hook, monkeypatch, stdin, capsys):
    monkeypatch.setattr("sys.stdin", type("S", (), {"read": staticmethod(lambda: stdin)}))
    monkeypatch.setattr("sys.argv", ["hook", "check"])

    assert hook.main() == 0
    assert capsys.readouterr().out == ""


def test_main_ignores_an_unknown_mode(hook, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", type("S", (), {"read": staticmethod(lambda: "{}")}))
    monkeypatch.setattr("sys.argv", ["hook", "bogus"])

    assert hook.main() == 0
    assert capsys.readouterr().out == ""


def test_git_returns_empty_when_the_command_fails(hook):
    """A missing upstream or detached HEAD must not raise into the session."""
    # `rev-parse` is no good here: it echoes unrecognised arguments and exits 0.
    assert hook.git("rev-parse", "--verify", "no/such/ref/exists") == ""
    assert hook.git("definitely-not-a-git-subcommand") == ""
