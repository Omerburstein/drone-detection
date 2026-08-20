"""Refuse to end a turn that left the project record incomplete.

Wired into `.claude/settings.json` as two hooks sharing one script:

  record  PostToolUse on Write|Edit -- notes which repo file the session touched
  check   Stop                      -- blocks the turn while any of those files
                                       is uncommitted or unpushed

**Why it tracks files instead of asking "is the tree dirty".** Several Claude
sessions run against this repo at once, and each sees the others' work in
progress as uncommitted changes. A bare `git status` check would fire on every
turn of every session for edits it did not make, and a hook that cries wolf gets
switched off. Scoping to the paths this session actually wrote makes the block
mean exactly one thing: *you* left something unrecorded.

The record is the point. `docs/todo.md` and `docs/experiments.md` are the
project's memory -- an unrecorded mission gets re-asked and re-run, which is
what happened to M2 on 2026-08-16. Tests and docs cannot be verified
mechanically, so they ride along in the message rather than in a condition.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

# Bumped only when the state-file format changes; old sessions' files are then
# simply never read again rather than misparsed.
STATE_DIR = Path(tempfile.gettempdir()) / "claude-dod-v1"

CHECKLIST = """\
Before ending this turn, finish the record (CLAUDE.md, "A mission is not
finished until the record is"):

  1. Tests    -- cover changed behaviour under src/ and run them:
                 py -3.13 -m pytest -m "not slow"
                 If tests genuinely do not apply, say so in one line.
  2. Docs     -- if a script's parameters changed, update its docs/ reference
                 in the SAME commit. The user reads those instead of the source.
  3. Record   -- move the finished task to Done in docs/todo.md with today's
                 date. If the work produced numbers, add the run to
                 docs/experiments.md.
  4. Commit   -- commit and push to origin main. This is standing approval;
                 do not ask for it.

Stage only the paths this task touched -- another session may have unrelated
work in this tree. If the work is genuinely incomplete, say what is left rather
than committing a half-change."""


def git(*args: str, cwd: Path | None = None) -> str:
    """Run a git command and return stdout, or "" if git fails for any reason.

    Silence is deliberate: a hook that raises on a detached HEAD or a missing
    remote would block every turn in a repo state that is nobody's fault.
    """
    try:
        done = subprocess.run(("git", *args), cwd=cwd, capture_output=True,
                              text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def repo_root() -> Path | None:
    """The working tree this script lives in, from its own location.

    Derived from `__file__` rather than `git rev-parse` on purpose: `record`
    runs after every single Write/Edit, and a git subprocess measures ~1 s on
    this machine, which is real latency for a bookkeeping step. The script's
    path is `<root>/.claude/hooks/`, so the root is two levels up; the `.git`
    check is what makes that assumption fail loudly-by-going-quiet if the file
    is ever moved.
    """
    root = Path(__file__).resolve().parents[2]
    return root if (root / ".git").exists() else None


def state_file(payload: dict) -> Path:
    """Where this session's touched-path list lives.

    Keyed by session id so parallel sessions never read each other's list. Falls
    back to a shared name only when the id is absent, which costs a little
    over-reporting rather than silently disabling the hook.
    """
    session = str(payload.get("session_id") or "unknown")
    safe = "".join(c for c in session if c.isalnum() or c in "-_")[:64] or "unknown"
    return STATE_DIR / f"{safe}.txt"


def edited_path(payload: dict) -> str | None:
    """The file a Write/Edit call wrote, as reported by the tool."""
    response = payload.get("tool_response") or {}
    tool_input = payload.get("tool_input") or {}
    for source in (response, tool_input):
        if isinstance(source, dict):
            value = source.get("filePath") or source.get("file_path")
            if value:
                return str(value)
    return None


def record(payload: dict) -> int:
    """Note one edited path, if it is a file inside this repository.

    Paths outside the tree -- scratchpad scribbles, files in another checkout --
    are dropped here rather than at check time, so the list only ever holds
    things a commit could actually contain.
    """
    root, raw = repo_root(), edited_path(payload)
    if root is None or raw is None:
        return 0
    try:
        resolved = Path(raw).resolve()
        relative = resolved.relative_to(root.resolve())
    except (OSError, ValueError):
        return 0

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = state_file(payload)
    known = set(path.read_text(encoding="utf-8").splitlines()) if path.exists() else set()
    known.add(relative.as_posix())
    path.write_text("\n".join(sorted(known)), encoding="utf-8")
    return 0


def dirty_paths(root: Path, tracked: list[str]) -> list[str]:
    """Which of `tracked` git still reports as modified, staged or untracked.

    Gitignored paths -- anything under /data/, /weights/, /runs/ -- report clean
    and drop out here, which is right: they are never meant to be committed.
    """
    status = git("status", "--porcelain", "--", *tracked, cwd=root)
    dirty = []
    for line in status.splitlines():
        # Split on the status code rather than slicing a fixed offset: the
        # leading space of " M path" does not survive every way this output is
        # read back, and an off-by-one here silently mangles every filename.
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        name = parts[1].strip()
        # Rename entries read "old -> new"; the new name is what needs
        # committing. Split before unquoting: a renamed path containing spaces
        # arrives as '"old name" -> "new name"', and stripping the outer quotes
        # first leaves the inner ones stranded on the split halves.
        if " -> " in name:
            name = name.split(" -> ", 1)[1]
        name = name.strip().strip('"')
        if name:
            dirty.append(name)
    return sorted(set(dirty))


def unpushed_count(root: Path) -> int:
    """Commits on HEAD that the upstream branch does not have.

    Zero when there is no upstream at all: a branch that was never meant to be
    pushed should not trap the session.
    """
    count = git("rev-list", "--count", "@{upstream}..HEAD", cwd=root)
    return int(count) if count.isdigit() else 0


def check(payload: dict) -> int:
    """Block the stop while this session's edits are not yet in the remote."""
    # Claude has already been told once this turn. Blocking again on the same
    # stop would loop the session instead of ending it.
    if payload.get("stop_hook_active"):
        return 0

    root = repo_root()
    path = state_file(payload)
    if root is None or not path.exists():
        return 0

    tracked = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not tracked:
        return 0

    dirty = dirty_paths(root, tracked)
    pending = unpushed_count(root)
    if not dirty and not pending:
        return 0

    problem = []
    if dirty:
        problem.append("Uncommitted, and edited by this session:\n  "
                       + "\n  ".join(dirty))
    if pending:
        problem.append(f"Unpushed commits on this branch: {pending}")

    print(json.dumps({
        "decision": "block",
        "reason": "\n\n".join(problem) + "\n\n" + CHECKLIST,
    }))
    return 0


def main() -> int:
    """Dispatch on argv[1]; unknown modes and bad input are no-ops.

    A hook that fails loudly on malformed stdin would break the session for a
    reason unrelated to the work, so every failure path here exits 0.
    """
    try:
        # A UTF-8 BOM survives some Windows shells' pipes and is not valid JSON.
        raw = sys.stdin.read().lstrip("﻿").strip()
        payload = json.loads(raw or "{}")
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return 0
    if not isinstance(payload, dict):
        return 0

    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    return {"record": record, "check": check}.get(mode, lambda _: 0)(payload)


if __name__ == "__main__":
    sys.exit(main())
