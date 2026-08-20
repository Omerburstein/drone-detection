# Working with Claude Code on this project

Derived on 2026-08-19 from the actual record: 16 sessions, 105 typed prompts,
43 commits, 2026-08-13 to 2026-08-19. Every rule below cites the thing that
prompted it, so it can be argued with rather than obeyed.

This is guidance for the user. The rules Claude must follow are in `CLAUDE.md`.

## Where the state lives

Three files are the project's memory. Nothing in a session transcript is durable;
these are.

| File | Holds |
|---|---|
| `docs/todo.md` | The mission plan, M1–M7, and every captured task |
| `docs/experiments.md` | The ledger: every run, its settings, its numbers |
| `CLAUDE.md` | The rules and traps that must survive into the next session |

If an answer matters next week, it belongs in one of these. **Asking the same
question twice is a docs bug, not a memory lapse** — *"are the results before or
after resizing?"* was asked in two different sessions five hours apart, and the
fix is to write the answer into the ledger, not to remember harder.

## Starting a task

**Name the artifact, not the nickname.** `M4a` cost an interrupt and a
clarification; `M4a in docs/todo.md` would not have. Mission ids, run ids and
file paths are unambiguous — nicknames are not.

**One task per prompt; keep blocking questions separate.** The habitual shape
here is a numbered list of four unrelated items. They all get answered, but the
answer to #1 often should have changed #4. Ask the question that gates the work,
then issue the work.

**Use plan mode for anything expensive.** Shift+Tab twice. A 27 GB download, a
rented-GPU run, or a refactor across `src/` should be a plan you approve, not a
diff you discover. This is already the working style — *"write the following
missions, and I will approve them one at a time"* — plan mode just makes it a
mechanism instead of a habit.

## Sessions

**One session per workstream, not per question.** On 2026-08-18 five sessions
were live between 05:38 and 06:10. Sessions cannot see each other, so the same
M4b question was typed into two of them, and the same rescaling question got
asked twice. Dataset acquisition in one session and model/eval in another maps
onto the same boundary the two agents split on.

Parallel sessions are correct when the topics are genuinely disjoint and one is
long-running. They are expensive when the topics touch, because neither session
knows what the other established.

**Closing a session costs nothing** once `todo.md` and the ledger are current.
That is the point of keeping them current.

## Delegation

**Skills should fire without being typed.** They match on their `description`
frontmatter, so that description has to contain the words actually used. `/eval`
sat unused through four requests for exactly what it does, because it advertised
*"evaluate", "score", "what's the mAP"* while the requests said *"a table of my
model's performance"* and *"a graph of precision per number of pixels"*. Fixed on
2026-08-19. **If a skill does not fire when it obviously should, the description
is the bug** — say so and it gets widened.

**Agents need naming more often than skills do.** A subagent starts cold: it has
none of the current session's context. That makes delegation a real trade, not a
free win.

- Worth it: read-heavy fan-out with a small answer — survey a dozen dataset
  pages, sweep the docs for every mention of X, audit labels across a split.
- Not worth it: anything depending on what was just established in this session.

Zero subagents were launched in the first 105 prompts. One clause fixes that:
*"use the dataset-agent to…"*.

## Long-running jobs

Two failures already happened and both are avoidable.

- A background run left **no completion record** because the session ended while
  it was running.
- A download died because **the machine went idle**.

So: background long jobs rather than blocking on them, insist the command is
resumable (`wget -c --tries=0 --waitretry=15`, as the ARD100 entry specifies),
and keep the laptop awake for anything measured in hours.

## Reducing friction

`.claude/settings.json` now exists but carries **hooks only, no permission
allowlist** — so roughly 550 shell calls still each raise a prompt.
`/fewer-permission-prompts` reads these same transcripts and writes the allowlist
into the `permissions.allow` array of that same file.

## The definition-of-done hook

Added 2026-08-20, because the instruction *"commit, push, test, document, tick the
todo"* had to be typed again after already being three separate bullets in
`CLAUDE.md`. A rule that has to be repeated is not a rule, it is a reminder — so it
is now a `Stop` hook that refuses to end a turn while a file the session edited is
uncommitted or unpushed (`.claude/hooks/definition_of_done.py`).

**What it does and does not catch.** Commits and pushes are checkable, so it blocks
on them. Tests, docs and the `todo.md` tick are not mechanically checkable, so they
ride in the message the block prints and still depend on Claude reading it.

**Why it tracks the session's own edits rather than `git status`.** Several sessions
run against this repo at once and each sees the others' work in progress as
uncommitted changes — the multi-session habit this doc already warns about. A bare
dirty-tree check would fire on every turn of every session for edits it did not
make, and a hook that cries wolf gets switched off. It also means the block message
can name the files, and that staging must be `git add <paths>`, never `git add -A`.

**If it gets in the way.** It never blocks twice in a row, so it cannot trap a
session — say what is left and the second stop goes through. `/hooks` reviews or
disables it; deleting the `Stop` entry in `.claude/settings.json` removes it
entirely. Config changes only take effect in sessions started afterwards.

## What already works — keep doing it

Recorded because it is the reason the numbers in this project can be trusted.

- **Demanding the raw data be kept.** *"Make sure to save all the data of the run
  so I will be able to extract later other metrics easily"* is what produced the
  per-object dump, and why any later metric is a `GROUP BY` instead of a re-run.
- **Refusing a meaningless comparison.** *"Testing it on its own test data is
  meaningless."* *"I don't care if we can compare it to a published thing, I want
  to compare M4a and M4b."*
- **Asking what a number means before accepting it** — IoU, recall, whether
  tracking was in play. Every one of those questions found something.
- **Missions with an approval gate.** M1–M7 in `todo.md` is the reason a
  six-day-old decision can still be reconstructed.
