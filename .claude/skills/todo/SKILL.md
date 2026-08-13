---
name: todo
description: Append a task to the project todo list in docs/todo.md. Takes a free-text description and files it as a dated, checkable entry under the right area. Use when the user says "todo", "add a todo", "remind me to", "note that down", "add to the list", or describes work to do later rather than now.
---

# todo

Capture a task in `docs/todo.md` so it survives the session. **Capture only — do not
start the work.** If the user wanted it done now they would have asked for that; turning
a note into a code change loses the note and produces an unrequested diff.

## Input

`$ARGUMENTS` is the task description, in whatever form the user typed it. If it is
empty, ask what the task is — do not invent one from conversation context.

If the description is a bare fragment ("tiling"), ask for one clarifying sentence. A
todo that cannot be understood next month is not a todo.

## The file

`docs/todo.md`, committed with the rest of `docs/`. Create it from this template if it
does not exist:

```markdown
# Todo

Captured tasks. Add via `/todo <description>`. Check an item off by moving it to Done
with the date it was finished.

## Open

## Done
```

Entry format, one line per task:

```markdown
- [ ] 2026-08-13 — [data] Convert ARD-MAV annotations to the canonical YOLO layout
```

- **Date** is the day it was captured, absolute (`YYYY-MM-DD`), never "today" or
  "next week" — a relative date in a file read months later is worse than no date.
- **Area tag** is one of `[data]`, `[algo]`, `[infra]`, `[docs]`. Pick from the
  description; use `[infra]` for tooling, environment, and GPU-rental work. If it
  genuinely spans two, tag the one that has to move first.
- **Description** is one line, imperative, specific enough to act on without this
  conversation. Rewrite a vague phrasing into a concrete one, but do not add scope the
  user did not ask for.

New entries go at the **bottom of `## Open`**, so the file reads in capture order.

## Procedure

1. Read `docs/todo.md` (create from the template if absent).
2. **Check for a duplicate or near-duplicate already open.** If one exists, say so and
   ask whether to sharpen the existing entry instead of adding a second — a list with
   two phrasings of one task gets trusted less every time it is read.
3. Append the entry under `## Open`.
4. Commit and push (`docs: add todo — <short description>`). This is standing
   instruction for this repo; do not ask first.
5. Reply with the single line you added and nothing else. A capture command that
   produces a paragraph of summary defeats its own purpose.

## When something else fits better

- A **finished experiment result** belongs in `docs/experiments.md` via the `/eval`
  skill or `algo-agent`, not here. The ledger is the record of what was measured; this
  file is the record of what has not been done yet.
- A **known trap or hard-won lesson** belongs in `CLAUDE.md` under *Project-specific
  traps*, so it is loaded every session rather than waiting to be read.

Say which one applies and offer to file it there instead.
