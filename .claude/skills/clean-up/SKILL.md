---
name: clean-up
description: Clean up code quality — remove unused imports, eliminate duplication, split oversized functions and files into single-purpose units, apply sound OOP structure, and ensure everything carries a description/docstring. Accepts an optional file or folder to scope the pass; defaults to the current diff. Use when the user asks to "clean up", "tidy", "refactor for quality", "remove duplication", "check OOP", or "add docstrings".
---

# clean-up

A quality pass over code. **Quality only — this is not a bug hunt.** If you spot a
correctness bug along the way, report it but do not silently fix it as part of the
cleanup; a cleanup diff that also changes behaviour is unreviewable.

## Scope

`$ARGUMENTS` may name a file or folder. Resolve scope in this order:

1. An explicit path in `$ARGUMENTS` → clean that file or folder recursively.
2. No argument → clean the current working-tree diff (`git status --porcelain` and
   `git diff`). If the tree is clean, ask what to scope to rather than sweeping the
   whole repo.

Never widen beyond the resolved scope. A cleanup that sprawls across untouched files
buries the intended change.

## The six checks

Work through these in order. Order matters — deleting dead code first means you don't
waste effort refactoring something that shouldn't exist.

### 1. Unused imports and dead code

Remove unused imports, unreachable branches, variables assigned but never read, and
functions with no callers.

Prefer tooling over eyeballing:

```
py -3.13 -m ruff check --select F401,F841,ARG,ERA <path>
```

If ruff is absent, offer to `pip install ruff` — it is fast and catches all of this
class. Falling back to manual reading is acceptable but note that you did.

**Care with two cases:** re-exports in `__init__.py` (an "unused" import there is often
the public API — check for `__all__`), and imports with registration side effects.
Confirm a symbol is genuinely unreferenced before deleting it.

### 2. Duplication

Find logic written more than once and give it a single home. Search for repeated
literals, near-identical blocks, and parallel functions differing only in a constant.

The bar for extracting is **three occurrences, or two that must always change
together.** Two superficially similar blocks that evolve independently are not
duplication, and merging them creates a coupling that later has to be undone. State
which of the two cases you are invoking when you extract.

### 3. Magic numbers

A bare literal carrying meaning is a parameter that was never given a name. Promote it,
in this order of preference:

1. **A parameter** — on `InferenceConfig`, or a function argument with a default. This
   is the right answer for anything that changes the *result*: confidence and NMS
   thresholds, `imgsz`, tile size and overlap, stride, frame budgets.
2. **A module-level named constant** — for a fixed value the caller has no business
   changing (box line thickness, JSONL schema version, a progress-log interval).
3. **An inline comment giving the unit or source** — the fallback when the value is
   genuinely local and naming it would not help.

**On this project the first option matters more than usual.** A threshold hardcoded
inside `algo/` is a parameter that never reaches the CLI, never lands in the run record,
and so never appears in the ledger entry — which makes the run unreproducible and any
comparison against it invalid. If a literal would change a metric, it belongs in
`InferenceConfig` and in the recorded run parameters, not in a function body.

Not every number is magic. Leave these alone:

- `0`, `1`, `-1`, `2` used as indices, counts, or offsets.
- Values fixed by an external format or convention — `255` for 8-bit pixel range, `3`
  for BGR channels, `100` for a percentage.
- A number used exactly once, immediately beside the name that explains it
  (`fps=30` in a call).

The signal is **meaning, not digits**: `if conf > 0.25` hides a decision, `range(3)`
over colour channels does not. Extracting a constant named `ZERO_POINT_TWO_FIVE` is
worse than the literal — if you cannot name what it *means*, comment the source instead.

When you promote a literal to a parameter, keep the current value as the default so the
cleanup stays behaviour-preserving, and say in your summary which values became
parameters — the user may want them exposed on the CLI as a follow-up.

### 4. Single purpose, small units

- **Functions** doing more than one thing get split. The signal is not raw line count
  but the presence of several distinct responsibilities — a function that parses *and*
  validates *and* writes wants to be three. Named intermediate steps beat a long body
  with comment headers.
- **Files** accumulating unrelated responsibilities get split along the same seam.
- Deep nesting is usually a missing function. Prefer early returns over `else` arms.

Do not split purely to hit a line target. A cohesive 60-line function that reads
straight through is better than three fragments that force the reader to jump.

### 5. OOP structure

Apply where it earns its place:

- **State that travels together** with the behaviour acting on it → a class.
  Dataset loaders, model wrappers, trackers, config objects.
- **Dependencies passed in, not constructed inside** — this is what makes a class
  testable, and it is the single highest-value OOP change in most codebases.
- **No god objects.** A class touching every part of the system is a namespace, not an
  abstraction.
- **Prefer composition to inheritance.** Reach for inheritance only for genuine
  is-a relationships with a stable base.

**This is a Python/ML codebase — do not convert pure functions into classes.** A
stateless transform is correctly a function; wrapping it in a class with one method
and no fields is strictly worse. Data-in-data-out numerical code should stay
functional. Use `@dataclass` for records rather than hand-written `__init__`
boilerplate. If a proposed OOP change makes the code longer without making it more
testable or less coupled, skip it and say why.

### 6. Descriptions

Every module, class, and public function gets a docstring. Cover **what it is for and
why it exists** — the signature already says what the arguments are, so restating them
in prose adds nothing:

```python
def tile_origins(total: int, tile: int, overlap: float) -> list[int]:
    """Start coordinates for overlapping tiles covering `total` pixels."""
```

Constants that survived check 3 get an inline comment giving their unit or source. A
comment explaining *why* is valuable; one restating *what* the line does is noise and
should be deleted, not preserved.

Private helpers whose names already say it need no docstring.

## Procedure

1. Resolve scope. State what you are cleaning before you touch anything.
2. Read the files in scope fully. Do not clean code you have not read.
3. Run the six checks, collecting findings.
4. Apply the fixes.
5. **Verify behaviour is unchanged** — run the test suite if one exists; otherwise
   exercise the affected entry point (for this repo, `-m src.baseline_detect --help`
   at minimum, or a short run against sample data). Cleanup that breaks the code is
   worse than no cleanup.
6. Summarise as a short list grouped by check, and state explicitly anything you chose
   *not* to change and why. The skipped items are often the more informative half.

## Reporting

Be concrete and terse. `scripts/baseline_detect.py:47 — removed unused `sys` import`
beats a paragraph. If a check found nothing, say so in one line rather than padding.
