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

## The five checks

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

### 3. Single purpose, small units

- **Functions** doing more than one thing get split. The signal is not raw line count
  but the presence of several distinct responsibilities — a function that parses *and*
  validates *and* writes wants to be three. Named intermediate steps beat a long body
  with comment headers.
- **Files** accumulating unrelated responsibilities get split along the same seam.
- Deep nesting is usually a missing function. Prefer early returns over `else` arms.

Do not split purely to hit a line target. A cohesive 60-line function that reads
straight through is better than three fragments that force the reader to jump.

### 4. OOP structure

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

### 5. Descriptions

Every module, class, and public function gets a docstring. Cover **what it is for and
why it exists** — the signature already says what the arguments are, so restating them
in prose adds nothing:

```python
def tile_origins(total: int, tile: int, overlap: float) -> list[int]:
    """Start coordinates for overlapping tiles covering `total` pixels."""
```

Non-obvious constants, magic numbers, and units get an inline comment. A comment
explaining *why* is valuable; one restating *what* the line does is noise and should be
deleted, not preserved.

Private helpers whose names already say it need no docstring.

## Procedure

1. Resolve scope. State what you are cleaning before you touch anything.
2. Read the files in scope fully. Do not clean code you have not read.
3. Run the five checks, collecting findings.
4. Apply the fixes.
5. **Verify behaviour is unchanged** — run the test suite if one exists; otherwise
   exercise the affected entry point (for this repo, `-m dronedet.baseline_detect --help`
   at minimum, or a short run against sample data). Cleanup that breaks the code is
   worse than no cleanup.
6. Summarise as a short list grouped by check, and state explicitly anything you chose
   *not* to change and why. The skipped items are often the more informative half.

## Reporting

Be concrete and terse. `scripts/baseline_detect.py:47 — removed unused `sys` import`
beats a paragraph. If a check found nothing, say so in one line rather than padding.
