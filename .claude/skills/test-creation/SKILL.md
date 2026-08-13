---
name: test-creation
description: Audit test coverage and write the missing tests — unit tests for individual functions and integration tests for the paths that wire them together. Accepts an optional file or folder to scope the audit. Use when the user asks to "add tests", "write unit tests", "check test coverage", "test this file/folder", or "make sure everything is tested".
---

# test-creation

Find what is untested, then write the tests. Two layers, and both are required:

- **Unit** — one function or method, its edge cases and boundaries, dependencies faked.
- **Integration** — several units wired together across a real seam (file I/O, a model
  load, an end-to-end CLI invocation). Catches the contract mismatches that unit tests
  structurally cannot.

A function covered only by a unit test is not done; a pipeline covered only end-to-end
is not done either.

## Scope

`$ARGUMENTS` may name a file or folder.

1. Explicit path → audit that file, or every source file under that folder recursively.
2. No argument → audit the current diff. If the tree is clean, ask what to scope to.

## Procedure

### 1. Inventory

List every module, class, function, and CLI entry point in scope. For each, record
whether a test exists and at which layer. Match by behaviour, not by name — a test
named `test_foo` that only asserts `foo` doesn't raise is not coverage.

Present this as a table before writing anything: **symbol | unit | integration**, with
gaps marked. This is the plan, and it lets the user redirect before you spend effort.

### 2. Prioritise

Not everything deserves equal effort. Rank by consequence of silent failure:

- **Highest** — logic with boundaries and off-by-one risk (tiling, coordinate
  transforms, box merging, index math), data parsing and label handling, anything whose
  wrongness produces plausible-looking-but-wrong output rather than a crash.
- **Medium** — orchestration, CLI argument handling, I/O paths.
- **Lowest** — thin pass-throughs and trivial property accessors. Say you are skipping
  these rather than padding the suite.

Silent-wrongness beats loud-crashing for priority. A miscomputed bounding-box offset
produces a number, not an exception, and can corrupt an entire training run unnoticed.

### 3. Write

Framework is **pytest**. If absent: `py -3.13 -m pip install pytest`.

Layout mirrors the source tree:

```
tests/
  unit/test_<module>.py
  integration/test_<flow>.py
  conftest.py          # shared fixtures
```

Rules:

- **One behaviour per test.** The name states the expectation:
  `test_tile_origins_covers_final_pixel_when_not_divisible`, not `test_tiles_2`.
- **Arrange–Act–Assert**, visually separated.
- **Assert on values, not on "it ran."** `assert result == [0, 512, 1024, 1520]` beats
  `assert result is not None`.
- **Cover the boundaries deliberately**: empty input, single element, exact-fit and
  off-by-one sizes, zero and negative values, the largest realistic input.
- **Parametrize** over cases rather than copy-pasting test bodies —
  `@pytest.mark.parametrize` keeps the failure output specific.
- **Test the contract, not the implementation.** A test that breaks on every harmless
  refactor is a liability.

### 4. ML-specific constraints

This repo does computer vision on a CPU-only machine. That shapes the suite:

- **Never download weights or datasets in a test.** Tests must pass offline. Build
  tiny synthetic inputs with numpy/OpenCV — a 64×64 array with a drawn shape exercises
  a full frame path in milliseconds.
- **Fake the model at the unit layer.** A stub returning canned boxes tests your
  merging, offsetting, and NMS logic without a forward pass. The heavy real-model run
  belongs in one integration test, marked `@pytest.mark.slow` and skipped by default:
  `pytest -m "not slow"`.
- **Seed everything** that samples randomly, and assert on exact values once seeded.
- **Use `tmp_path`** for anything writing files. Never write into `data/`, `runs/`, or
  `weights/`.
- **Compare floats with `pytest.approx`**, never `==`.

### 5. Verify

Run the suite. Report the actual result — pass counts, and the full output of any
failure.

Then confirm the tests are worth having: **make a test fail on purpose.** Briefly
break the code under test (or reason precisely about what change would break it) to
confirm the assertion actually catches it. A test that passes against both correct and
broken code is worse than none, because it manufactures confidence. If a written test
cannot fail, say so and rewrite it.

## Reporting

Give the coverage table again with gaps now closed, the pass/fail counts, and an
explicit list of what you deliberately left untested and why.
