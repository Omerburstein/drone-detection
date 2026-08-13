"""Air-to-air drone detection tooling.

Split into three layers along the same seam the project's agents divide on, so
data work and model work can move independently:

- `src.data` — where frames come from: source classification, decoding,
  striding, and frame budgets. Owned by dataset concerns.
- `src.algo` — what the detector does with a frame: configuration, tiled
  inference, NMS merging. Owned by model concerns.
- `src.output` — what a run leaves behind: the JSONL record, the run
  counters, and annotated media.

`src.baseline_detect` is the step-1 CLI that wires the three together. See
docs/research-notes.md for the plan and docs/baseline_detect.md for the
parameter reference.
"""
