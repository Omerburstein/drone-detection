"""Scoring layer: comparing a recorded run against ground truth.

Depends only on `detections.jsonl` and label files, never on the detector, so
evaluation runs without torch or a checkpoint present. Split as:

- `labels` — reading ground truth and pairing it with recorded predictions
- `metrics` — IoU matching, AP, and the scored `Metrics` record
- `report` — printing the metric block

`src.evaluate` is the CLI over the three.
"""
