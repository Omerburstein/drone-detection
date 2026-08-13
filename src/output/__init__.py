"""Output layer: what a run leaves on disk.

The JSONL record and the run counters live in `recording`; annotated video and
images in `annotate`. Both are written incrementally so a long run that is
interrupted still leaves usable partial output.
"""
