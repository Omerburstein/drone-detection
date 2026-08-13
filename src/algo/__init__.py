"""Model layer: detector configuration, tiled inference, and result merging.

Nothing here touches the filesystem or the CLI — it takes frames as arrays and
returns `Detections`, so an experiment can drive it directly from a notebook.
"""
