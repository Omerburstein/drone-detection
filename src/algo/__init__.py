"""Model layer: detector configuration, tiled inference, and result merging.

The detection path takes frames as arrays and returns `Detections`, so an
experiment can drive it directly from a notebook without a command line.

This used to claim nothing here touches the filesystem or the CLI. That is no
longer true and cannot cheaply be made true: `glad/vendor.py` mutates `sys.path`
and monkeypatches `torch.load` so a vendored third-party checkpoint will import
at all. It is the one deliberate exception, and it is isolated there precisely so
nothing else in this package has to know about it.
"""
