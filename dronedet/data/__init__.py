"""Data layer: classifying a source and turning it into frames to process.

Video decoding, image loading, frame striding and frame budgets all live here,
so the run loop is identical whether the source is a video or a directory of
stills.
"""
