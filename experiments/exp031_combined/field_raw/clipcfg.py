"""EXP-031 on the FIELD capture's `.raw`: the sensor's own 4128x3008 pixels, not the 4x-shrunk mp4.

Same recording as `exp025_top3/clips/clipcfg.py`'s `field`, frame for frame (see
`src/data/raw_bayer.py` for how the layout was measured), and the same stage-1 settings:
colour on, brightness test off, texture cut at 60. Every pixel constant in the shared
modules is quoted at 1440 px wide and scaled by width/1440, so a wider picture rescales
them, it does not re-tune them.

`FIELD_RAW_WIDTH` picks the working width (default the sensor's 4128); the height follows
the 4128x3008 aspect. Outputs go to `runs/field/exp031_combined/raw<W>/`.

The experiment scripts open `CLIP["video"]` with `cv2.VideoCapture`, which cannot read a
headerless Bayer file. Importing this config routes a `.raw` path to
`src.data.raw_bayer.RawBayerCapture` at the working size; every other path still goes to
OpenCV, so the drivers' own mp4 joins are untouched.
"""
import os

import cv2

from src.data.raw_bayer import SENSOR_H, SENSOR_W, RawBayerCapture

WIDTH = int(os.environ.get("FIELD_RAW_WIDTH", SENSOR_W))
HEIGHT = round(WIDTH * SENSOR_H / SENSOR_W)
STEM = "captured_raw_20260616_040253_004"

CLIP = dict(
    name=STEM,
    video=f"data/raw/FIELD/videos/{STEM}.raw",
    labels=None,
    hud=None,
    out=f"runs/field/exp031_combined/raw{WIDTH}/",
    width=WIDTH,
    height=HEIGHT,
    n_frames=3600,
    fps=30.0,
    stretches=[(2, 3600)],
    source_videos=f"data/raw/FIELD/videos/{STEM}.raw",
    prop_mask=None,
    ladder_mask=None,
    use_colour=True,
    sky_luma_pctl=0.0,
    sky_texture_pctl=60.0,
)
CLIPS = {"field_raw": CLIP}

_opencv_capture = cv2.VideoCapture
if not getattr(_opencv_capture, "_raw_aware", False):
    def _capture(path, *args):
        if str(path).lower().endswith(".raw"):
            return RawBayerCapture(str(path), (WIDTH, HEIGHT))
        return _opencv_capture(path, *args)

    _capture._raw_aware = True
    cv2.VideoCapture = _capture
