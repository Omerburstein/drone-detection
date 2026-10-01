"""EXP-016 clip settings: SOFA-ANALOG `catch_2` (960x720, 890 frames, 30 fps).

Every pixel parameter in `mf.py` is scaled by width/1440 = 0.667, so nothing here is a
re-tuned threshold -- it is the same rule read on a smaller picture. The speed cap is
*not* scaled: it is derived from physical units and the target's own apparent size, so it
is already camera-independent.
"""
CLIP = dict(
    name="catch_2",
    video="data/raw/SOFA-ANALOG/videos/catch_2.mp4",
    labels="data/processed/SOFA-ANALOG/annotations/catch_2.json",
    hud="data/processed/SOFA-ANALOG/hud_mask.png",
    out="runs/sofa_analog/exp016_multiframe/",
    # Built by screenfixed.py on this clip's own empty frames (same rule as EXP-015's).
    screen_fixed="runs/sofa_analog/exp016_multiframe/screen_fixed_frac.npy",
    width=960,
    height=720,
    n_frames=890,
    fps=30.0,
    stretches=[(491, 591), (647, 678), (687, 735), (746, 785), (491, 785)],
    osd_twins=True,    # compass letters and horizon dashes move; a static mask cannot hold them
)
