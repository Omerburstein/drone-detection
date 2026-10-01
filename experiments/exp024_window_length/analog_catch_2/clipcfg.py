"""EXP-024 clip settings: SOFA-ANALOG `catch_2` (960x720, 890 frames, 30 fps).

Every pixel parameter in the shared modules is quoted at 1440 px wide and scaled by
width/1440 = 0.667, so nothing here is a re-tuned threshold -- it is the same rule read on
a smaller picture.
"""
CLIP = dict(
    name="catch_2",
    video="data/raw/SOFA-ANALOG/videos/catch_2.mp4",
    labels="data/processed/SOFA-ANALOG/annotations/catch_2.json",
    hud="data/processed/SOFA-ANALOG/hud_mask.png",
    out="runs/sofa_analog/exp024_window_length/",
    width=960,
    height=720,
    n_frames=890,
    fps=30.0,
    stretches=[(491, 591), (647, 678), (687, 735), (746, 785), (491, 785)],

    # --- structural masks (EXP-017) -------------------------------------
    source_videos="data/raw/SOFA-ANALOG/videos/*.mp4",
    prop_mask="runs/sofa_analog/exp017_motion_first/prop_mask.npy",
    ladder_mask="runs/sofa_analog/exp017_motion_first/ladder_mask.npy",

    # --- sky/scene split --------------------------------------------------
    # CVBS is close to greyscale and carries chroma crawl, so blue-excess is noise here.
    # The split runs on luminance and texture alone.
    use_colour=False,
)
