"""EXP-016 clip settings: SOFA-O4 `first_catch` (1440x1080, 964 frames, 30 fps)."""
CLIP = dict(
    name="first_catch",
    video="data/processed/SOFA-O4/videos/first_catch.avi",
    labels="data/processed/SOFA-O4/annotations/first_catch.json",
    hud="data/processed/SOFA-O4/hud_mask.png",
    out="runs/sofa_o4/exp016_multiframe/",
    # EXP-015 already calibrated the screen-fixed map on this clip's empty frames.
    screen_fixed="data/processed/SOFA-O4/screen_fixed_frac.npy",
    width=1440,
    height=1080,
    n_frames=964,
    fps=30.0,
    # Stretches quoted in EXP-015, for per-stretch coverage.
    stretches=[(708, 800), (801, 900), (901, 953), (954, 964), (708, 964)],
    osd_twins=False,   # the O4 HUD does not move; the static mask holds it
)
