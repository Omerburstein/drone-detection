"""EXP-017 clip settings: SOFA-O4 `first_catch` (1440x1080, 964 frames, 30 fps).

Same shape as EXP-016's `clipcfg`, plus the two things EXP-017 adds: the *source* (every
clip from the same airframe, which is what the structural masks are calibrated over) and
the sky/scene split's per-clip knobs.
"""
CLIP = dict(
    name="first_catch",
    video="data/processed/SOFA-O4/videos/first_catch.avi",
    labels="data/processed/SOFA-O4/annotations/first_catch.json",
    hud="data/processed/SOFA-O4/hud_mask.png",
    out="runs/sofa_o4/exp017_motion_first/",
    width=1440,
    height=1080,
    n_frames=964,
    fps=30.0,
    stretches=[(708, 800), (801, 900), (901, 953), (954, 964), (708, 964)],

    # --- structural masks (EXP-017) -------------------------------------
    # Every clip shot through the same airframe and camera mount. The prop and ladder
    # masks are calibrated over ALL of these, which is what makes them a property of the
    # aircraft rather than of one clip's scene -- and therefore out-of-sample for any one
    # clip's drone. Contrast EXP-015's screen-fixed map, which is fitted on the same
    # clip it is then applied to.
    source_videos="data/processed/SOFA-O4/videos/*.avi",
    prop_mask="data/processed/SOFA-O4/prop_mask.npy",
    ladder_mask="data/processed/SOFA-O4/ladder_mask.npy",

    # --- sky/scene split --------------------------------------------------
    # Colour footage: the blue-excess channel carries real information. See skyline.py.
    use_colour=True,
)
