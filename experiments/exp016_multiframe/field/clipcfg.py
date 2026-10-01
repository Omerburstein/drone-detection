"""EXP-017 clip settings: FIELD `captured_raw_20260616_040253_004` (1032x752, 3600 frames, 30 fps).

Our own airframe, arid hillside, target multirotor against sky. The third clip EXP-016's
machinery has been read on, and the first that is neither O4 goggles nor analog goggles.

Three things differ from both SOFA clips and each is a `CLIP` key rather than a branch in
`mf.py`:

* **No HUD.** This is a raw camera capture, not a recording of a goggles display. There is
  nothing burned into the picture, so `hud=None` and the mask is empty.
* **No labels.** `data/raw/FIELD/PROVENANCE.md` is explicit that nothing here is annotated,
  so `labels=None`, `BOXES` is empty and no rank, recall or false-alarm number can be
  computed from this run. The deliverable is the overlay video.
* **`drone_ranges` instead of labels.** PROVENANCE's three episodes, read off the EXP-010
  GLAD boxes and confirmed by eye. They are used for exactly one thing -- keeping those
  frames out of the screen-fixed map's calibration set, the same way the SOFA clips keep
  their labelled frames out. They are **not** ground truth: PROVENANCE warns in terms that
  a gap may be a stretch the detector missed rather than a stretch with no drone.

Every pixel constant in `mf.py` is quoted at 1440 px wide and scaled by 1032/1440 = 0.717,
so this is the same rule read on a smaller picture, not a re-tuned one.
"""
CLIP = dict(
    name="captured_raw_20260616_040253_004",
    video="data/raw/FIELD/videos/captured_raw_20260616_040253_004.mp4",
    labels=None,        # there are none; see PROVENANCE.md
    hud=None,           # raw capture, nothing burned in
    out="runs/field/exp017_multiframe/",
    screen_fixed="runs/field/exp017_multiframe/screen_fixed_frac.npy",
    width=1032,
    height=752,
    n_frames=3600,
    fps=30.0,
    # PROVENANCE's three episodes, plus the whole clip last (draw_video reads [:-1]).
    stretches=[(2, 180), (1101, 1553), (3140, 3600), (2, 3600)],
    drone_ranges=[(2, 180), (1101, 1553), (3140, 3600)],
    osd_twins=False,    # no overlay at all, moving or otherwise
    overlay_name="overlay_exp017.mp4",
)
