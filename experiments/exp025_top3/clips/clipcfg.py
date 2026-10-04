"""EXP-025e clip settings: the labelled analog clips other than catch_2 (catch_4, catch_5).

One config for both, chosen by the `EXP025_CLIP` environment variable, so the sky-branch
dump, the window dump and the split overlay of a clip all read the same settings and write
to the same folder. `run_clips.py` sets the variable; by hand:

    EXP025_CLIP=catch_4 PYTHONPATH="experiments/exp025_top3/clips;..." py -3.13 -m overlay_sky

Same camera, HUD and airframe masks as catch_2, and the same luminance-only sky split.
Every pixel constant in the shared modules is quoted at 1440 px wide and scaled by
width/1440, so nothing here is re-tuned. Unlabelled clips are left out on purpose: the
point of this run is the drone's rank, which needs ground truth.
"""
import os

_ANALOG = dict(
    hud="data/processed/SOFA-ANALOG/hud_mask.png",
    width=960,
    height=720,
    fps=30.0,
    source_videos="data/raw/SOFA-ANALOG/videos/*.mp4",
    prop_mask="runs/sofa_analog/exp017_motion_first/prop_mask.npy",
    ladder_mask="runs/sofa_analog/exp017_motion_first/ladder_mask.npy",
    use_colour=False,   # CVBS: luminance and texture only, as catch_2
)


def _analog(name: str, n_frames: int, start: int, end: int) -> dict:
    """`start`-`end` is the span EXP-023 already ran the sky branch over, with the settings
    catch_2's dump used, so its candidate dump (`sky_dump`) is reused, not recomputed. It
    covers every labelled frame with a lead-in, as catch_2's 441-800 does."""
    return dict(_ANALOG, name=name, video=f"data/raw/SOFA-ANALOG/videos/{name}.mp4",
                labels=f"data/processed/SOFA-ANALOG/annotations/{name}.json",
                out=f"runs/sofa_analog/exp025_top3/{name}/", n_frames=n_frames,
                stretches=[(start, end)], start=start, end=end,
                sky_dump=f"runs/sofa_analog/exp023_sky_branch/{name}/"
                         f"candidates_{name}_{start}_{end}.csv")


# labelled frames: catch_4 209-295, catch_5 119-394
CLIPS = {n: _analog(n, *v) for n, v in (("catch_4", (326, 159, 326)),
                                        ("catch_5", (465, 69, 444)))}

CLIP = CLIPS[os.environ.get("EXP025_CLIP", "catch_4")]
