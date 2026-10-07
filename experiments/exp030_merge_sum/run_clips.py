"""EXP-030: a merged object scored by the sum of its members' c, not the strongest one's, on
catch_2, catch_4 and catch_5.

    py -3.13 experiments/exp030_merge_sum/run_clips.py [clip ...] [--merge 20 30] [--jobs 4]

EXP-028's split overlay (top 3, circles >= 20 px, `--osd-grid`; merge 20, or each radius
given to `--merge`) from the dumps
EXP-028 used -- catch_2's own, EXP-025e's for catch_4 and catch_5. No detector re-runs.
Four overlays per clip, so the score is the only difference inside each pair:

  * `max_nogate` / `sum_nogate` -- the pooled ranking alone, as EXP-027 runs it;
  * `max_gate`   / `sum_gate`   -- through EXP-028's default kinematic gate (c-keep 6,
                                   ranked by track evidence). `sum_gate` is the overlay.

At merge 20 the `max_` runs re-create EXP-028's `nogate` and `gate`; their CSVs must match
EXP-028's byte for byte, which checks that `--merge-score` changed nothing when it is off.
At any other radius they are that radius's own baseline.

Each run is skipped when its `_frames.csv` exists, so a killed batch resumes. Outputs go to
`runs/sofa_analog/exp030_merge_sum/merge<R>/<clip>/`, and the table at the end is read from
the CSVs.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, "experiments/exp025_top3/clips")
from clipcfg import CLIPS  # noqa: E402

TAIL = ("experiments/exp025_top3;experiments/exp023_sky_branch;"
        "experiments/exp017_motion_first;experiments/exp015_normalised_motion;.")
CATCH_2 = dict(name="catch_2", start=441, end=800,
               sky_dump="runs/sofa_analog/exp023_sky_branch/candidates_catch_2_441_800.csv",
               window_dump="runs/sofa_analog/exp024_window_length/seeds_k4_catch_2_441_800.csv",
               config="experiments/exp023_sky_branch/analog_catch_2")
BASE = ["--top", "3", "--min-draw", "20", "--osd-grid"]
VARIANTS = {
    "max_nogate": [],
    "sum_nogate": ["--merge-score", "sum"],
    "max_gate": ["--kinematic"],
    "sum_gate": ["--kinematic", "--merge-score", "sum"],
}
ANALOG = ("catch_2", "catch_4", "catch_5")
JUMP_PX = 29.0   # reach(1) at the default limit: 25 px/frame + 4 px slack


def clip_cfg(clip: str) -> dict:
    """Span, dumps and clip-config folder for one clip."""
    if clip == "catch_2":
        return CATCH_2
    c = CLIPS[clip]
    span = f"{c['name']}_{c['start']}_{c['end']}"
    return dict(name=clip, start=c["start"], end=c["end"], sky_dump=c["sky_dump"],
                window_dump=f"{c['out']}seeds_k4_{span}.csv",
                config="experiments/exp025_top3/clips")


def paths(clip: str, variant: str, merge: int) -> tuple[str, str]:
    """The overlay's video and its log for one clip, variant and merge radius."""
    c = clip_cfg(clip)
    out = f"runs/sofa_analog/exp030_merge_sum/merge{merge}/{clip}/"
    return f"{out}{variant}_{clip}_{c['start']}_{c['end']}.mp4", f"{out}{variant}_{clip}.log"


def run(clip: str, variant: str, merge: int) -> None:
    c = clip_cfg(clip)
    video, log = paths(clip, variant, merge)
    stem = os.path.splitext(video)[0]
    if os.path.exists(stem + "_frames.csv"):
        print(f"[{clip}:{variant}] have {stem}_frames.csv", flush=True)
        return
    os.makedirs(os.path.dirname(video), exist_ok=True)
    args = ["overlay_split", "--start", str(c["start"]), "--end", str(c["end"]), *BASE,
            "--merge", str(merge), *VARIANTS[variant], "--sky-dump", c["sky_dump"],
            "--window-dump", c["window_dump"], "--out", video]
    env = dict(os.environ, EXP025_CLIP=clip, PYTHONPATH=f"{c['config']};{TAIL}",
               PYTHONUNBUFFERED="1")
    print(f"[{clip}:{variant}] running -> {log}", flush=True)
    with open(log, "w", encoding="utf-8") as fh:
        subprocess.run([sys.executable, "-m", *args], env=env, stdout=fh,
                       stderr=subprocess.STDOUT, check=True)


def score(clip: str, variant: str, merge: int) -> dict:
    """One row of the table, from the run's three CSVs."""
    stem = os.path.splitext(paths(clip, variant, merge)[0])[0]
    read = lambda suffix: list(csv.DictReader(open(stem + suffix, encoding="utf-8")))
    shown, drone, frames = read(".csv"), read("_drone.csv"), read("_frames.csv")
    ranked = [r for r in drone if r["outcome"] == "ranked"]
    first = {int(r["frame"]): (float(r["x"]), float(r["y"])) for r in shown if r["rank"] == "1"}
    pairs = [(first[f - 1], first[f]) for f in first if f - 1 in first]
    return dict(labelled=len(drone), shown=len(shown) / max(len(frames), 1),
                fa=sum(r["on_target"] == "0" for r in shown),
                top3=sum(int(r["rank"]) <= 3 for r in ranked),
                first=sum(r["rank"] == "1" for r in ranked),
                jumps=sum(np.hypot(p[0] - q[0], p[1] - q[1]) > JUMP_PX for p, q in pairs),
                pairs=len(pairs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="*", default=list(ANALOG))
    ap.add_argument("--merge", type=int, nargs="+", default=[20],
                    help="merge radius in px; one subfolder `merge<R>/` per value")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    jobs = [(c, v, m) for m in a.merge for c in a.clips for v in VARIANTS]
    with ThreadPoolExecutor(a.jobs) as pool:
        for f in [pool.submit(run, c, v, m) for c, v, m in jobs]:
            f.result()
    print("\n| merge | clip | run | shown/frame | FA shown | drone in top 3 | drone #1 "
          "| #1 jumps > 29 px |\n| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for c, v, m in jobs:
        r = score(c, v, m)
        print(f"| {m} | {c} ({r['labelled']} labelled) | {v} | {r['shown']:.2f} | {r['fa']} | "
              f"{r['top3']} | {r['first']} | {r['jumps']} / {r['pairs']}"
              f" ({r['jumps'] / max(r['pairs'], 1):.0%}) |")


if __name__ == "__main__":
    main()
