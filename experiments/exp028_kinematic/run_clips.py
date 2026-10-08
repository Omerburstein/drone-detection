"""EXP-028 on catch_4 and catch_5: the kinematic gate out of sample, against the same split
with no gate.

    py -3.13 experiments/exp028_kinematic/run_clips.py [clip ...] [--jobs 4]

EXP-025e's dumps are reused: the sky-branch candidates and the 2-of-4 window seeds over
each clip's labelled span with a lead-in (`exp025_top3/clips/clipcfg.py`). No detector
re-runs. Four overlays per clip, all top 3, merge 20, circles >= 20 px, `--osd-grid`:

  * `nogate`   -- the split as EXP-027 runs it on catch_2, the comparison every row needs
                  (EXP-025e's catch_4/5 videos predate `--osd-grid`, so they are not it);
  * `gate`     -- EXP-028's default: c-keep 6, ranked by track evidence (catch_2's row 2);
  * `ckeep3`   -- weak continuation down to c 3, ranked by track evidence (row 5);
  * `noweak`   -- only strong candidates continue a track (row 6).

Each run is skipped when its `_frames.csv` exists, so a killed batch resumes. Outputs go to
`runs/sofa_analog/exp028_kinematic/<clip>/`, and the table at the end is read from the CSVs.
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

STACK = ("experiments/exp025_top3/clips;experiments/exp025_top3;experiments/exp023_sky_branch;"
         "experiments/exp017_motion_first;experiments/exp015_normalised_motion;.")
BASE = ["--top", "3", "--merge", "20", "--min-draw", "20", "--osd-grid", "--ref-width", "960"]
VARIANTS = {
    "nogate": [],
    "gate": ["--kinematic"],
    "ckeep3": ["--kinematic", "--c-keep", "3"],
    "noweak": ["--kinematic", "--c-keep", "1e9"],
}
ANALOG = ("catch_4", "catch_5")
JUMP_PX = 29.0   # reach(1) at the default limit: 25 px/frame + 4 px slack


def paths(clip: str, variant: str) -> tuple[str, str]:
    """The overlay's video and its log for one clip and variant."""
    c = CLIPS[clip]
    out = f"runs/sofa_analog/exp028_kinematic/{clip}/"
    return f"{out}{variant}_{clip}_{c['start']}_{c['end']}.mp4", f"{out}{variant}_{clip}.log"


def run(clip: str, variant: str) -> None:
    c = CLIPS[clip]
    video, log = paths(clip, variant)
    stem = os.path.splitext(video)[0]
    if os.path.exists(stem + "_frames.csv"):
        print(f"[{clip}:{variant}] have {stem}_frames.csv", flush=True)
        return
    os.makedirs(os.path.dirname(video), exist_ok=True)
    span = f"{c['name']}_{c['start']}_{c['end']}"
    args = ["overlay_split", "--start", str(c["start"]), "--end", str(c["end"]), *BASE,
            *VARIANTS[variant], "--sky-dump", c["sky_dump"],
            "--window-dump", f"{c['out']}seeds_k4_{span}.csv", "--out", video]
    env = dict(os.environ, EXP025_CLIP=clip, PYTHONPATH=STACK, PYTHONUNBUFFERED="1")
    print(f"[{clip}:{variant}] running -> {log}", flush=True)
    with open(log, "w", encoding="utf-8") as fh:
        subprocess.run([sys.executable, "-m", *args], env=env, stdout=fh,
                       stderr=subprocess.STDOUT, check=True)


def score(clip: str, variant: str) -> dict:
    """One row of the table, from the run's three CSVs."""
    stem = os.path.splitext(paths(clip, variant)[0])[0]
    read = lambda suffix: list(csv.DictReader(open(stem + suffix, encoding="utf-8")))
    shown, drone, frames = read(".csv"), read("_drone.csv"), read("_frames.csv")
    ranked = [r for r in drone if r["outcome"] == "ranked"]
    first = {int(r["frame"]): (float(r["x"]), float(r["y"])) for r in shown if r["rank"] == "1"}
    pairs = [(first[f - 1], first[f]) for f in first if f - 1 in first]
    return dict(clip=clip, variant=variant, frames=len(frames), labelled=len(drone),
                shown=len(shown) / max(len(frames), 1),
                fa=sum(r["on_target"] == "0" for r in shown),
                top3=sum(int(r["rank"]) <= 3 for r in ranked),
                first=sum(r["rank"] == "1" for r in ranked),
                overruled=sum(r["outcome"] == "overruled" for r in drone),
                jumps=sum(np.hypot(p[0] - q[0], p[1] - q[1]) > JUMP_PX for p, q in pairs),
                pairs=len(pairs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="*", default=list(ANALOG))
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    jobs = [(c, v) for c in a.clips for v in VARIANTS]
    with ThreadPoolExecutor(a.jobs) as pool:
        for f in [pool.submit(run, c, v) for c, v in jobs]:
            f.result()
    print("\n| clip | run | shown/frame | FA shown | drone in top 3 | drone #1 | held back "
          "by gate | #1 jumps > 29 px |\n| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for c, v in jobs:
        r = score(c, v)
        print(f"| {c} ({r['labelled']} labelled) | {v} | {r['shown']:.2f} | {r['fa']} | "
              f"{r['top3']} | {r['first']} | {r['overruled']} | {r['jumps']} / {r['pairs']}"
              f" ({r['jumps'] / max(r['pairs'], 1):.0%}) |")


if __name__ == "__main__":
    main()
