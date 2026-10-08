"""EXP-032: the drone selector against EXP-031's ranking, on catch_2, catch_4 and catch_5.

    py -3.13 experiments/exp032_drone_selector/run_clips.py [clip ...] [--merge 30] [--jobs 4]

EXP-031 ranks the kinematic gate's survivors by one number, the track's decayed sum of c.
`--rank-by drone` (`src.algo.selection.DroneSelector`) ranks the same survivors by six
cues: evidence, colour against the previous #1s, closeness to recent #1s, hit rate and age,
constant-velocity smoothness, and speed against the scene. Everything before the ranking
is unchanged, so the load and the gate are the same within each pair, and the only
difference is which survivor comes first.

Variants per clip, at each `--merge` radius (analog px, `--ref-width 960`):

  * `all`, `exp030_ck10` -- EXP-031's runs, read from `runs/sofa_analog/exp031_combined/`,
                            not re-run (the code reproduces them byte for byte);
  * `all_drone`, `exp030_ck10_drone` -- the same with `--rank-by drone`, default weights;
  * `all_drone_no<cue>` -- `all_drone` with one cue's weight at 0, so each row's difference
                           from `all_drone` is what that cue contributed.

Each run is skipped when its `_frames.csv` exists. Outputs go to
`runs/sofa_analog/exp032_drone_selector/merge<R>/<clip>/`. The table at the end is read from
the CSVs. The new column is **#1 precision**: of the frames whose #1 was shown, how many
had it on the drone. That is what "which candidate is the drone" asks.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/exp030_merge_sum")
sys.path.insert(0, "experiments/exp031_combined")
import run_all as exp031  # noqa: E402
import run_clips as exp030  # noqa: E402

from src.algo.selection import CUES  # noqa: E402

DRONE = ["--rank-by", "drone"]
VARIANTS = {
    "all_drone": [*exp031.VARIANTS["all"], *DRONE],
    "exp030_ck10_drone": [*exp031.VARIANTS["exp030_ck10"], *DRONE],
    **{f"all_drone_no{c}": [*exp031.VARIANTS["all"], *DRONE, "--cues", f"{c}=0"]
       for c in CUES},
}
BASELINES = ("all", "exp030_ck10")


def stem(clip: str, variant: str, merge: int) -> str:
    """A run's output path without extension; EXP-031's own for a baseline."""
    if variant in BASELINES:
        return exp031.stem(clip, variant, merge)
    c = exp030.clip_cfg(clip)
    return (f"runs/sofa_analog/exp032_drone_selector/merge{merge}/{clip}/"
            f"{variant}_{clip}_{c['start']}_{c['end']}")


def run(clip: str, variant: str, merge: int) -> None:
    if variant in BASELINES:
        exp031.run(clip, variant, merge)
        return
    c = exp030.clip_cfg(clip)
    out = stem(clip, variant, merge)
    if os.path.exists(out + "_frames.csv"):
        print(f"[{merge}:{clip}:{variant}] have {out}_frames.csv", flush=True)
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    args = ["overlay_split", "--start", str(c["start"]), "--end", str(c["end"]), *exp031.BASE,
            "--merge", str(merge), *VARIANTS[variant], "--sky-dump", c["sky_dump"],
            "--window-dump", c["window_dump"], "--out", out + ".mp4"]
    env = dict(os.environ, EXP025_CLIP=clip, PYTHONPATH=f"{c['config']};{exp030.TAIL}",
               PYTHONUNBUFFERED="1")
    print(f"[{merge}:{clip}:{variant}] running -> {out}.log", flush=True)
    with open(out + ".log", "w", encoding="utf-8") as fh:
        subprocess.run([sys.executable, "-m", *args], env=env, stdout=fh,
                       stderr=subprocess.STDOUT, check=True)


def score(clip: str, variant: str, merge: int) -> dict:
    """One row of the table, from the run's three CSVs."""
    out = stem(clip, variant, merge)
    read = lambda suffix: list(csv.DictReader(open(out + suffix, encoding="utf-8")))
    shown, drone, frames = read(".csv"), read("_drone.csv"), read("_frames.csv")
    labelled = {int(r["frame"]) for r in drone}
    first = [r for r in shown if r["rank"] == "1"]
    first_lab = [r for r in first if int(r["frame"]) in labelled]
    ranked = [r for r in drone if r["outcome"] == "ranked"]
    xy = {int(r["frame"]): (float(r["x"]), float(r["y"])) for r in first}
    pairs = [(xy[f - 1], xy[f]) for f in xy if f - 1 in xy]
    return dict(labelled=len(drone), shown=len(shown) / max(len(frames), 1),
                fa=sum(r["on_target"] == "0" for r in shown),
                top3=sum(int(r["rank"]) <= 3 for r in ranked),
                first=sum(r["rank"] == "1" for r in ranked),
                first_shown=len(first_lab),
                precision=sum(r["on_target"] == "1" for r in first_lab) / max(len(first_lab), 1),
                jumps=sum(np.hypot(p[0] - q[0], p[1] - q[1]) > exp030.JUMP_PX
                          for p, q in pairs),
                pairs=len(pairs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="*", default=list(exp030.ANALOG))
    ap.add_argument("--merge", type=int, nargs="+", default=[30],
                    help="merge radius in analog px; one subfolder `merge<R>/` per value")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    names = [*BASELINES, *VARIANTS]
    jobs = [(c, v, m) for m in a.merge for c in a.clips for v in names]
    with ThreadPoolExecutor(a.jobs) as pool:
        for f in [pool.submit(run, c, v, m) for c, v, m in jobs]:
            f.result()
    print("\n| merge | clip | run | shown/frame | FA shown | drone in top 3 | drone #1 "
          "| #1 shown (labelled) | #1 precision | #1 jumps > 29 px |\n"
          "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for c, v, m in jobs:
        r = score(c, v, m)
        print(f"| {m} | {c} ({r['labelled']} labelled) | {v} | {r['shown']:.2f} | {r['fa']} | "
              f"{r['top3']} | {r['first']} | {r['first_shown']} | {r['precision']:.0%} | "
              f"{r['jumps']} / {r['pairs']} ({r['jumps'] / max(r['pairs'], 1):.0%}) |")
    print("\n| merge | run | drone in top 3 | drone #1 | #1 precision | #1 jumps |\n"
          "| ---: | --- | ---: | ---: | ---: | ---: |")
    for m in a.merge:
        for v in names:
            rows = [score(c, v, m) for c in a.clips]
            first, shown = sum(r["first"] for r in rows), sum(r["first_shown"] for r in rows)
            print(f"| {m} | {v} | {sum(r['top3'] for r in rows)} | {first} | "
                  f"{first / max(shown, 1):.0%} | {sum(r['jumps'] for r in rows)} / "
                  f"{sum(r['pairs'] for r in rows)} |")


if __name__ == "__main__":
    main()
