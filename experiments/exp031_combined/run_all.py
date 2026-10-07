"""EXP-031: every stage from EXP-025d to EXP-030 on at once, on catch_2, catch_4 and catch_5.

    py -3.13 experiments/exp031_combined/run_all.py [clip ...] [--merge 20 30] [--jobs 4]

The stages, in the order `overlay_split` applies them:

  * EXP-025d -- the frame split: sky branch (c >= 6) on stage-1 sky, the 2-of-4 motion
                window on the ground, one pooled ranking, top 3;
  * EXP-025d -- blobs within `--merge` px folded into the strongest (default 20);
  * EXP-027  -- `--osd-grid`, the OSD horizon dashes dropped by the character grid;
  * EXP-028  -- `--kinematic`, the speed limit and hysteresis track, ranked by evidence;
  * EXP-029  -- `--min-move 20`, the moving factor, the speed limit's bound from below;
  * EXP-030  -- `--merge-score sum`, a merged object scored by the sum of its members' c.

Variants per clip, so each stage's contribution is the difference between two rows:

  * `exp028`     -- max, gate.                    At merge 20 = EXP-028's default;
  * `exp029`     -- max, gate, min-move 20.       At merge 20 on catch_2 = EXP-029 stacked;
  * `exp030`     -- sum, gate.                    = EXP-030's `sum_gate` at the same radius;
  * `all`        -- sum, gate, min-move 20.       Everything, at c-keep 6;
  * `all_ck<N>`  -- the same at c-keep N, the matched-load sweep EXP-030 left open.

No detector re-runs: the dumps are EXP-030's. Each run is skipped when its `_frames.csv`
exists, so a killed batch resumes. Outputs go to
`runs/sofa_analog/exp031_combined/merge<R>/<clip>/`, and the table at the end is read from
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

sys.path.insert(0, "experiments/exp030_merge_sum")
import run_clips as exp030  # noqa: E402

BASE = ["--top", "3", "--min-draw", "20", "--osd-grid", "--kinematic"]
MOVE = ["--min-move", "20"]
SUM = ["--merge-score", "sum"]
C_KEEP = (8, 10, 12, 15)
VARIANTS = {
    "exp028": [],
    "exp029": MOVE,
    "exp030": SUM,
    "all": [*MOVE, *SUM],
    **{f"all_ck{k}": [*MOVE, *SUM, "--c-keep", str(k)] for k in C_KEEP},
}


def stem(clip: str, variant: str, merge: int) -> str:
    """The run's output path without extension; the log sits beside it."""
    c = exp030.clip_cfg(clip)
    return (f"runs/sofa_analog/exp031_combined/merge{merge}/{clip}/"
            f"{variant}_{clip}_{c['start']}_{c['end']}")


def run(clip: str, variant: str, merge: int) -> None:
    c = exp030.clip_cfg(clip)
    out = stem(clip, variant, merge)
    if os.path.exists(out + "_frames.csv"):
        print(f"[{merge}:{clip}:{variant}] have {out}_frames.csv", flush=True)
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    args = ["overlay_split", "--start", str(c["start"]), "--end", str(c["end"]), *BASE,
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
    ranked = [r for r in drone if r["outcome"] == "ranked"]
    first = {int(r["frame"]): (float(r["x"]), float(r["y"])) for r in shown if r["rank"] == "1"}
    pairs = [(first[f - 1], first[f]) for f in first if f - 1 in first]
    top3 = sum(int(r["rank"]) <= 3 for r in ranked)
    fa = sum(r["on_target"] == "0" for r in shown)
    return dict(labelled=len(drone), shown=len(shown) / max(len(frames), 1), fa=fa, top3=top3,
                first=sum(r["rank"] == "1" for r in ranked), per_fa=top3 / max(fa, 1),
                jumps=sum(np.hypot(p[0] - q[0], p[1] - q[1]) > exp030.JUMP_PX
                          for p, q in pairs),
                pairs=len(pairs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="*", default=list(exp030.ANALOG))
    ap.add_argument("--merge", type=int, nargs="+", default=[20],
                    help="merge radius in px; one subfolder `merge<R>/` per value")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    jobs = [(c, v, m) for m in a.merge for c in a.clips for v in VARIANTS]
    with ThreadPoolExecutor(a.jobs) as pool:
        for f in [pool.submit(run, c, v, m) for c, v, m in jobs]:
            f.result()
    print("\n| merge | clip | run | shown/frame | FA shown | drone in top 3 | drone #1 "
          "| top 3 per FA | #1 jumps > 29 px |\n"
          "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for c, v, m in jobs:
        r = score(c, v, m)
        print(f"| {m} | {c} ({r['labelled']} labelled) | {v} | {r['shown']:.2f} | {r['fa']} | "
              f"{r['top3']} | {r['first']} | {r['per_fa']:.2f} | {r['jumps']} / {r['pairs']}"
              f" ({r['jumps'] / max(r['pairs'], 1):.0%}) |")


if __name__ == "__main__":
    main()
