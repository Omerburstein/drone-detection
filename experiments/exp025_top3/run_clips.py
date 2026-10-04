"""EXP-025e: the EXP-025d split overlay (merge 20 px, circles >= 20 px) on catch_4 and catch_5.

    py -3.13 experiments/exp025_top3/run_clips.py [clip ...] [--jobs 4]

Per clip, three steps, each skipped when its output already exists (so a killed batch
resumes where it stopped):

  1. EXP-023's sky branch, `--no-video`, for its candidate dump -- skipped when the clip
     config names an existing `sky_dump` (EXP-023's own, same settings);
  2. EXP-024's window at k=4, 2 appearances, 9 px, `--no-video`, for its seed dump;
  3. `overlay_split --merge 20 --min-draw 20` from those two dumps, then the rank
     histogram when the clip is labelled.

Every output, the dumps included, goes to the clip's `out` in `clips/clipcfg.py`
(`runs/<source>/exp025_top3/<clip>/`), with one log per step.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = "experiments/exp025_top3"
sys.path.insert(0, os.path.join(HERE, "clips"))
from clipcfg import CLIPS  # noqa: E402

BASE = "experiments/exp017_motion_first;experiments/exp015_normalised_motion;."
STACKS = {
    "sky": f"{HERE}/clips;experiments/exp023_sky_branch;{BASE}",
    "window": f"{HERE}/clips;experiments/exp024_window_length;{BASE}",
    "split": f"{HERE}/clips;{HERE};experiments/exp023_sky_branch;{BASE}",
}
MERGE, MIN_DRAW = 20, 20


def step(key: str, stack: str, args: list[str], out: str, log: str) -> None:
    if os.path.exists(out):
        print(f"[{key}] have {out}", flush=True)
        return
    # Unbuffered, so a step's log shows its progress while it runs, not only at exit.
    env = dict(os.environ, EXP025_CLIP=key.split(":")[0], PYTHONPATH=STACKS[stack],
               PYTHONUNBUFFERED="1")
    print(f"[{key}] running -> {log}", flush=True)
    with open(log, "w", encoding="utf-8") as fh:
        subprocess.run([sys.executable, "-m", *args], env=env, stdout=fh,
                       stderr=subprocess.STDOUT, check=True)


def run(key: str) -> str:
    c = CLIPS[key]
    out, start, end = c["out"], c.get("start", 2), c.get("end", c["n_frames"])
    os.makedirs(out, exist_ok=True)
    span = f"{c['name']}_{start}_{end}"
    se = ["--start", str(start), "--end", str(end)]
    sky = c.get("sky_dump") or f"{out}candidates_{span}.csv"
    seeds = f"{out}seeds_k4_{span}.csv"
    video = f"{out}split_sky_window2of4_top3_merge{MERGE}_{span}.mp4"
    step(f"{key}:sky", "sky", ["overlay_sky", *se, "--no-video", "--dump", sky], sky,
         f"{out}sky_{key}.log")
    step(f"{key}:window", "window", ["overlay_window", *se, "--k", "4", "--min-appear", "2",
                                     "--appear-radius", "9", "--no-video", "--dump", seeds],
         seeds, f"{out}window_{key}.log")
    step(f"{key}:split", "split", ["overlay_split", *se, "--merge", str(MERGE), "--min-draw",
                                   str(MIN_DRAW), "--sky-dump", sky, "--window-dump", seeds,
                                   "--out", video], video, f"{out}split_{key}.log")
    if c["labels"]:
        subprocess.run([sys.executable, f"{HERE}/split_rank_hist.py",
                        video.replace(".mp4", "_drone.csv")], check=True)
    print(f"[{key}] done -> {video}", flush=True)
    return key


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="*", default=None, help=f"default: all of {list(CLIPS)}")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    keys = a.clips or sorted(CLIPS, key=lambda k: -CLIPS[k]["n_frames"])
    with ThreadPoolExecutor(a.jobs) as pool:
        for f in [pool.submit(run, k) for k in keys]:
            f.result()


if __name__ == "__main__":
    main()
