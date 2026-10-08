"""EXP-025e: the EXP-025d split overlay (merge 20 px, circles >= 20 px) on catch_4, catch_5 and FIELD.

    py -3.13 experiments/exp025_top3/run_clips.py [clip ...] [--jobs 4]

Per clip, three steps, each skipped when its output already exists (so a killed batch
resumes where it stopped):

  1. EXP-023's sky branch, `--no-video`, for its candidate dump -- skipped when the clip
     config names an existing `sky_dump` (EXP-023's own, same settings);
  2. (alongside 1) EXP-024's window at k=4, 2 appearances, 9 px at 960 wide (scaled to the
     clip's width), `--no-video`, for its seed dump;
  3. `overlay_split --merge 20 --min-draw 20 --ref-width 960` from those two dumps -- in
     `split_chunks` parallel ranges, joined, when the clip config sets it -- then the rank histogram when
     the clip is labelled.

Every output, the dumps included, goes to the clip's `out` in `clips/clipcfg.py`
(`runs/<source>/exp025_top3/<clip>/`), with one log per step.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2

HERE = "experiments/exp025_top3"
sys.path.insert(0, os.path.join(HERE, "clips"))
from clipcfg import CLIPS  # noqa: E402

BASE = "experiments/exp017_motion_first;experiments/exp015_normalised_motion;."
STACKS = {
    "sky": f"{HERE}/clips;experiments/exp023_sky_branch;{BASE}",
    "window": f"{HERE}/clips;experiments/exp024_window_length;{BASE}",
    "split": f"{HERE}/clips;{HERE};experiments/exp023_sky_branch;{BASE}",
}
MERGE, MIN_DRAW = 20, 20   # px at REF_WIDTH, the analog clips' width
REF_WIDTH = 960
APPEAR_RADIUS = 9.0        # the window's seed radius, px at REF_WIDTH


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


def sky_step(key: str, c: dict, start: int, end: int, sky: str, out: str) -> None:
    """The sky branch, in `sky_chunks` frame ranges run side by side, joined into one dump.

    Exact, not approximate: each frame reads only itself and the frame before it (for the
    homography), and a chunk starting at s reads s-1 first, as a whole-clip run does."""
    if os.path.exists(sky):
        print(f"[{key}:sky] have {sky}", flush=True)
        return
    n = c.get("sky_chunks", 1)
    edges = [start + round(i * (end - start + 1) / n) for i in range(n + 1)]
    parts = [(a, b - 1) for a, b in zip(edges, edges[1:])]
    paths = [f"{sky}.part{i}" for i in range(n)]
    with ThreadPoolExecutor(n) as pool:
        for f in [pool.submit(step, f"{key}:sky[{a}-{b}]", "sky",
                              ["overlay_sky", "--start", str(a), "--end", str(b),
                               "--no-video", "--dump", p], p, f"{out}sky_{key}_{a}_{b}.log")
                  for (a, b), p in zip(parts, paths)]:
            f.result()
    with open(sky + ".tmp", "w", encoding="utf-8", newline="") as dst:
        for i, p in enumerate(paths):
            with open(p, encoding="utf-8", newline="") as src:
                lines = src.readlines()
            dst.writelines(lines if i == 0 else lines[1:])
    os.replace(sky + ".tmp", sky)
    for p in paths:
        os.remove(p)


def split_step(key: str, c: dict, start: int, end: int, base: list[str], video: str,
               out: str) -> None:
    """The split overlay, in `split_chunks` frame ranges side by side, joined into one video,
    one set of CSVs and one report (`overlay_split --report-only` over the joined tables).

    Exact, like `sky_step`: a frame reads its own stage-1 split (from itself and the frame
    before), its own dump rows and its own window survivors, nothing carried from earlier
    frames. Done-ness is the `_frames.csv`, written last, not the video: a killed run leaves
    an unplayable video behind (FIELD, 2026-10-04)."""
    stem = os.path.splitext(video)[0]
    log = f"{out}split_{key}.log"
    if os.path.exists(stem + "_frames.csv"):
        print(f"[{key}:split] have {stem}_frames.csv", flush=True)
        return
    n = c.get("split_chunks", 1)
    if n == 1:
        step(f"{key}:split", "split", ["overlay_split", "--start", str(start), "--end",
                                       str(end), *base, "--out", video], stem + "_frames.csv",
             log)
        return
    edges = [start + round(i * (end - start + 1) / n) for i in range(n + 1)]
    parts = [(a, b - 1) for a, b in zip(edges, edges[1:])]
    stems = [f"{stem}.part{i}" for i in range(n)]
    with ThreadPoolExecutor(n) as pool:
        for f in [pool.submit(step, f"{key}:split[{a}-{b}]", "split",
                              ["overlay_split", "--start", str(a), "--end", str(b), *base,
                               "--out", ps + ".mp4"], ps + "_frames.csv",
                              f"{out}split_{key}_{a}_{b}.log")
                  for (a, b), ps in zip(parts, stems)]:
            f.result()
    print(f"[{key}:split] joining {n} parts", flush=True)
    writer = None
    for ps in stems:
        cap = cv2.VideoCapture(ps + ".mp4")
        if writer is None:
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            writer = cv2.VideoWriter(stem + ".tmp.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                                     cap.get(cv2.CAP_PROP_FPS), size)
        while True:
            ok, img = cap.read()
            if not ok:
                break
            writer.write(img)
        cap.release()
    writer.release()
    os.replace(stem + ".tmp.mp4", video)
    for suffix in (".csv", "_drone.csv", "_frames.csv"):
        with open(stem + suffix + ".tmp", "w", encoding="utf-8", newline="") as dst:
            for i, ps in enumerate(stems):
                with open(ps + suffix, encoding="utf-8", newline="") as src:
                    lines = src.readlines()
                dst.writelines(lines if i == 0 else lines[1:])
    for suffix in (".csv", "_drone.csv"):
        os.replace(stem + suffix + ".tmp", stem + suffix)
    os.replace(stem + "_frames.csv.tmp", stem + "_frames.csv")   # last: marks the step done
    step(f"{key}:report", "split", ["overlay_split", "--start", str(start), "--end", str(end),
                                    *base, "--out", video, "--report-only"],
         log + ".absent", log)
    for ps in stems:
        for suffix in (".mp4", ".csv", "_drone.csv", "_frames.csv"):
            os.remove(ps + suffix)


def run(key: str) -> str:
    c = CLIPS[key]
    out, start, end = c["out"], c.get("start", 2), c.get("end", c["n_frames"])
    os.makedirs(out, exist_ok=True)
    span = f"{c['name']}_{start}_{end}"
    se = ["--start", str(start), "--end", str(end)]
    sky = c.get("sky_dump") or f"{out}candidates_{span}.csv"
    seeds = f"{out}seeds_k4_{span}.csv"
    video = f"{out}split_sky_window2of4_top3_merge{MERGE}_{span}.mp4"
    # The two detectors read only the video, not each other, so they run side by side.
    with ThreadPoolExecutor(2) as pool:
        for f in [pool.submit(sky_step, key, c, start, end, sky, out),
                  pool.submit(step, f"{key}:window", "window",
                              ["overlay_window", *se, "--k", "4", "--min-appear", "2",
                               "--appear-radius", f"{APPEAR_RADIUS * c['width'] / REF_WIDTH:g}",
                               "--no-video", "--dump", seeds],
                              seeds, f"{out}window_{key}.log")]:
            f.result()
    split_step(key, c, start, end, ["--merge", str(MERGE), "--min-draw", str(MIN_DRAW),
                                    "--ref-width", str(REF_WIDTH), "--sky-dump", sky,
                                    "--window-dump", seeds], video, out)
    if c.get("labels"):
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
