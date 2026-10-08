"""EXP-031 `all` on the FIELD `.raw` -- the sensor's 4128x3008 pixels.

    py -3.13 experiments/exp031_combined/run_field_raw.py [--merge 30] [--spans all] [--full]
        [--rank-by track|drone]

The analog runs in `run_all.py` reuse EXP-030's dumps; there are none for the raw, so this
builds them first. Per span:

  1. EXP-023's sky branch, `--no-video`, in `SKY_CHUNK`-frame ranges joined into one dump
     (exact: a frame reads only itself and the one before, as `exp025_top3/run_clips.py`);
  2. EXP-024's window at k=4, 2 appearances, 9 px at 960 wide (38.7 at 4128), `--no-video`,
     one pass per span (it carries a window across frames, so it is not chunked);
  3. `overlay_split` with `run_all.py`'s `all` flags -- top 3, OSD grid, kinematic gate,
     moving factor, summed c, c-keep 6 -- at each `--merge` radius, ranked by `--rank-by`
     (default `track`, EXP-031's; `drone` is EXP-032's selector, which lost on analog);
  4. an H.264 copy of each overlay at 1080 rows, `_1080p.mp4`, the one a Windows player opens;
     the full-size `mp4v` is deleted after it unless `--keep-full-size`.

`--spans episodes` (default) runs the three episodes in `data/raw/FIELD/PROVENANCE.md`, with a
lead-in for the gate to confirm a track; they are where the detector was right on the mp4,
not where the drone is, so a quiet stretch outside them is not a negative. `--spans all`
adds the two gaps between them, so the five spans tile frames 2-3600. `--full` (implies
`all`) joins the five spans' dumps and runs step 3 once over 2-3600 instead of per span, so
the kinematic tracker runs unbroken through the clip. The window dumps are still per span,
so the window restarts at 181, 1090, 1554 and 3130 -- its first 3 frames there have a short
history; the sky dumps are exact.

**Every pixel number is quoted at 960 wide**, the analog width `run_all.py`'s were tuned
at, and scaled to `--width` (`--ref-width 960` in `BASE`). So `--merge 30` is 129 px at 4128,
the same fraction of the picture as on analog. Until 2026-10-09 they were absolute pixels,
which made merge 20 and 30 identical here and the speed ceiling 4x stricter than on analog.
The outputs from then, under `runs/field/exp031_combined/raw<W>/`, are not reproducible by
this driver. Its own go to `runs/field/exp032_drone_selector/raw<W>/<a>_<b>/`, and `full/`
for `--full`. The sky dumps are not affected (the sky branch always scaled), so an existing
one under the old folder is read in place rather than rebuilt; the window dumps are rebuilt.

Every step is skipped when its output exists, so a killed batch resumes.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2

sys.path.insert(0, "experiments/exp031_combined")
from run_all import BASE, VARIANTS  # noqa: E402

EPISODES = ((2, 180), (1090, 1553), (3130, 3600))
GAPS = ((181, 1089), (1554, 3129))
FULL = (2, 3600)
SKY_CHUNK = 60
ANALOG_WIDTH = 960   # BASE's pixel numbers are quoted here (`--ref-width 960`), and so is:
APPEAR_RADIUS = 9.0  # the window's seed radius
ROOT = "runs/field/exp032_drone_selector"
SKY_FROM = "runs/field/exp031_combined"  # sky dumps from before 2026-10-09, still exact
PLAY_HEIGHT = 1080
CFG = "experiments/exp031_combined/field_raw"
TAIL = "experiments/exp017_motion_first;experiments/exp015_normalised_motion;."
STACKS = {
    "sky": f"{CFG};experiments/exp023_sky_branch;{TAIL}",
    "window": f"{CFG};experiments/exp024_window_length;{TAIL}",
    "split": f"{CFG};experiments/exp025_top3;experiments/exp023_sky_branch;{TAIL}",
}


def step(stack: str, args: list[str], done: str, log: str, width: int) -> None:
    """One module run, skipped when `done` exists; output to `log`, unbuffered."""
    if os.path.exists(done):
        print(f"have {done}", flush=True)
        return
    env = dict(os.environ, PYTHONPATH=STACKS[stack], FIELD_RAW_WIDTH=str(width),
               PYTHONUNBUFFERED="1")
    print(f"running -> {log}", flush=True)
    with open(log, "w", encoding="utf-8") as fh:
        subprocess.run([sys.executable, "-m", *args], env=env, stdout=fh,
                       stderr=subprocess.STDOUT, check=True)


def playable(video: str, height: int = PLAY_HEIGHT) -> None:
    """An H.264 copy of `video` at `height` px, `<stem>_<height>p.mp4`, skipped when present.

    `overlay_split` writes `mp4v`, which Windows players will not decode at 4128x3008 (the
    files are intact: OpenCV reads every frame). Windows' own Media Foundation encoder
    writes H.264 without an OpenH264 or ffmpeg install; 1080 rows keeps it in H.264's
    common levels. `INTER_AREA` keeps the circles and the caption legible."""
    out = f"{os.path.splitext(video)[0]}_{height}p.mp4"
    if os.path.exists(out):
        print(f"have {out}", flush=True)
        return
    cap = cv2.VideoCapture(video)
    w0, h0 = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    size = (int(round(w0 * height / h0 / 2)) * 2, height)
    writer = cv2.VideoWriter(out + ".tmp.mp4", cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*"H264"),
                             cap.get(cv2.CAP_PROP_FPS), size)
    if not writer.isOpened():
        raise RuntimeError(f"no H.264 writer for {out}")
    while True:
        ok, img = cap.read()
        if not ok:
            break
        writer.write(cv2.resize(img, size, interpolation=cv2.INTER_AREA))
    cap.release()
    writer.release()
    os.replace(out + ".tmp.mp4", out)
    print(f"wrote {out}", flush=True)


def sky_dump(width: int, name: str, own: str) -> str:
    """An existing sky dump from the EXP-031 runs when there is one, else `own`."""
    old = f"{SKY_FROM}/raw{width}/{name}/sky_candidates.csv"
    return old if os.path.exists(old) else own


def chunks(a: int, b: int) -> list[tuple[int, int]]:
    return [(s, min(s + SKY_CHUNK - 1, b)) for s in range(a, b + 1, SKY_CHUNK)]


def join(parts: list[str], dst: str) -> None:
    """CSV parts into one file, the header kept once; the parts are left in place."""
    with open(dst + ".tmp", "w", encoding="utf-8", newline="") as out:
        for i, p in enumerate(parts):
            with open(p, encoding="utf-8", newline="") as src:
                lines = src.readlines()
            out.writelines(lines if i == 0 else lines[1:])
    os.replace(dst + ".tmp", dst)


def split(span: tuple[int, int], merge: int, sky: str, seeds: str, out: str, width: int,
          keep: bool, rank_by: str) -> None:
    """Step 3 and 4 for one span and radius: the `all` overlay, then its playable copy."""
    step("split", ["overlay_split", "--start", str(span[0]), "--end", str(span[1]), *BASE,
                   "--merge", str(merge), *VARIANTS["all"], "--rank-by", rank_by,
                   "--sky-dump", sky, "--window-dump", seeds, "--out", out + ".mp4"],
         out + "_frames.csv", out + ".log", width)
    playable(out + ".mp4")
    if not keep and os.path.exists(out + ".mp4"):
        os.remove(out + ".mp4")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width", type=int, default=4128)
    ap.add_argument("--merge", type=int, nargs="+", default=[30],
                    help="merge radius, px at 960 wide (129 px at 4128 for 30)")
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--spans", choices=("episodes", "all"), default="episodes")
    ap.add_argument("--full", action="store_true",
                    help="one overlay over 2-3600 from every span's dumps (implies --spans all)")
    ap.add_argument("--keep-full-size", action="store_true",
                    help="keep the 4128-wide mp4v next to its _1080p copy")
    ap.add_argument("--rank-by", choices=("track", "drone"), default="track",
                    help="EXP-031's track evidence, or EXP-032's drone selector")
    a = ap.parse_args()
    spans = sorted(EPISODES + GAPS) if a.spans == "all" or a.full else list(EPISODES)
    root = f"{ROOT}/raw{a.width}"
    dirs = {s: f"{root}/{s[0]}_{s[1]}/" for s in spans}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    sky = {s: sky_dump(a.width, f"{s[0]}_{s[1]}", f"{dirs[s]}sky_candidates.csv")
           for s in spans}
    seeds = {s: f"{dirs[s]}window_seeds_k4.csv" for s in spans}
    appear = APPEAR_RADIUS * a.width / ANALOG_WIDTH
    tag = "all_drone" if a.rank_by == "drone" else "all"

    # Windows first: one long job per span, the sky chunks fill the pool around them.
    jobs = [("window", ["overlay_window", "--start", str(a0), "--end", str(b0), "--k", "4",
                        "--min-appear", "2", "--appear-radius", f"{appear:g}", "--no-video",
                        "--dump", seeds[s]], seeds[s], f"{dirs[s]}window.log")
            for s in spans if not os.path.exists(seeds[s]) for a0, b0 in [s]]
    parts = {s: [] for s in spans}
    for s in spans:
        if os.path.exists(sky[s]):
            continue
        for c0, c1 in chunks(*s):
            p = f"{sky[s]}.part{c0}"
            parts[s].append(p)
            jobs.append(("sky", ["overlay_sky", "--start", str(c0), "--end", str(c1),
                                 "--no-video", "--dump", p], p, f"{dirs[s]}sky_{c0}_{c1}.log"))
    with ThreadPoolExecutor(a.jobs) as pool:
        for f in [pool.submit(step, *j, a.width) for j in jobs]:
            f.result()
    for s in spans:
        if parts[s]:
            join(parts[s], sky[s])
            for p in parts[s]:
                os.remove(p)

    if a.full:
        d = f"{root}/full/"
        os.makedirs(d, exist_ok=True)
        full_sky = sky_dump(a.width, "full", d + "sky_candidates.csv")
        for name, per in ((full_sky, sky), (d + "window_seeds_k4.csv", seeds)):
            if not os.path.exists(name):
                join([per[s] for s in spans], name)
        for m in a.merge:
            split(FULL, m, full_sky, d + "window_seeds_k4.csv",
                  f"{d}{tag}_merge{m}_{FULL[0]}_{FULL[1]}", a.width, a.keep_full_size,
                  a.rank_by)
    else:
        for s in spans:
            for m in a.merge:
                split(s, m, sky[s], seeds[s], f"{dirs[s]}{tag}_merge{m}_{s[0]}_{s[1]}",
                      a.width, a.keep_full_size, a.rank_by)
    print(f"done -> {root}", flush=True)


if __name__ == "__main__":
    main()
