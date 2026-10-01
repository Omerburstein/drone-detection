"""EXP-016 scoring, against the bars fixed in the plan before anything was run.

Metrics, all on one `verify_*.pkl`:
  * **acquisition latency** -- frames from a span's first labelled frame to the first
    confirmed track sitting on the drone;
  * **coverage** -- labelled frames in a stretch with a confirmed track on the drone;
  * **fragments** -- distinct confirmed track ids that were ever on the drone;
  * **false tracks per minute** -- confirmed tracks whose confirming frame carries no
    label, over the clip's empty-frame duration;
  * **top-1 / top-5 / top-10 by z** -- where the best on-drone hypothesis ranks among all
    live hypotheses, which is the figure directly comparable with EXP-015's peak ranks.

Match criterion, unchanged from EXP-015: the position lies inside the label box grown by
max(10 px, 25% of its longest side).
"""
import argparse
import pickle

import numpy as np

import common
import mf
from mf import CLIP, OUT


def on_drone_at(trail: np.ndarray, n: int):
    row = trail[trail[:, 0] == n]
    if not len(row):
        return False
    return mf.on_drone(float(row[0, 1]), float(row[0, 2]), n)


def summarise(res: dict, tag: str):
    tracks = res["tracks"]
    ranks = res["ranks"]
    zstar = res["zstar"]
    print(f"\n=== {CLIP['name']}   {tag}   z* = {zstar:g} ===")
    print(f"hypotheses born {res['n_hyp']}, confirmed tracks {len(tracks)}, "
          f"OSD-twin vetoes {res['twin_vetoes']}")
    f, t = res["fb"]
    print(f"hypothesis forward-backward failure rate {f}/{t} = {f / max(t, 1):.4f} "
          f"(gate {mf.FB_MAX:.2f} px)")

    # --- which tracks are the drone's -------------------------------------
    drone_tracks, false_tracks = {}, []
    for hid, tr in tracks.items():
        frames = [int(n) for n in tr["trail"][:, 0]]
        hit = [n for n in frames if on_drone_at(tr["trail"], n)]
        if hit:
            drone_tracks[hid] = (tr, hit)
        if int(tr["confirmed"]) not in mf.BOXES:
            false_tracks.append(tr)

    empty_minutes = len(mf.EMPTY_FRAMES) / CLIP["fps"] / 60.0
    print(f"\nempty frames {len(mf.EMPTY_FRAMES)} = {empty_minutes * 60:.1f} s; "
          f"confirmed false tracks {len(false_tracks)} = "
          f"**{len(false_tracks) / empty_minutes:.1f} per minute**")

    # --- acquisition latency per span -------------------------------------
    print("\nacquisition, per labelled span:")
    covered_any = set()
    for lo, hi in mf.label_spans():
        first = None
        for hid, (tr, hit) in drone_tracks.items():
            h = [n for n in hit if lo <= n <= hi]
            if h and (first is None or min(h) < first[0]):
                first = (min(h), hid)
        if first is None:
            print(f"  {lo:4d}-{hi:4d}  never confirmed on the drone")
        else:
            print(f"  {lo:4d}-{hi:4d}  first confirmed on-drone frame {first[0]} "
                  f"= +{first[0] - lo} frames ({(first[0] - lo) / CLIP['fps']:.2f} s), track #{first[1]}")
    for hid, (tr, hit) in drone_tracks.items():
        covered_any |= set(hit)

    # --- coverage and rank per stretch ------------------------------------
    print(f"\n{'stretch':>12} {'n':>4} {'coverage':>9} {'frags':>6} "
          f"{'top1':>6} {'top5':>6} {'top10':>6} {'medrank':>8} {'seen':>6}")
    rows = []
    for lo, hi in CLIP["stretches"]:
        fs = [f for f in mf.DRONE_FRAMES if lo <= f <= hi]
        if not fs:
            continue
        cov = np.mean([f in covered_any for f in fs])
        frags = len({hid for hid, (tr, hit) in drone_tracks.items()
                     if any(lo <= n <= hi for n in hit)})
        r = [ranks[f][0] for f in fs if f in ranks]
        seen = len(r) / len(fs)
        arr = np.array(r) if r else np.array([np.inf])
        row = dict(stretch=f"{lo}-{hi}", n=len(fs), coverage=cov, fragments=frags,
                   top1=np.mean(arr <= 1) * seen if r else 0.0,
                   top5=np.mean(arr <= 5) * seen if r else 0.0,
                   top10=np.mean(arr <= 10) * seen if r else 0.0,
                   median_rank=float(np.median(arr)) if r else np.inf, seen=seen)
        rows.append(row)
        print(f"{row['stretch']:>12} {row['n']:4d} {cov:9.2f} {frags:6d} "
              f"{row['top1']:6.2f} {row['top5']:6.2f} {row['top10']:6.2f} "
              f"{row['median_rank']:8.0f} {seen:6.2f}")
    print("  coverage = labelled frames with a CONFIRMED track on the drone")
    print("  top-k    = fraction of labelled frames where the best on-drone hypothesis "
          "(confirmed or not) ranks in the top k by z, over ALL live hypotheses")
    print("  seen     = fraction of labelled frames carrying any live hypothesis on the drone")

    # --- what the surviving false tracks are ------------------------------
    print("\nconfirmed false tracks, by where they died:")
    why = {}
    for tr in false_tracks:
        why[tr["why"]] = why.get(tr["why"], 0) + 1
    for k, v in sorted(why.items(), key=lambda kv: -kv[1]):
        print(f"  {k:14s} {v:4d}")
    if false_tracks:
        life = np.array([tr["end"] - tr["confirmed"] for tr in false_tracks])
        sz = np.array([tr["size"] for tr in false_tracks])
        print(f"  lifetime frames: median {np.median(life):.0f} p90 {np.percentile(life, 90):.0f} "
              f"max {life.max()}")
        print(f"  seed size px:    median {np.median(sz):.0f} p10 {np.percentile(sz, 10):.0f} "
              f"p90 {np.percentile(sz, 90):.0f}")
    return rows, drone_tracks, false_tracks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="frozen")
    a = ap.parse_args()
    res = pickle.load(open(OUT + f"verify_{a.tag}.pkl", "rb"))
    summarise(res, a.tag)


if __name__ == "__main__":
    main()
