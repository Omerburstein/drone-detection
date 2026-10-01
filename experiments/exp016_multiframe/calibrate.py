"""EXP-016: choose z* on empty frames only, two-fold, and freeze it before any drone frame is scored.

The ladder passes (`verify --zstar Z --light`) run over the whole clip because a tracker
needs consecutive frames, but **nothing here reads a drone frame**: a confirmed track
counts only if the frame it confirmed on carries no label. The empty frames are split into
two contiguous halves; z* is tuned on one and the rate is reported on the other, both ways
round. The frozen value is the more conservative of the two, so neither half can have been
the lucky one.

Selection rule, fixed in advance: the **smallest** ladder value whose false-track rate on
the tuning fold is at or under the budget -- most sensitive threshold that still fits.

    py -3.13 -m calibrate --budget 5
"""
import argparse
import pickle

import numpy as np

import mf
from mf import CLIP, OUT

LADDER = (4, 5, 6, 8, 12, 16, 20, 25, 30, 40, 60)


def folds():
    e = sorted(mf.EMPTY_FRAMES)
    half = len(e) // 2
    return set(e[:half]), set(e[half:])


def rate(tracks: dict, fold: set) -> tuple[int, float]:
    n = sum(1 for tr in tracks.values()
            if int(tr["confirmed"]) in fold and int(tr["confirmed"]) not in mf.BOXES)
    minutes = len(fold) / CLIP["fps"] / 60.0
    return n, n / minutes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=5.0, help="false tracks per minute")
    a = ap.parse_args()
    A, B = folds()
    print(f"{CLIP['name']}: empty frames {len(mf.EMPTY_FRAMES)}  "
          f"fold A {min(A)}-{max(A)} ({len(A)})  fold B {min(B)}-{max(B)} ({len(B)})")
    table = {}
    print(f"\n{'z*':>5} {'A n':>5} {'A /min':>8} {'B n':>5} {'B /min':>8}")
    for z in LADDER:
        try:
            res = pickle.load(open(OUT + f"verify_z{z:g}.pkl", "rb"))
        except FileNotFoundError:
            continue
        na, ra = rate(res["tracks"], A)
        nb, rb = rate(res["tracks"], B)
        table[z] = (ra, rb)
        print(f"{z:5g} {na:5d} {ra:8.1f} {nb:5d} {rb:8.1f}")
    if not table:
        print("no ladder passes found")
        return
    pick = {}
    for name, idx, other in (("A", 0, 1), ("B", 1, 0)):
        ok = [z for z in sorted(table) if table[z][idx] <= a.budget]
        if ok:
            pick[name] = ok[0]
            print(f"\ntuned on fold {name}: z* = {ok[0]:g} "
                  f"({table[ok[0]][idx]:.1f}/min in-fold) -> held-out fold gives "
                  f"**{table[ok[0]][other]:.1f} false tracks/min**")
        else:
            print(f"\ntuned on fold {name}: no ladder value meets {a.budget}/min; "
                  f"best is z* = {max(table):g} at {table[max(table)][idx]:.1f}/min")
            pick[name] = max(table)
    frozen = max(pick.values())
    print(f"\nFROZEN z* = {frozen:g}   (max of the two folds' choices)")
    with open(OUT + "zstar.txt", "w") as fh:
        fh.write(f"{frozen}\n")


if __name__ == "__main__":
    main()
