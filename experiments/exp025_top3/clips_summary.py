"""EXP-025e: one Markdown row per clip from `run_clips.py`'s split logs.

    py -3.13 experiments/exp025_top3/clips_summary.py

Reads each clip's `split_<clip>.log` (the report `overlay_split.py` prints), so it adds no
number of its own. Drone columns are blank on unlabelled clips.
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, "experiments/exp025_top3/clips")
from clipcfg import CLIPS  # noqa: E402


def parse(path: str) -> dict:
    text = open(path, encoding="utf-8").read()
    row = {"frames": re.search(r"\[video\] (\d+) frames", text).group(1)}
    m = re.search(r"sky fraction: median (\d+)%, no sky in (\d+) frames", text)
    row["sky"], row["skyless"] = m.groups()
    for key in ("ranked", "shown"):
        row[key] = re.search(rf"{key}/frame\s+sky\s+([\d.]+)\s+ground\s+([\d.]+)\s+total\s+"
                             r"([\d.]+)", text).groups()
    row["labelled"] = int(re.search(r"drone, of (\d+) labelled", text).group(1))
    for key, pat in (("top3", r"in the top 3"), ("1", r"#1"), ("2", r"#2"), ("3", r"#3"),
                     ("dropped", r"dropped by its section"), ("noblob", r"no blob on it")):
        row[key] = re.search(rf"{pat}\s+(\d+)\s+(\d+)\s+(\d+)", text).groups()
    return row


def main() -> None:
    print("| clip | frames | sky, median / skyless frames | ranked/frame (sky + ground) "
          "| shown/frame | drone in top 3 | #1 / #2 / #3 | dropped (sky, ground) |")
    print("| --- | ---: | :---: | :---: | ---: | ---: | :---: | :---: |")
    for key, c in CLIPS.items():
        log = os.path.join(c["out"], f"split_{key}.log")
        if not os.path.exists(log):
            print(f"| {key} | not run | | | | | | |")
            continue
        r = parse(log)
        drone = ("| | |" if not r["labelled"] else
                 f"| {r['top3'][2]} of {r['labelled']} | {r['1'][2]} / {r['2'][2]} / "
                 f"{r['3'][2]} | {r['dropped'][0]}, {r['dropped'][1]} |")
        print(f"| {key} | {r['frames']} | {r['sky']}% / {r['skyless']} | {r['ranked'][2]} "
              f"({r['ranked'][0]} + {r['ranked'][1]}) | {r['shown'][2]} {drone}")


if __name__ == "__main__":
    main()
