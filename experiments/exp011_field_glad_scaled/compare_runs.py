"""EXP-010 (native) against EXP-011 (--scale auto), on unlabelled footage.

Neither run is scored, so nothing here is accuracy. What it compares is what the
detector *did*: which branch handled each frame, how many boxes came out, and
where in the video they clustered.
"""
import json
import sys
from collections import Counter

FRAMES = 3600
DRIFT = (19, 101, 176)          # box on a bush while the drone is visible
EPISODES = ((2, 176), (1101, 1553), (3140, 3600))   # EXP-010's three lock-ons


def load(path):
    by_frame = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            idx = int(row["image"].rsplit("_", 1)[-1].split(".")[0])
            by_frame[idx] = row
    return by_frame


def episodes(by_frame, gap=15):
    """Contiguous runs of detecting frames, merging gaps shorter than `gap`."""
    hits = sorted(i for i, r in by_frame.items() if r["detections"])
    if not hits:
        return []
    runs, start, prev = [], hits[0], hits[0]
    for i in hits[1:]:
        if i - prev > gap:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    return runs


def summarise(name, by_frame):
    n = len(by_frame)
    dets = sum(len(r["detections"]) for r in by_frame.values())
    empty = sum(1 for r in by_frame.values() if not r["detections"])
    print(f"\n=== {name} ===")
    print(f"  frames recorded    {n}")
    print(f"  detections         {dets}  ({dets / max(n,1):.3f} per frame)")
    print(f"  empty frames       {empty}  ({100*empty/max(n,1):.1f}%)")
    runs = episodes(by_frame)
    covered = sum(b - a + 1 for a, b in runs)
    print(f"  episodes (gap<=15) {len(runs)}, covering {covered} frames "
          f"({100*covered/max(n,1):.1f}%)")
    for a, b in runs[:12]:
        inside = sum(1 for i in range(a, b + 1) if by_frame.get(i, {}).get("detections"))
        print(f"      {a:>5}-{b:<5} {b-a+1:>5} frames, {inside:>4} with a box")
    if len(runs) > 12:
        print(f"      ... and {len(runs)-12} more")
    return Counter(r["branch"] for r in by_frame.values()), dets, empty


def main(native_path, scaled_path):
    native, scaled = load(native_path), load(scaled_path)
    bn, dn, en = summarise("EXP-010  native 1032x752", native)
    bs, ds, es = summarise("EXP-011  --scale auto -> 1780x1297", scaled)

    print("\n=== branch mix ===")
    tn, ts = max(sum(bn.values()), 1), max(sum(bs.values()), 1)
    print(f"{'branch':<14}{'EXP-010':>9}{'':>3}{'share':>7}{'EXP-011':>10}{'':>3}{'share':>7}")
    for b in sorted(set(bn) | set(bs), key=lambda k: -bn.get(k, 0)):
        print(f"{b:<14}{bn.get(b,0):>9}{'':>3}{100*bn.get(b,0)/tn:>6.1f}%"
              f"{bs.get(b,0):>10}{'':>3}{100*bs.get(b,0)/ts:>6.1f}%")

    print("\n=== the three drift frames (box on a bush, drone visible) ===")
    for i in DRIFT:
        def show(d):
            r = d.get(i)
            if r is None:
                return "not recorded"
            if not r["detections"]:
                return f"no box ({r['branch']})"
            x1, y1, x2, y2 = r["detections"][0]["bbox"]
            return (f"box ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f}) "
                    f"{x2-x1:.0f}x{y2-y1:.0f} [{r['branch']}]")
        print(f"  frame {i:<5} EXP-010: {show(native):<46} EXP-011: {show(scaled)}")

    print("\n=== detections inside EXP-010's episodes ===")
    print(f"{'span':<16}{'EXP-010':>9}{'EXP-011':>10}")
    for a, b in EPISODES:
        cn = sum(len(native.get(i, {}).get("detections", [])) for i in range(a, b + 1))
        cs = sum(len(scaled.get(i, {}).get("detections", [])) for i in range(a, b + 1))
        print(f"{f'{a}-{b}':<16}{cn:>9}{cs:>10}")
    outside_n = dn - sum(sum(len(native.get(i, {}).get("detections", []))
                             for i in range(a, b + 1)) for a, b in EPISODES)
    outside_s = ds - sum(sum(len(scaled.get(i, {}).get("detections", []))
                             for i in range(a, b + 1)) for a, b in EPISODES)
    print(f"{'outside':<16}{outside_n:>9}{outside_s:>10}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
