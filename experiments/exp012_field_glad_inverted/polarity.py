"""Is a detection darker or brighter than what surrounds it?

`scene_stats.measure_frame` takes `abs()` of the box-minus-ring contrast, which
is right for a lighting axis and wrong for this question: the sign is the whole
point. Reuses its `_crop` / `_ring_mean` so the annulus is defined identically.

    py -3.13 polarity.py <detections.jsonl> <video.mp4> [label]
"""
import json
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, ".")
from src.data.scene_stats import _crop, _ring_mean  # noqa: E402


def rows(pred):
    out = []
    with open(pred, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            idx = int(r["image"].rsplit("_", 1)[-1].split(".")[0])
            for d in r["detections"]:
                out.append((idx, d["bbox"], r["branch"]))
    return out


def measure(pred, video, label):
    found = rows(pred)
    want = {i for i, _, _ in found}
    cap, i, per_frame = cv2.VideoCapture(video), 0, defaultdict(list)
    for idx, bbox, branch in found:
        per_frame[idx].append((bbox, branch))

    recs = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i not in want:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        for bbox, branch in per_frame[i]:
            ring = _ring_mean(gray, np.array(bbox, dtype=float))
            if ring is None:
                continue
            inside = float(_crop(gray, np.array(bbox, dtype=float)).mean())
            recs.append((i, branch, inside, ring, inside - ring))
        if i > max(want):
            break
    cap.release()

    print(f"\n=== {label} — {len(recs)} detections measured ===")
    bright = [r for r in recs if r[4] > 0]
    dark = [r for r in recs if r[4] <= 0]
    print(f"  brighter than background : {len(bright):>4}  ({100*len(bright)/len(recs):.1f}%)")
    print(f"  darker  than background : {len(dark):>4}  ({100*len(dark)/len(recs):.1f}%)")

    for name, group in (("brighter", bright), ("darker", dark)):
        if not group:
            continue
        inside = np.array([r[2] for r in group])
        ring = np.array([r[3] for r in group])
        delta = np.array([abs(r[4]) for r in group])
        print(f"  {name:<9} box mean {inside.mean():6.1f}  ring mean {ring.mean():6.1f}  "
              f"|contrast| {delta.mean():5.1f}  (box p10 {np.percentile(inside,10):.0f}, "
              f"p90 {np.percentile(inside,90):.0f})")
    return recs


def split_by_span(recs, spans, name):
    """Same measurement, restricted to frame ranges."""
    sel = [r for r in recs if any(lo <= r[0] <= hi for lo, hi in spans)]
    if not sel:
        return
    bright = sum(1 for r in sel if r[4] > 0)
    inside = np.array([r[2] for r in sel])
    print(f"    {name:<28} {len(sel):>4} dets, {100*bright/len(sel):5.1f}% brighter, "
          f"box mean {inside.mean():6.1f}")


if __name__ == "__main__":
    recs = measure(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "run")
    print("  by region:")
    split_by_span(recs, [(1101, 1553), (3140, 3600)], "the two sky episodes")
    split_by_span(recs, [(1833, 2464)], "EXP-011's clutter block")
    split_by_span(recs, [(1555,1578),(1625,1644),(2008,2070),(1823,1830)],
                  "EXP-012's new regions")
    split_by_span(recs, [(12, 176)], "EXP-010's bush lock")
