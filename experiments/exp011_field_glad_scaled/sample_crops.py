"""Seeded random sample of a run's detections, as one contact sheet.

Replicates the protocol EXP-010 reported 23/24 from: sample N detections, crop a
zoomed window around each, and look at them. Precision *conditional on firing* is
the only direction measurable without labels -- it says nothing about recall.

    py -3.13 sample_crops.py <detections.jsonl> <video.mp4> <out.png> [N] [seed]
"""
import json
import random
import sys

import cv2
import numpy as np

SPAN = 96        # source pixels around the box centre
CELL = 192       # rendered cell edge
COLS = 6


def detections(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            idx = int(row["image"].rsplit("_", 1)[-1].split(".")[0])
            for d in row["detections"]:
                out.append((idx, d["bbox"], row["branch"]))
    return out


def main(pred, video, out_path, n=24, seed=0):
    picks = detections(pred)
    print(f"{len(picks)} detections in {pred}")
    if not picks:
        return
    rng = random.Random(seed)
    chosen = sorted(rng.sample(picks, min(n, len(picks))))

    cap = cv2.VideoCapture(video)
    cells = []
    for idx, bbox, branch in chosen:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx - 1)   # keys are one-based
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        left = int(max(0, min(w - 2 * SPAN, cx - SPAN)))
        top = int(max(0, min(h - 2 * SPAN, cy - SPAN)))
        crop = frame[top:top + 2 * SPAN, left:left + 2 * SPAN].copy()
        # the box, in crop coordinates
        cv2.rectangle(crop, (int(x1 - left), int(y1 - top)),
                      (int(x2 - left), int(y2 - top)), (0, 255, 255), 1)
        crop = cv2.resize(crop, (CELL, CELL), interpolation=cv2.INTER_NEAREST)
        cv2.putText(crop, f"{idx} {branch}", (3, CELL - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        cells.append(crop)
    cap.release()

    rows = []
    for i in range(0, len(cells), COLS):
        row = cells[i:i + COLS]
        while len(row) < COLS:
            row.append(np.zeros((CELL, CELL, 3), dtype=np.uint8))
        rows.append(np.hstack(row))
    cv2.imwrite(out_path, np.vstack(rows))
    print(f"wrote {out_path}: {len(cells)} crops, seed {seed}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2], int(a[3]) if len(a) > 3 else 24,
         int(a[4]) if len(a) > 4 else 0)
