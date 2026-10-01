"""What does the sky branch actually see at the labelled target? A diagnostic, not a run.

    PYTHONPATH="experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
        py -3.13 -m probe_sky --start 700 --end 760 --every 10

`overlay_stage2` reports that the drone was or was not detected. When it was not, that can
mean the contrast is genuinely low, or that the peak was there and something upstream threw
it away -- the candidate cap, the NMS radius, a stage-0 mask, or the ring refusing. Those
have completely different fixes, and guessing between them is how a tuning session turns
into a week. This prints all four for each labelled frame.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

import common
import masks
import silhouette
import skyline
from clipcfg import CLIP
from overlay_video import load_boxes

W, H = CLIP["width"], CLIP["height"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=700)
    ap.add_argument("--end", type=int, default=760)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--cap", type=int, default=600)
    a = ap.parse_args()

    stage0 = masks.Stage0()
    boxes = load_boxes()
    cap = cv2.VideoCapture(CLIP["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.start - 2)
    ok, prev = cap.read()
    if not ok:
        raise SystemExit("cannot read")

    print(f"{'frame':>6} {'box':>18} {'sz':>5} {'label':>9} {'masked':>7} "
          f"{'resp':>7} {'rank':>6} {'sigma':>6} {'c':>7} {'c_sky':>7}  note")
    for f in range(a.start - 1, a.end + 1):
        ok, cur = cap.read()
        if not ok:
            break
        if f < a.start or f % a.every or f not in boxes:
            prev = cur
            continue
        box = boxes[f]
        bx, by, bw, bh = [float(v) for v in box]
        cx, cy = bx + bw / 2.0, by + bh / 2.0
        hmat = common.homography(common.prep(prev, 11), common.prep(cur, 11))
        valid = stage0.valid(hmat)
        sl = skyline.split(cur, valid)

        grey = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
        resp, sig_map, blurs = silhouette.dog_pyramid(grey)
        sigmas = silhouette.scale_ladder()

        # The strongest response anywhere inside the labelled box, and where it is.
        x0, x1 = max(0, int(bx)), min(W, int(bx + bw) + 1)
        y0, y1 = max(0, int(by)), min(H, int(by + bh) + 1)
        sub = resp[y0:y1, x0:x1]
        if sub.size == 0:
            prev = cur
            continue
        iy, ix = np.unravel_index(int(np.argmax(sub)), sub.shape)
        px_, py_ = x0 + ix, y0 + iy
        r_here = float(resp[py_, px_])
        s_here = float(sig_map[py_, px_])

        # Rank of that response among ALL valid local maxima: is the cap throwing it away?
        allpk = silhouette._local_maxima(resp, valid, int(round(silhouette.px(9.0))),
                                         0.0, 10 ** 7)
        rank = int((allpk[:, 0] > r_here).sum()) + 1 if len(allpk) else -1

        core_src = blurs[int(np.argmin(np.abs(sigmas - s_here)))]
        got = silhouette.contrast_at(core_src, grey.astype(np.float32),
                                     float(px_), float(py_), s_here, None)
        sky_only = silhouette.contrast_at(core_src, grey.astype(np.float32),
                                          float(px_), float(py_), s_here, sl.sky | sl.uncertain)
        note = []
        if not valid[py_, px_]:
            note.append("MASKED by stage 0")
        if rank > a.cap:
            note.append(f"CUT BY CAP (rank {rank} > {a.cap})")
        if got is None:
            note.append("ring unusable")
        print(f"{f:>6} {f'{bx:.0f},{by:.0f} {bw:.0f}x{bh:.0f}':>18} "
              f"{max(bw, bh):>5.0f} {sl.label(cx, cy):>9} "
              f"{'yes' if not valid[py_, px_] else 'no':>7} "
              f"{r_here:>7.2f} {rank:>6} "
              f"{(got[3] if got else float('nan')):>6.2f} "
              f"{(got[0] if got else float('nan')):>7.1f} "
              f"{(sky_only[0] if sky_only else float('nan')):>7.1f}  "
              f"{'; '.join(note)}")
        prev = cur
    cap.release()


if __name__ == "__main__":
    main()
