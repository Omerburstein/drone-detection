"""Build the structural prop and ladder masks over EVERY clip from one airframe.

Run from the repo root, e.g.

    PYTHONPATH="experiments/exp017_motion_first;." py -3.13 -m calibrate_masks
    PYTHONPATH="runs/sofa_analog/exp017_motion_first;experiments/exp017_motion_first;." \
        py -3.13 -m calibrate_masks

Why over every clip, and not over one clip's empty frames
---------------------------------------------------------
EXP-015's screen-fixed map is fitted on the same clip it is applied to, so it can and does
absorb scene content: on analog it masks a patch of tree line the drone later crosses
(EXP-016, frame 585). Calibrating across clips removes that. Every statistic below is
combined across clips with a **minimum**, so a pixel must behave the same way in *all* of
them to qualify. The scene differs between clips and the airframe does not, which is the
entire discriminant -- and no clip's drone can survive an intersection with clips it does
not appear in.

Two rules, and both need a shape prior as well as a statistic
-------------------------------------------------------------
A first attempt used statistics alone (departure from a clip's temporal median for the
props, near-white frequency for the ladder) and produced masks covering 46% and 97% of the
frame. Both failed the same way: over a whole clip the camera moves, so *every* textured
pixel is restless and a bright sky is near-white as often as a glyph. The statistic says
"something happens here", which is true nearly everywhere.

So each rule now pairs a short-timescale statistic with the structural prior that actually
identifies the thing:

  * **props** -- blades enter from the left and right picture edges and are attached to the
    airframe, so only components *touching a side edge* are kept. Statistic: flicker
    between consecutive frames, which a blade chopping at rotor rate produces and a scene
    sliding under a moving camera does not.
  * **ladder** -- glyphs are thin bright marks on a darker background, so the statistic is
    a white top-hat response (bright *relative to its surroundings*), which a large bright
    sky region cannot produce. A column's swept span is then filled, capped so one stray
    row cannot swallow the column.

Both masks are written as .npy and both are drawn by `debug_video.py`. Look at them before
trusting them.
"""
from __future__ import annotations

import glob

import cv2
import numpy as np

from clipcfg import CLIP

W, H = CLIP["width"], CLIP["height"]
S = W / 1440.0

BURSTS = 24                # places in the clip to sample from
BURST_LEN = 6              # consecutive frames per burst -> BURSTS*(BURST_LEN-1) pairs

# --- prop rule ---
PROP_PCTL = 95.5           # keep the top slice of the cross-clip flicker floor
PROP_EDGE_FRAC = 0.22      # a component must reach within this fraction of a side edge
PROP_MIN_AREA = int(round((40 * S) ** 2))
PROP_VERT_BAND = 0.12      # a component reaching this close to the top or bottom is scene
PROP_DILATE = max(3, int(round(11 * S)) | 1)

# --- ladder rule ---
TOPHAT = max(3, int(round(9 * S)) | 1)     # glyph stroke width scale
LADDER_TOPHAT_MIN = 45     # grey levels brighter than the local surround
LADDER_FRAC = 0.15         # ... in this share of sampled frames, in EVERY clip
LADDER_MIN_ROWS = 4
LADDER_MAX_SPAN = 0.35     # a cluster's filled span may not exceed this fraction of height
LADDER_GAP = 0.05          # lit rows further apart than this belong to different elements
LADDER_DILATE = max(3, int(round(7 * S)) | 1)


def bursts(path: str) -> list[np.ndarray]:
    """Short runs of consecutive frames, spread through the clip."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    out = []
    for start in np.linspace(0, max(total - BURST_LEN - 1, 0), BURSTS).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
        run = []
        for _ in range(BURST_LEN):
            ok, f = cap.read()
            if not ok or f.shape[:2] != (H, W):
                break
            run.append(f)
        if len(run) == BURST_LEN:
            out.append(np.asarray(run, np.uint8))
    cap.release()
    return out


def flicker(runs: list[np.ndarray]) -> np.ndarray:
    """Median |f_t - f_t+1| per pixel, normalised by this clip's own median.

    Normalising makes the map comparable between clips shot at different exposures, so the
    cross-clip minimum below compares like with like.
    """
    diffs = []
    for run in runs:
        g = np.asarray([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in run], np.int16)
        diffs.append(np.abs(np.diff(g, axis=0)).astype(np.float32))
    d = np.median(np.concatenate(diffs, axis=0), axis=0)
    return d / max(float(np.median(d)), 1e-3)


def tophat_freq(runs: list[np.ndarray]) -> np.ndarray:
    """Share of sampled frames where a pixel is much brighter than its local surround."""
    k = np.ones((TOPHAT, TOPHAT), np.uint8)
    hits = []
    for run in runs:
        for f in run:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            hits.append(cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k) >= LADDER_TOPHAT_MIN)
    return np.asarray(hits, np.float32).mean(axis=0)


def edge_touching(mask: np.ndarray) -> np.ndarray:
    """Keep the components that look like a blade: in from a side, not off the top or bottom.

    The side test alone is not enough. Near ground at the bottom of the frame flickers
    hard -- it is the fastest-moving thing in the picture -- and its component reaches the
    left edge too, so at a lower threshold it was being masked as a propeller. A blade
    enters from a side and stays in the picture; ground runs off the bottom. Rejecting
    components that touch the top or bottom row separates them without a height constant.
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    band = int(PROP_EDGE_FRAC * W)
    vband = int(PROP_VERT_BAND * H)
    out = np.zeros_like(mask)
    for i in range(1, n):
        x, y, w_, h_ = stats[i, :4]
        if stats[i, cv2.CC_STAT_AREA] < PROP_MIN_AREA:
            continue
        if y <= vband or (y + h_) >= (H - vband):   # runs off the top or bottom: scene
            continue
        if x <= band or (x + w_) >= (W - band):
            out[lab == i] = True
    return out


def sweep_span(lit: np.ndarray) -> np.ndarray:
    """Fill each column between lit rows, but only within one cluster of them.

    A global topmost-to-bottommost fill is wrong wherever the overlay is not a single
    swept element. On analog it bridged a top OSD row to a bottom telemetry row and buried
    a third of the frame -- including the drone, from frame 491, which is the acquisition
    frame the whole experiment is judged on. The verification pass caught it; that is what
    the verification pass is for.

    So lit rows are split into clusters separated by more than `LADDER_GAP`, and each
    cluster is filled only across its own extent. A ladder that genuinely sweeps has its
    positions close together and still fills; two unrelated glyph rows do not merge.
    """
    out = np.zeros_like(lit, bool)
    cap = int(LADDER_MAX_SPAN * H)
    gap = max(2, int(round(LADDER_GAP * H)))
    for x in range(lit.shape[1]):
        rows = np.nonzero(lit[:, x])[0]
        if len(rows) < LADDER_MIN_ROWS:
            continue
        for grp in np.split(rows, np.nonzero(np.diff(rows) > gap)[0] + 1):
            if len(grp) < LADDER_MIN_ROWS:
                continue
            lo, hi = int(grp.min()), int(grp.max())
            if hi - lo > cap:
                lo, hi = int(np.percentile(grp, 5)), int(np.percentile(grp, 95))
            out[lo:hi + 1, x] = True
    return out


def main() -> None:
    paths = sorted(glob.glob(CLIP["source_videos"]))
    if not paths:
        raise SystemExit(f"no clips matched {CLIP['source_videos']!r}")
    print(f"calibrating over {len(paths)} clips from this airframe")

    flick_min = np.full((H, W), np.inf, np.float32)
    lit_min = np.ones((H, W), np.float32)
    used = 0
    for p in paths:
        runs = bursts(p)
        if len(runs) < 4:
            print(f"  [skip] {p}: only {len(runs)} usable bursts at {(H, W)}")
            continue
        flick_min = np.minimum(flick_min, flicker(runs))
        lit_min = np.minimum(lit_min, tophat_freq(runs))
        used += 1
        print(f"  {p}: {len(runs)} bursts")
    if used < 2:
        raise SystemExit("need at least 2 clips; a single clip is the in-sample case again")

    thr = np.percentile(flick_min, PROP_PCTL)
    prop = edge_touching(flick_min >= thr)
    prop = cv2.dilate(prop.astype(np.uint8), np.ones((PROP_DILATE,) * 2, np.uint8)) > 0

    ladder = sweep_span(lit_min >= LADDER_FRAC)
    ladder = cv2.dilate(ladder.astype(np.uint8), np.ones((LADDER_DILATE,) * 2, np.uint8)) > 0

    np.save(CLIP["prop_mask"], prop)
    np.save(CLIP["ladder_mask"], ladder)
    print(f"\nprop mask   {prop.mean() * 100:5.2f}% of frame  (flicker floor >= {thr:.2f}x median)")
    print(f"ladder mask {ladder.mean() * 100:5.2f}% of frame")
    print(f"union       {(prop | ladder).mean() * 100:5.2f}%   (calibrated over {used} clips)")
    print("\nLook at these in debug_video.py before trusting either.")


if __name__ == "__main__":
    main()
