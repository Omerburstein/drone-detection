"""EXP-016 verify: the local-ring differential-motion hypothesis test with sequential confirmation.

One pass over the clip at one value of z*. Every frame:

1. Advance every live hypothesis by forward-backward LK.
2. Take its **differential vector** -- its motion, minus the homography's prediction for
   static background at that point, minus the median residual of the grid tracks in its
   25-120 px ring. The parallax the homography cannot model is shared with the ring and
   cancels; the drone's own motion does not.
3. Accumulate:  z_k = |sum d| / sqrt(sum_j (1.48 * ring_MAD_j + 0.3 px)^2).
   With a constant ring this is exactly the plan's |sum d| / (sqrt(k) * sigma), and it
   degrades gracefully when the ring quality changes frame to frame.
4. Confirm at the first k >= 3 with z >= z* and |sum d| >= 6 px.
   Kill at k = 15 (the user's 0.5 s latency ceiling), after two forward-backward failures,
   or on the speed cap.
5. Veto an overlay: raw image motion < 0.5 px/f while the ring flows > 3 px/f means
   screen-fixed -- props, HUD, ladder. On analog, also veto an OSD twin.
6. Hold a confirmed track by LK with a template re-lock, and drop it when its trailing
   15-frame z has sat below z*/2 for 10 frames.

Seeds are blind: the top 200 peaks of `win_b5_e4` anywhere in the valid frame, with no
centre prior. The prior the user allowed after a first confirmation is not used here --
stage 1 is the acquisition question.

    PYTHONPATH="<clip dir>;experiments/exp016_multiframe;experiments/exp015_normalised_motion;." \
    py -3.13 -m verify --zstar 6.0 --tag frozen
"""
import argparse
import pickle
import time
from collections import deque

import cv2
import numpy as np

import common
import mf
from mf import CLIP, OUT, Hypothesis
from src.algo.masking import has_twin

HOLD_DROP_FRAMES = 10          # frames of weak trailing z before a confirmed track is dropped
STORE_TOP = 300                # hypotheses stored per frame for the overlay video


class Tracker:
    def __init__(self, zstar: float, light: bool = False):
        self.zstar = zstar
        self.light = light
        self.fixed = mf.screen_fixed()
        self.live: list[Hypothesis] = []
        self.next_id = 0
        self.tracks: dict[int, dict] = {}
        self.per_frame: dict[int, dict] = {}
        self.twin_vetoes = 0
        self.fb_fail_hyp = [0, 0]      # [failures, attempts] on hypothesis tracks
        self.windows: dict[int, deque] = {}
        self.weak: dict[int, int] = {}

    # --- spawning ---------------------------------------------------------
    def spawn(self, n: int, seeds: np.ndarray, valid: np.ndarray):
        if len(self.live):
            lp = np.array([[h.x, h.y] for h in self.live], np.float32)
            gate = np.array([max(mf.px(8.0), 0.5 * h.size) for h in self.live], np.float32)
        else:
            lp, gate = np.zeros((0, 2), np.float32), np.zeros(0, np.float32)
        for v, x, y, size in seeds:
            if not valid[int(y), int(x)]:
                continue
            if len(lp) and (np.hypot(lp[:, 0] - x, lp[:, 1] - y) < gate).any():
                continue
            h = Hypothesis(self.next_id, n, float(size), float(x), float(y), float(v))
            self.next_id += 1
            self.live.append(h)
            lp = np.vstack([lp, [[x, y]]]).astype(np.float32)
            gate = np.append(gate, max(mf.px(8.0), 0.5 * float(size)))

    # --- one hypothesis, one frame ---------------------------------------
    def step(self, h: Hypothesis, n: int, q: np.ndarray, fb: float,
             pred: np.ndarray, flow: mf.Flow, prev_gray, gray, raw_gray, valid):
        px0, py0 = h.x, h.y
        self.fb_fail_hyp[1] += 1
        evidence = True
        if not np.isfinite(fb) or fb > mf.FB_MAX:
            self.fb_fail_hyp[0] += 1
            h.fb_fail += 1
            limit = HOLD_DROP_FRAMES if h.confirmed else mf.FB_STRIKES
            relock = self.relock(h, gray, pred) if h.confirmed else None
            if relock is None:
                if h.fb_fail >= limit:
                    h.dead = "fb"
                    return
                q = pred                 # coast on the background prediction
                evidence = False
            else:
                q = relock
                evidence = False

        med, sigma, ring_raw, nring = flow.ring(px0, py0)
        if nring < mf.MIN_RING_PTS:
            evidence = False

        if evidence:
            d = (q - pred) - med
            h.sum_d += d
            h.var += (sigma + mf.NOISE_FLOOR) ** 2
            h.k += 1
            if h.confirmed is not None:
                self.windows[h.hid].append((d.copy(), sigma + mf.NOISE_FLOOR))
        h.raw_path += float(np.hypot(q[0] - px0, q[1] - py0))
        h.ring_path += ring_raw
        h.x, h.y = float(q[0]), float(q[1])
        h.trail.append((n, h.x, h.y))
        h.update_z()

        ix, iy = int(round(h.x)), int(round(h.y))
        if not (0 <= ix < mf.W and 0 <= iy < mf.H_PX) or not valid[iy, ix]:
            h.dead = "edge"
            return

        if h.k >= 1 and h.disp / max(h.k, 1) > h.cap:
            h.dead = "speed"
            return

        if h.k >= mf.K_MIN and h.raw_path / h.k < mf.SCREEN_FIXED_VMAX \
                and h.ring_path / h.k > mf.RING_FLOW_MIN:
            h.dead = "screen-fixed"
            return

        if h.confirmed is None:
            if h.k >= mf.K_MIN and h.z >= self.zstar and h.disp >= mf.D_MIN:
                if CLIP["osd_twins"] and has_twin(raw_gray, (h.x - h.size / 2, h.y - h.size / 2,
                                                             h.size, h.size)):
                    self.twin_vetoes += 1
                    h.dead = "osd-twin"
                    return
                h.confirmed = n
                h.template = self.grab(gray, h)
                self.windows[h.hid] = deque(maxlen=mf.K_MAX)
                self.weak[h.hid] = 0
            elif h.k >= mf.K_MAX:
                h.dead = "timeout"
            return

        # confirmed: hold while the trailing window still says it is moving
        win = self.windows[h.hid]
        if len(win):
            s = np.sum([d for d, _ in win], axis=0)
            var = float(np.sum([g * g for _, g in win]))
            zw = float(np.linalg.norm(s)) / np.sqrt(var) if var > 0 else 0.0
            self.weak[h.hid] = self.weak[h.hid] + 1 if zw < self.zstar / 2 else 0
            if self.weak[h.hid] >= HOLD_DROP_FRAMES:
                h.dead = "faded"

    def grab(self, gray, h: Hypothesis):
        r = int(max(6, round(h.size)))
        x, y = int(round(h.x)), int(round(h.y))
        t = gray[max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1]
        return t.copy() if t.size and t.std() >= 4.0 else None

    def relock(self, h: Hypothesis, gray, pred):
        """Template correlation inside the speed-cap gate, as `Follower`/`TemplateTracker` do."""
        if h.template is None:
            return None
        r = int(max(8, round(h.cap)))
        cx, cy = int(round(pred[0])), int(round(pred[1]))
        th, tw = h.template.shape
        x0, y0 = max(cx - r - tw // 2, 0), max(cy - r - th // 2, 0)
        strip = gray[y0:cy + r + th // 2 + 1, x0:cx + r + tw // 2 + 1]
        if strip.shape[0] < th or strip.shape[1] < tw:
            return None
        m = cv2.matchTemplate(strip.astype(np.float32), h.template.astype(np.float32),
                              cv2.TM_CCOEFF_NORMED)
        _, best, _, loc = cv2.minMaxLoc(m)
        if best < mf.HOLD_TEMPLATE_MIN:
            return None
        return np.array([x0 + loc[0] + tw / 2.0, y0 + loc[1] + th / 2.0], np.float32)

    # --- the pass ---------------------------------------------------------
    def run(self, col: dict):
        cap = cv2.VideoCapture(CLIP["video"])
        prev_gray = prev_raw = None
        n = 0
        t0 = time.time()
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            n += 1
            raw_gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            gray = common.prep(fr, 3)
            rec = col.get(n)
            if prev_gray is not None and rec is not None:
                Hm = rec["H"].astype(np.float64)
                flow = mf.Flow(Hm, mf.GRID_PTS, rec["resid"], rec["raw"], rec["ok"],
                               mf.GRID_ROWS, mf.GRID_COLS, mf.GRID_ORIGIN)
                valid = common.valid_mask(Hm, fr.shape[:2]) & ~self.fixed
                if self.live:
                    pts = np.array([[h.x, h.y] for h in self.live], np.float32)
                    p1, fb = mf.track(prev_gray, gray, pts)
                    pred = mf.predict(Hm, pts)
                    for h, q, f, pr in zip(list(self.live), p1, fb, pred):
                        self.step(h, n, q, float(f), pr, flow, prev_gray, gray, raw_gray, valid)
                self.reap(n)
                self.spawn(n, rec["seeds"], valid)
                self.snapshot(n)
                if n % 100 == 0:
                    nc = sum(1 for h in self.live if h.confirmed)
                    print(f"{n}  {time.time() - t0:.0f}s  live {len(self.live):4d}  "
                          f"confirmed-live {nc:3d}  tracks {len(self.tracks)}", flush=True)
            prev_gray, prev_raw = gray, raw_gray
        cap.release()
        for h in self.live:
            self.retire(h, n, "eof")
        print(f"pass done: {n} frames, {time.time() - t0:.0f}s, "
              f"{len(self.tracks)} confirmed tracks, {self.twin_vetoes} OSD-twin vetoes")

    def reap(self, n: int):
        keep = []
        for h in self.live:
            if h.dead is None:
                keep.append(h)
            elif h.confirmed is not None:
                self.retire(h, n, h.dead)
        self.live = keep

    def retire(self, h: Hypothesis, n: int, why: str):
        if h.confirmed is None or h.hid in self.tracks:
            return
        trail = [t for t in h.trail if t[0] >= h.confirmed]
        self.tracks[h.hid] = dict(hid=h.hid, born=h.born, confirmed=h.confirmed, end=n,
                                  size=h.size, z=h.z, why=why,
                                  trail=np.array(trail, np.float32))

    def snapshot(self, n: int):
        rows = [(h.hid, h.x, h.y, h.z, h.k, h.size,
                 1.0 if h.confirmed is not None else 0.0) for h in self.live if h.k >= 1]
        if not rows:
            self.per_frame[n] = np.zeros((0, 7), np.float32)
            return
        a = np.array(rows, np.float32)
        a = a[np.argsort(-a[:, 3])]
        if self.light:
            keep = (a[:, 6] > 0)
            if mf.BOXES.get(n) is not None:
                x0, y0, x1, y1 = common.drone_region(mf.BOXES[n])
                keep |= (a[:, 1] >= x0) & (a[:, 1] <= x1) & (a[:, 2] >= y0) & (a[:, 2] <= y1)
            a = np.vstack([a[:20], a[keep]])
        else:
            a = a[:STORE_TOP] if len(a) > STORE_TOP else a
            if mf.BOXES.get(n) is not None:
                pass
        self.per_frame[n] = a

    # --- the rank the plan asks for, computed over *all* live hypotheses --
    def rank_on_drone(self, n: int):
        if n not in mf.BOXES:
            return None
        x0, y0, x1, y1 = common.drone_region(mf.BOXES[n])
        best, off = None, []
        for h in self.live:
            if h.k < 1:
                continue
            if x0 <= h.x <= x1 and y0 <= h.y <= y1:
                best = h.z if best is None else max(best, h.z)
            else:
                off.append(h.z)
        if best is None:
            return None
        return 1 + int(sum(1 for z in off if z > best)), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zstar", type=float, required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--light", action="store_true", help="store little; for the z* ladder")
    a = ap.parse_args()
    col = pickle.load(open(OUT + "collect.pkl", "rb"))

    tr = Tracker(a.zstar, light=a.light)
    ranks = {}
    orig_snapshot = tr.snapshot

    def snapshot(n):
        r = tr.rank_on_drone(n)
        if r is not None:
            ranks[n] = r
        orig_snapshot(n)
    tr.snapshot = snapshot

    tr.run(col)
    tag = a.tag or f"z{a.zstar:g}"
    path = OUT + f"verify_{tag}.pkl"
    with open(path, "wb") as fh:
        pickle.dump(dict(zstar=a.zstar, tracks=tr.tracks, frames=tr.per_frame, ranks=ranks,
                         twin_vetoes=tr.twin_vetoes, fb=tr.fb_fail_hyp,
                         n_hyp=tr.next_id), fh, protocol=4)
    f, t = tr.fb_fail_hyp
    print(f"hypothesis forward-backward failures: {f}/{t} = {f / max(t, 1):.4f}")
    print(f"hypotheses born: {tr.next_id}; confirmed tracks: {len(tr.tracks)}")
    print("wrote", path)


if __name__ == "__main__":
    main()
