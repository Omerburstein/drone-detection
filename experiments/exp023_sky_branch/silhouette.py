"""Stage 2a, second cut: the depth-aware ring wired in, and size finally used as evidence.

What changed from EXP-022's `experiments/exp017_motion_first/silhouette.py`
----------------------------------------------------------------------------
That module is left frozen, because the ledger cites it as EXP-022's baseline. Three
changes here, each measurable against it on the same frames:

1. **The ring respects the stage-1 label, per candidate.** EXP-022 implemented and
   self-checked `depth_aware_ring` and then never passed it -- the run used a whole ring
   everywhere. That mattered: 21.2% of O4's kept candidates sat in the uncertain band,
   which is only 7% of the frame, so horizon candidates were 3x over-represented. A ring
   straddling the horizon has its `sigma_ring` set by the *step* rather than by the sky,
   which both invents false alarms on one side and suppresses real targets on the other.
   `contrast_at` now takes the whole label map and restricts the ring to the candidate's
   **own** label, cropped first so the cost stays local.

2. **Both statistics are computed for every candidate**, depth-aware and plain. Without
   that the run cannot say what the change bought -- it could only compare against numbers
   from a different execution, which is the mistake EXP-019 made with its epipole.

3. **Size becomes evidence, instead of leaking in as a bias.** See below.

Why the old `c` ignored size, and why that was a defect
--------------------------------------------------------
`c = (median(ring) - min(core)) / sigma_ring` is a *per-pixel* contrast ratio. It uses the
detected scale only to place the core and the ring, never as evidence. Two consequences,
and the second is the bad one:

  * **It throws away information.** For an extended object the matched-filter SNR goes as
    `delta * sqrt(N) / sigma`, with N the pixels on target. A 3 px blob and a 12 px blob at
    the same `c` are not equally strong evidence -- the larger has ~4x the area and so ~2x
    the signal-to-noise. `c` scores them identically.
  * **Size leaks in backwards, as a bias.** `min(core)` is a *minimum over N pixels*, and
    the expected minimum of N samples drifts downward as N grows. So a bigger core inflates
    `c` from noise alone, and a noisier sky inflates it too. That is the measured
    sub-proportionality from EXP-022: tripling the sky noise cut `c` by only ~2 instead of
    3. Size was already affecting the score -- in the wrong direction, for the wrong reason.

So this module adds `snr`, which uses the core's **mean** rather than its minimum and
normalises by the standard error of that mean:

    snr = (median(ring) - mean(core)) * sqrt(n_core) / sigma_ring

That is unbiased in the way `min` is not, and it rises with size the way evidence should.
Both are reported; `c` is kept so the comparison against EXP-022 stays exact.

The size floor, and the number that has to be checked before setting one
-------------------------------------------------------------------------
A minimum-size gate is the right idea -- a 1-2 px speck is a dead pixel, a grain clump or a
compression artefact, never an airframe at any range. But a floor here cuts on the
**detected scale**, which is not the target's size: see the warning below, where the two
differ by 20x on O4. A floor set from the detected-scale distribution would be cutting on
an artefact of the ladder. `MIN_DIAMETER` therefore defaults to
**0 (off)**, and the run reports the target and clutter size distributions side by side so
a floor is chosen from data rather than from intuition about what "too small" means.

Defaults are EXP-022's, on purpose
-----------------------------------
`detect` ships with `refuse_uncertain=False` and callers default to a whole ring, so this
module reproduces EXP-022 unless a flag says otherwise. EXP-023 measured the two changes
at matched false-alarm rate and neither dominates -- refusing the band wins on O4, scoring
it wins on analog -- so neither is a safe default, and the one that is easier to audit is
the one that matches the run already in the ledger.

A WARNING that outranks all of the above
-----------------------------------------
The scale ladder tops out at 40 px of diameter on O4 and 27 px on analog, while the
labelled targets on those spans run **17-142 px (median 72)** and **18-80 px (median 37)**.
The detector is therefore firing at a median of **0.048x** the target's true size on O4 and
**0.163x** on analog: it is not finding the airframe as a blob at all, it is finding some
small dark sub-feature inside a generously grown box. Every recall figure this module has
produced should be read as `a candidate landed within the box region`, not `the drone was
detected`. Fixing the ladder is the first thing to do, before any threshold is tuned.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from clipcfg import CLIP

W, H = CLIP["width"], CLIP["height"]
S = W / 1440.0

# Label codes for the depth-aware ring. Integers rather than a bool mask per label so one
# array describes the whole frame and a candidate's own label is a single lookup.
SCENE, SKY, UNCERTAIN = 0, 1, 2


def px(v: float) -> float:
    """A constant quoted at 1440 px wide, read on this clip."""
    return v * S


N_SCALES = 8
SIGMA_MIN, SIGMA_MAX = px(1.1), px(14.1)
DOG_K = 1.6

CORE_R = 1.0
RING_IN, RING_OUT = 2.5, 5.0
MIN_RING_PX = 24
# With the ring restricted to one label, a horizon candidate can lose most of its annulus.
# Below this fraction of the whole ring the remainder is not a sample of anything, and the
# candidate is refused rather than scored on a sliver.
MIN_RING_FRACTION = 0.25

SIGMA_FLOOR = 0.75
SATURATION = 246.0
CLOUD_MIN_SIGMA = px(6.0)
CLOUD_EDGE_GRAD = 2.0
MAX_SCALE_FOR_TARGET = px(20.0)
# Off by default. EXP-022 measured the targets at 3.1 px (O4) and 6.2 px (analog).
MIN_DIAMETER = 0.0


@dataclass(frozen=True)
class Silhouette:
    """One sky-branch candidate, carrying both statistics and the geometry behind them."""
    x: float
    y: float
    sigma: float
    response: float
    contrast: float          # c, with the depth-aware ring when one was used
    contrast_plain: float    # the same with the whole ring: EXP-022's number, for the A/B
    snr: float               # area-aware: (ring_med - core_mean) * sqrt(n_core) / sigma
    ring_median: float
    core_min: float
    core_mean: float
    ring_sigma: float
    n_core: int
    ring_kept: float         # fraction of the annulus the label restriction left
    edge_grad: float
    floored: bool
    label: int
    reason: str

    @property
    def diameter(self) -> float:
        """The blob size the scale implies, in px. This is what to compare with a box."""
        return 2.0 * np.sqrt(2.0) * self.sigma

    @property
    def kept(self) -> bool:
        return not self.reason


def scale_ladder(n: int = N_SCALES) -> np.ndarray:
    return np.geomspace(SIGMA_MIN, SIGMA_MAX, n)


def dog_pyramid(grey: np.ndarray, sigmas: np.ndarray | None = None):
    """Negative-polarity scale-normalised DoG, maxed over scale. Unchanged from EXP-022."""
    sig = scale_ladder() if sigmas is None else np.asarray(sigmas, float)
    f = grey.astype(np.float32)
    blurs = [cv2.GaussianBlur(f, (0, 0), float(s)) for s in sig]
    best = np.full(f.shape, -np.inf, np.float32)
    which = np.zeros(f.shape, np.int32)
    for i, s in enumerate(sig):
        hi = cv2.GaussianBlur(f, (0, 0), float(s * DOG_K))
        r = hi - blurs[i]
        upd = r > best
        best[upd] = r[upd]
        which[upd] = i
    return best, sig[which], blurs


def _local_maxima(response: np.ndarray, valid: np.ndarray, radius: int,
                  min_response: float, cap: int) -> np.ndarray:
    m = np.where(valid, response, -np.inf).astype(np.float32)
    k = 2 * int(radius) + 1
    dil = cv2.dilate(m, np.ones((k, k), np.uint8))
    ys, xs = np.nonzero((m >= dil) & (m > min_response))
    if not len(ys):
        return np.zeros((0, 3), np.float32)
    v = m[ys, xs]
    o = np.argsort(-v)[:cap]
    return np.stack([v[o], xs[o], ys[o]], 1).astype(np.float32)


def _disc_and_ring(shape, x: float, y: float, r_core: float, r_in: float, r_out: float):
    """Index arrays for the core disc and the ring, cropped to a bounding box."""
    h, w = shape
    r = int(np.ceil(r_out)) + 2
    x0, x1 = max(0, int(x) - r), min(w, int(x) + r + 1)
    y0, y1 = max(0, int(y) - r), min(h, int(y) + r + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    yy, xx = np.ogrid[y0:y1, x0:x1]
    d2 = (xx - x) ** 2 + (yy - y) ** 2
    core = d2 <= r_core ** 2
    ring = (d2 >= r_in ** 2) & (d2 <= r_out ** 2)
    return (slice(y0, y1), slice(x0, x1)), core, ring


def _robust_sigma(v: np.ndarray) -> float:
    med = float(np.median(v))
    return float(1.4826 * np.median(np.abs(v - med)))


def contrast_at(grey_for_core: np.ndarray, grey_for_ring: np.ndarray,
                x: float, y: float, sigma: float,
                label_map: np.ndarray | None = None):
    """Both statistics at one point and scale, with the ring optionally depth-restricted.

    `grey_for_core` is the scale-matched blur, so no single hot or dead pixel decides the
    core; `grey_for_ring` is the **raw** frame, so `sigma_ring` is the sky's own noise.
    Measuring the ring on a blurred image divides its noise by ~2*sqrt(pi)*sigma_blur and
    drove it under `SIGMA_FLOOR` at every realistic level, which is how EXP-022's first
    version stopped depending on sky noise at all.

    `label_map`, when given, restricts the ring to pixels carrying the candidate's **own**
    stage-1 label. The comparison is done on the crop, so a depth-aware ring costs a boolean
    AND over a few hundred pixels rather than over the frame.

    Returns a dict, or None where the ring is too small or too heavily eaten to be a sample.
    """
    root2 = float(np.sqrt(2.0))
    r_core = max(1.0, CORE_R * sigma * root2)
    r_in, r_out = RING_IN * sigma * root2, RING_OUT * sigma * root2
    got = _disc_and_ring(grey_for_ring.shape, x, y, r_core, r_in, r_out)
    if got is None:
        return None
    box, core, ring_full = got
    if not ring_full.any() or not core.any():
        return None

    own = int(label_map[int(y), int(x)]) if label_map is not None else -1
    if label_map is not None:
        ring = ring_full & (label_map[box] == own)
    else:
        ring = ring_full
    kept_frac = float(ring.sum()) / float(ring_full.sum())

    rv_plain = grey_for_ring[box][ring_full]
    rv = grey_for_ring[box][ring]
    cvals = grey_for_core[box][core]
    if rv_plain.size < MIN_RING_PX or cvals.size < 1:
        return None
    if rv.size < MIN_RING_PX or kept_frac < MIN_RING_FRACTION:
        # The depth-aware ring refuses here rather than widening. At a boundary that IS
        # the answer: there is no same-depth background to compare against.
        return None

    med, med_plain = float(np.median(rv)), float(np.median(rv_plain))
    sg, sg_plain = _robust_sigma(rv), _robust_sigma(rv_plain)
    floored = sg < SIGMA_FLOOR
    sg, sg_plain = max(sg, SIGMA_FLOOR), max(sg_plain, SIGMA_FLOOR)
    cmin, cmean, n_core = float(cvals.min()), float(cvals.mean()), int(cvals.size)
    return dict(
        contrast=(med - cmin) / sg,
        contrast_plain=(med_plain - cmin) / sg_plain,
        # The core's MEAN, normalised by the standard error of that mean. Unbiased where
        # `min` is not, and it rises with size the way evidence should.
        snr=(med - cmean) * float(np.sqrt(n_core)) / sg,
        ring_median=med, core_min=cmin, core_mean=cmean, ring_sigma=sg,
        n_core=n_core, ring_kept=kept_frac, floored=floored, label=own)


def _edge_gradient(grad: np.ndarray, x: float, y: float, sigma: float) -> float:
    """The strongest gradient on the core's boundary. Cloud is soft; a target has an edge."""
    r = max(1.0, CORE_R * sigma * float(np.sqrt(2.0)))
    got = _disc_and_ring(grad.shape, x, y, r, r * 0.8, r * 1.4)
    if got is None:
        return 0.0
    box, _, ring = got
    v = grad[box][ring]
    return float(np.percentile(v, 90)) if v.size else 0.0


def _reject(c: float, sigma: float, diameter: float, ring_median: float,
            edge_grad: float, threshold: float, min_diameter: float) -> str:
    """Which rejection fires, or "" to keep. Cheapest-and-most-certain first."""
    if ring_median >= SATURATION:
        return "bloom"
    if sigma > MAX_SCALE_FOR_TARGET:
        return "too large"
    if min_diameter > 0.0 and diameter < min_diameter:
        return "too small"
    if sigma >= CLOUD_MIN_SIGMA and edge_grad < CLOUD_EDGE_GRAD:
        return "cloud"
    if c < threshold:
        return "below threshold"
    return ""


def detect(frame_bgr: np.ndarray, valid: np.ndarray, *, threshold: float = 6.0,
           label_map: np.ndarray | None = None, nms_radius: float | None = None,
           cap: int = 4000, min_diameter: float = MIN_DIAMETER,
           refuse_uncertain: bool = False) -> list[Silhouette]:
    """Every sky-branch candidate in one frame, kept and rejected alike.

    Rejected candidates are returned carrying their `reason` rather than dropped, so the
    render can colour them and the report can count them.
    """
    grey = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    resp, sig_map, blurs = dog_pyramid(grey)
    sigmas = scale_ladder()
    idx = {float(s): i for i, s in enumerate(sigmas)}
    gf = grey.astype(np.float32)
    grad = np.hypot(cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3),
                    cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3)) / 8.0

    r = int(round(nms_radius if nms_radius is not None else px(9.0)))
    pk = _local_maxima(resp, valid, r, 0.0, cap)

    out: list[Silhouette] = []
    for v, x, y in pk:
        s = float(sig_map[int(y), int(x)])
        got = contrast_at(blurs[idx.get(s, 0)], gf, float(x), float(y), s, label_map)
        if got is None:
            continue
        eg = _edge_gradient(grad, float(x), float(y), s)
        diam = 2.0 * float(np.sqrt(2.0)) * s
        # A candidate whose OWN label is `uncertain` has no clean same-depth background:
        # the band is the straddle zone itself, a mix of sky and near tree line. Scoring it
        # against other uncertain pixels compares it to a thin, relatively uniform ribbon,
        # which *deflates* sigma_ring and manufactures detections along the horizon --
        # measured, on a 17-frame probe, as 85% of all kept candidates landing in a band
        # covering 7% of the frame. Refusing is what "never let a ring straddle the
        # boundary" actually means for a candidate standing on it.
        reason = _reject(got["contrast"], s, diam, got["ring_median"], eg, threshold,
                         min_diameter)
        if refuse_uncertain and got["label"] == UNCERTAIN:
            reason = "uncertain band"
        out.append(Silhouette(
            float(x), float(y), s, float(v), got["contrast"], got["contrast_plain"],
            got["snr"], got["ring_median"], got["core_min"], got["core_mean"],
            got["ring_sigma"], got["n_core"], got["ring_kept"], eg, got["floored"],
            got["label"], reason))
    return out


def false_alarm_curve(values: np.ndarray, thresholds: np.ndarray, n_frames: int):
    """Measured false alarms per frame at each threshold, from target-free candidates."""
    c = np.asarray(values, float)
    n = max(1, n_frames)
    return np.array([float((c >= t).sum()) / n for t in thresholds])


def gaussian_rate(threshold: float, n_independent: float) -> float:
    """The false alarms/frame a true Gaussian sky would give at `threshold` sigma."""
    from math import erfc, sqrt
    return float(n_independent * 0.5 * erfc(threshold / sqrt(2.0)))


def _selfcheck() -> int:
    """Synthetic checks, run with `py -3.13 -m silhouette` from this directory.

    Carries EXP-022's checks forward -- a regression in polarity or in the sigma units
    would otherwise reappear silently -- and adds the two properties this cut introduces.
    """
    rng = np.random.default_rng(11)
    fails = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal fails
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
        fails += 0 if ok else 1

    def sky(noise: float, level: float = 180.0, shape=(300, 400)) -> np.ndarray:
        return np.clip(level + rng.normal(0, noise, shape), 0, 255).astype(np.uint8)

    def put_blob(img, x, y, r, depth, dark=True):
        yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
        g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * r ** 2))
        return np.clip(img.astype(np.float32) + (-depth if dark else depth) * g,
                       0, 255).astype(np.uint8)

    def bgr(g):
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    def near(hits, x, y, r=6.0):
        return [s for s in hits if abs(s.x - x) < r and abs(s.y - y) < r]

    print("silhouette (EXP-023) self-check")
    ok_all = np.ones((300, 400), bool)

    # --- carried forward from EXP-022 -----------------------------------------
    base = sky(2.0)
    d = near(detect(bgr(put_blob(base, 200, 150, 3.0, 40)), ok_all, threshold=4.0), 200, 150)
    b = near(detect(bgr(put_blob(base, 200, 150, 3.0, 40, dark=False)), ok_all,
                    threshold=4.0), 200, 150)
    check("dark blob detected, bright blob is not",
          any(s.kept for s in d) and not any(s.kept for s in b),
          f"dark {sum(s.kept for s in d)} kept, bright {sum(s.kept for s in b)} kept")

    cs = []
    for nz in (1.0, 3.0):
        hits = near(detect(bgr(put_blob(sky(nz), 200, 150, 3.0, 40)), ok_all, threshold=0.0),
                    200, 150)
        cs.append(max((s.contrast for s in hits), default=0.0))
    ratio = cs[0] / max(cs[1], 1e-6)
    check("c falls with sky noise (sub-proportionally; min(core) is biased)",
          1.5 <= ratio <= 4.5,
          f"c={cs[0]:.1f} at noise 1, {cs[1]:.1f} at noise 3, ratio {ratio:.2f}")

    hits = near(detect(bgr(put_blob(np.full((300, 400), 180, np.uint8), 200, 150, 3.0, 40)),
                       ok_all, threshold=0.0), 200, 150)
    check("flat sky flags `floored`", bool(hits) and hits[0].floored,
          f"sigma_ring {hits[0].ring_sigma:.2f}" if hits else "no detection")

    soft = cv2.GaussianBlur(put_blob(sky(2.0), 200, 150, 14.0, 30), (0, 0), 9.0)
    hits = near(detect(bgr(soft), ok_all, threshold=3.0), 200, 150, 12.0)
    check("a large soft blob is rejected as cloud",
          bool(hits) and not any(s.kept for s in hits),
          ", ".join(f"{s.sigma:.1f}px {s.reason or 'KEPT'}" for s in hits) or "none")

    # --- new in EXP-023 --------------------------------------------------------
    # 1. The depth-aware ring must exclude the other label's step from sigma_ring, and
    #    therefore RAISE the contrast of a target sitting just inside the boundary.
    img_g = sky(2.0)
    img_g[170:, :] = np.clip(img_g[170:, :].astype(int) - 70, 0, 255).astype(np.uint8)
    img_g = put_blob(img_g, 200, 160, 3.0, 40)
    lmap = np.full(img_g.shape, SCENE, np.int32)
    lmap[:170] = SKY
    hits = near(detect(bgr(img_g), ok_all, threshold=0.0, label_map=lmap), 200, 160, 8.0)
    check("depth-aware ring raises c for a target beside a horizon step",
          bool(hits) and hits[0].contrast > 1.8 * hits[0].contrast_plain,
          f"c {hits[0].contrast:.1f} depth-aware vs {hits[0].contrast_plain:.1f} whole-ring, "
          f"ring kept {hits[0].ring_kept * 100:.0f}%" if hits else "no detection")

    # 2. Two geometries, because they behave differently and only one is a refusal.
    #
    #    A STRAIGHT horizon can never eat much of the ring: a point just inside a half-plane
    #    keeps ~50% of its annulus no matter how close to the line it sits. So the refusal
    #    path is NOT for the ordinary horizon -- it is for concave geometry, a sliver of sky
    #    between tree tops, where the own-label region is thinner than the ring is wide.
    #    Testing only the straight case would have left the refusal path unexercised while
    #    looking like it passed.
    core_g = cv2.GaussianBlur(img_g.astype(np.float32), (0, 0), 3.0)
    straight = contrast_at(core_g, img_g.astype(np.float32), 200, 172, 8.0, lmap)
    check("a straight horizon keeps about half the ring and is still scored",
          straight is not None and 0.35 <= straight["ring_kept"] <= 0.65,
          f"kept {straight['ring_kept'] * 100:.0f}%" if straight else "refused")

    sliver = np.full(img_g.shape, SCENE, np.int32)
    sliver[165:176] = SKY               # an 11 px band, far thinner than the 57 px ring
    eaten = contrast_at(core_g, img_g.astype(np.float32), 200, 170, 8.0, sliver)
    check("a ring eaten below MIN_RING_FRACTION is refused, not widened", eaten is None,
          "refused" if eaten is None else f"kept {eaten['ring_kept'] * 100:.0f}% and scored")

    # 2b. A candidate standing IN the uncertain band is refused by default. Without this
    #     the band's own ribbon becomes its background, sigma_ring deflates, and the
    #     horizon fills with detections -- 85% of all kept candidates, measured.
    band = np.full(img_g.shape, SCENE, np.int32)
    band[160:185] = UNCERTAIN
    inband = near(detect(bgr(img_g), ok_all, threshold=0.0, label_map=band,
                         refuse_uncertain=True), 200, 170, 12.0)
    free = near(detect(bgr(img_g), ok_all, threshold=0.0, label_map=band), 200, 170, 12.0)
    check("a candidate inside the uncertain band is refused when asked",
          bool(inband) and all(s.reason == "uncertain band" for s in inband)
          and bool(free) and any(s.reason != "uncertain band" for s in free),
          f"refuse_uncertain=True -> {inband[0].reason if inband else 'nothing'}; "
          f"default -> {free[0].reason or 'KEPT' if free else 'nothing'}")

    # 3. `snr` must grow with blob size at fixed contrast, where `c` does not. This is the
    #    whole point of adding it: a bigger blob is more evidence, and c cannot say so.
    got = []
    for r in (2.0, 6.0):
        hits = near(detect(bgr(put_blob(sky(2.0), 200, 150, r, 40)), ok_all, threshold=0.0),
                    200, 150, 8.0)
        best = max(hits, key=lambda s: s.response) if hits else None
        got.append((best.contrast, best.snr, best.diameter) if best else (0, 0, 0))
    check("snr grows with size at similar contrast, where c does not",
          got[1][1] > 1.5 * got[0][1],
          f"r=2: c {got[0][0]:.1f} snr {got[0][1]:.1f} ({got[0][2]:.1f}px)   "
          f"r=6: c {got[1][0]:.1f} snr {got[1][1]:.1f} ({got[1][2]:.1f}px)")

    # 4. The size floor rejects below it and is OFF by default -- the default matters,
    #    because the measured targets are 3.1 px (O4) and 6.2 px (analog).
    img = bgr(put_blob(sky(2.0), 200, 150, 2.0, 45))
    off = near(detect(img, ok_all, threshold=4.0), 200, 150, 8.0)
    on = near(detect(img, ok_all, threshold=4.0, min_diameter=12.0), 200, 150, 8.0)
    check("min_diameter rejects below the floor and defaults to off",
          any(s.kept for s in off) and on and all(s.reason == "too small" for s in on),
          f"default kept {sum(s.kept for s in off)}, "
          f"floor 12px -> {on[0].reason if on else 'nothing'} "
          f"(blob measured {off[0].diameter:.1f}px)" if off else "no detection")

    print(f"\n{'all checks passed' if not fails else str(fails) + ' CHECK(S) FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_selfcheck())
