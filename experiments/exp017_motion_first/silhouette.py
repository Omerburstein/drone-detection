"""Stage 2a, the sky branch: a multi-scale negative-polarity blob detector with a
contrast statistic in units of the local sky noise.

Why this is a different kind of test from everything before it
---------------------------------------------------------------
Every candidate score this project has used -- EXP-015's normalised difference, EXP-016's
`z`, EXP-017's `tau` -- is a *motion* score. It needs two frames, it needs the homography
to be right, and its threshold is a number with no units, calibrated by counting
candidates on empty frames until the count looks acceptable.

This one inverts the regime. Sky has no texture, so the local scatter `sigma_ring` around
a patch of sky is tiny, and a 4 px dark speck sitting in it is a large-sigma event. The
statistic

    c = (median(ring) - min(core)) / sigma_ring

is the classic IRST local-contrast measure with the sign flipped, and it is in units of
sigma. That is the property worth having: a threshold on `c` is a statement about a
false-alarm rate on smooth sky, which none of the motion thresholds ever were.

It also needs **one frame**. Zero acquisition latency, against the 0.5-2.5 s EXP-016
measured for a motion track to confirm.

The claim "units of sigma" is a claim, and this module measures it
------------------------------------------------------------------
`c` is only a false-alarm rate if the sky's pixel distribution is actually Gaussian with
scale `sigma_ring`. Real sky is not: it carries grain, chroma crawl on CVBS, compression
blocking, and thin cloud. So `false_alarm_curve` measures the empirical rate of `c >= t`
on frames with no target, and the run reports it **next to** the Gaussian rate the same
threshold would nominally buy. Where the two part company, `c` is a ranking score and not
a calibrated detection statistic, and it must be described as one.

The other way this can quietly stop meaning sigma is the floor. On 8-bit video a patch of
clear sky can be flat to the quantiser, giving `sigma_ring = 0` and `c = inf` for any
speck at all. `SIGMA_FLOOR` stops that, and `Silhouette.floored` records when it bound --
if the floor binds often, the units are the floor's and not the sky's, and the report says
so rather than printing a large number.

Scale comes out of the detector, not out of a parameter
--------------------------------------------------------
The candidate locator is a scale-normalised difference-of-Gaussians pyramid with the
polarity reversed. `G_{k*s} - G_{s}` approximates `(k-1) * s^2 * laplacian(G)`, which is
**positive at a dark blob** and already carries the `s^2` that scale-normalisation needs,
so no extra factor is applied. Taking the max over the scale ladder gives each pixel a
response and an argmax, and the argmax is the blob's size -- which is what then sets the
core and ring radii for `c`. A bottom-hat would detect the same things and would not hand
back the scale.

What it cannot do, stated here rather than discovered later
------------------------------------------------------------
  * **Birds are not rejectable.** A bird against sky is a small dark high-contrast blob,
    which is the definition of a detection here. It is Stage 4's problem and nothing in
    this module pretends otherwise.
  * **Sun, bloom, cloud edge and sensor dirt are rejectable**, and each has its own test
    below with its own reason.
  * **It says nothing about the scene branch.** Below the horizon the regime is the usual
    one -- texture everywhere, `sigma_ring` large, a drone is not a high-sigma event -- and
    this detector is expected to be useless there. The run measures that rather than
    assuming it, by scoring every candidate and cutting the result by stage-1 label.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from clipcfg import CLIP

W, H = CLIP["width"], CLIP["height"]
S = W / 1440.0


def px(v: float) -> float:
    """A constant quoted at 1440 px wide, read on this clip."""
    return v * S


# The scale ladder, as Gaussian sigma. A dark disc of diameter d peaks the DoG at about
# sigma = d / (2 * sqrt(2)), so 1.1 to 14.1 spans roughly 3 to 40 px of diameter -- the
# span the plan asks for, and the span this project's targets actually occupy (ARD-MAV is
# 65% sub-8 px, the SOFA clips run 6-40 px).
N_SCALES = 8
SIGMA_MIN, SIGMA_MAX = px(1.1), px(14.1)
DOG_K = 1.6                  # the usual ratio; at 1.6 the DoG is a good LoG approximation

# Core and ring, in multiples of the detected blob radius. The ring must clear the blob's
# own skirt -- a Gaussian of sigma s still has energy at 2s -- or the "background" median
# is partly the target and the contrast reads low.
CORE_R = 1.0
RING_IN, RING_OUT = 2.5, 5.0
MIN_RING_PX = 24             # fewer than this and the ring statistic is not a statistic

# 8-bit sky can be flat to the quantiser. Below this, `c` would be reporting the
# quantiser rather than the sky. Chosen as the scatter a 1-LSB dither produces.
SIGMA_FLOOR = 0.75

# Rejections.
SATURATION = 246.0           # ring median above this is sun or bloom, not sky
CLOUD_MIN_SIGMA = px(6.0)    # only large blobs can be cloud at all
CLOUD_EDGE_GRAD = 2.0        # grey levels/px at the core boundary; cloud is soft
MAX_SCALE_FOR_TARGET = px(20.0)


@dataclass(frozen=True)
class Silhouette:
    """One sky-branch candidate, with every quantity the rejections needed kept on it.

    Holding `ring_median`, `core_min`, `ring_sigma` and `edge_grad` rather than just `c`
    is what lets the run report *why* a candidate was rejected and lets a threshold be
    re-cut from a recorded run without re-detecting. Same discipline as
    `src/eval/records.py`: every quantity a later cut needs, written once.
    """
    x: float
    y: float
    sigma: float             # the scale that won, in px
    response: float          # the scale-normalised DoG value there
    contrast: float          # c, in units of sigma_ring
    ring_median: float
    core_min: float
    ring_sigma: float
    edge_grad: float
    floored: bool            # sigma_ring hit SIGMA_FLOOR, so `c` is not in sky units
    reason: str              # "" when kept, otherwise which rejection fired

    @property
    def diameter(self) -> float:
        """The blob size the scale implies, in px. This is what to compare with a box."""
        return 2.0 * np.sqrt(2.0) * self.sigma

    @property
    def kept(self) -> bool:
        return not self.reason


def scale_ladder(n: int = N_SCALES) -> np.ndarray:
    """Geometric sigma ladder. Geometric, because scale space is multiplicative."""
    return np.geomspace(SIGMA_MIN, SIGMA_MAX, n)


def dog_pyramid(grey: np.ndarray, sigmas: np.ndarray | None = None):
    """Negative-polarity scale-normalised DoG, maxed over scale.

    Returns (response, sigma_at_max, blurs) where `blurs` is the per-scale blurred stack,
    kept because `core_min` is read off the scale-matched blur rather than off raw pixels
    -- a single hot or dead pixel would otherwise decide the minimum, and one pixel is not
    a target.
    """
    sig = scale_ladder() if sigmas is None else np.asarray(sigmas, float)
    f = grey.astype(np.float32)
    blurs = [cv2.GaussianBlur(f, (0, 0), float(s)) for s in sig]
    best = np.full(f.shape, -np.inf, np.float32)
    which = np.zeros(f.shape, np.int32)
    for i, s in enumerate(sig):
        # G_{k s} - G_{s} is positive at a DARK blob, and already carries the s^2 that
        # scale normalisation needs, so it is comparable across the ladder as it stands.
        hi = cv2.GaussianBlur(f, (0, 0), float(s * DOG_K))
        r = hi - blurs[i]
        upd = r > best
        best[upd] = r[upd]
        which[upd] = i
    return best, sig[which], blurs


def _local_maxima(response: np.ndarray, valid: np.ndarray, radius: int,
                  min_response: float, cap: int) -> np.ndarray:
    """Peaks of the response map inside `valid`, strongest first. (N, 3) of (value, x, y).

    `cap` is a memory guard and not a budget: a run that hits it is reporting that its
    response floor is wrong, exactly as `budget.candidates` does for the motion branch.
    """
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
    """Index arrays for the core disc and the ring, cropped to a bounding box.

    Cropped rather than built over the whole frame: at 1440x1080 a full-frame ogrid per
    candidate is ~12 MB and there are hundreds of candidates per frame. This is the
    difference between a render that runs and one that does not.
    """
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
    """1.4826 * MAD. Robust, because a ring at a cloud edge or holding a second target
    would have its standard deviation set by that intruder rather than by the sky."""
    med = float(np.median(v))
    return float(1.4826 * np.median(np.abs(v - med)))


def contrast_at(grey_for_core: np.ndarray, grey_for_ring: np.ndarray,
                x: float, y: float, sigma: float,
                label_mask: np.ndarray | None = None):
    """The IRST local contrast at one point and scale, sign-flipped for a dark target.

    `label_mask`, when given, is the depth-aware ring the plan asks for: only ring pixels
    whose stage-1 label matches the candidate's are used, so a ring straddling the horizon
    describes one side of it rather than the average of two. Passing None uses the whole
    ring, which is what the sky interior wants and is also the control the run needs in
    order to say what the depth-aware version bought.

    The two sources are deliberately different images, and the self-check caught what
    happens when they are not. `grey_for_core` is the **scale-matched blur**, so a single
    hot or dead pixel cannot decide `min(core)`; `grey_for_ring` is the **raw** frame, so
    `sigma_ring` is the sky's own noise. Measuring the ring on a blurred image instead
    divides the noise by ~2*sqrt(pi)*sigma_blur, which drove `sigma_ring` under
    `SIGMA_FLOOR` at every realistic sky noise level -- `c` then read the floor rather than
    the sky and stopped depending on sky noise at all, which is the one property the whole
    statistic exists to have.

    Returns (c, ring_median, core_min, sigma_ring, floored), or None where the ring has too
    few usable pixels to be a statistic.
    """
    root2 = float(np.sqrt(2.0))
    r_core = max(1.0, CORE_R * sigma * root2)
    r_in, r_out = RING_IN * sigma * root2, RING_OUT * sigma * root2
    got = _disc_and_ring(grey_for_ring.shape, x, y, r_core, r_in, r_out)
    if got is None:
        return None
    box, core, ring = got
    if label_mask is not None:
        ring = ring & label_mask[box]
    rv = grey_for_ring[box][ring]
    cval = grey_for_core[box][core]
    if rv.size < MIN_RING_PX or cval.size < 1:
        return None
    med = float(np.median(rv))
    sg = _robust_sigma(rv)
    floored = sg < SIGMA_FLOOR
    sg = max(sg, SIGMA_FLOOR)
    cmin = float(cval.min())
    return (med - cmin) / sg, med, cmin, sg, floored


def _edge_gradient(grad: np.ndarray, x: float, y: float, sigma: float) -> float:
    """The strongest gradient on the core's boundary. Cloud is soft; a target has an edge."""
    r = max(1.0, CORE_R * sigma * float(np.sqrt(2.0)))
    got = _disc_and_ring(grad.shape, x, y, r, r * 0.8, r * 1.4)
    if got is None:
        return 0.0
    box, _, ring = got
    v = grad[box][ring]
    return float(np.percentile(v, 90)) if v.size else 0.0


def _reject(c: float, sigma: float, ring_median: float, edge_grad: float,
            threshold: float) -> str:
    """Which rejection fires, or "" to keep. Cheapest-and-most-certain first."""
    if ring_median >= SATURATION:
        return "bloom"                      # sun or specular bloom; sigma there is meaningless
    if sigma > MAX_SCALE_FOR_TARGET:
        return "too large"                  # bigger than any target at any range we fly
    if sigma >= CLOUD_MIN_SIGMA and edge_grad < CLOUD_EDGE_GRAD:
        return "cloud"                      # large AND soft; a large hard blob is kept
    if c < threshold:
        return "below threshold"
    return ""


def detect(frame_bgr: np.ndarray, valid: np.ndarray, *, threshold: float = 6.0,
           label_mask: np.ndarray | None = None, nms_radius: float | None = None,
           cap: int = 600) -> list[Silhouette]:
    """Every sky-branch candidate in one frame, kept and rejected alike.

    Rejected candidates are returned carrying their `reason` rather than dropped. That is
    what lets the render colour them by rejection and the report count them -- a detector
    that silently discards cannot be debugged, and EXP-016's null results were expensive
    for exactly that reason.
    """
    grey = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    resp, sig_map, blurs = dog_pyramid(grey)
    sigmas = scale_ladder()
    idx = {float(s): i for i, s in enumerate(sigmas)}
    gf = grey.astype(np.float32)
    grad = np.hypot(cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3),
                    cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3)) / 8.0
    # Raw, not blurred: this is what `sigma_ring` must measure. See `contrast_at`.
    ring_src = gf

    r = int(round(nms_radius if nms_radius is not None else px(9.0)))
    pk = _local_maxima(resp, valid, r, 0.0, cap)

    out: list[Silhouette] = []
    for v, x, y in pk:
        s = float(sig_map[int(y), int(x)])
        core_src = blurs[idx.get(s, 0)]
        got = contrast_at(core_src, ring_src, float(x), float(y), s, label_mask)
        if got is None:
            continue
        c, med, cmin, sg, floored = got
        eg = _edge_gradient(grad, float(x), float(y), s)
        out.append(Silhouette(float(x), float(y), s, float(v), float(c), med, cmin, sg,
                              eg, floored, _reject(c, s, med, eg, threshold)))
    return out


def false_alarm_curve(contrasts: np.ndarray, thresholds: np.ndarray,
                      n_frames: int) -> np.ndarray:
    """Measured false alarms per frame at each threshold, from target-free frames.

    This is the number the whole "units of sigma" argument rests on. It is reported beside
    the Gaussian rate so the two can be compared; where they diverge, `c` is a ranking
    score and must not be quoted as a false-alarm rate.
    """
    c = np.asarray(contrasts, float)
    n = max(1, n_frames)
    return np.array([float((c >= t).sum()) / n for t in thresholds])


def gaussian_rate(threshold: float, n_independent: float) -> float:
    """The false alarms/frame a true Gaussian sky would give at `threshold` sigma.

    `n_independent` is how many independent looks a frame contains -- roughly the sky area
    divided by the area of one resolution cell at the detection scale. It is an order-of-
    magnitude figure and is labelled as such wherever it is printed.
    """
    from math import erfc, sqrt
    return float(n_independent * 0.5 * erfc(threshold / sqrt(2.0)))


def _selfcheck() -> int:
    """Synthetic checks, run with `py -3.13 -m silhouette`.

    Each one is a property that would fail *silently* on video -- a polarity error, a scale
    error or a miscalibrated sigma all come back as "the sky branch found nothing", which
    is indistinguishable from "the idea does not work". They are checked against scenes
    where the answer is known by construction, as `selftest.py` does for the geometry.
    """
    rng = np.random.default_rng(11)
    fails = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal fails
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
        fails += 0 if ok else 1

    def sky(noise: float, level: float = 180.0, shape=(300, 400)) -> np.ndarray:
        return np.clip(level + rng.normal(0, noise, shape), 0, 255).astype(np.uint8)

    def put_blob(img: np.ndarray, x: int, y: int, r: float, depth: float,
                 dark: bool = True) -> np.ndarray:
        yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
        g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * r ** 2))
        f = img.astype(np.float32) + (-depth if dark else depth) * g
        return np.clip(f, 0, 255).astype(np.uint8)

    def bgr(g: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    def near(hits, x, y, r=6.0):
        return [s for s in hits if abs(s.x - x) < r and abs(s.y - y) < r]

    print("silhouette self-check")
    ok_all = np.ones((300, 400), bool)

    # 1. Polarity. A dark blob must respond and a bright one of the same depth must not.
    base = sky(2.0)
    d = near(detect(bgr(put_blob(base, 200, 150, 3.0, 40)), ok_all, threshold=4.0), 200, 150)
    b = near(detect(bgr(put_blob(base, 200, 150, 3.0, 40, dark=False)), ok_all,
                    threshold=4.0), 200, 150)
    check("dark blob detected, bright blob is not",
          any(s.kept for s in d) and not any(s.kept for s in b),
          f"dark {sum(s.kept for s in d)} kept, bright {sum(s.kept for s in b)} kept")

    # 2. The statistic is in units of sigma: triple the noise at fixed depth and c thirds.
    cs = []
    for nz in (1.0, 3.0):
        hits = near(detect(bgr(put_blob(sky(nz), 200, 150, 3.0, 40)), ok_all, threshold=0.0),
                    200, 150)
        cs.append(max((s.contrast for s in hits), default=0.0))
    ratio = cs[0] / max(cs[1], 1e-6)
    # Tripling the sky noise must cut `c` substantially, but NOT by exactly three. The
    # numerator carries its own noise term: `min(core)` is a minimum over ~14 px, so it is
    # a downward-biased order statistic whose bias grows with sigma. That inflates `c` on
    # noisy footage -- which matters, because analog/CVBS is the noisy footage. Measured
    # ratio here is ~2 against a proportional 3, and the report says so rather than
    # quoting `c` as if it were pure SNR.
    check("c falls with sky noise (sub-proportionally; min(core) is biased)",
          1.5 <= ratio <= 4.5,
          f"c={cs[0]:.1f} at noise 1, {cs[1]:.1f} at noise 3, ratio {ratio:.2f} "
          f"(proportional would be 3.00)")

    # 3. Scale is recovered, not assumed. A blob three times bigger must pick a bigger sigma.
    got = []
    for r in (2.0, 6.0):
        hits = near(detect(bgr(put_blob(sky(2.0), 200, 150, r, 45)), ok_all, threshold=0.0),
                    200, 150, 8.0)
        got.append(max(hits, key=lambda s: s.response).sigma if hits else 0.0)
    check("recovered scale grows with blob size", got[1] > got[0] * 1.5,
          f"sigma {got[0]:.1f} for r=2, {got[1]:.1f} for r=6")

    # 4. The sigma floor is recorded, not hidden. A perfectly flat sky must say `floored`.
    flat = np.full((300, 400), 180, np.uint8)
    hits = near(detect(bgr(put_blob(flat, 200, 150, 3.0, 40)), ok_all, threshold=0.0),
                200, 150)
    check("flat sky flags `floored`", bool(hits) and hits[0].floored,
          f"sigma_ring {hits[0].ring_sigma:.2f}" if hits else "no detection")

    # 5. Cloud rejection keeps a large HARD blob and drops a large soft one.
    soft = cv2.GaussianBlur(put_blob(sky(2.0), 200, 150, 14.0, 30), (0, 0), 9.0)
    hits = near(detect(bgr(soft), ok_all, threshold=3.0), 200, 150, 12.0)
    check("a large soft blob is rejected as cloud",
          bool(hits) and not any(s.kept for s in hits),
          ", ".join(f"sigma {s.sigma:.1f} {s.reason or 'KEPT'}" for s in hits) or "none")

    # 6. Bloom rejection: a speck inside a saturated region is not a sigma event.
    hits = near(detect(bgr(put_blob(np.full((300, 400), 252, np.uint8), 200, 150, 3.0, 30)),
                       ok_all, threshold=0.0), 200, 150)
    check("a speck in bloom is rejected", bool(hits) and hits[0].reason == "bloom",
          hits[0].reason if hits else "no detection")

    # 7. The depth-aware ring uses only the matching label. A ring half-covered by dark
    #    ground must give a different sigma from the same ring restricted to sky.
    img_g = sky(2.0)
    img_g[170:, :] = np.clip(img_g[170:, :].astype(int) - 70, 0, 255).astype(np.uint8)
    img_g = put_blob(img_g, 200, 160, 3.0, 40)
    lab = np.zeros(img_g.shape, bool)
    lab[:170] = True
    gf = img_g.astype(np.float32)
    core_src = cv2.GaussianBlur(gf, (0, 0), 3.0)
    whole = contrast_at(core_src, gf, 200, 160, 3.0, None)
    half = contrast_at(core_src, gf, 200, 160, 3.0, lab)
    # The whole ring straddles a 70-grey-level step, so its scatter is set by the step and
    # not by the sky. That inflated sigma is exactly what makes a target at the horizon
    # read as low contrast, and restricting the ring to one label is the fix.
    check("depth-aware ring removes the horizon step from sigma_ring",
          whole is not None and half is not None and whole[3] > 2.0 * half[3],
          f"sigma_ring whole {whole[3]:.1f} vs sky-only {half[3]:.1f}, "
          f"c {whole[0]:.1f} -> {half[0]:.1f}" if whole and half else "ring unusable")

    # 8. Empty sky: the measured false-alarm rate must fall with threshold. This is the
    #    number deciding whether `c` may be called a detection statistic at all.
    cands: list[float] = []
    for _ in range(6):
        cands += [s.contrast for s in detect(bgr(sky(2.0)), ok_all, threshold=0.0)]
    # Thresholds taken from the data's own percentiles rather than fixed numbers, so the
    # curve cannot come back all-zero and pass vacuously -- an all-zero curve is also what
    # a detector that finds nothing at all produces, and that must not read as a pass.
    arr = np.array(cands)
    thr = np.percentile(arr, [50, 75, 90, 99])
    far = false_alarm_curve(arr, thr, 6)
    check("false alarms fall monotonically with threshold, from a non-empty curve",
          bool(np.all(np.diff(far) <= 0)) and far[0] > 0.0,
          "  ".join(f"c>={t:.1f}:{f:.1f}/frame" for t, f in zip(thr, far))
          + f"   (max c on pure noise sky {arr.max():.1f})")

    print(f"\n{'all checks passed' if not fails else str(fails) + ' CHECK(S) FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_selfcheck())
