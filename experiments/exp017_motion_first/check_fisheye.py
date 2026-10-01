"""Does undistortion change stage 1, and is there any distortion to undo in the first place?

    PYTHONPATH="experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
        py -3.13 -m check_fisheye [--frames 60] [--lam L]

The measurement, and why it is this one
---------------------------------------
Stage 1 already emits a horizon: the lowest sky row per column. A horizon is the best
plumb line this footage offers -- but a tree line has real relief, so its bow is NOT by
itself evidence of a lens. The discriminator is that the two sources behave differently
as the camera moves:

  * a **lens** bows a line by an amount set by where the line sits in the IMAGE. The bow
    is proportional to the line's distance from the optical axis and changes sign as the
    line crosses it. It is pinned to image coordinates and does not care what is imaged.
  * **terrain** is pinned to the scene. As the camera pitches, a ridge keeps its shape and
    simply translates; its bow has no relation to image height.

So step 2 regresses each frame's sag on the horizon's height above the frame centre. A
lens gives a straight line through the centre with a consistent slope; terrain gives a
cloud. That regression, not the sag itself, is the evidence.

Step 3 then does the plumb-line fit properly -- sweep lambda, undistort the horizon
points, refit, and take the lambda that flattens every frame at once. Step 4 re-runs
stage 1 on genuinely resampled frames at that lambda and reports what moved.

The guard that matters
----------------------
If the sag is small next to the tree line's own roughness (the fit residual), lambda is
unidentifiable from this footage and any value the sweep returns is fitting noise. Step 1
reports both numbers side by side so that verdict is visible before the sweep runs.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

import masks
import skyline
import undistort
from clipcfg import CLIP

W, H = CLIP["width"], CLIP["height"]
MIN_SPAN_FRAC = 0.55      # the horizon must cross this much of the width to be fittable
MIN_COLS = 200


def robust_quad(x: np.ndarray, y: np.ndarray, iters: int = 4):
    """Quadratic fit in x normalised to [-1, 1], trimming outliers by MAD.

    Trimming is not cosmetic. A single column where the sky mask runs to the bottom of the
    frame through a gap in the trees puts a 600 px outlier into a least-squares fit and
    swamps a bow of a few px.
    """
    lo, hi = float(x.min()), float(x.max())
    xn = 2.0 * (x - lo) / max(hi - lo, 1.0) - 1.0
    keep = np.ones(len(x), bool)
    cf = np.polyfit(xn, y, 2)
    for _ in range(iters):
        res = y - np.polyval(cf, xn)
        s = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        keep = np.abs(res) < 2.5 * max(s, 1.0)
        if keep.sum() < 20:
            break
        cf = np.polyfit(xn[keep], y[keep], 2)
    res = y - np.polyval(cf, xn)
    rms = float(np.sqrt(np.mean(res[keep] ** 2))) if keep.sum() else float("nan")
    # With xn in [-1, 1] the chord midpoint is a + c and the curve at 0 is c, so the bow
    # (curve minus chord, at mid-span) is exactly -a. No further algebra needed.
    return dict(sag=float(-cf[0]), rms=rms, n=int(keep.sum()),
                y_mid=float(np.polyval(cf, 0.0)), span=float(hi - lo),
                lo=lo, hi=hi, keep=keep, coef=cf)


def horizon_points(sl):
    """(x, y) of the horizon, or None if it does not cross enough of the frame."""
    if not sl.has_sky:
        return None
    x = np.nonzero(sl.horizon >= 0)[0]
    if len(x) < MIN_COLS:
        return None
    if (x.max() - x.min()) < MIN_SPAN_FRAC * W:
        return None
    return x.astype(float), sl.horizon[x].astype(float)


def sample_frames(n: int) -> list[int]:
    return [int(v) for v in np.linspace(2, CLIP["n_frames"] - 1, n).astype(int)]


def collect(frames, valid):
    """Stage 1 on each frame, keeping the frames whose horizon is fittable."""
    cap = cv2.VideoCapture(CLIP["video"])
    out = []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        sl = skyline.split(img, valid)
        hp = horizon_points(sl)
        if hp is None:
            continue
        x, y = hp
        out.append(dict(frame=f, x=x, y=y, fit=robust_quad(x, y),
                        sky=sl.sky_fraction, sky_mask=sl.sky))
    cap.release()
    return out


def positive_control(rows, lam_inject, lams):
    """Can this measurement recover a distortion it is handed? If not, nothing else counts.

    A perfectly straight horizon is synthesised at each real frame's own height and span,
    bent by `lam_inject`, and put through the same sweep. The argmin must come back at
    `lam_inject`. Without this, "the sweep found nothing" and "the sweep cannot find
    anything" are the same output, which is exactly the trap EXP-012a fell into.
    """
    synth = []
    for r in rows:
        f = r["fit"]
        x = np.arange(f["lo"], f["hi"] + 1.0)
        y = np.full(len(x), f["y_mid"])
        bent = undistort.distort_points(np.column_stack([x, y]), lam_inject)
        synth.append(dict(x=bent[:, 0], y=bent[:, 1],
                          fit=dict(keep=np.ones(len(x), bool))))
    res = sweep(synth, lams)
    return res, min(res, key=lambda t: t[1])


def sweep(rows, lams):
    """Mean |sag| over the same frames and the same inlier columns, per lambda."""
    res = []
    for lam in lams:
        sags = []
        for r in rows:
            k = r["fit"]["keep"]
            pts = np.column_stack([r["x"][k], r["y"][k]])
            u = undistort.undistort_points(pts, lam)
            sags.append(abs(robust_quad(u[:, 0], u[:, 1], iters=2)["sag"]))
        res.append((float(lam), float(np.mean(sags))))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--lam", type=str, default=None,
                    help="comma-separated lambdas to run step 4 at, instead of the "
                         "sweep's own answer. Use this to try a physically plausible "
                         "barrel value when the sweep has no interior minimum.")
    ap.add_argument("--range", type=float, default=0.60,
                    help="sweep lambda over +/- this. Widen it if the best value lands on "
                         "the boundary -- a boundary answer is not a fit.")
    ap.add_argument("--inject", type=float, default=-0.16,
                    help="positive control: bend the measured horizons by this lambda and "
                         "check the sweep recovers it. 0 disables.")
    a = ap.parse_args()

    stage0 = masks.Stage0()
    valid = ~stage0.static
    frames = sample_frames(a.frames)
    print(f"[fisheye] {CLIP['name']}: stage 1 on {len(frames)} frames, "
          f"centre {undistort.CENTRE.round(1)}, r=1 at {undistort.SCALE:.0f} px\n")

    rows = collect(frames, valid)
    if len(rows) < 8:
        raise SystemExit(f"only {len(rows)} frames have a fittable horizon; "
                         f"nothing can be concluded")

    # --- step 1: is there a bow at all, next to the tree line's own roughness? ----
    sag = np.array([r["fit"]["sag"] for r in rows])
    rms = np.array([r["fit"]["rms"] for r in rows])
    ymid = np.array([r["fit"]["y_mid"] for r in rows])
    cy = float(undistort.CENTRE[1])
    print("1. THE BOW, AND WHAT IT COMPETES WITH")
    print(f"   frames with a fittable horizon      {len(rows)} of {len(frames)}")
    print(f"   sag (bow at mid-span), px           median {np.median(sag):+7.2f}"
          f"   p10 {np.percentile(sag, 10):+7.2f}   p90 {np.percentile(sag, 90):+7.2f}")
    print(f"   |sag|, px                           median {np.median(np.abs(sag)):7.2f}")
    print(f"   fit residual RMS (tree relief), px  median {np.median(rms):7.2f}")
    print(f"   ratio |sag| / residual                     "
          f"{np.median(np.abs(sag)) / np.median(rms):7.2f}")
    print(f"   horizon height, row                 {ymid.min():.0f} to {ymid.max():.0f}"
          f"   (optical centre row {cy:.0f})")

    # --- step 2: is the bow pinned to the image or to the scene? -----------------
    print("\n2. IS IT A LENS? sag regressed on horizon height above the optical centre")
    dy = ymid - cy
    if np.ptp(dy) < 60:
        print(f"   horizon height varies by only {np.ptp(dy):.0f} px over these frames --")
        print("   too little leverage to separate a lens from terrain. INCONCLUSIVE.")
    else:
        slope, icpt = np.polyfit(dy, sag, 1)
        r_p = float(np.corrcoef(dy, sag)[0, 1])
        zero = (-icpt / slope + cy) if abs(slope) > 1e-9 else float("nan")
        print(f"   slope   {slope:+.5f} px of sag per px of height  (a lens: one sign)")
        print(f"   r       {r_p:+.3f}  over {len(rows)} frames, height range "
              f"{np.ptp(dy):.0f} px")
        print(f"   sag crosses zero at row {zero:.0f}  "
              f"(a lens: the optical centre, row {cy:.0f})")
        print("   -> " + ("consistent with a lens" if abs(r_p) > 0.6 else
                          "NOT a lens signature: the bow does not track image height"))

    # The direct refutation, which needs no regression and no fitted intercept. A radial
    # model bows a line by an amount proportional to its distance from the optical axis,
    # so a horizon running THROUGH the axis must come back flat. If those frames bow, the
    # bow is not radial, whatever the correlation happens to be.
    on_axis = np.abs(dy) < 30
    print(f"\n   the decisive frames: horizons within 30 px of the optical centre row")
    if on_axis.sum() == 0:
        print("   none in this sample -- no direct refutation available")
    else:
        print(f"   {on_axis.sum()} frames, |sag| median {np.median(np.abs(sag[on_axis])):.1f} px, "
              f"max {np.abs(sag[on_axis]).max():.1f} px")
        print("   a radial lens must bow these by ZERO. " +
              ("It does not -- the bow is scene structure."
               if np.median(np.abs(sag[on_axis])) > 3 * np.median(rms) / 2 else
               "They are flat, which is what a lens would give."))

    # --- step 3: plumb-line fit -------------------------------------------------
    lams = np.round(np.linspace(-a.range, a.range, 31), 4)

    if a.inject:
        print(f"\n3a. POSITIVE CONTROL: bend a straight horizon by lambda = {a.inject:+.3f}")
        pres, pbest = positive_control(rows, a.inject, lams)
        flat = [t for t in pres if abs(t[0]) < 1e-9][0]
        print(f"    uncorrected mean |sag| {flat[1]:7.2f} px   "
              f"(a straight line bent by the model)")
        print(f"    recovered lambda {pbest[0]:+.3f}, residual |sag| {pbest[1]:.3f} px")
        ok = abs(pbest[0] - a.inject) <= 2 * float(np.diff(lams)[0])
        print(f"    -> {'RECOVERED' if ok else 'FAILED TO RECOVER'}: the sweep "
              f"{'can' if ok else 'CANNOT'} see a real lens through this horizon.")
        if not ok:
            print("    Everything below is uninterpretable until this passes.")

    print("\n3. PLUMB-LINE FIT: the lambda that flattens every frame's horizon at once")
    res = sweep(rows, lams)
    worst = max(m for _, m in res)
    best = min(res, key=lambda t: t[1])
    base = [t for t in res if abs(t[0]) < 1e-9][0]
    for lam, m in res:
        bar = "#" * int(round(m / worst * 40))
        mark = ("   <- best" if lam == best[0] else
                ("   <- uncorrected" if abs(lam) < 1e-9 else ""))
        print(f"   lam {lam:+.2f}   mean |sag| {m:7.2f} px  {bar}{mark}")
    print(f"\n   best lambda {best[0]:+.3f}: mean |sag| {best[1]:.2f} px against "
          f"{base[1]:.2f} px uncorrected ({(1 - best[1] / max(base[1], 1e-9)) * 100:+.1f}%)")
    print(f"   for comparison, the median tree-line residual is {np.median(rms):.2f} px")

    lam_list = ([float(s) for s in a.lam.split(",")] if a.lam else [best[0]])

    # --- step 4: what undistortion actually does to stage 1 ---------------------
    print("\n4. STAGE 1 RE-RUN ON RESAMPLED FRAMES")
    for lam_star in lam_list:
        stage1_under(lam_star, rows, valid, sag, rms)


def stage1_under(lam_star, rows, valid, sag, rms):
    print(f"\n   --- lambda = {lam_star:+.3f} "
          f"({'barrel correction' if lam_star < 0 else 'pincushion correction'}) ---")
    if abs(lam_star) < 1e-9:
        print("   lambda is zero; the remap is the identity and stage 1 cannot move.")
        return
    mx, my, inside = undistort.undistort_maps(lam_star)
    valid_u = (cv2.remap(valid.astype(np.uint8), mx, my, cv2.INTER_NEAREST) > 0) & inside
    fov = undistort.fov_retained(lam_star)
    print(f"   field of view retained {100.0 * fov:.1f}% of the source frame "
          f"({100.0 * (1 - fov):.1f}% cropped away)")
    print(f"   destination pixels with a source: {100.0 * inside.mean():.1f}%; "
          f"valid after masks {100.0 * valid_u.mean():.1f}% "
          f"(was {100.0 * valid.mean():.1f}%)")

    # The crop confounds the sky fraction: the before-frame sees picture the after-frame
    # has thrown away. Compare over the source region that survives, not over the whole
    # frame, or a crop that happens to cut ground reads as "more sky was found".
    vis_src = undistort.visible_source(lam_star)

    cap = cv2.VideoCapture(CLIP["video"])
    comp = []
    for r in rows:
        cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"])
        ok, img = cap.read()
        if not ok:
            continue
        sl = skyline.split(undistort.remap(img, mx, my), valid_u)
        hp = horizon_points(sl)
        fit = robust_quad(*hp) if hp else None
        # Does the SPLIT move, or only the pixels? Undistortion is a bijection, so a
        # pixel that looked like tree still looks like tree; carrying the before-split
        # through the same remap and intersecting it with the after-split separates
        # "stage 1 decided differently" from "stage 1 decided the same thing elsewhere".
        sky_b = cv2.remap(r["sky_mask"].astype(np.uint8), mx, my, cv2.INTER_NEAREST) > 0
        inter = float((sky_b & sl.sky).sum())
        union = float((sky_b | sl.sky).sum())
        comp.append(dict(sky_before=float(r["sky_mask"][vis_src].mean()),
                         sky_after=sl.sky_fraction,
                         iou=inter / union if union else float("nan"),
                         sag_after=fit["sag"] if fit else float("nan"),
                         rms_after=fit["rms"] if fit else float("nan")))
    cap.release()

    sb = np.array([c["sky_before"] for c in comp]) * 100
    sa = np.array([c["sky_after"] for c in comp]) * 100
    g_a = np.array([c["sag_after"] for c in comp], float)
    r_a = np.array([c["rms_after"] for c in comp], float)
    lost = int(np.isnan(g_a).sum())
    print(f"   sky fraction        {np.median(sb):6.2f}% -> {np.median(sa):6.2f}%   "
          f"(median, both over the retained field of view)")
    print(f"   |sag| of horizon    {np.median(np.abs(sag)):6.2f}  -> "
          f"{np.nanmedian(np.abs(g_a)):6.2f} px")
    print(f"   tree-line residual  {np.median(rms):6.2f}  -> {np.nanmedian(r_a):6.2f} px")
    print(f"   frames that lost a fittable horizon after the remap: {lost} of {len(comp)}")
    iou = np.array([c["iou"] for c in comp], float)
    print(f"   IoU of the split against the OLD split carried through the same remap: "
          f"{np.nanmedian(iou):.3f}")
    print("   (1.000 would mean undistortion relocated every sky pixel and relabelled none.")
    print("    Stage 1 is a per-pixel appearance decision, so this is the number that says")
    print("    whether undistortion changed stage 1 at all, as opposed to moving it.)")
    print("\n   The sag after a REMAP is not step 3's measurement: step 3 moved coordinates")
    print("   only. A gap between the two is resampling, not optics.")


if __name__ == "__main__":
    main()
