# EXP-017 — motion-first, with the parallax corrections

Stage 0, stage 1, and the stage-2 geometry (steps 1-3). Nothing here detects anything
end to end yet: the multi-frame tracker is the next build.

## Why this exists

EXP-016 accumulated differential motion over up to 15 frames and failed its bars. The
reason it failed is geometric, not statistical. A static point at depth `Z_t`, while the
homography fits a plane at `Z_g`, has leftover image motion of about

    parallax ≈ f · T⊥ · (1/Z_t − 1/Z_g)

With the camera translating consistently, that accumulates **linearly and in a consistent
direction** — exactly like a translating object. EXP-016's `z = |Σd| / (√k·σ)` only
measures *how much*, so it cannot separate them, and it did not: clutter reached z = 62
against the drone's 15-frame ceiling of ~14.

The user's observation stands and is the reason motion stays the detector: over 10–15
frames the drone becomes obvious to the eye. What the eye uses and the statistic throws
away is **direction** — a static point's parallax points along the line to the epipole,
always, by geometry — and **structure consistency**, that one scalar depth explains a
static point's whole track.

Both need to know where the depth discontinuities are, which is what stage 1 provides.

## Stage gate

Only the **depth-aware ring** waits on the sky/scene split being verified -- it is only
as good as that boundary, and building it on an unverified segmentation is how EXP-012a
happened. The epipolar and structure tests need the epipole and the KLT grid, not the
sky mask, so steps 1-3 below were built in parallel and do not depend on that session.

## Files

| file | what it is |
| --- | --- |
| `clipcfg.py` | per-clip settings; one copy per clip directory, first on `PYTHONPATH` wins |
| `masks.py` | stage 0: composing the static masks, held as named layers so a veto can be attributed |
| `calibrate_masks.py` | builds the prop and ladder masks across **every** clip from one airframe |
| `skyline.py` | stage 1: the sky/scene split, the uncertain band, and the horizon |
| `debug_video.py` | renders both stages back onto the clip for a human to check |

## Running it

```bash
# O4
PYTHONPATH="experiments/exp017_motion_first;." py -3.13 -m calibrate_masks
PYTHONPATH="experiments/exp017_motion_first;." py -3.13 -m debug_video --start 650 --end 964

# analog -- its clipcfg goes first, the shared modules come from the O4 directory
PYTHONPATH="experiments/exp017_motion_first/analog_catch_2;experiments/exp017_motion_first;." \
    py -3.13 -m calibrate_masks
PYTHONPATH="experiments/exp017_motion_first/analog_catch_2;experiments/exp017_motion_first;." \
    py -3.13 -m debug_video --start 441 --end 800
```

## What stage 0 changes, and what it does not

Measured 2026-09-28 on `first_catch`, top-200 `win_b5_e4` peaks over 15 frames:

| masks applied | share of peaks in the real scene |
| --- | ---: |
| warp border + 24 px margin + HUD | 40.2% |
| ... + EXP-015's `screen_fixed` (4.36% of frame) | 41.2% |

**The existing screen-fixed map buys one point.** Not because it masks the wrong pixels,
but because the candidate budget is a fixed top-N *by rank*: masking 4.36% of the frame
promotes the next 4.36% of clutter into the budget instead of removing load. All 200
requested peaks were still available in every frame.

So stage 0 is only half the fix. The other half is a **threshold** candidate budget rather
than top-N, which is a stage-2 decision and is recorded here so it does not get lost.

## Known gaps, as of the first calibration pass

These are real and unresolved. They are listed rather than tuned away because the
verification pass is what should settle them.

- **O4: only the right propeller is masked** (prop mask 1.28%). The left blade survives
  the cross-clip minimum, so its flicker is genuinely lower in at least one of the six
  clips. Lowering the threshold to catch it pulled in near ground at the bottom of the
  frame instead, which is why `edge_touching` now also rejects components that run off the
  top or bottom.
- **O4: the swept centre dashed bar is not masked.** The MANIFEST already records that the
  HUD mask misses it; the ladder rule does not catch it either.
- **analog: the ladder mask covers the labelled drone in 55 frames, first at 588.** Down
  from 98 frames first at **491** — the acquisition frame the bars are set on — after the
  span-fill was changed to cluster lit rows instead of bridging topmost to bottommost.
  55 frames is still recall lost before the detector runs, and it is the first thing to
  fix.
- **analog's own HUD mask is 15.85% of the frame** and stage 0's union is 23.4%. That is a
  lot of picture to give up on a 960×720 capture.

## What this deliberately does not do

- No detection, no scoring, no `detections.jsonl`. Nothing here can be compared to
  EXP-015/016 numbers yet.
- The sky/scene split is a **two-level depth prior**, not semantic segmentation. Sun,
  bloom and thin cloud read as sky, which is correct for parallax purposes — they are at
  infinity too.
- Thresholds in `skyline.py` are percentiles of each frame's own distribution, so nothing
  is a fixed grey level. That is what lets one rule serve a digital link and CVBS.

---

# Steps 1–3 (2026-09-28)

## Files added

| file | what it is |
| --- | --- |
| `geometry.py` | homography (EXP-015's, unchanged), residual field in **current-frame** coords, epipole estimation with its own reliability flag |
| `criteria.py` | step 2 (epipolar direction) and step 3 (structure consistency), pure functions |
| `budget.py` | step 1: threshold candidate budget, two-fold calibrated on empty frames |
| `calibrate_budget.py` | fits the threshold and checks masking now reduces load |
| `measure_tests.py` | single-pair preview of step 2 on real footage |
| `selftest.py` | synthetic geometry checks — no video, no weights |

## Step 1 — threshold budget: works, but the masks are not the lever

O4 `first_catch`, 40 empty frames, one frozen `tau = 1.661`:

| mask set | threshold budget | top-200 budget |
| --- | ---: | ---: |
| edge + HUD (EXP-015 baseline) | 150.4 | 200.0 |
| + prop + ladder (EXP-017) | **143.3** | 200.0 |

The structural fix is real: under top-N the count is pinned at exactly 200 whatever you
mask, and under a threshold it falls. **But it only falls 4.7%** while the added masks
cover 6.7% of the frame — so the prop and ladder regions carry roughly *average* candidate
density, not concentrated clutter.

**Read this as a negative result for stage 0 as a lever.** The load is ~150 candidates per
frame spread across the real scene, which is EXP-012b's ~115 terrain candidates again. No
amount of overlay masking addresses it. The geometry has to.

## Step 2 — epipolar direction, measured single-pair on real footage

30 empty and 30 drone frame pairs, `tau = 1.661`:

| | empty | drone |
| --- | ---: | ---: |
| candidates/frame (trackable) | 68.4 | 92.4 |
| rejected as epipolar (static parallax) | **27.3%** | 34.7% |
| unjudged (FOE / rotation, passed on) | **50.4%** | 16.1% |
| surviving load | 49.7 | 60.3 |
| labelled drone survives | — | **14 of 16 (88%)** |

**The asymmetry is in the right direction and it is the first one this project has
measured**: the test removes about a third of the clutter while keeping the target in 88%
of the frames where the target is a candidate at all.

**But it is a 27% cut, not the several-fold I predicted in the plan.** That prediction is
not supported by this measurement and should not be repeated until the tracker exists.
Two reasons it understates, neither of them an excuse:

  * This is **one frame pair**. The test that matters runs on accumulated `Sum d` over
    10–15 frames, where a static point must read epipolar every time and the direction
    estimate has far better SNR. Single-pair rejection is a floor, not the result.
  * **50.4% of empty-frame candidates are unjudged** — the pair is rotation-dominated or
    the candidate sits near the focus of expansion. The test refuses rather than guessing,
    which is correct, but it means half the empty-frame load is currently untouched by it.
    That 50% is the next thing to understand.

## Step 3 — structure consistency: maths verified, not yet measurable

`selftest.py` passes on synthetic scenes where the answer is known by construction:

  * epipole recovered to **0.0 px** of truth; 100% of static parallax reads epipolar
  * a crossing target reads as moving (sin 0.44)
  * a banking flight separates static (drift 2.1σ) from moving (**9.5σ**)
  * all three refusals fire: rotation-dominated pair, track at the FOE, too few steps

**The named degeneracy is now an asserted test, not a surprise.** On a straight
constant-velocity flight the structure test *cannot* see a collinear constant-velocity
target (drift 1.2σ) — the classic plane+parallax blind spot. The direction test has to
carry that case, and `confirm()` runs them as an OR for exactly this reason.

Margin to watch: the static case reads 2.1σ against a 3.0σ threshold. That is thin, and on
real footage with worse tracking it may not hold.

Step 3 needs the multi-frame tracker before it can be measured on video. That is the next
build.

## Next

1. **The tracker.** Accumulate `Sum d` over k ≤ 15 with the speed cap, run both tests at
   confirmation, re-run EXP-016's protocol and bars unchanged otherwise.
2. **Understand the 50% unjudged.** If empty frames are rotation-dominated, the detector
   should say so per frame rather than carrying an untested load.
3. Depth-aware ring, once the skyline session verifies the boundary.

---

# Stage 1 under fisheye correction (2026-09-29) — EXP-018 in the ledger

## Files added

| file | what it is |
| --- | --- |
| `undistort.py` | division-model radial undistortion, both directions, plus `py -3.13 -m undistort` self-check |
| `check_fisheye.py` | is there a distortion to correct, and does stage 1 care |

```bash
# O4
PYTHONPATH="experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
    py -3.13 -m check_fisheye --frames 40 --range 1.2 --lam=-0.20
# analog
PYTHONPATH="experiments/exp017_motion_first/analog_catch_2;experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
    py -3.13 -m check_fisheye --frames 40 --lam=-0.60
```

## The short version

**O4 has no fisheye to correct. Analog does. Neither changes stage 1.**

Stage 1 already emits a horizon, which is the only plumb line this footage offers. A tree
line has relief, so a bow is not by itself evidence of a lens — but a lens bows a line by
an amount set by its distance from the optical axis, and is **zero** when the line runs
through it, while terrain is pinned to the scene instead.

| | O4 | analog |
| --- | ---: | ---: |
| median \|sag\| / tree-line residual | 159.5 / 40.1 px | 38.7 / 13.7 px |
| sag crosses zero at row | 325 (axis 540) | 368 (axis 360) |
| best lambda | none over ±1.2, wants *pincushion* | **-0.60**, a clean bowl |
| \|sag\| at best lambda | — | 38.7 -> **14.2** px |
| FOV retained at that lambda | — | **59.3%** |
| split IoU vs the old split, remapped | **0.989** (at -0.20) | 0.885 |

The decisive O4 number needs no regression: **11 frames whose horizon runs within 30 px of
the optical centre bow by a median 209 px.** A radial model cannot do that. Rendering one
shows why — tall near trees at both frame edges, distant tree line low in the middle.

**Why stage 1 cannot benefit, on either clip.** Undistortion is a bijection on pixels. It
relocates the sky boundary; it cannot relabel it. Stage 1 thresholds texture, luminance and
blue excess against percentiles of the frame's own distribution, so a pixel that looked
like tree still looks like tree at its new coordinates. The 0.989 IoU is the measurement of
that, and the residual 1.1% is boundary resampling, not a better decision.

## The positive control, and why it is not optional

A synthetic straight horizon at each real frame's own height and span is bent by a known
lambda and put through the same sweep; the argmin must come back at that lambda.

    injected -0.160, recovered -0.160, residual |sag| 0.000 px    (both clips)

Without it, "the sweep found nothing" and "the sweep cannot find anything" print the same
output. That is the EXP-012a failure mode exactly.

## One number that was wrong first

The first pass reported "remap keeps 100.0% of the frame" and a sky fraction **rising**
28.0% -> 30.6%. Both were artefacts of the same mistake: `undistort_maps`'s `inside` flag
says which *destination* pixels have a source, and for a barrel correction that is always
all of them, because the map zooms in. The discarded picture sits at the *source* edges and
only the forward map sees it — so a 20% crop read as "keeps 100%", and the sky fraction
rose only because the crop removed ground. `fov_retained` / `visible_source` fix it, and
the comparison is now made over the retained field of view, where it does not rise.

## What is left open

The one place a radial error actually bites is **stage 2**, not stage 1: the grid-KLT
homography, the epipole and the parallax residual all assume a pinhole camera.
`undistort_points` corrects coordinates with **no resampling and no FOV loss**, so on
analog it is nearly free to try. EXP-018 did not measure it.

No undistorted copy was written to `data/processed/`. On O4 there is nothing to undo; on
analog the only honest lambda costs 41% of the frame on a fixed canvas, and if it is ever
wanted it should expand the canvas rather than crop.

---

# Stage 1 and stage 2 on one picture (2026-09-29) — EXP-019 in the ledger

## Files added

| file | what it is |
| --- | --- |
| `overlay_video.py` | stages 0, 1 and 2 rendered together, plus a health report over the span. `--no-video` gives the report alone. |

`debug_video.py` stays: it is stage 0/1 only and is the right tool when stage 2 is not
the question.

```bash
PYTHONPATH="experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
    py -3.13 -m overlay_video --start 650 --end 964
PYTHONPATH="experiments/exp017_motion_first/analog_catch_2;experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
    py -3.13 -m overlay_video --start 441 --end 800
```

Output: `stages_first_catch_650_964.mp4` (315 frames) and
`stages_catch_2_441_800.mp4` (360 frames).

## Stage 1, measured rather than eyeballed

| | O4 650-964 | analog 441-800 |
| --- | ---: | ---: |
| sky fraction, median (p10 / p90) | 21.2% (17.4 / 31.0) | 21.5% (**0.0** / 24.0) |
| uncertain band | 7.0% of frame | 8.9% of frame |
| frames with **no sky at all** | 0 of 315 | **75 of 360** |
| frame-to-frame sky IoU, median | 0.960 | 0.960 |
| consecutive pairs below 0.90 IoU | 30 of 314 | 56 of 359 |
| **where the labelled drone is put** | **scene 257, sky 0** | **scene 158, sky 44, uncertain 22** |

Three things, in order of how much they matter.

**1. The split calls the airborne target "scene", nearly always.** On O4 that is 257 of 257
labelled frames — the drone is never once put in sky. On analog it is 158 of 224. The user
said the tree line is sky for their purposes, and this is that correction with a number on
it: the split is separating *blue sky* from everything, when what the ring test needs is
**far from near**, and a drone at shallow elevation against a distant tree line is far.
This is a definition error, not a threshold to tune.

**2. Analog loses the sky entirely in 21% of frames.** A depth prior that is absent on a
fifth of frames cannot be a precondition for the ring test; the ring test would have to
refuse on those frames, and refusing on a fifth of the clip is not a usable design.

**3. Flicker is real but secondary** -- 0.960 median IoU is healthy, and ~10-16% of
consecutive pairs dipping under 0.90 is the boundary breathing, not the split chattering.

## Stage 2 on real video, and it is weaker than step 2 suggested

| | O4 | analog |
| --- | ---: | ---: |
| epipole reported reliable | 297 of 315 | 310 of 360 |
| epipole anisotropy, median | 0.38 (threshold 0.15) | 0.29 |
| **epipole frame-to-frame jump, median** | **126.8 px** (p90 293.4) | **98.9 px** (p90 232.5) |
| **epipole agreement with the background (angular)** | **34.2%** | **29.4%** |
| the same for random directions | 22.8% | 22.8% |
| candidates/frame | 87.4 | 47.2 |
| rejected as epipolar | 32.5% | 22.8% |
| unjudged (FOE / rotation) | 18.2% | 28.6% |
| surviving load | 59.0/frame | 36.5/frame |
| **rejection among JUDGED candidates** | **39.7%** | **31.9%** |
| **the same for random directions** | 22.8% | 22.8% |
| **lift over chance** | **x1.74** | **x1.40** |
| drone survives | 134 of 163 (82%) | 65 of 67 (97%) |

**The chance baseline is the number that was missing.** A candidate is rejected when
|sin| to the epipolar line is under 0.35, so a candidate pointing in a **random** direction
is rejected with probability `2*asin(0.35)/pi = 22.8%`. Any rejection rate has to be read
against that floor. O4's is 1.74x chance and analog's is 1.40x. The direction test is doing
something real, and it is doing much less than the several-fold I claimed in the plan.

**The epipole hops ~100-130 px between consecutive frames.** A real focus of expansion
drifts smoothly with the manoeuvre. This is single-pair noise: EXP-012b measured the
post-homography residual at ~1 px median, and a 1 px residual has a direction dominated by
tracking noise, so each point votes weakly and the fit wanders. The fix is the one already
planned -- **accumulate over the window and fit the epipole once per window, not per pair**
-- which should cut the noise as sqrt(k). Nothing here says the epipole is unfindable; it
says a single frame pair cannot find it well.

**A robust fit is worse, so this is not an estimator bug.** RANSAC over the same points
(200 hypotheses, 3 px tolerance, inlier refit) gave a median frame-to-frame jump of
**454.7 px** against least squares' 93.9 px on frames 700-758, and the two answers differed
by a median 282 px. Least squares averaging 1000 weak votes beats a robust fit chasing a
handful of strong ones, which is the signature of noise-limited rather than
outlier-limited data.

## A metric of my own that was broken

`geometry.estimate_epipole`'s `inlier_fraction` counted voting lines within a fixed 3 px
of the solution, on `|n . (p - x)|` -- the epipole's offset from the line through x along
mu. **That offset grows with `|p - x|` for a fixed angular error**, so a fixed pixel
tolerance is only meaningful next to the epipole and read ~0.008 everywhere else. It was
measuring distance, not agreement. The RANSAC run above maximised that same quantity and
landed 282 px away, which is what optimising a meaningless objective looks like.

It is now the **angular** form -- the fraction of background points whose residual is
within the same 0.35 sin threshold the direction test uses -- so it states directly what
share of the background the fitted epipole would itself call static. It is reporting only
and was never part of `reliable`, so no earlier verdict changes.

Measured that way the epipole explains **34.2% of the O4 background and 29.4% of
analog's**, against the same 22.8% floor a random direction would hit -- x1.50 and
x1.29. The epipole is real and it is weak, and on analog it is barely locating anything
at all. This is the same story the candidate lift tells (x1.74 / x1.40) arriving by an
independent route, which is the reason to trust both.

## Where the survivors sit (the reason both stages are drawn together)

| | band's share of frame | survivors in the band |
| --- | ---: | ---: |
| O4 | 7.0% | **12.3%** of 13,582 |
| analog | 8.9% | **7.9%** of 8,264 |

**EXP-016's horizon failure does not appear at the candidate stage.** O4 is 1.76x
over-represented at the boundary, analog is 0.89x -- *under*-represented. EXP-016's 57.5%
was about **confirmed multi-frame tracks**, so these are not in contradiction, but they
place the horizon problem in the ring test and the accumulation, not in candidate
generation. **That is an argument for deprioritising the depth-aware ring** relative to the
tracker, and it is the first evidence either way.

## What the render shows that a table cannot

Frame 946 (O4), checked by eye: red rejections cluster correctly on the near ground and the
left canopy, the drone is boxed and green, and the epipole's degenerate circle sits
directly under the target -- the intercept geometry the whole design is fighting, visible
in one frame. Two defects also show up:

- **Green survivors sit in the gaps between HUD glyph boxes.** The dilated HUD mask covers
  each glyph but not the space between them, and the glyph edges leak difference energy
  into it. Visible around `4.04v` and `24.3V`.
- **The uncertain band is a thick ribbon** at 7-9% of the frame. That is `UNCERTAIN_PX`
  = 24 px dilate/erode as designed, but it is a lot of picture to declare unjudgeable when
  the drone spends its time near that boundary.

## Next, in the order this changes

1. **Redefine stage 1 as far/near, not sky/ground.** It is currently wrong on the target
   257 times out of 257 on O4. Belongs with the skyline session.
2. **Fit the epipole per window, not per pair.** The 100-130 px hop is the largest single
   thing limiting the direction test, and the tracker needs the window anyway.
3. **Tracker.** Unchanged as the next build.
4. Depth-aware ring, now with less urgency -- see the survivor table above.

---

# The 5-frame window: persistence works, the direction test's premise does not (2026-09-29)

EXP-020 in the ledger. Asked for: average each candidate's direction over 5 frames, and
drop anything that does not appear in at least 4 of them.

## Files added

| file | what it is |
| --- | --- |
| `window.py` | the accumulation: chaining, per-point residual summing, the pooled epipole, persistence. `py -3.13 -m window` self-checks it. |
| `overlay_window.py` | the k-frame renderer and its report. `overlay_video.py` stays as the single-pair baseline to compare against. |

```bash
PYTHONPATH="experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
    py -3.13 -m overlay_window --start 650 --end 964 --appear-radius 14
```

### Two additions made for EXP-024 (the window-length sweep)

`--direction` (default **off**) applies the epipolar direction test on top of persistence.
It matches `overlay_stage2.py`'s flag of the same name and the same default. Before this,
`overlay_window` always applied the test, so its videos painted red rejections from a test
EXP-021 measured as rejecting at chance -- and its "kept" figure was the direction-gated
one, not the persistence-alone one the standing recommendation refers to. With the test off
the epipole/lift block is skipped rather than printed as a 0.00 lift.

The report now prints **every `min_appear` operating point from one pass**:

```
  OPERATING POINTS (persistence alone, every threshold from this one pass)
    need  load/frame     drone kept  seeds held
     4/5        25.8     79 ( 42%)       19.4%
```

`Track.appearances` does not depend on the threshold -- `Track.verdict` only compares
against it -- so one render measures the whole load-versus-recall curve for that window
length. Sweeping `--min-appear` by re-rendering costs k times as much for the same table,
and invites reading thresholds off different spans. Only the **video** is drawn at the one
`--min-appear` passed; the table covers the rest.


## The headline

**Persistence is the win. The direction test has to go, and not because it is tuned
wrong -- because the residual field on this footage is not a parallax field at all.**

## 1. The accumulation does exactly what it was supposed to

Measured over 61 windows on `first_catch` 900-960:

| | |
| --- | ---: |
| coherence `\|Sum mu\| / Sum\|mu\|` | **0.920** |
| ... a perfectly coherent (pure parallax) accumulation | 1.000 |
| ... a random walk (pure tracking noise) | 0.447 |
| `\|Sum mu\|` over 5 frames | 3.97 px |
| per-step `\|mu\|` | 0.93 px |
| direction anisotropy, windowed vs single-pair | **0.389** vs 0.355 |
| epipole frame-to-frame jump, windowed vs single-pair | **46.1 px** vs 126.8 px |

The residuals add up rather than cancelling, the directions sharpen, and the epipole stops
hopping. Every mechanical claim made for the window holds.

## 2. And the epipole still explains nothing

    agreement with the background it was fitted to:  22.1%
    the same figure for random directions:           22.8%

**The pooled epipole is at chance on the very points it was fitted to.** Single-pair was
34.2% (EXP-019) -- so reducing the noise made the fit *worse*, which is only possible if
the thing being fitted is not there.

## 3. Why: the field is a uniform shift, not a parallax field

Two models, scored identically on the same accumulated field, with the same 0.35 |sin|
the direction test uses:

| model | frames 700-760 | frames 900-960 |
| --- | ---: | ---: |
| **A. uniform shift** -- every residual points the same way | **37.2%** | **38.5%** |
| **B. parallax from a focus of expansion** | 30.3% | 23.8% |
| chance | 22.8% | 22.8% |
| `\|mu\|` vs distance to the epipole, correlation | **+0.171** | **+0.061** |

Parallax means `mu = gamma * (e - x)`, so `|mu|` **must** grow with distance from the
epipole. It does not. The field is better described as a small near-uniform misalignment
left by the homography than as depth-induced parallax -- which is the same thing as saying
the epipole is at or near infinity, where fitting it as a *point* is meaningless.

**This reinterprets EXP-019.** The x1.74 lift was weak evidence that survived only because
single-pair noise hid the structure. With the noise reduced, the structure is visible, and
it is parallel rather than radial.

## 4. What persistence buys, on its own

`first_catch` 900-964, appearance radius 14 px:

| | load/frame | drone kept |
| --- | ---: | ---: |
| single-pair, EXP-019 (650-964) | 59.0 | 82% |
| **persistence alone (>= 4 of 5)** | **31.2** | **78%** |
| persistence + direction test | 25.7 | **45%** |

**Persistence roughly halves the load for ~4 points of recall.** Adding the direction test
buys 5.5 more candidates/frame and costs **33 points of recall** -- and since its rejections
are at chance, those 22 rejected drone frames are lost at random. That is not a trade worth
making, so the direction test is off by default until the field is modelled properly.

Appearance radius sweep on the same span (drone kept / load per frame):

    8 px  31% / 18.2      14 px  45% / 25.7      22 px  46% / 38.4      32 px  49% / 49.3

It saturates by 14 px, which is why 14 is the O4 default. On the 960-wide analog clip the
same physical radius is 9 px.

## 5. Three errors of mine the self-test caught first

None of these would have been visible on video -- they would have come back as a null
result indistinguishable from "the idea does not work".

1. **Pooling votes instead of accumulating per point.** The first `fit_window` threw all k
   pairs' votes into one fit. That gives k times as many votes of the *same* poor quality,
   so directional concentration never improves and the reliability gate still refuses: at
   0.8 px noise, 5 of 20 pooled windows came back reliable against 9 of 24 single pairs.
   Accumulating each point's residual first is different in kind, and fixed it.
2. **The synthetic's noise model was wrong.** It drew fresh noise for each pair, but a real
   frame is measured *once* -- its error enters one pair as the current endpoint and the
   next as the previous one, with opposite sign, so intermediate noise largely cancels
   under accumulation. Independent draws destroy that cancellation and made accumulation
   look worse than it is.
3. **Selection bias in the comparison.** Errors were being compared only where each
   estimator called itself reliable, so the per-pair fit was scored only on the windows it
   found easy. Scoring both on every window reversed the result.

After the fixes, at 0.8 px of noise: per-pair reliable in **1 of 9** windows with **121 px**
of error; accumulated reliable in **9 of 9** with **5.7 px**. The machinery is right. The
footage just does not contain the field it is designed to exploit.

## 6. What should happen next

1. **Replace the direction test with a local-parallel test.** The field is parallel, so the
   question worth asking is whether a candidate's accumulated residual differs from the
   residual of *its own surroundings* -- EXP-016's ring, but on direction instead of
   magnitude, and with no epipole to estimate. That matches what the field actually is.
2. **Keep persistence.** It rests only on "a real thing is visible in consecutive frames",
   needs no geometry, and is the only part of stage 2 currently earning its cost.
3. **Re-examine whether a bigger baseline exists.** Parallax at 1/30 s may simply be too
   small. Comparing frames k apart rather than adjacent, at the same total window length,
   would raise `T_perp` without raising the frame rate cost -- and if parallax is genuinely
   sub-pixel at this baseline, the whole plane+parallax line of attack is misconceived for
   this footage and should be dropped rather than tuned.

# Stage 2, both branches: a sky silhouette detector and a two-plane scene test (2026-09-29)

EXP-022 in the ledger. Asked for: run stage 2 over many frames with an overview video,
to the plan's two-branch design — 2a multi-scale negative-polarity blob detection on sky
with an IRST local-contrast statistic, 2b parallax-discounted motion with a layered
homography, an epipolar direction rejector and depth-aware rings.

## Files added

| file | what it is |
| --- | --- |
| `silhouette.py` | stage 2a. Negative-polarity scale-normalised DoG pyramid, the `c = (median(ring) - min(core)) / sigma_ring` statistic, and the cloud / bloom / size rejections. `py -3.13 -m silhouette` self-checks it (8 synthetic properties). |
| `layers.py` | stage 2b. Two-plane RANSAC fit with a separation guard, per-layer residuals, the epipolar **rejector**, and the depth-aware ring. `py -3.13 -m layers` self-checks it (10 properties). |
| `overlay_stage2.py` | the two-branch renderer and its report, including a single-plane control on identical frames. |
| `probe_sky.py` | diagnostic: what the sky branch sees at the labelled target, and which of the four upstream stages threw it away when it saw nothing. |

```bash
PYTHONPATH="experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
    py -3.13 -m overlay_stage2 --start 650 --end 964
```

## Two deviations from the plan, both forced by measurements already in this repo

1. **The sky branch runs ungated, not on stage-1 sky.** EXP-019 measured `skyline.split`
   putting the labelled airborne target in `scene` in **257 of 257** O4 frames. Gated as
   the plan specifies, this branch scores zero on O4 and the run says nothing about the
   detector. It therefore runs over the whole valid frame, and the report cuts the result
   by stage-1 label — which measures the detector *and* what the gate would have cost.

2. **The epipolar direction test is off by default** (`--direction` enables it). EXP-021
   measured a uniform-shift model outscoring a focus of expansion on every span, with
   `|mu|` essentially uncorrelated with distance from the epipole. The plan's claim that
   this test "kills the 57.5% of analog false tracks at the horizon" is separately
   contradicted by EXP-019, which located that failure in the tracker rather than in
   candidate generation. It is still implemented and still measured, because this run
   changes its input — a two-plane residual field instead of one — and the
   single-plane control in the report is what says whether that mattered.

## Four defects the self-checks and the probe caught before any long run

None would have raised an exception; each returns a plausible-looking number.

1. **`sigma_ring` was measured on a blurred image.** A Gaussian blur divides white noise
   by ~`2*sqrt(pi)*sigma_blur`, which drove `sigma_ring` under `SIGMA_FLOOR` at every
   realistic sky noise level. `c` then read the *floor* rather than the sky and stopped
   depending on sky noise at all — the one property the statistic exists to have. Caught
   by the check that triples the sky noise and expects `c` to fall: it did not move
   (33.3 → 33.4). The ring is now measured on raw pixels, the core on the scale-matched
   blur, and the two sources are different images for stated reasons.
2. **The candidate cap was acting as a rank budget.** The sky branch returned exactly
   600 peaks every frame — `budget.py`'s documented failure mode, where masking clutter
   promotes the next clutter instead of reducing load. Cap raised to 4000 and the report
   now says when it binds.
3. **A frame-indexing error in the direction path.** `_track_back` returns positions
   newest-first, so indexing from the far end reached the *oldest* pair while using the
   *newest* pair's homography and layer fit — a residual measured across five frames
   against a one-frame plane. Only ever a wrong number, never an exception.
4. **"No opinion" was being counted as "passed".** `epipolar_reject` returns False both
   when it judges a candidate off-epipolar and when it refuses to judge at all (no usable
   epipole, at the FOE, barely moved). Conflating them put unjudged candidates into the
   denominator of the rejection rate, so the lift-over-chance measured how often the test
   *declined* rather than how well it discriminates. It now returns
   `(reject, judged, reason)` and the run reports `unjudged` separately.

## `c` is sub-proportional to sky noise, and that is a property of the formula

Tripling the sky noise cuts `c` by ~2, not by 3. `min(core)` is a minimum over ~14 px, so
it is a downward-biased order statistic whose bias grows with sigma, and that inflates `c`
on noisy footage — which is exactly the analog/CVBS case. `c` may be quoted as a ranking
score; it may not be quoted as pure SNR.
