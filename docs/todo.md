# Todo

Captured tasks. Add via `/todo <description>`. Check an item off by moving it to Done
with the date it was finished. A task overtaken by a later decision moves to Superseded
rather than being deleted — the reasoning is why the current plan is the current plan.

Mission ids (M1–M7) refer to the approved evaluation plan: baseline through edge-ready
model. They are ordered — each assumes the previous one landed. Two constraints run
through all of them: the **15 official test videos only** until fine-tuning needs more,
and **field deployment is a hard requirement**, so every experiment from M3 on records
inference cost alongside accuracy.

`## Open` is ordered by mission id, with unscheduled backlog last. Dates are capture
dates, not priorities.

## Open

- [ ] 2026-10-05 — [algo] **Tighten the chain, then re-run EXP-029 at EXP-028's operating
  point.** EXP-029 measured the moving factor working as designed and still not helping below
  ~12 candidates/frame. Two specific causes, both to attack:
  1. **Clutter accumulates travel through permissive association.** The 25 px/frame speed
     limit allows 125 px across a 5-frame window, so a chain that hops between two objects
     banks the hop as displacement — measured clutter median 19.97 px against the target's
     p10 of 15.63. Try a smaller reach, and requiring direction consistency *along the chain*
     (successive steps roughly collinear) rather than against an epipole, which EXP-029 and
     EXP-021 both found worthless.
  2. **A 5.42 kept/frame floor.** 25.8% of confirmed-track rows have no span yet and are kept
     as `unjudged`, so `--min-move` cannot reduce load past that. Those rows carry 42 of the
     drone frames, so refusing them outright is not free — needs a graded treatment.
  Then re-run with `--top 3` so the load sits at EXP-028's 0.93 shown/frame rather than 15x
  above it; EXP-029's numbers are not comparable to the shipping pipeline without it.
  **2026-10-07, EXP-031:** a third cause, found out of sample. On catch_5 the camera chases
  the drone, and measuring travel against the static scene removes it: 13 drone frames lost
  (37 → 24 in the top 3). Measure travel in raw picture coordinates, or take the larger of
  raw and compensated, before re-running.

- [ ] 2026-10-05 — [algo] **Take the EXP-028 kinematic gate to O4, then into the CLIs.**
  catch_4/5 are done (2026-10-05, the default holds; see Done). O4 remains: its fps and
  lens differ, so the limit must be re-derived. Then wire `KinematicTracker` into
  `baseline_detect` / `glad_detect` / `live_detect` on box centres, behind an off-by-default
  flag. Check the 16 frames where it held the drone back for a pattern first.

- [ ] 2026-10-04 — [algo] **Put the persistence gate on the sky branch's candidates instead
  of the motion branch's.** Measured in EXP-024's 2026-10-04 addendum: on `catch_2` 441-800,
  recut to the common 224 labelled frames, EXP-023's 2-frame sky branch beats the 5-frame
  motion window at *every* matched load by ~1.7x (32.6% at 5.3 false alarms/frame against
  the window's 26.8% at 18.0/frame). The cause is upstream of the gate — the motion
  front-end makes the target a candidate in 174 of 224 frames (77.7%) against the sky
  branch's 223 (99.6%) — so no window length or threshold can close it. The sky branch is
  also immune to the 1-in-6 duplicate frames, finding the target in 12 of the 60 where the
  motion branch finds none. **So the open question is whether persistence adds anything to
  the branch that is already winning**, where it would start from 99.6% rather than 77.7%.
  Needs candidate association across frames for sky-branch hits (`window._appears` by peak
  proximity already does this without flow, and EXP-024 established LK is unusable on analog
  at median 0 usable steps). Compare against EXP-023's c>=6 and EXP-025's top-3 cap at
  matched load, on the 224 denominator.
  **2026-10-05:** EXP-028's kinematic gate is a form of this, on the pooled answers: a
  2-hit confirmation with a speed-gated chain. At matched-or-lower load it holds recall
  (69 vs 67 in the top 3), so persistence-by-track does not cost what EXP-025b's 4-of-5 did.

- [ ] 2026-10-04 — [algo] **Chain candidate peaks instead of tracking pixels, and use the
  accumulated displacement as the statistic.** Measured in EXP-024's 2026-10-04 addendum:
  the target's ego-compensated displacement over 5 frames is median 28.4 px with p10 13.9,
  against the background field's median 3.18 px and p90 5.60 — non-overlapping tails, where
  the epipolar direction test on the same data runs at chance (30.7% vs 22.8%). So magnitude
  is the better statistic and `Track.total_mu` already holds it; only its direction is
  tested, with magnitude reduced to a 2 px veto. **What blocks it is association, not the
  statistic**: the target has a full-depth LK track in 2 of 19 sampled windows, and window
  length does not help (47% of seeds reach full depth at k=5, 32% at k=15). Peak-chaining by
  proximity is arithmetically viable — target step median 4.12 px, p90 12.67, so a ~13 px
  radius catches 90% of steps and admits 0.13 spurious candidates per step at 167.6
  candidates/frame — needs no texture, and a duplicate frame contributes no candidate rather
  than killing a tracker. One chain yields the displacement *and* the appearance count, so it
  replaces both halves of the present gate. **First, the cheap prerequisite:** add `total_mu`
  to the seed dump and measure magnitude separation against *surviving candidates* rather
  than background grid points, on O4 where tracks reach median depth 3 — grid points are a
  biased-optimistic comparison and that gap is the one unmeasured step.

- [ ] 2026-10-04 — [algo] **Drop duplicate frames before the window is built, then
  re-measure analog persistence.** `catch_2` repeats 1 frame in 6 (148 of 890, at
  `f % 6 == 4` exactly — 25 fps resampled to 30), and a duplicate pair produces zero
  candidates, so a nominal k-frame window carries only `k * 5/6` live frames. Measured in
  EXP-024: 7-of-7 keeps 0 of 60338 seeds because every 7-window contains a duplicate, 5-of-5
  is reachable in 17% of windows, and 4-of-5 is a perfect-run requirement rather than the
  4-in-5 it reads as. **Every analog persistence number in EXP-021/024 is measured through
  this.** The fix is in how the window is assembled, not a flag: skip pairs whose candidate
  list is empty (or whose frame diff is under ~2.0) when filling the deque, so k counts live
  pairs. Then re-run the k=5 operating curve and compare against EXP-024's. Unknown whether
  it recovers recall — the target may simply not be a candidate often enough — but no
  window-length or threshold question on analog is answerable until it is done.
  **Scanned: all 9 analog clips carry it (16.8-17.8%, gap 6); all 6 O4 clips are clean
  (0-0.8%, uncadenced), so O4's numbers stand.** The phase differs per clip (`catch_2` at
  `f % 6 == 4`, `catch_6` at `% 6 == 2`), so detect duplicates rather than assume a phase.
  Since it is the capture chain and not one file, the drop belongs at decode in
  `src/data/` — which makes it a change under `src/` with tests, not an experiment script.


- [ ] 2026-09-30 — [algo] **Extend the sky branch's scale ladder to cover the real target
  sizes, then re-measure everything.** The ladder spans 3–40 px of diameter on O4 and
  3–27 px on analog, while the labelled targets on those spans are a median **71.6 px
  (17–142)** and **37.0 px (18–80)**. The detector therefore fires at **0.048×** and
  **0.163×** the target's true size — it is matching a small dark sub-feature inside a box
  `on_drone` has grown by `max(10 px, 25%)`, not the airframe. **Nothing about the sky
  branch's recall means what it says until this is fixed**, and no threshold, size floor or
  large-blob ceiling should be tuned before it. `MAX_SCALE_FOR_TARGET` is mis-set for the
  same reason: at ~56 px of diameter it would reject the real O4 airframe at close range,
  so ceiling and ladder move together. Raised by the user, who was right that the target is
  ~20 px and not 3. Full write-up in the correction section of [experiments.md](experiments.md).

- [ ] 2026-09-30 — [algo] **Give the sky branch a persistence gate.** It is still purely
  single-frame, while the motion branch's k-of-5 rule cut that branch's load by 80% for ~4
  points of recall (EXP-021). A real target is visible in consecutive frames at a
  consistent scale; grain, chroma crawl and compression blocking are not. Likely a better
  false-alarm lever than any threshold, and it costs the branch's zero-latency property
  only if the single-frame hits are withheld — report them immediately and mark them
  confirmed on persistence instead. `window.py` already has the machinery.
  **2026-10-05:** `src/algo/kinematics.KinematicTracker` (EXP-028) is now the reusable
  machinery: tracks with a speed limit, hysteresis and evidence.

- [ ] 2026-09-29 — [algo] **Make the stage-1 uncertain-band policy a per-clip operating
  point.** EXP-023 measured both settings at matched false-alarm rate and **neither
  dominates**: refusing the band wins on O4 (40 vs 30 drone frames at 9.35 FA/frame),
  scoring it wins on analog (60 vs 49 at 3.30 FA/frame). The split is explained — EXP-019
  put the analog drone inside that band in 22 of 224 frames, so refusing it discards real
  targets there, while on O4 the band is mostly clutter. Worth ~10 recall points either
  way, so it must be chosen from the dump per clip rather than frozen in code.
  `overlay_sky.py --keep-uncertain` is the switch; the CSVs make the re-cut seconds.

- [ ] 2026-09-29 — [algo] **Drop the layered homography; one plane is measurably better.**
  EXP-022 fitted `H_ground` by RANSAC and refitted on the outliers for `H_far`, as the
  motion-first plan specifies, and measured it against a single-plane control on identical
  frames. **Two planes are worse on both clips and on every diagnostic**: epipole
  reliability 91% vs 94% on O4 and **51% vs 86%** on analog, background agreement 0.294 vs
  0.309 and 0.246 vs 0.277, `corr(|mu|, dist)` 0.104 vs 0.135 and 0.104 vs 0.174. The
  second plane is not imaginary — two distinct planes are found in 98% of O4 frames and 82%
  of analog at median separations of 5.06 and 3.31 px, well clear of the guard — so the
  mechanism is that splitting the points between two fits leaves each with less evidence,
  while the field being fitted was never radial to begin with. Delete the branch rather
  than tune it; `layers.py` keeps the depth-aware ring and the rejector, which are separate.

- [ ] 2026-09-29 — [algo] **`c` is a ranking score, not a false-alarm rate — stop quoting it
  as one.** The sky branch's whole appeal was that a threshold in units of sigma implies a
  false-alarm rate on smooth sky. EXP-022 measured the gap: at `c >= 6`, **14.65 false
  alarms/frame on O4 and 5.16 on analog against a Gaussian 6.5e-05** — six orders of
  magnitude. The synthetic control explains it rather than excusing it: on pure Gaussian
  noise sky the maximum `c` seen anywhere was **1.8**, so everything above that on real
  footage is *structure* (cloud edge, grain, blocking, chroma crawl) and structure has no
  Gaussian tail. Second reason the units are not what they look like: **sky sigma_ring is
  exactly 1.4826 on both clips**, the MAD of one grey level — sky is flat to the 8-bit
  quantiser, so `c` on sky is contrast in units of 1.5 grey levels. Any operating point
  must be read off the measured curve, per clip.

- [ ] 2026-09-29 — [algo] **The sky branch needs a second test after `c`.** EXP-022:
  **89.8% of O4 clutter and 76.6% of analog clutter reaches the target's p10 contrast**, so
  `c` alone cannot separate target from clutter on either clip even where the target is
  against sky. The branch is still worth keeping — it is the only part of stage 2 that
  acquires from a single frame, and on analog it found the drone in 32.6% of labelled
  frames against the motion branch's 9.8% — but it is a *candidate generator*, not a
  detector. Shape, persistence across frames at the same scale, and the bird question
  (Stage 4) are the candidates for that second test.

- [ ] 2026-09-29 — [algo] **Redefine stage 1 as far/near, not sky/ground.** EXP-019 measured
  it: `skyline.split` puts the labelled airborne drone in `scene` in **257 of 257** O4
  frames and **158 of 224** analog frames. The user's correction stands — "the drone is in
  the sky, just shallow one (for me tree line is sky as well)" — and this is it with a
  number. The rule separates *blue sky* from everything (texture + luminance + blue excess,
  each on a per-frame percentile), while what the ring test needs is a **depth** split, and
  a distant tree line is far. A parallax-magnitude rule is the obvious replacement, since
  the residual field stage 2 already computes is exactly a depth signal. Second defect on
  the same module: analog finds **no sky at all in 75 of 360 frames**, so anything gated on
  stage 1 would refuse on a fifth of that clip. **Belongs with the skyline session** — tell
  them before they verify the current boundary, or they will be checking a definition
  already known to be wrong.

- [ ] 2026-09-29 — [algo] **The HUD mask leaves the gaps between glyph boxes open.**
  Visible on O4 frame 946 in EXP-019's overlay: candidates survive in the spaces between
  `4.04v` and between `24.3V`/`19 Mbps`, because the mask covers each glyph box and the
  dilation does not close the gaps, while the glyph edges leak difference energy into them.
  Small, but it is false-alarm load in a region known to be worthless, and it is cheap to
  fix by closing the mask across a text block rather than dilating each glyph.

- [ ] 2026-09-23 — [data] **Label a FIELD episode with `/annotate` — now ahead of the second
  analog clip.** EXP-017 ran EXP-016's motion test unchanged on
  `data/raw/FIELD/videos/captured_raw_20260616_040253_004.mp4` and it **acquired the target
  in all three PROVENANCE episodes** — 8 tracks sustained-co-located with EXP-010 GLAD
  boxes, one holding 183 of 185 overlapping frames — which neither goggles clip managed.
  None of it is a *result*, because FIELD has no ground truth: z\* was imported from the
  analog clip rather than calibrated, "on target" means "near a box another detector drew",
  and the false-alarm rate is an upper bound. Labels convert every one of those into a real
  number. FIELD also carries **83.5 s** of believed-empty footage against 22–23 s on either
  SOFA clip, which is the count EXP-016 said was the binding constraint. Start with
  **episode 3 (frames 3140–3600, 15.4 s)**, where the target falls from ~18 px to ~6 px —
  the most informative span in the clip. Episodes 1 (2–180) and 2 (1101–1553) next.
  **Caveat carried from PROVENANCE:** the episode bounds come from where a detector fired,
  so the gaps are *not* verified empty and labelling must judge them rather than assume.

- [ ] 2026-09-23 — [algo] **Test the host-angular-rate false-alarm hypothesis out-of-sample.**
  EXP-017's false alarms have no spatial structure (21/42/38% across the frame thirds) but
  are concentrated almost entirely in time: **50 of 53 fall in frames 1800–2699**, the one
  hard banking manoeuvre in the clip, where median background image motion is 5.4–7.8
  px/frame against 1.1–3.3 elsewhere. The homography still fits there (residual 0.16–0.25
  px); what breaks is raw motion against an 11 px KLT window. Excluding that window the rate
  is **3.4/min, inside the 5/min budget, while still acquiring** — but that split was chosen
  after seeing where the false tracks fell, on this clip's own frames, so **it is in-sample
  and must not be quoted**. The test that would make it real needs no labels: derive
  per-frame background motion from `collect.pkl`, pick a gate threshold on **one** clip, read
  the false-alarm rate on **another**. If it holds it is an IMU gate on the aircraft, not an
  algorithm change — hand the flight-side half to `deploy-agent`.

### Analog `catch 2`: a confirmed target passage

- [ ] 2026-09-18 — [data] **Label `catch_2` frames ~510–585 with `/annotate`.** The drone closes from a speck to ~25 px against sky, and GLAD boxed it once (EXP-014). Until this is labelled, the 1-in-75 figure is eyeballed. Then try `--invert` on the passage as the cheap probe of the appearance prior.
  *(Labelling landed 2026-09-23 — 890 frames judged, 224 boxes; `--invert` on the passage is what remains.)*

- [ ] 2026-09-23 — [data] **Label a *second* analog clip with long-range spans, as the held-out
  test.** `data/raw/SOFA-ANALOG/videos/` has `catch_3`…`catch_8`, `miss_1`, `miss_2` and
  none is labelled. **This is now the binding constraint on the whole motion branch, not
  the algorithm.** EXP-016's verdict on `catch_2` turns on single-digit event counts over
  **22 seconds** of drone-free footage — the frozen false-alarm rate is *one track*, so ±1
  moves it by 5.4/min, the same order as the budget it is measured against. Every threshold
  in EXP-016 was also chosen on `catch_2`'s own empty frames, so nothing in that entry is
  out-of-sample for this clip's clutter. `/annotate` does it. The user prioritises analog,
  so this comes before more O4 labelling. Pick a clip with a **long-range span against the
  horizon and one against ground** — EXP-016 found those two backgrounds behave completely
  differently (49% coverage against sky, 0% against terrain).
  *(Reprioritised 2026-09-23 by EXP-017: **labelling a FIELD episode comes first.** FIELD is
  the target domain, carries 83.5 s of believed-empty footage against this clip's 22 s, and
  is the one clip where the motion test actually acquires. This item stays open — analog
  remains the user's priority for deployment — but it is no longer the cheapest thing that
  could change the motion branch's verdict.)*
  *(2026-10-07, EXP-031: catch_4 and catch_5 are labelled now, and both were used to pick
  EXP-031's `--merge 30 --merge-score sum --c-keep 10`. A fourth labelled analog clip is
  the only held-out test of that setting, and defaults stay unchanged until it exists. Run
  `experiments/exp031_combined/run_all.py` on it first.)*

### EXP-012: measure whether the O4 fixes actually work

The tooling landed on 2026-09-17 (see Done). What is left is the measurement.

- [ ] 2026-09-17 — [data] **Seed labels for the O4 target passages.** Nothing about
  EXP-012 can be stated as recall until this exists. `src.data.seed_track` is built and
  documented; the work is placing seeds — or, easier, `/annotate` the passage in the
  window (`src.data.annotate`, 2026-09-18), which re-seeds on every correction. Budget **several seeds per passage** — one seed
  held 10 frames on `first_catch`, because an intercept target grows and rotates fast.
  Write to `data/processed/SOFA-O4/labels/test/` and `verified.jsonl`, declare
  `--negatives` for confirmed target-free stretches, and **review the proposal sheet**
  before scoring. Confirmed passage so far: `first_catch` ~955-964, target 80x34 px.

- [ ] 2026-09-17 — [algo] **Run EXP-012 over all six clips** with
  `--hud-mask data/processed/SOFA-O4/hud_mask.png --motion-profile clutter`, then score
  against those labels and put it beside EXP-011. The branch mix is the diagnostic: if
  `global miss` falls from 46.8% and `global mod` rises, the motion fix worked; if
  HUD-overlapping detections fall from 83.1% to ~0, the mask worked.

  **`--motion-profile clutter` should not be adopted on this evidence.** EXP-012a ran it
  over `first_catch`: recall on the confirmed passage stayed at **zero**, all 9 detections
  were HUD or ground, and it cost **2.3x the compute** (3.11 -> 1.37 fps). Run EXP-012
  with the mask alone unless a labelled measurement says otherwise. EXP-012b traced the
  zero to candidate ranking and the shape test, not to missing signal — see the next item.

- [ ] 2026-09-22 — [algo] **Make the global motion branch keep the drone it already sees.**
  EXP-012b: the labelled drone is in `MOD2_global`'s difference image in 199 of 257
  `first_catch` frames and is found in 0. It is lost to the >50-candidate bail (126
  frames upstream), to `candidate_score`'s shape ranking pushing it out of the top 50 (90
  under `clutter`), and to the 0.6–3.0 aspect test (71). Try, in order: rank by difference
  strength instead of shape; loosen the aspect test in the global branch; then re-run the
  EXP-012a command and score it against the labels. Measure it on `first_catch` first,
  because that clip has the labels.
  **Refined by EXP-015 (2026-09-22):**
  - **Ranking score.** Rank by the *edge-normalised* difference, D / max(G, 2) or its
    windowed form, not raw strength. At equal empty-frame load it halves the candidates
    carried per drone frame (26 → 14), and top-10 goes 0.11 → 0.27.
  - **Mask the host airframe.** Add a mask for the host drone's own prop blades (left
    and right edges, y≈500–620). They are the loudest thing in the difference image.
  - **Mask the pitch ladder where it is.** It moves with pitch, so a fixed-position
    mask misses much of it.

  After normalisation, 69% of the top false peaks are frame-edge or overlay, not scene.
  This will not reach 708–800, where no motion map ranks the drone in the top 10.

  **Refined again by EXP-016 (2026-09-23), and one sub-item is now a prerequisite:**
  - **Multi-frame accumulation does reach 708–800 for *ranking*.** Verifying each
    `win_b5_e4` seed over time by its motion relative to its own 25–120 px ring puts the
    drone in the top 10 of **0.22–0.30** of those frames, against EXP-015's **0%**. If the
    global branch is ever re-ranked, accumulated differential z is the score to rank by.
  - **It does not reach it for *detection*.** The z\* the false-alarm budget demands on
    `first_catch` is **60**; the drone's maximum z on 708–800 is **13.9**. Do not build a
    confirm/reject stage on this statistic without fixing the two items below first.
  - **The pitch-ladder mask is a prerequisite, not an optimisation.** **59% of EXP-016's
    surviving confirmed false tracks on O4 confirm within 30 px of a burned-in overlay**,
    and 81% of them score at or above EXP-014's 0.70 OSD-twin threshold. The twin test
    itself **must not** be turned on for O4: the on-drone tracks score a median **0.76** on
    it, because O4's HUD is a digital overlay with no MAX7456 character grid and the test
    fires on scene texture. See the separate pitch-ladder item below.
  - **The ring needs to respect depth.** On analog, 57% of false tracks confirm at the
    sky/tree-line boundary, where the ring straddles two depths and its median describes
    neither. Segmenting the ring by flow magnitude, or rejecting a bimodal ring, is the
    named fix.
    **Deprioritised 2026-09-29 by EXP-019**, which drew stage 1 and stage 2 on the same
    frames and asked where the surviving candidates sit. The sky/scene boundary band is
    7.0% of the O4 frame and holds 12.3% of survivors (x1.76), and 8.9% of the analog frame
    holding 7.9% (x0.89 -- *under*-represented). EXP-016's 57.5% was about **confirmed
    multi-frame tracks**, so this does not contradict it, but it puts the horizon problem in
    the accumulation rather than in candidate generation. Fix the tracker and the epipole
    window first; re-measure this after.

- [ ] 2026-09-29 — [algo] **Does analog's lambda = -0.60 help stage 2?** Narrowed from the
  fisheye item below, which EXP-018 closed for stage 1. A radial error bites where a
  *pinhole camera is assumed* — the grid-KLT homography, the epipole, the parallax
  residual — not where a per-pixel appearance threshold is applied. `undistort_points`
  corrects coordinates with no resampling and **no field-of-view loss at all**, so on
  analog this is nearly free to try: undistort the KLT correspondences before fitting the
  homography and re-measure the residual and the epipole's anisotropy. O4 needs no such
  test; it has no radial signature.

- [ ] 2026-09-17 — [algo] **Fisheye undistortion before differencing.** *Premise withdrawn
  2026-09-22 by EXP-012b:* the residual after the homography is ~1 px median and 3–7 px
  p90, below the drone's 4–22 px differential motion, so compensation error is not what
  hides the target. Undistortion might still cut the ~115 terrain candidates per frame.
  Low priority until the ranking item above is done.
  **Settled for stage 1 on 2026-09-29 by EXP-018, and the two airframes differ.** O4 has
  **no radial signature at all**: 11 frames whose horizon runs within 30 px of the optical
  axis — where a radial model demands zero bow — bow by a median 209 px, and the plumb-line
  sweep never finds an interior minimum over ±1.2, wanting *pincushion*. The bow is near
  trees at the frame edges against a distant tree line in the middle. Analog **does** have
  one, lambda = -0.60, which straightens the horizon from 38.7 px of sag to 14.2 px against
  a 11.5 px tree-line residual. **Neither helps stage 1**, because undistortion is a
  bijection on pixels: it relocates the sky boundary and cannot relabel it (split IoU 0.989
  on O4 against the old split carried through the same remap), while costing 20-41% of the
  field of view on a fixed canvas. No undistorted copy was written to `data/processed/`.
  **Superseded within this item:** `motion_compensate`'s 50 px flow-rejection cap is *not*
  binding on this footage — measured flow is under 50 everywhere — so porting it is no
  longer a priority.

- [ ] 2026-09-17 — [algo] **Mask the pitch ladder, which sweeps with pitch.** No longer a
  caveat: in EXP-012a **five of the nine surviving detections sit on it**. It is rarely
  white at any one position so the frequency threshold misses it. Either lower
  `--min-fraction`, or mask the union of white pixels over the swept band, which is
  cheaper than masking more of the frame everywhere.

### M4b — generalisation: does GLAD hold up on video it has never seen?

- [ ] 2026-08-18 — [M4b] [algo] **Stretch, only after ARD100 lands: FL-Drones** (14
  videos / 38,948 frames, air-to-air, via the TransVisDrone repo). Genuinely held out —
  GLAD published on ARD-MAV and NPS-Drones but never on FL-Drones — but its confounds are
  *unbounded*, which is why it is second and not first: three annotation lineages, and
  752×480 against 1080p-tuned constants. Two blockers must be cleared before any
  FL-Drones number is believable:

  - **Resolution.** GLAD's motion module is tuned in absolute pixels for 1920×1080
    (`area 30–3000`, search region `a=160`, `dist_ref=200`, blur 11). Running as-is
    measures our failure to rescale, not GLAD's generalisation. Normalise by frame
    diagonal first: 752×480 → ×0.405 linear, ×0.164 for areas.
  - **Target size.** FL-Drones targets run up to 259×197 px while the motion branch caps
    blob area at 3000 px² (~55×55). Those targets are invisible to GMD/LMD *at any
    rescaling*, leaving only the YOLO appearance branch — and the motion branches carry
    recall (0.51→0.81 in the ablation). That is a structural handicap masquerading as a
    generalisation failure.

  Then sweep the IoU threshold, which M4a showed is the dominant variable.

### M5–M7

- [ ] 2026-08-20 — [M5-1] [algo] **Re-score EXP-001/002/003 to pick up `far`, `loc_err` and `loc_by_size`.** The metrics landed on 2026-08-20 and were applied to EXP-004 only; the three baseline runs still carry pre-M5 JSON, so the ledger's resize-vs-tile comparison has no false-alarm rate on either side. No inference — all three `detections.jsonl` are persisted. **The cost is labels, not scoring:** ~25 min wall-clock per criterion for the per-frame label read, so six re-scorings is an evening, and doing it *after* the per-video label store below turns it into minutes. Expect the localisation error to be the interesting column: these runs box 1.7x too large, so a large offset beside their near-zero recall would separate "never found it" from "found it and drew the wrong box".
- [ ] 2026-08-13 — [M7] [algo] [deploy] Fine-tune GLAD from its released weights on ARD-MAV's official training split, on a rented GPU (Kaggle 2×T4 free, or RunPod ~$1–2 for 3–4 h). Extract the 45 training videos **on the instance**, not locally. Score on the same held-out 15 videos; record whether it still fits the edge budget. Split across agents: `algo-agent` owns the training recipe and the ledger entry, `deploy-agent` owns provisioning, staging and the cost/wall-clock report.
- [ ] 2026-08-24 — [deploy] [algo] **Measure 10 fps (`--sample-n 3`), and a 3-frame burst (`--burst-length 3`).** EXP-006/008 showed half rate costs only 8.5% of recall with the tracking lock intact and `global miss` unmoved, so the 50 px `MAX_DISTANCE` gate has margin left — one run establishes whether the penalty is linear in interval or has a knee. Separately, every 2-frame burst spends half its frames on a structurally guaranteed miss (measured: 0 TP from 471 and 560 targets); a 3-frame burst buys two usable frames for 1.5× the cost and might let the local regime engage once, which no 2-frame burst can. Same code, no new flags. Cheap: ~1 h and ~15 min respectively per dataset.

- [ ] 2026-09-23 — [deploy] **Analog optics give a 4× shorter detection range than
  `edge-budget.md` assumes — re-derive §1 and the latency budget that hangs off it.**
  Raised by `algo-agent` out of EXP-016; it is `deploy-agent`'s to resolve.
  [edge-budget.md §1](edge-budget.md) assumes **1920 px across ~60°** (0.031°/px) and puts
  a 0.6 m target at **10 px ≈ 110 m, 2.8 s from contact**. The analog feed the user cares
  about most is **960 px across 120°** — 8 px per degree, 0.125°/px — so the same target is
  **275/R px**: **10 px at ~27 m** and 5 px at ~55 m. At 40 m/s closing that is **~0.7 s**,
  not 2.8 s. Every range, time-to-contact and "first detectable" figure in §1, §4.2 and
  §4.3 needs re-deriving for this optic, and the recall projections that cite them
  re-checked. **It bites immediately:** EXP-016's confirmation latency on analog is
  **0.6–2.5 s** depending on the threshold, which is the *whole* budget or more. Related to the
  open "pin down the airframe's actual camera" backlog item, but this one does not wait for
  hardware — the numbers are already in hand.

- [ ] 2026-08-23 — [deploy] **Profile `GladPipeline.step` per stage on idle hardware.** `time.perf_counter()` around GAD, LAD, `MOD2_global`, `MOD2_local` and the classifier gate in `src/algo/glad/pipeline.py`, over ~2,000 contiguous frames. [edge-budget.md §2.3](edge-budget.md) currently *estimates* the split by solving `0.884*T_LAD + 0.116*(T_LAD + T_MOD) = 374 ms` and infers `T_MOD ~ 1.5 s`; that is arithmetic, not a profile, and it is the one number the whole edge budget rests on. **Could not be taken on 2026-08-23** — `exp005_glad_ard100` was occupying the CPU and a contended timing run violates the benchmarking rules in `deploy-agent`'s own charter. Cheap: minutes, no GPU. Also worth emitting p50/p95/p99 and the worst video rather than a run mean.
- [ ] 2026-08-23 — [deploy] **Export the two GLAD `yolov5s` checkpoints to ONNX and re-run through ONNX Runtime / OpenVINO on this host.** [edge-budget.md §3](edge-budget.md) item 1: expected 1.5-3x on the stage that dominates 88.4% of frames, and the ONNX artifact is the same one a Jetson TensorRT build would start from, so the work is not throwaway. **Not free of accuracy risk despite being fp32->fp32** — the NMS implementation changes — so it needs an `src.evaluate` re-score at the same criterion before the number is carried anywhere. Blocked on nothing.
- [ ] 2026-08-23 — [deploy] **Measure the pixel-scaling exponent of GLAD's motion path — no camera needed.** [edge-budget.md §4.2.4](edge-budget.md) prices a 12 MP Alvium by assuming every full-frame stage scales **linearly with pixel count (5.88x)**, and the entire "naive 12 MP is not viable" verdict rests on that one untested assumption. It is free to check: run `src.glad_detect` over a downscaled copy of two ARD-MAV test videos (960x540, i.e. 0.25x the pixels) and compare per-branch timings against 1080p. If the motion path does **not** fall ~4x, the 12 MP projection is wrong and must be redone. **Fold this into the per-stage profile above rather than running it separately** — same instrumentation, one extra resolution. Timings only: GLAD's constants are absolute pixels tuned for 1920x1080 ([glad-model.md §6](glad-model.md) item 4), so **the accuracy of a downscaled run is meaningless** and must not be recorded as a result.
- [ ] 2026-08-23 — [deploy] **Get the Alvium ROI mode-switch latency from Allied Vision.** [edge-budget.md §4.2.6](edge-budget.md) identifies sensor-side ROI readout as potentially better than the recommended hybrid — GLAD's local regime already *is* a 320x320 window, so reading only that window off the sensor would cut capture latency to near zero. Blocked on one number: how many frames a mode switch costs. EXP-004 says the regime is stable (acquisition fires 42 times in 28,337 frames), so even a several-frame switch may be affordable. A vendor email, not an experiment.
- [ ] 2026-08-23 — [deploy] [algo] **Decide 12 MP vs a narrow lens — needs one answer from the user: is the sensor cued or searching?** [edge-budget.md §4.2.8](edge-budget.md). A 28.6-degree lens on 1080p gives **identical pixels-on-target** to 12 MP at 60 degrees for **5.88x less compute, no retrain and no new camera** — it only costs field of view. If a bearing is handed over by radar/RF/GCS, 12 MP is buying FOV nobody uses. **Blocked on the user, and it is the cheapest open question in the project.** Now recorded as the third unresolved input alongside closing speed and persistence frames.


### Backlog — no mission, revisit when the trigger fires

- [ ] 2026-09-28 — [algo] **Pin the tiled coordinate mapping with a test.** `src/algo/tiling.py` (`tile_origins`, `crop_grid`, `merge_boxes`) has **no test file**, and it implements the invariant CLAUDE.md calls load-bearing: tiled results are mapped back to absolute original-frame pixels before they leave `algo`, so tiled and whole-frame output are directly comparable. Nothing currently checks that. A run that silently returned tile-local coordinates would produce a plausible-looking mAP that is fiction — and CLAUDE.md tells you to *always* run both ways when benchmarking, so the broken number would be the one you compare against. Wants: origins tile the frame with the asked-for overlap, a box in a tile maps back to where it is in the source, and NMS merges a target straddling a seam into one box rather than two. Raised while restructuring `src/` (2026-09-28); the baseline-YOLO chain was deliberately out of scope for that pass, which is why this is filed rather than done. **Trigger:** before the next tiled benchmark, or any change to `tiling.py`.
- [ ] 2026-08-23 — [deploy] **Pin down the airframe's actual camera, then re-run this projection against real frames.** [edge-budget.md §4.3](edge-budget.md#43-the-opposite-direction--an-analog-fpv-camera-iflight-racecam-r1-mini) shows an analog FPV camera (720 px, 130–165°) drops projected recall from 0.688 to ≈0.005 on optics alone, and every accuracy number this project owns comes from someone else's gimbal. **Trigger:** a camera is chosen, or any airframe hardware is bought. Target size is now supplied (10-inch quad, ~0.59 m) and every range table in edge-budget.md was re-derived for it on 2026-08-23; the camera and the cued-vs-searching question are still open. Cheap version — capture a few minutes off the candidate camera, label it, run `src.glad_detect` + `src.evaluate` exactly as EXP-005 did.
- [ ] 2026-08-16 — [data] Derive per-frame **target motion** (fast/slow) from inter-frame box displacement in the ground truth, and add it as a second breakdown dimension alongside scene category. Nothing published provides this, so it has to be computed. Deferred from M2a, which delivered the scene-category half.
- [ ] 2026-08-13 — [data] Store labels per-video (one file, one row per frame) instead of one `.txt` per frame, and expand to the per-image tree only on the training instance. 28,337 tiny files cost minutes per full read — measured: two `find` calls and the MANIFEST regeneration all blew a 120 s timeout — and the per-image layout is only actually required by the ultralytics dataloader at M7, which runs on the rented GPU, not here. Space is not the issue (NTFS keeps sub-700-byte files resident in the MFT); per-file syscall latency is. **Trigger:** label reading starts dominating the M5 re-scoring loop, or the 45 training videos push the tree past ~100k files.

- [ ] 2026-08-24 — [deploy] **Get a *live* feed from the drone's downlink.** Narrowed on 2026-09-17: a **recording** now exists (`data/raw/FIELD/`, EXP-010), which settles the optics question below — recall on our own camera is emphatically not near zero. What is still missing is the **live** path: `src.live_detect` has never seen a real feed. The FPV goggles do not enumerate on this laptop in any device class over USB-C-to-USB, and the goggles' HDMI is input-only, so there is no video-out path yet. **Trigger:** the goggles expose a UVC device, or an HDMI capture card is bought. Note EXP-010 measured 3.05 fps on this footage, so `--policy auto` would pick **burst pairs** (36% retention), not half rate — the live path's first real test should expect the fallback.

- [x] 2026-09-17 — [algo] **Tested the polarity hypothesis — EXP-012, and it found a drone nobody else did.** Inverting the frame (`255 - pixel`, new `--invert`) did **not** do what was predicted: cold appearance acquisition went 6 → 11 frames of 3,600, so GAD is useless on this footage whichever way up it is. The prior is confirmed anyway, and more sharply — across EXP-010 and EXP-012 the detector consistently picks **whatever is brighter than its local background in the image it is given** (93.5% and 98.6% respectively, measured as-seen). Inversion changed which physical objects satisfy that, and the effect was large: it **found a real drone episode at frames ~1990–2120 that EXP-010 and EXP-011 both recorded zero boxes in** — 40 detections, median 22 px, payload clearly slung beneath the airframe — while **eliminating** EXP-011's white-structure clutter (75 → 0) and cutting the bush lock 41 → 14. Acquisition came from `global mod`, a motion candidate confirmed by LAD, not from appearance. So the prior gates **which motion candidates survive confirmation**, not appearance acquisition. **Not a deployment fix** — it would invert the problem on ARD-MAV-like footage, and the sky episodes lost ground (458 → 394). Also the slowest configuration measured, 0.70 fps against 3.05. Full entry in [experiments.md](experiments.md).

- [ ] 2026-09-17 — [data] **Annotate the FIELD capture's three target episodes** — *now the critical path: EXP-011 exhausted what unlabelled footage can answer, and both surviving questions (the ground-clutter false-alarm rate, and recall on our own camera) are measurements.* — frames ~2–180, ~1101–1553 and ~3140–3600 of `captured_raw_20260616_040253_004.mp4`, roughly 1,100 frames. This is the cheapest real score available to the project: EXP-010 is already keyed at `data/processed/FIELD/images/test/`, so labels there turn **an existing JSONL into AP, precision and recall with no second inference pass**. It is also the only way to measure **recall** on our own camera, which EXP-010 leaves unmeasured and which is the number the edge budget actually needs. Two warnings: the episodes' bounds come from where the *detector* fired, so annotating only those frames would score a set chosen by the thing being scored — extend each episode outward until the target is genuinely absent. And **do not eyeball full frames**: a 14×11 px drone at frame 1350 was missed by eye and caught by the detector.

- [ ] 2026-09-29 — [algo] **Score a HUD mask against labels, not against its coverage percentage.** EXP-020: the tool prints `7.71% of the frame`, which is the wrong number — 13.15% sounded acceptable and was 20% of the ground truth. Add `--labels <dir>` to `src.data.hud_mask` so the build reports how many labelled boxes the mask would veto at `HUD_VETO_FRACTION`, and refuse silently-expensive masks. Cheap: the scoring loop is `overlap_fraction` over YOLO txt files, ~20 lines, and it is the only check that would have caught this.

## Done

- [x] 2026-10-07 — [algo] **EXP-031: every stage at once, and EXP-030 at matched load.**
  `experiments/exp031_combined/run_all.py` runs EXP-025d to EXP-030 together on catch_2/4/5
  at merge 20 and 30, with c-keep swept 6–15. The `exp028`/`exp030` rows match EXP-030's CSVs
  byte for byte.
  - Pooled, **`--merge 30 --merge-score sum --c-keep 10` beats EXP-028 on every column**:
    0.65 against 0.75 shown/frame, 434 against 541 false alarms, drone top 3 142 against 128,
    #1 123 against 109. catch_2 and catch_4 win, but catch_5 loses #1 (21 against 31).
    The c-keep was chosen in-sample, so the defaults are unchanged.
  - **The moving factor fails out of sample:** on catch_5, `--min-move 20` loses 13 drone
    frames where the camera chases the drone. Once c-keep is calibrated it adds nothing.
    EXP-029's shipping recommendation is withdrawn.

- [x] 2026-10-07 — [algo] **The moving factor stacked on EXP-028's gate: strictly better on
  every column.** The user pointed out the stages are meant to come one on top of the other.
  `--min-move` is now a flag on `experiments/exp025_top3/overlay_split.py` (default 0 and
  **inert** — at 0 the shown and drone CSVs are byte-identical to EXP-028's), and
  `experiments/exp029_moving_factor/overlay_stacked.py` runs EXP-028's defaults plus
  `--min-move 20`: one `KinematicTracker` with the speed limit bounding motion from above
  and the moving factor from below. On catch_2 441-800, top 3, against EXP-028: **false
  alarms 264 -> 209 (-21%), load 0.93 -> 0.78 shown/frame (-16%), drone #1 60 -> 62, drone in
  top 3 unchanged at 69 of 224.** 12 is nearly as good, 30 starts costing recall, 45 is past
  the knee. This retires EXP-029's earlier conclusion that the factor only pays above ~12
  candidates/frame and cannot reach EXP-028's load — both were properties of
  `overlay_moving.py`'s standalone rebuild, which discarded the merge, the OSD veto, c-keep
  and evidence ranking, and so reproduced EXP-027's weaker #1 column (36 of 224) before
  losing on it. Video and three CSVs in `runs/sofa_analog/exp029_moving_factor/`; ledger
  entry has the dated addendum and its recommendation replaced.

- [x] 2026-10-07 — [algo] **EXP-029's moving factor defaulted to 20 px, and the direction
  test's verdict corrected.** At the user's suggestion `--min-move` went from 8 to 20:
  **940 fewer false alarms (5626 -> 4686) for one drone frame** (105 -> 104 of 224), load
  16.14 -> 13.52/frame. Sweeping the combination properly then overturned the 2026-10-05
  conclusion that the epipolar direction test "adds nothing": that was measured at
  `--min-move 8` alone, where the magnitude and direction cuts remove overlapping sets. Past
  ~16 it is worth +1, **+4 at 20** and **+18 at 30** drone frames at matched load, because
  magnitude pushed harder starts cutting the target while the veto removes clutter by an
  uncorrelated criterion. **Shipping config is now `--min-move 20 --mode veto`:** 12.33
  kept/frame, 103 of 224, +9 over EXP-023 at matched load. Unchanged: the 5.42/frame
  unjudged floor still keeps EXP-029 away from EXP-023's 5.59/frame and EXP-028's 0.93, so
  it remains a gain in a band above where anything ships. Ledger entry retitled and its
  recommendation replaced.

- [x] 2026-10-07 — [algo] **EXP-030 at merge 30.** `run_clips.py --merge 30`; outputs in
  `runs/sofa_analog/exp030_merge_sum/merge30/`, with merge 20 moved to `merge20/`.
  - Without the gate, the summed score at 30 px is the best #1 at identical load so far:
    94 frames vs 90 (sum, 20 px) and 81 (EXP-027's max).
  - With the gate it is worse than 20 px: load +65–83% over the max, and on catch_5 the
    drone is #1 in 32 frames, below the max gate's 33.
  - The max alone barely moves between 20 and 30 px.

- [x] 2026-10-05 — [algo] **EXP-030: merged objects scored by the sum of their members' c.**
  `overlay_split.py --merge-score sum` (default `max`, byte-identical when off), run by
  `experiments/exp030_merge_sum/run_clips.py` on catch_2/4/5 at merge 20, with and without
  the kinematic gate.
  - Without the gate, at identical load, the drone is #1 in 90 frames vs 81, because the
    drone fragments into 2–3 blobs and clutter does not.
  - With the gate, the load rises 32–73% (summed clusters clear c-keep 6), so it is not
    comparable to EXP-028. Next: the same at matched load.

- [x] 2026-10-05 — [algo] **EXP-029: the moving factor, and the direction test measured out
  of the pipeline.** Added the moving factor to `src/algo/kinematics.py` — the complement of
  EXP-028's speed limit, a lower bound on how far a confirmed track must travel against the
  static scene, with anchors carried through each frame's homography. `min_move = 0` is inert
  so EXP-028's 32 tests pass untouched; 8 new tests cover clutter kept forever and the target
  dropped by premature judging or by being charged for the camera's pan. The direction half
  is in `experiments/exp029_moving_factor/moving.py` (off / veto / require, 9-of-9
  self-check) because it needs an epipole and `src/` must not import `experiments/`.
  **Result: the direction test adds nothing.** `veto` drops 555 false alarms for one drone
  frame, but magnitude alone at `--min-move 15` reaches the same 14.57/frame load and the
  identical 104 of 224 drone frames; its epipole agreement is 28.9% against a ~23% floor.
  `require` is worse on both axes. Against plain EXP-023 thresholding the chain+factor wins
  by 2-8 drone frames above ~12 candidates/frame and loses below it, and EXP-023 ships at
  5.59/frame — so **as built it is not an improvement at any load the project wants.** The
  cause is measured: clutter's real travel is a median 19.97 px against the target's p10 of
  15.63, so EXP-024's predicted non-overlapping tails do not survive contact with the actual
  false alarms — the filed caveat that the background grid is not the clutter. Video and
  dumps in `runs/sofa_analog/exp029_moving_factor/`; full entry in `docs/experiments.md`.

- [x] 2026-10-05 — [algo] **EXP-028 out of sample on catch_4 and catch_5.**
  `experiments/exp028_kinematic/run_clips.py` (no gate / gate / c-keep 3 / no weak, from
  EXP-025e's dumps). The default holds:
  - catch_5: 0.37 vs 0.71 shown/frame, drone top 3 37 vs 29, #1 31 vs 16, #1 jumps 30% vs 71%.
  - catch_4: 1.21 vs 1.11 shown/frame (9% more), top 3 22 vs 12, #1 18 vs 8, jumps 51% vs 82%.

- [x] 2026-10-05 — [algo] **EXP-028: the kinematic gate — answers held to a drone's top
  speed.** `src/algo/kinematics.py` (`SpeedLimit`, `KinematicTracker`, unit-tested), run as
  `overlay_split.py --kinematic` through `experiments/exp028_kinematic/`. The m/s limit is
  converted at an assumed minimum range (40 m/s, 10 m, 130 deg = 29.8 px/frame), capped at
  25 px/frame. Tracks use hysteresis (strong births, weak continues, 5-frame coast), and
  #1 is ranked by track evidence. On catch_2 441-800 against EXP-027's top 3: 0.93 vs 1.08
  shown/frame, 264 vs 318 FAs, drone top 3 69 vs 67, #1 60 vs 57, and #1 jumps > 29 px
  43 vs 103. Off by default, and byte-identical to EXP-027 when off.

- [x] 2026-10-04 — [algo] **EXP-025e on the FIELD capture.** The split overlay (merge 20,
  circles >= 20 px) over all 3,599 frames, with stage 1's brightness test off for FIELD
  (`sky_luma_pctl` / `sky_texture_pctl`, new per-clip knobs in `skyline.py`). 0.31
  shown/frame: 0.64 and 0.77 in the two main drone episodes, 0.17 outside them, and the
  ground section is nearly silent. No labels, so no rank. Video in
  `runs/field/exp025_top3/`.

- [x] 2026-10-04 — [algo] **EXP-025e: the sky/ground split overlay on catch_4 and catch_5.**
  `experiments/exp025_top3/run_clips.py` (merge 20 px, circles >= 20 px), reusing EXP-023's
  sky dumps. Drone in the top 3: catch_4 14 of 82 (sky alone 33), catch_5 31 of 188 (sky
  alone 30). Stage 1 calls a median 0% and 7% of those frames sky, so the window decides
  almost everything, and the catch_2 gain does not carry over. Unlabelled clips and FIELD
  dropped at the user's request.

- [x] 2026-10-04 — [algo] **EXP-027: drop the OSD horizon dashes ("white dots") from the
  sky/window split.** New `src.algo.masking.grid_twins` / `on_osd_grid`: a blob with copies
  at 2 of the positions ±1, ±2 OSD columns away is a dash. GLAD's `has_twin` would have vetoed
  the drone in 32 of 53 frames. Wired in as `overlay_split.py --osd-grid` and run by
  `experiments/exp027_osd_grid/overlay_grid.py`. On catch_2 441–800, top 1: dash-band false
  alarms 120 → 49, shown/frame 0.79 → 0.64, drone #1 53 → 57 (+5, −1 at 585). Not yet on
  in `run_clips.py`. The end dashes of the row still get through.
  **Top 3** (`overlay_grid --top 3`): dash-band false alarms 254 → 92, shown/frame 1.62 → 1.08,
  drone in the top 3 68 → 67. Lost 4: 590 was a merge artefact, 499 and 585 are the drone
  in the dash row, and 660 is a false veto on blurred grass. Gained 3.

- [x] 2026-10-04 — [algo] **EXP-026: EXP-025d's sky/ground split overlay with only the #1
  per frame.** `experiments/exp026_top1/overlay_top1.py` runs `overlay_split.py --top 1
  --merge 20 --min-draw 20`. The drone is #1 in 53 of 224 labelled frames (36 sky,
  17 ground), at 0.79 shown/frame. 53 of the 284 circles drawn are on the drone. Video:
  `runs/sofa_analog/exp026_top1/split_sky_window2of4_top1_merge20_catch_2_441_800.mp4`.

- [x] 2026-10-04 — [algo] **EXP-025d re-run with blobs merged at 20 px (40 px across),
  circles drawn at least 20 px.** `overlay_split.py --merge 20 --min-draw 20`. The drone
  is in the top 3 in 68 of 224 frames and #1 in 53, against 66 / 52 unmerged. Of the 3
  new top-3 frames, 2 are a clutter anchor absorbing the drone. One clean gain (585, #2 to
  #1).

- [x] 2026-10-04 — [algo] **EXP-025d re-run with blobs merged at 10 px (20 px across).**
  `overlay_split.py --merge 10` folds every candidate within 10 px of a stronger one before
  the sections' tests (231 folded, 0.64/frame). The drone's result is unchanged: 66 of 224
  in the top 3, 52 at #1. The ground section's 145 drops have no 2-of-4 survivor near the
  drone (median 79 px away), so the motion window, not fragmentation, is the bottleneck.

- [x] 2026-10-04 — [algo] **EXP-025d: sky/ground split, the sky branch on the sky, the
  2-of-4 window on the ground, top 3 by sky c.** `experiments/exp025_top3/overlay_split.py`
  recomputes stage 1 per frame and pools both sections into one ranking. The drone is in
  the top 3 in 66 of 224 frames and #1 in 52, at 1.66 shown/frame. That beats EXP-025
  (61 / 31 at 2.88) and EXP-025c (49 / 39 at 1.29). In the sky it is #1 in every frame it
  is ranked (36); the ground section drops it in 145. `split_rank_hist.py` draws the
  histogram stacked by section. Video:
  `runs/sofa_analog/exp025_top3/split_sky_window2of4_top3_catch_2_441_800.mp4`.

- [x] 2026-10-04 — [algo] **EXP-025c: the 2-of-4 motion window, ranked by sky contrast,
  top 3, rendered.** `experiments/exp025_top3/overlay_window_skyc.py` draws it from the
  EXP-024 and EXP-023 dumps. A sky blob is ranked when a 2-of-4 survivor lies within 9 px,
  with no c threshold and cloud vetoes dropped. The drone is in the top 3 in 49 of 224
  frames and #1 in 39, against EXP-025's 61 and 31, at 1.29 shown/frame against 2.88. It
  beats EXP-025b's sky 4-of-5 gate on every column. Dropping cloud vetoes also corrected the
  EXP-024 intersection numbers: 48 vs 39 (2/3) and 52 vs 42 (2/4) against the sky branch
  alone at matched load. Video:
  `runs/sofa_analog/exp025_top3/window2of4_skyc_top3_catch_2_441_800.mp4`.

- [x] 2026-10-04 — [algo] **2/3 and 2/4 window survivors ranked by the sky branch's
  contrast.** `rank_hist.py --rank-by sky`: each survivor takes the `c` of the nearest
  EXP-023 blob within 9 px, else last. The drone is #1 in 38 and 39 frames, against 5 and 4
  by motion score, but 90% of survivors have no sky blob nearby, so this is mostly the
  intersection of the two detectors. That intersection keeps the drone in 48 frames at
  1.43/frame against the sky branch alone's 42 at the same load (51 vs 43 for 2/4): the
  first sign that persistence adds to the sky branch. `docs/experiments.md` EXP-024
  addendum.

- [x] 2026-10-04 — [algo] **Drone-rank histograms for 2/3, 2/4 and the sky branch on
  analog.** `experiments/exp024_window_length/rank_hist.py`, which reads the dumps, no
  re-render. Window gates: median rank 5, #1 in 5 and 4 frames, flat from rank 1 to 8. Sky
  c>=6: median 2, #1 in 31, top 3 in 61. Loosened to the window's load, the sky branch
  gains 22 frames, only one of them in the top 3. `docs/experiments.md` EXP-024 addendum.

- [x] 2026-10-04 — [algo] **EXP-023's sky branch rendered without the c >= 6 cut.** At the
  dump floor, c >= 3, clean look, `catch_2` 441-800:
  `runs/sofa_analog/exp023_sky_branch/clean_c3_catch_2_441_800.mp4`. 45.5 kept/frame
  against 5.5, and the drone lands in 130 of 224 frames against 73. Bold circles are the
  c >= 6 set (drawn thicker at 2x threshold). `docs/experiments.md` EXP-023 addendum.

- [x] 2026-10-04 — [algo] **EXP-024 on analog at 2/3, 2/4, 2/5 and 3/5, rendered for
  viewing.** Four clean-look videos of `catch_2` 441-800, plus the first k=3 and k=4 curves
  and dumps. At two appearances the window length barely matters to the target: 57, 60 and
  60 of 174 frames at 13.8, 16.4 and 18.1 load/frame, so 2/3 is the cheapest of the three.
  3/5 keeps 36 (21%) at 4.2. The sky branch beats every one of them at matched load by
  1.6-1.8x, so the analog verdict stands. `docs/experiments.md` EXP-024 addendum.

- [x] 2026-10-04 — [algo] **EXP-024's window gate against EXP-023's 2-frame sky branch, at
  matched load.** Recut both from dumps, no re-run, and **both out of the same 224 labelled
  frames** — EXP-024's printed recall divides by 174, which is the motion front-end's
  ceiling rather than a denominator, and that mismatch was why the ledger had called the two
  incomparable. The sky branch wins at every matched load by ~1.7x: 99.6% vs 77.7% at the
  floor, 44.2% vs 26.8% at 18/frame, 17.0% vs 9.8% at 1.1/frame. Its shipping c>=6 is 32.6%
  at 5.27 false alarms/frame; the window needs 3.2x that load for a lower 26.8%. Cause is
  upstream of the gate (174/224 vs 223/224 candidate ceiling), and the sky branch is immune
  to the duplicate frames, finding the target in 12 of the 60 where the motion branch finds
  none. The window's contrast does order its own survivors better (53.0% of clutter above
  the target's p10 against the sky's 82.0%), noted as a hint given 22 target frames.
  Different detectors, so a pipeline comparison and not a window-length ablation. EXP-025's
  "not comparable" caveat amended to point at the fixed denominator. Filed the follow-up:
  put persistence on the sky branch instead. `docs/experiments.md` EXP-024 addendum.

- [x] 2026-10-04 — [algo] **EXP-025b: the top 3 behind a 4-of-5 persistence gate, and a
  table of every frame the drone is ranked.** `experiments/exp025_top3/overlay_gate.py`
  chains each sky-branch candidate back 4 frames (nearest kept candidate within 9 px) and
  ranks only those seen in 4 of 5. Chained **in picture coordinates** it keeps the drone in
  the top 3 in 27 of 224 frames (#1 in 20) at 1.71 shown/frame, against 61 at 2.88 with no
  gate; 46 is the ceiling. Chained camera-compensated, which was the option asked for, it
  keeps 3, because the camera follows the drone and compensation moves the chain off it.
  `drone_ranks.py` writes the per-frame rank table (it is also in the folder README). Video:
  `runs/sofa_analog/exp025_top3/gate4of5_image_top3_catch_2_441_800.mp4`.
  `docs/experiments.md` EXP-025 addendum.

- [x] 2026-10-04 — [algo] **EXP-024: 4-of-5 vs 5-of-7 on analog, with contrast — and
  catch_2 turns out to repeat 1 frame in 6.** Added `--dump` to the EXP-024 overlay (one
  row per seed per frame: contrast, appearances, steps, verdict, on_drone), so false-alarm
  count and the drone's score are readable at every threshold from one pass; counting rows
  with `appearances >= m` reproduces the OPERATING POINTS row for `m/k`. 4-of-5 gives 1.03
  false alarms/frame and keeps the drone in 22 of 174 frames at median contrast 4.44
  (median rank #1, strongest in frame half the time); 5-of-7 gives 0.79 and 18 frames at
  4.48. At matched load the two are within one frame of each other. The drone sits at ~1.6x
  the false alarms' median contrast but below their p90, so ~1 in 10 outscores it; ranking
  after persistence halves load and recall together (6% everywhere) and is not an operating
  point. **The find: 148 of 890 frames are near-duplicates at `f % 6 == 4` exactly** — 25
  fps resampled to 30 — and a duplicate pair yields zero candidates, so the effective
  window is `k * 5/6`. 7-of-7 kept 0 of 60338 seeds by arithmetic, 5-of-5 is reachable in
  17% of windows, and 4-of-5 is really a perfect-run requirement. Also rendered
  `window5_catch_2_441_800.mp4` at the defaults, the video `curve_k5.log` named but never
  wrote. `docs/experiments.md` EXP-024 addendum; artifacts in
  `runs/sofa_analog/exp024_window_length/`.


- [x] 2026-10-01 — [algo] **EXP-024 at 5/7 and 5/10 on analog, and EXP-025: top 3 per
  frame.** Rendered `window7_need5_` and `window10_need5_catch_2_441_800.mp4` (clean look).
  5/7 keeps the same 18 of 174 drone frames (10%) as 5/10 at 0.8 vs 1.3 candidates/frame.
  EXP-025 (`experiments/exp025_top3/`) shows only each frame's 3 highest-contrast sky-branch
  candidates, tagged `#rank c`. Load drops from 5.48 to 2.88/frame while keeping 61 of
  EXP-023's 73 drone frames (27.2% of 224). The drone is #1 in 31, #2 in 18, #3 in 12.
  Video: `runs/sofa_analog/exp025_top3/top3_catch_2_441_800.mp4`. Both are in
  `docs/experiments.md`.

- [x] 2026-10-01 — [algo] **EXP-024's overlay drawn in the clean look.** The analog
  EXP-024 videos had been rendered in EXP-017's diagnostic style (sky tint, horizon dots,
  epipole, flash dots, arrows, label box), the opposite of the standing request that every
  overlay look like EXP-023's `clean_catch_2_441_800.mp4`. `overlay_window.py` in
  `experiments/exp024_window_length/` now draws that look by default: stage-0 tints, kept
  tracks as red 10 px circles, two-line caption. `--diagnostic` restores the old drawing.
  Re-rendered `runs/sofa_analog/exp024_window_length/window10_` and
  `window15_catch_2_441_800.mp4`; the reports are byte-identical to the earlier ones apart
  from the first line, so only the drawing changed.

- [x] 2026-10-01 — [algo] **Clean overlays on both clips, red circles, and no scripts
  in `runs/`.** Rendered `runs/sofa_o4/clean_first_catch_650_964.mp4` the same way as the
  analog `clean_catch_2_441_800.mp4` (`overlay_sky --no-sky --no-truth`, EXP-022
  defaults), and re-rendered both after changing the drawing: kept circles are now red and
  at least 10 px across (`overlay_sky.MIN_DRAW_DIAMETER`; drawing only, the report and dump
  are unchanged). Deleted every folder under `runs/sofa_o4/` (~1.1 GB). They held the
  **only** copy of the EXP-012b–EXP-024 code, so all 64 scripts under `runs/` moved first to
  `experiments/` (tracked; the was→now table is in `experiments/README.md`), and the O4
  prop/ladder masks and EXP-015's screen-fixed map moved to `data/processed/SOFA-O4/`.
  Checked: both clips reproduce their original `[stage 0]` mask percentages from the new
  location, and `silhouette`'s self-checks pass. Ledger paths naming a script under
  `runs/<clip>/expNNN_*/` now mean `experiments/expNNN_*/`; the O4 outputs they name are
  gone. The default overlay is now analog `catch_2` 441–800 unless another clip is named.

- [x] 2026-10-01 — [algo] **Window length on both clips: render a 10-frame and a 15-frame
  gate and recommend one.** Four overlay videos on `first_catch` 650-964 (matched-load 5/10 and
  6/15, low-load 8/10 and 10/15), plus a k=5 control on the same span so the table is
  internal to one run. **EXP-024. Recommendation: keep k=5 at 4-of-5.** k=5 is at least as
  good at every load it can reach and strictly better below ~40 candidates/frame -- 42%
  recall at 25.8/frame against 33% (k=10) and 29% (k=15) at matched load. A longer window
  buys only *reach*: 5/5 floors k=5 at 14.3/frame, and below that k=10 at 9/10 is the only
  option, at 17%.
  **This reverses my own recommendation from the design discussion that preceded it** --
  I argued it from motion-magnitude SNR, which does grow with window length, and applied
  it to the persistence gate, which shrinks with it. Two different tests.
  **The durable finding:** `usable residual steps per seed: median 3` at k=5, 10 and 15
  alike. LK tracks die at three frames regardless of window length, so the accumulated
  residual a long window exists to compute is mostly unavailable, and any motion-magnitude
  test must chain candidate *peaks* rather than optical flow.
  **Instrument:** `overlay_window.py` gained `--direction` (default off, matching
  `overlay_stage2.py`; it previously always applied a test EXP-021 measured at chance) and
  an operating-point table printing every `min_appear` from one pass.
  **Caveat recorded:** on 900-964 alone the strict settings look tied (52/51/48%); on the
  full span they are 25/19/17%. EXP-021's 78% and 31.2/frame are that easy sub-span.
  **Analog (2026-10-01, same entry):** the O4 conclusion does not transfer -- at matched
  load the three window lengths are indistinguishable on `catch_2`, so longer is not worse,
  merely not better. What dominates is that **persistence has no usable setting on analog**:
  1/k keeps 100% of the target at 167.6 candidates/frame and 2/k keeps 37% at ~20, with the
  shipping 4-of-5 at 13%. Probed the suspected cause: `--fb-max 3.0` against the 0.67 px
  default more than doubles surviving tracks but buys only ~3 points at matched load, so LK
  throttling is real and is **not** the explanation. Analog's problem is upstream of the
  gate. Scripts now live in `experiments/exp024_window_length/` with a clipcfg per clip;
  EXP-017's `overlay_window.py` was reverted to its frozen state after I had edited it in
  place.

- [x] 2026-09-30 — [algo] **Roll back the EXP-022/EXP-023 difference, and record every
  candidate.** Both done, plus three defects the user's questions exposed.
  **Rollback:** defaults reproduce EXP-022 (whole ring, uncertain band scored); the two
  EXP-023 changes are now `--depth-ring` and `--refuse-uncertain`. Verified — the
  rolled-back analog run returns **73/224 at 5.16 FA/frame**, EXP-022 exactly.
  **Dump:** every candidate, kept *and* rejected, with `reason`, `kept`, `sigma`,
  `response` and all ring/core quantities; the old dump held kept rows only, so a threshold
  could be raised but never lowered. `--dump-floor` (default 3.0) always keeps structural
  rejections and on-target rows, cutting analog from 530,273 rows / 59.6 MB to 44,988 /
  4.8 MB — the dropped rows sat at c ≈ 1.4, below the **c = 1.8 ceiling measured on pure
  Gaussian noise sky**, so they carried no information.
  **Defects fixed:** the render drew ~190 rejected candidates per frame against ~10 kept,
  which is why the video looked full while the report said 5/frame (now `--show-rejected`);
  report sections printed for conditions the run had not applied, reading as measured null
  results (now gated); and the "target size" figures were the detected scale — see the
  correction section in [experiments.md](experiments.md), filed above as a blocking item.

- [x] 2026-09-29 — [algo] **Wire in the depth-aware ring and re-run the sky branch.**
  Done — **EXP-023**, in a new folder `runs/sofa_o4/exp023_sky_branch/` so EXP-022 stays
  frozen as the baseline the ledger cites. Videos: `sky_first_catch_650_964.mp4` (315
  frames) and `sky_catch_2_441_800.mp4` (360). **The whole-ring control reproduces EXP-022
  to the digit** (48/257 at 14.65 FA/frame; 73/224 at 5.16), which is what makes the rest
  readable. **The depth-aware ring alone is a recall win on O4** — 48 → 57 frames, because a
  target beside the horizon no longer has its `sigma_ring` set by the step instead of the
  sky (synthetic: `c` 8.2 whole-ring → 16.4 label-restricted). **The uncertain-band refusal
  is what cuts false alarms** (−47% O4, −31% analog) **and it costs 17 and 22 drone frames**;
  filed above as a per-clip choice. Also answers the size question: `c` is a per-pixel ratio
  that ignores extent, and `min(core)` was leaking size in *backwards* as a downward-biased
  order statistic, so `snr = (ring_med − core_mean)·sqrt(n)/sigma` is now recorded beside it
  (not a clear separation win: 85.9% vs 92.3% clutter-at-p10 on O4, 74.7% vs 71.9% on
  analog). A minimum-diameter floor **pays on analog and never on O4** — 4 px removes 22.1%
  of clutter for 4.1% of target on analog, while O4's target and clutter share a 3.1 px
  median — so `--min-diameter` defaults to off. Each run now writes a per-candidate CSV;
  every number in the ledger entry after the renders is a re-cut of those, in seconds.

- [x] 2026-09-29 — [algo] **Run stage 2 over a span with an overview video.** Done —
  **EXP-022**. Both branches built and rendered on two clips:
  `stage2_first_catch_650_964.mp4` (315 frames) and `stage2_catch_2_441_800.mp4` (360).
  New `silhouette.py` (2a: negative-polarity scale-normalised DoG over 3–40 px, the IRST
  contrast statistic, cloud/bloom/size rejections; 8 synthetic self-checks) and `layers.py`
  (2b: two-plane RANSAC with a separation guard, per-layer residuals, an epipolar
  **rejector** that can never confirm, depth-aware rings; 10 checks), driven by
  `overlay_stage2.py` with a single-plane control on identical frames.
  **The regime inversion is real** — sigma_ring 1.48 on sky against 11.86 (O4) and 7.41
  (analog) on scene — **and the two branches fail on different frames**, which is the only
  reason to have two: O4 sky 18.7% / scene 30.7% / either 37.4%, analog sky 32.6% / scene
  9.8% / either 33.9%. Three findings are filed as open items above: drop the layered
  homography, stop quoting `c` as a false-alarm rate, and add a second test after `c`.
  Four defects the self-checks caught before any long run are written up in the ledger —
  none would have thrown an exception.

- [x] 2026-09-29 — [algo] **Fit the epipole once per window, not once per frame pair.**
  Done — **EXP-021**. `runs/sofa_o4/exp017_motion_first/window.py` + `overlay_window.py`,
  k = 5 with 4 appearances required. **Every mechanical claim for the window holds** —
  residual coherence 0.920 (pure parallax 1.000, random walk 0.447), anisotropy 0.389 vs
  0.355, and the epipole's frame-to-frame jump falls to **40.9 px** from EXP-019's 126.8.
  **And it did not rescue the direction test**, because the field is not the one the test
  assumes: a *uniform shift* model outscores a focus of expansion on every span measured
  (37.2% vs 30.3% on 700–760, 38.5% vs 23.8% on 900–960, chance 22.8%), and `|mu|` is
  essentially uncorrelated with distance from the epipole (+0.17, +0.06) where parallax
  requires proportionality. **The win is persistence**: 80.6% of seeds dropped as flashes,
  load 132.9 → 25.8/frame for ~4 points of recall, while adding the direction test costs
  **33 points of recall** for 5.5 candidates/frame. Direction test off by default.
  Three errors the synthetic self-test caught before any video ran are written up in the
  ledger. One caveat recorded there too: the write-up's "epipole at chance (22.1%)" is a
  61-frame sub-span figure and the full span logged 35.4% — the comparative conclusion
  survives, the absolute one does not.

- [x] 2026-09-29 — [algo] **The analog HUD mask was deleting a fifth of the ground truth (EXP-020).** It covered 13.15% of the frame while the visible OSD is a fraction of that, because `osd_blocks` fills each block to its **bounding rectangle** — only 14k of its 91k px were ever called OSD. Scored against the 494 labelled boxes: the mask vetoed **100 of them (20.2%)**. The two bottom rectangles alone were 66% of the mask, cost **74 real targets**, and vetoed **zero** false alarms — no EXP-013 detection ever landed below row 274. Cause: `--picture-rows 150:530` read as "the sky is the picture", leaving rows 530–720 open to the fill, but 118 labelled boxes are below row 530. Fix is the parameter, not the code: `--picture-rows 150:720` gives **53,257 px (7.71%)**, **26/494 vetoed**, and a veto set **bit-for-bit identical** on EXP-013, so EXP-014 is untouched. It also closes **47 of the 55 frames** in EXP-017's standing `stage-0 masks cover the labelled drone` warning. Two dead ends recorded so they are not retried: glyph-tight masking (no better, loses 8 vetoes) and a median-OSD-plate residual (distributions overlap completely). [experiments.md](experiments.md) · [hud_mask.md](hud_mask.md#-picture-rows-must-name-all-the-picture-not-just-the-sky)

- [x] 2026-09-28 — [algo] **Restructured `src/` — eight stages, all landed.** De-duplication,
  layer-boundary fixes, shared interfaces and file splits. Plan agreed 2026-09-28; each
  stage is its own commit so the work can stop after any of them. The baseline-YOLO chain
  (`baseline_detect.py`, `algo/detector.py`, `algo/tiling.py`, `algo/config.py`,
  `data/frames.py`, `output/annotate.py`) is **deliberately out of scope** — hence the
  separate `tiling.py` test item in the backlog.
  - [x] S1 — CLAUDE.md describes packages, not files; tiling test filed
  - [x] S2 — characterization tests: `data/sources.py`, `plot_eval` CLI, `live_detect`
  - [x] S3 — `eval/tables.py` + `eval/vocabulary.py`: one dump parser, one CSV writer,
        one `rounded`, one set of outcome labels
  - [x] S3b — one `NO_TARGET` across `scene_stats`/`crosscut`/`alarms`, plus the
        `conditions.json` migration (1,412 values in two tracked files)
  - [x] S4 — `data/sampling.py` → `algo/sampling.py`, fixing the algo→data inversion
  - [x] S5 — `UsageError`: library modules stop calling `sys.exit`, so they are callable
        from a notebook as this file promises
  - [x] S6 — split `eval/metrics.py` into `matching.py` + `metrics.py`
  - [x] S7 — `print_branches`/`GLAD_CONFIDENCE` shared; `live.open_source` → `open_feed`
  - [x] S8 — `seed_track.py` split into `template_track.py` + `labels_io.py` + a thin
        CLI. Gate lifted 2026-09-28 (no labelling in progress, no other sessions).

- [x] 2026-09-23 — [algo] **Ran EXP-016's motion test unchanged on our own FIELD footage (EXP-017), and it acquires where the goggles clips did not.** User request. No constant re-tuned: EXP-016's `mf.py`/`collect.py`/`verify.py` run against a new `clipcfg.py`, every pixel parameter scaled by 1032/1440 = 0.717. Three defaults-preserving changes in the shared modules (`labels=None`, `hud=None`, `drone_ranges`); both SOFA clips re-import byte-identical, verified against the ledger's own frame counts.
  - **It acquires in all three PROVENANCE episodes** at the imported z\* = 20: 97 confirmed tracks over 3600 frames, **8 sustained-co-located with EXP-010 GLAD boxes** (≥50% of overlapping frames, ≥10 frames), one holding **183 of 185**. On O4 the drone's peak z was 13.9 against a required z\* of 60; here a target track reaches **z = 111.8** and holds 262 frames.
  - **The clip is genuinely easier for the flow stage**, which is most of the story: usable grid points **0.661** vs 0.489 (O4) and 0.389 (analog), FB failure **0.338** vs 0.461/0.492, masked **0.6%** vs 9.2%/18.2%. No HUD, no analog grain. Part of what EXP-016 measured as algorithm failure was a property of **goggles recordings**.
  - **False alarms are concentrated in time, not space** — 50 of 53 in the single hard banking manoeuvre (frames 1800–2699), where background image motion is 5.4–7.8 px/frame against 1.1–3.3 elsewhere. New failure mode: **host angular rate**, unlike O4 (overlay) and analog (horizon depth). It is the only one of the three detectable from the IMU without the video.
  - **Nothing here is a calibrated result and the entry says so first, before the numbers.** No labels ⇒ no Pd, no precision, no recall; z\* borrowed from the analog clip; "on target" means near a box another detector drew; the false-alarm rate is an upper bound; and the 3.4/min under-budget figure comes from an in-sample post-hoc split. **EXP-016's "do not build stage 2" stands.**
  - Overlay video `runs/field/exp017_multiframe/overlay_exp017.mp4`, frames 1–3600 at z\* = 20. Two follow-ups filed: label a FIELD episode (now ahead of the second analog clip), and test the angular-rate gate out-of-sample.

- [x] 2026-09-23 — [algo] **Stage 1 of the multi-frame motion plan: built it, ran it on both clips, and it does not pass (EXP-016).** Seed the top 200 `win_b5_e4` peaks per frame, then verify each over time by its motion relative to its own **25–120 px ring** — the 3–5 px parallax that swamps a whole-frame map is shared by the surroundings and cancels — with sequential confirmation (k_min 3, k_max 15 = the user's 0.5 s ceiling), the overlay veto, and a **speed cap derived in physical units** (0.6 m airframe, 30 m/s, doubled for closing ⇒ 3.33 × apparent size px/frame, camera-independent). z\* calibrated on empty frames only, two-fold, frozen per clip before any drone frame was scored. Analog `catch_2` carried equal weight to O4 `first_catch` throughout.
  - **Every pre-set pass bar fails.** O4: never confirms on the drone at the frozen z\* = 60 (bar: by frame 760), 0% coverage of 708–800 (bar: 50%), top-10 **0.24** (bar: ≥ 0.30). Analog: first on-drone confirmation **+76 frames** (bar: 30), **15%** span coverage (bar: 40%), at 2.7 false tracks/min.
  - **The one real win is ranking.** On `first_catch` 708–800, EXP-015 put the drone in the top 10 of **0 of 93** frames for every two-frame map. Accumulated differential z puts it in the top 10 of **0.22–0.30** at every threshold from z\* = 6 up (0.24 at the frozen z\* = 60) and in the top 5 of 0.12. The ~9σ premise re-derived clean on hand-placed labels: **4.2 px/frame against a 0.63 px ring scatter**.
  - **The mechanism of failure is measured, not guessed.** The z\* the false-alarm budget demands on O4 is **60**; the drone's maximum z on 708–800 is **13.9** — a 4× gap that no threshold closes. The clutter produces the same statistic: on analog **57% of false tracks confirm at the sky/tree-line boundary**, where the ring straddles two depths; on O4 **59% confirm within 30 px of a burned-in overlay**. Both bars *are* reachable on analog, but only at 11–76 false tracks/min against a 5/min budget.
  - **The speed cap is correct and nearly inert.** It scales with apparent size and clutter seeds are large — 53% of O4 seeds sit at the size clamp — so it rejects almost nothing. It is free, keep it, do not expect it to discriminate.
  - **The cheap comparison arm is dead.** A long-baseline difference through chained homographies (k = 3, 6) is *worse* than two frames on both clips: analog top-10 0.19 → 0.02 → 0.00. It does not need trying again.
  - **Two design-pass figures corrected.** Seed coverage of 708–800 is **76%**, not 92% (91% over 708–964); and the ~9σ ring result holds only on hand-placed labels — on follower boxes the same measurement reads **1.1σ**, because the follower smooths and lags.
  - **Stage 2 is not authorised.** `src/algo/temporal/` is not built. The binding constraint is now the evidence base, not the algorithm: both verdicts rest on ~22 s of drone-free footage per clip and one approach each. See the new open items — label a second analog clip, and the analog-range finding for `deploy-agent`.
  - Nothing under `src/` changed, so no tests were added; everything lives in the gitignored `runs/sofa_o4/exp016_multiframe/` and `runs/sofa_analog/exp016_multiframe/`. Full entry, tables, caveats and overlay-video notes in [experiments.md](experiments.md).

- [x] 2026-09-22 — [algo] **Tried an edge-normalised frame difference on `first_catch` (EXP-015).** The compensated difference is divided by the local gradient, both pointwise and as a windowed normal-flow magnitude. The map is scored as a peak generator against the 257 labelled frames and compared with MOD2's plain difference, with the threshold set to 10 candidates per empty frame (1–707).
  - **It suppresses static textured clutter as predicted.** Roofs and canopies go dark, and scene statics read their 5–6 px misalignment.
  - **At the same empty-frame load it halves the candidates per drone frame** (26 → 14). Median drone rank goes 56 → 29, and top-10 goes 0.11 → 0.27, reaching 1.00 at 954–964.
  - **It gets 0 of 93 frames in 708–800,** the long-range stretch, and so does every variant: lighter blur, ε, windowed, and DIS dense flow, which scored 0 everywhere.
  - **The loudest clutter was the host drone's own prop blades and the moving pitch ladder,** not terrain.

  Scripts and stills are in `runs/sofa_o4/exp015_normalised_motion/` (gitignored). The recommendation is to fold normalised ranking and airframe/ladder masks into the global branch (see the open item), and to label more long-range O4 footage before any learned-model decision.

- [x] 2026-09-22 — [algo] **Deleted the `runs/field/_scratch_*` folders (~120 MB).** They held throwaway output from 2026-09-17: codec trials, sample frames, early HUD-mask attempts, smoke runs and trial labels. Nothing in the repo referenced them. The codec results are in `data/processed/SOFA-O4/MANIFEST.md` and the real mask is `data/processed/SOFA-O4/hud_mask.png`. The one keeper, the lossless crop script that produced `data/processed/SOFA-O4/videos/`, moved beside that MANIFEST as `make_processed.py`.


- [x] 2026-09-22 — [algo] **Grouped `runs/` by footage.** It now has `field/` (EXP-010–012 and the two `_scratch_*` folders), `sofa_o4/` (EXP-011 O4, 012a, 012b) and `sofa_analog/` (EXP-013, 014). Each experiment kept its folder name as a subfolder, because runs share file names such as `detections.jsonl` and `overlay.mp4`. Logs and previews sit beside their run. Every path cited in `docs/` and in the `src/glad_detect.py` and `src/render_video.py` docstrings was rewritten to match.

- [x] 2026-09-22 — [algo] **Pruned `runs/` to the FIELD and SOFA runs.** Deleted the EXP-004–009 run folders (ARD-MAV / ARD100), both `compare_*` folders and their logs — about 1.9 GB. The EXP-001–003 metric JSONs were already gone. Every number those runs produced was already in `docs/experiments.md`, which now notes that the `runs/` paths cited in EXP-001–009 no longer exist and must be regenerated from each entry's recorded command. Kept: EXP-010–014, 012a/b, and the two FIELD `_scratch_*` folders.

- [x] 2026-09-22 — [algo] **Re-measured EXP-012a's motion claim on the `first_catch` labels (EXP-012b), and it was wrong.** EXP-012a compared the target's raw image motion (4.5 px) with the raw background spread. The right comparison is the target's motion relative to the compensated background, against what the homography leaves behind. On hand-placed boxes that is **4.3 / 10.1 / 13.0 / 22.2 px/frame** through the approach, against a residual of ~1 px median and 3–7 px p90. The drone is in `MOD2_global`'s difference image in 199 of 257 frames and is found in 0: it is lost to the crowding bail (126 frames), shape ranking (90 under `clutter`) and the aspect test (71). A correction note sits on EXP-012a. `docs/annotate.md` and the `Follower` docstring repeated the old claim and are fixed. The fisheye item's premise is withdrawn, and a new item targets ranking and the aspect test.

- [x] 2026-09-18 — [algo] **Vetoed the moving analog OSD, and GLAD found the drone in `catch 2` (EXP-014).** Detections fell from 82 to 8. The drone was boxed at frame 547 of a passage visible from about 510 to 585, which EXP-013 missed entirely while locked on telemetry. Two additions made it work. `src.data.hud_mask --block-fraction/--picture-rows` masks OSD text blocks as whole rectangles, but only outside the picture rows. `src.glad_detect --osd-twins` vetoes a box with an identical copy one OSD column away, which catches the artificial-horizon dashes. The drone's own twin score was measured at 0.42–0.63 against a 0.70 veto. 21 new tests. Also corrected EXP-013, which said no drone was found.

- [x] 2026-09-18 — [algo] **Ran GLAD on the analog `catch 2` clip and rendered it — EXP-013.** No drone was found. All 82 detections were inspected: 38 are the compass tape's "N", 31 are telemetry digits, 15 are one dash of the artificial horizon, 3 are ground clutter and 1 is undetermined. The overlay is at `runs/sofa_analog/exp013_analog_catch2/overlay.mp4`. Staged the nine 960×720 analog clips under `data/raw/SOFA-ANALOG/` and built an analog HUD mask. Analog OSD needs `--white-level 180`, because the O4 default of 225 caught only the date strip. The next step is to find the target in `catch_2` by hand, since a miss and an empty clip currently look the same.

- [x] 2026-09-18 — [data] **An annotation window: `src.data.annotate` and the `/annotate` skill.** Play, pause and step a clip; drag a box, move or resize it by its body, edges or corners; the box then **follows the target** and the user only corrects it — each correction re-seeds the template, and loss stops playback. `x` marks a frame empty and the negative carries forward while playing, so target-free spans label themselves. Writes `seed_track`'s formats (YOLO + `verified.jsonl`) plus a resumable session recording `human`/`tracked`/`carried` and a score per frame. **Following is correlation plus a constant-velocity prior, deliberately not frame differencing** — EXP-012a measured the background residual on this camera above the target's own motion; the prior is pinned by a test where 60 px/frame loses correlation alone at step one and the prior holds it every step. **One measured trap, now guarded:** a box drawn on empty sky (grey std 1.1) "followed" open sky for 20 frames at **~0.8** — normalised correlation stretches a flat patch to full contrast — so featureless templates (std < 4) and matches under 0.35x the template's contrast are refused; the 80x34 drone at 962 has std 46.9. On that drone the follower held 962→955 backward (0.94→0.47) and stopped at 954 rather than guess. Backward stepping refills the frame cache half a buffer at a time: 192 ms/step against 454 ms seeking per keypress. Also corrected `seed_track`'s docstring example, whose `962 690 236 46 26` box is empty sky in the processed clip. 26 new tests; reference in [annotate.md](annotate.md).

- [x] 2026-09-17 — [algo] **Measured the two O4 fixes on `first_catch` — EXP-012a.** **The HUD veto works:** 166 detections fall to 9 and every glyph lock is gone. **The motion fix does not buy recall:** frames 950-964 are `global miss` without exception, the unmistakable 80x34 px quadcopter at 962 is still missed, and all 9 survivors are HUD ladder dashes or ground. The reason is measured rather than guessed — the target moves **4.5 px/frame** while the background moves 21 px with a **~10 px spread between median and p90**, so the parallax a single homography cannot model leaves a residual *larger than the target's entire differential motion*. Intercept geometry does this: closing on a target puts it near the focus of expansion where image motion is least, while the near ground streams past. GLAD's motion premise is inverted here, and no threshold recovers a signal quieter than the noise around it. **This corrects the EXP-011 write-up**, which blamed the 50 px flow-rejection cap; measured flow is 18.9-24.3 px median, under the cap everywhere.

- [x] 2026-09-17 — [data] [algo] **Built the three fixes EXP-011's diagnosis called for.** (1) `src.data.seed_track` — labels our own footage from a hand-placed box, since nothing on it is measurable otherwise. Two measured defaults: template blending is **off** (with it on, a track that slipped onto terrain adopted the terrain and matched it at 0.99 — the highest-confidence proposals were the worst ones), and `--min-score` stays at 0.45 (0.35 bought 118 frames sitting on trees). Reads sequentially: a seek costs 1,109 ms against 26 ms. (2) `src.data.hud_mask` + `src.glad_detect --hud-mask` — finds the burned-in overlay (28,002 px, **1.80% of the frame**) and vetoes boxes lying mostly on it, at every point one can be emitted *or locked onto*. On EXP-011's detections it vetoes **83.1%**, while the one confirmed drone scores **0.000** overlap. Nothing is inpainted. (3) `src.algo.glad.motion` — `MOD2` ported so its absolute-pixel constants can move, with `--motion-profile clutter` keeping ranked candidates instead of discarding all of them. Measured cause: `first_catch` frames 901-907 produce **105-170 candidates against a cap of 50**, so upstream discarded everything on every frame. Equivalence against the vendored module is pinned on **both** paths — the saturation path on O4 and the accept path on ARD-MAV — plus a test asserting the two fixtures really are different regimes, so it cannot pass vacuously. The default still loads the vendored `MOD2`, not the port.

- [x] 2026-09-17 — [algo] **Made the run-inspection workflow real — `src.crops` and the `/inspect` skill.** The contact sheets that carried EXP-010's 23-of-24, EXP-011's white-structure verdict and EXP-012's drone-with-payload were all produced by throwaway scripts in gitignored run directories. EXP-010's sampling script was lost that way once already and recorded as a loss at the time; this is the second occurrence, so the tooling became code. [`src.crops`](crops.md) does the three things actually being done — every detection grouped and coloured by branch, a seeded sample for precision-conditional-on-firing, and one span rendered large enough to settle what an object is — over a **single forward decode**, since seeking to 800 scattered positions in a 388 MB mp4 costs minutes and a regression to per-detection seeking would still produce a correct sheet (`tests/integration/test_crops.py` pins it). `open_video` moved from `src.render_video` into `src/output/video.py` so both CLIs share one definition. The `/inspect` skill covers both tools and, more usefully, **how to read what they produce**: a sheet is a judgement and never a score, a 4–8 px blob supports no confident verdict either way, a second identical unboxed object means the detector is picking scene features, and a negative from eyeballing is not evidence. Also rendered EXP-012's `overlay.mp4`.

- [x] 2026-09-17 — [data] [algo] **Filed the O4 intercept trials and ran GLAD over them — EXP-011.** Six clips from EXP Sofa Base 24-08-26, copied to `data/raw/SOFA-O4/videos/`. They are **goggles screen recordings**: 2520×1080 of which only a 1440×1080 window is picture. Cropped into `data/processed/SOFA-O4/videos/` as **FFV1 lossless** — benchmarked first (mp4v 37.5 dB, avc1 37.8, MJPG 41.2, FFV1 lossless) because a lossy generation would smear the 10–30 px targets being measured — and **verified bit-exact on all 7,386 frames**. Cropped but **not rescaled**: the picture is already 1080 lines, and widening 1440→1920 would stretch 4:3 into 16:9 and turn a round drone into an ellipse. Added `--crop` to `src.glad_detect`/`src.render_video` and `resolve_video` so a stem resolves in any container (FFV1 cannot live in MP4). **The run's headline is a negative result:** 1,347 detections at 3.11 fps, of which a seeded sample of 24 held **21 HUD glyphs, 3 ground, 0 drones** — see the open item above. Six overlay videos in `runs/sofa_o4/exp011_sofa_o4_glad/examples/`.

- [x] 2026-09-17 — [algo] **Made GLAD runnable at the right scale — `--scale` on `src.glad_detect`.** Every constant in the motion branch is absolute pixels tuned at 1920×1080 (blob area 30–3000 and the 50-blob cap in `MOD2_global`, `dist_ref = 200` and the 30-blob cap in `MOD2_local`, blur kernel 11, `REGION_HALF = 160`, `MAX_DISTANCE` 50 and 10); none is a ratio, so on our 1032×752 FIELD capture — **0.579× linear, 0.335× in area** — every one of them was being applied to a target a third the size it was calibrated for. EXP-010 therefore measured our failure to rescale alongside the detector. `src.algo.glad.scaling.ScaledPipeline` resizes each frame to the reference **diagonal** and maps boxes back, so the record stays in original pixels and a scaled run's JSONL compares directly against an unscaled one's — unlike `--crop`, which declares its coordinate change. Aspect is preserved rather than forcing 1920×1080: upstream's own commented-out attempt distorts anything not 16:9, and this capture is 1.372:1, while MOD2's flow-coherence tests threshold on exactly the spread anisotropic scaling perturbs. **Frame-side, not constant-side** — the constants are function-local literals inside the vendored, gitignored `MOD2.py`, with nothing `import_motion` can rebind, so doing it properly means porting the motion module and putting every EXP-004–010 number behind a code change; [glad-model.md](glad-model.md) improvement #4 now records the half that is done and the half that is not. A factor leaving the dimensions unchanged delegates with the original array, asserted on **object identity** — a resize to the same size would compare equal while still having run an interpolation kernel — which is what keeps EXP-004–010 citable. Costs 2.98× the pixels on FIELD, measured 2.15 fps against 3.05 native. Reference: [glad_detect.md](glad_detect.md) § "Footage that is not 1080p".

- [x] 2026-09-17 — [algo] **Ran EXP-011 and settled the resolution question — it is not the cause.** `--scale auto` on the FIELD capture (1032×752 → 1780×1297, 2.975× pixels) moved `global miss` from 66.3% to **66.0%**, which is noise. The composition changed a great deal and inspecting it is what decided things: the three known drift frames (19/101/176, box on a bush) are **fixed**, but the run **invented two larger clutter locks** at frames ~1832–1885 and ~2374–2464 — a seeded 24-crop sample of those 113 detections was **24/24 ground clutter, not one drone** — while **losing 186 in-episode detections** whose crops show large, unmistakable multirotors against blue sky. Median box fell 12.4 → 9.6 px √area, the signature of a population shifting from drones to clutter blobs. **Likely mechanism:** the blob-area gate (`30 < area < 3000` px²) was doing real clutter rejection at native resolution, and 2.975× in area lifts sub-threshold rocks and bush crowns into the admissible window while the drone, already inside it, gains nothing. If that reading holds, the **constant-side** fix — scaling thresholds down rather than the frame up — is not equivalent to this run and could still help; it remains [glad-model.md](glad-model.md) improvement #4's expensive half. Also worth keeping: **2.66 fps at triple the pixels against 3.05 native**, only 13% slower, so the motion branch is not where this pipeline spends its time — the YOLO passes are, and optimising MOD2 would buy almost nothing. Full entry in [experiments.md](experiments.md).
- [x] 2026-09-17 — [data] [algo] **Filed our own capture and ran GLAD over it — EXP-010.** `captured_raw_20260616_040253_004.mp4` moved from `data/` into `data/raw/FIELD/videos/`, mirroring the `data/raw/<NAME>/videos/` layout ARD-MAV and ARD100 use; provenance in [datasets.md](datasets.md), since `data/` is gitignored and a record that only exists on this laptop is not a record. **1032×752, 30 fps, 3,600 frames (120 s)**, arid hillside, target multirotor against sky in three episodes. Required one change: `src.glad_detect` records a frame only when a label file exists — the gate that keeps a prepared split honest — which on unlabelled footage writes an empty JSONL, so `--record-all` lifts it. **GLAD works on our footage**: 866 detections, 96% of them inside three sustained lock-ons, and a seeded random sample of 24 crops contained 23 drones and one bush. **No score is possible and none was reported** — there are no labels, so recall in particular is unmeasured, and demonstrably not 1 (frames 19–176 hold a box on a bush while the drone is in clear sky). Two numbers the prepared datasets could not have given: **3.05 fps** at this resolution, and **66.3% of frames in `global miss`** — the expensive branch, and the regime the edge budget has to survive. Also added `src.render_video --no-labels`, which draws the boxes without judging them: the scored renderer colours by match outcome, so unlabelled footage would have come out as 866 red false alarms — a measurement nobody made. Video at `runs/field/exp010_field_glad/overlay.mp4`.

- [x] 2026-08-24 — [deploy] [algo] **Measured what duty-cycled inference costs — EXP-006 to EXP-009**, four runs plus four zero-inference controls. Frame *striding* was already ruled out ([edge-budget.md §3](edge-budget.md) item 7) because both GLAD motion branches difference against the previous frame, but **that objection is to the gap, not to processing fewer frames**: half rate keeps every processed frame adjacent to its predecessor, and a burst differences inside a 33 ms pair. Added `--sample nth|burst` to `src.glad_detect` (`src/algo/sampling.py`) and `--keys-from` to `src.evaluate`, which restricts a full-rate run's persisted JSONL to a duty-cycled run's exact frame set — the only legal comparison, and it costs no inference. **Half rate retains 91.5% of recall on *both* ARD-MAV and ARD100** (0.8182/0.8943 and 0.6290/0.6876), so its cost is a property of the policy and does not compound with the generalisation gap. **Burst pairs retains 48.2% and 36.0%** — it does compound, because cold acquisition leans on GAD, which is what fails to generalise. Burst is disqualified by latency rather than recall: at 25.0% per burst the expected wait is four bursts, so the proposed 30-second period means **120 s and 4,806 m of closing at 40 m/s**. Compute saving is content-dependent and is *not* the frame ratio — 2.71× on ARD-MAV against 1.55× on ARD100, because losing lock more often means paying the motion path more often. Full entries in [experiments.md](experiments.md); lever table row 11 in [edge-budget.md](edge-budget.md).
- [x] 2026-08-24 — [deploy] **Built the live detection path — `src.live_detect`**, which turns those measurements into a decision: warm up, time the pipeline against the feed's real rate, then take the most accurate policy the machine can sustain (full rate → half rate → burst pairs). Burst pairs is the fallback because it is the only policy that stays *coherent* when the pipeline is slower than the feed — taking whatever frame is newest leaves gaps that are large and unknown, which is item 7's striding, whereas two adjacent frames keep the difference meaningful however long the sleep. Source layer (`src/data/live.py`) serves both an HDMI-to-USB capture card and **saved footage replayed at the recording's own frame rate**, dropping what the pipeline was too slow to collect — which makes a replay a simulation of the deployed system rather than an offline run. Verified on `phantom05.mp4`: frame age holds flat at ~370 ms instead of climbing, the property a naive `read()` loop gets wrong while appearing to work. The window carries the policy and what it cost, because a box drawn without its policy invites comparison against a full-rate number. Reference: [live_detect.md](live_detect.md). **Never run against the intended camera** — see the open item above.
- [x] 2026-08-24 — [deploy] **Pinned `opencv-python` below 5.0 and banned `opencv-python-headless`.** Both share the `cv2` namespace, and the headless build was shadowing the GUI one, so `imshow` raised and no window could open; uninstalling it also removed files the GUI build needs. Repairing pulled in OpenCV 5.0.0, which **changes `resize` sampling — and that call sits inside the detector's letterbox (`src/algo/glad/yolo.py`), so it would silently shift every detection away from what EXP-004 to EXP-009 measured.** Caught by `tests/unit/test_overlay.py`, which failed on the inset magnification. Reverted to 4.12.0.88; reason recorded in `requirements.txt` beside the constraint.

- [x] 2026-08-23 — [data] **Derived the `lighting` and `relative_range` axes for the
  ARD-MAV test split** — `py -3.13 -m src.data.scene_stats --processed
  data/processed/ARD-MAV --split test`, 28,337 frames, ~55 min. Open since 2026-08-18 and
  abandoned twice for competing with a scoring pass; run alone this time. `conditions.json`
  now carries all three axes (`scene_category` preserved), so `src.evaluate --conditions`
  breaks any ARD-MAV run down along them.

  **Done because M4b's backlit control needed it to be symmetric**, and it immediately
  corrected two things now fixed in [experiments.md](experiments.md) and
  [prepare_ardmav.md](prepare_ardmav.md):

  1. **ARD-MAV test is 18.6% `backlit`, not the "~0%" EXP-005 first assumed.** The
     0.0–0.2% figure was a per-video mean `pct_blown`; `backlit` is a per-frame test at 2%.
     Different quantities. Real gap against ARD100 is 18.6% vs 29.7%.
  2. **`backlit` is the *easiest* bucket on ARD-MAV** (R 0.9165, above its 0.8946 aggregate)
     and the second-hardest on ARD100 (0.6277), so the label does not carry an intrinsic
     difficulty.

  EXP-005's conclusion is unchanged: the symmetric non-backlit comparison is 0.8896 →
  0.7126, **17.7 points**, against 20.7 aggregate. Also settled that `relative_range` is
  **not** cross-dataset comparable — ARD-MAV has a `very far (>5x)` bucket (2,758 frames)
  that ARD100 lacks entirely, and 36.5% `far` against ARD100's 3.0%.

- [x] 2026-08-23 — [M4b] [algo] **Scored GLAD on ARD100's 15 unseen videos — M4b is
  answered. Recorded as EXP-005 in [experiments.md](experiments.md).** 34,287 frames /
  33,517 boxes, 2.65 h at 3.60 fps on CPU, byte-identical settings to EXP-004 so the one
  variable is the video content.

  **GLAD retains roughly three-quarters of its recall on unseen video from the same
  campaign.** At `centre@1x` — recall **0.895 → 0.688**, precision 0.993 → 0.946, false
  alarms **188 → 1,316** (0.0066 → 0.0384 per frame). At IoU@0.50, 0.834 / 0.606 / 0.702.
  The honest sentence is *"retains 77% of its recall on unseen video from the same
  campaign"*, never *"GLAD generalises"* — same lab, same rigs, likely the same capture
  campaign, so this is the optimistic end of any generalisation estimate.

  **Both confounds the plan warned about were checked, and neither explains the drop:**
  1. **Backlighting** (29.7% of frames, against ARD-MAV's ~0%) — cutting every backlit
     frame lifts recall only **2.5 points, 0.688 → 0.713**. An **18.2-point gap survives
     the control**, so exposure is ~1/8 of the effect.
  2. **Size composition** — recall falls in the **medium (32–96 px)** control bucket too,
     0.956 → 0.658. A composition shift reweights buckets; it cannot move them.

  **The mechanism is lock-loss, not localisation.** `global miss` rises 2.9% → 12.3% and
  `local yolo` falls 88.4% → 66.6%: GLAD loses lock four times as often and re-acquires
  worse. Meanwhile mean IoU on matched boxes is **0.6805 against 0.6829** — when it finds
  a drone it places the box exactly as well as on its training campaign. Range collapses:
  near 0.784 / mid 0.561 / **far 0.164**.

  **What this hands M7:** the lever is acquisition — GAD failing to re-acquire after lock
  loss, and at range — not a box-regression head and not a tighter matching rule.

  Two smaller notes carried into the ledger: ARD100 publishes no
  `ordinary`/`complex`/`small_mav` grouping, so EXP-004's headline cut has no counterpart
  and the comparison is aggregate plus **size in pixels** (never `relative_range`, which is
  scaled per split); and the motion module's 50-candidate cap
  (`third_party/GLAD/MOD2.py:60,183`, which returns an *empty* list rather than degrading)
  **fired 35 times** — never triggered on ARD-MAV, negligible here, but a silent recall
  floor on genuinely cluttered data.

- [x] 2026-08-23 — [M6] [deploy] **Added §4.2 to `docs/edge-budget.md`: the camera and the
  capture stage.** Answers "what FPS with a 12 MP Allied Vision 1800?" — and the honest
  answer is **it depends which of four cameras that is, and feeding GLAD the full 12 MP
  does not work.** The doc previously had **no camera model at all**, a genuine gap:
  capture is the first term in the end-to-end chain `deploy-agent`'s charter defines, and
  at 12 MP it stops being a rounding error (~30-35 ms, comparable to the whole inference —
  so §1's assumed 60 ms of pipeline overhead is now flagged as too low).

  **"12 MP Alvium 1800" is four products.** U-1240 (IMX226 rolling, USB3, **29 fps**),
  C-1240 (IMX226 rolling, CSI-2 4-lane, **41 fps**), U-1236 / C-1236 (IMX304 **global
  shutter**, **22-23 fps**). Budgeted the C-1240 as the compute worst case. **The U-1240's
  29 fps is a USB3 payload cap at 8-bit, not a sensor figure** — ask for RAW10 and it falls
  to ~23. On CSI-2 the same sensor needs 5.00 Gbit/s against a 10 Gbit/s 4-lane link, so
  there the sensor binds and the link does not.

  **Ingest is fine, and is the least risky part of the whole deployment:** Orin NX offers
  two 4-lane D-PHY groups at 2.5 Gbit/s per lane, its ISP runs 1.75 GPixel/s against the
  500 MPixel/s needed (29%), and Allied Vision ship a **JetPack 6.2 `.deb` driver covering
  all Orin modules** — the exact opposite of the dead TensorRT 7.2 path in §4. Caveat: the
  driver is **4-lane only**, no 2-lane fallback. USB3 would additionally tax 15-30% of a
  core on the branch that is **CPU-bound OpenCV**, so CSI-2 wins on two counts.

  **The crux, decomposed against EXP-004's branch split.** 12.20 MP is **5.88x** 1080p.
  The modal frame (88.4%, LAD on a fixed 320x320 crop) is **inference-cost-invariant to
  sensor resolution** — but its per-frame full-frame overhead is not, and the motion path
  (11.6%, all `O(pixels)`) takes the full 5.88x on the branch that was **already** the
  bottleneck. Projected Orin NX: **~13-14 fps sustained but 2.4-3.2 fps when the motion
  path fires**, which spends 38% of the whole engagement in latency. **That row fails.**
  Worse, GAD's 640 letterbox goes from a 3x reduction to **6.3x**, so a 21 px target
  reaches the acquisition detector at 3.3 px — the founding trap, doubled, on the branch
  EXP-004 already named as the real ceiling.

  **The upside is real and was stated fairly.** At the same FOV, 4024 px is **2.096x finer
  linearly**: first-detectable range moves **55 m -> 115 m**, the engagement window
  **1.4 s -> 2.9 s**, and — the number that matters — **the 0.98-recall boundary moves from
  ~34 m out to ~72 m.** 18,265 of ARD-MAV's 28,160 targets are under 16 px, so this attacks
  the project's central failure mode more directly than any architecture on the table.

  **Recommendation: never feed GLAD the full frame.** Option (c) — native-resolution 320
  crop for LAD, downsampled copy for the motion module *and* GAD — projects **~25 fps
  sustained / 7-9 fps hard scene** and is the only configuration that improves both range
  and the latency ratio. **Flagged loudly that this is a protocol change, not a config:**
  `yolov5s_GLAD-crop.pt` was trained on 320x320 crops from 1080p, so a native crop of a
  12 MP frame is a 2.1x input-scale change needing a **retrain, not a re-score** — and
  every absolute-pixel constant in `MOD2.py` breaks with it. No accuracy number in this
  repo may be read as applying to a 12 MP build, and that verdict is `algo-agent`'s.

  **Two findings that outrank the frame rate.** (1) **Rolling vs global shutter.** IMX226
  is rolling; GLAD compensates ego-motion with a **RANSAC homography**, which assumes all
  pixels were captured at one instant. Under rolling shutter with fast rotation that
  assumption fails and the residual — *which is exactly what GLAD thresholds as motion* —
  is corrupted, worst during hard manoeuvre. Counter-argument recorded honestly: ARD-MAV
  was shot on rolling-shutter Mavics, so it demonstrably works **on a gimbal**; nothing
  here measures hard-mounted. Since the pipeline sustains only ~13-25 fps anyway, **the
  C-1240's extra frame rate is unusable and the global-shutter C-1236 is probably the
  better buy** — its 1.1-inch sensor also attacks M2b's measured 19-point contrast loss.
  (2) **The lens is a competing answer and it is cheaper**: a 28.6-degree lens on 1080p
  gives identical pixels-on-target for 5.88x less compute. Which wins turns entirely on a
  question nobody has answered — **is the sensor cued or searching?** — now recorded as the
  **third unresolved user input** alongside closing speed and persistence frames.

  §8's three tables updated (camera/ingest specs as published, the 12 MP projections as
  [EXTRAPOLATED], six new assumptions including the untested linear-pixel-scaling
  exponent), §1 and §6 cross-linked. **No tests: docs-only, no behaviour or parameters
  under `src/`.** Three follow-ups filed — the pixel-scaling check (free, no camera), the
  ROI mode-switch latency (a vendor email), and the cued-vs-searching decision (blocked on
  the user, and the cheapest open question in the project).

- [x] 2026-08-23 — [M6] [deploy] **Wrote `docs/edge-budget.md`**, answering "this model is
  very slow, is there a faster one I can deploy?" — and the answer is **no, and you do not
  need one.** The slowness decomposes almost entirely onto the **host**: GLAD is published
  at 23.6 FPS on a 2019 Jetson Xavier NX and 146.5 FPS on an RTX 3070 against our **2.67
  fps** sustained (EXP-004, 28,337 frames, fp32 PyTorch, CPU-only laptop) — **~9x and ~55x,
  closed by a $249 board, not by research.** Protocol contributes ~1x (GLAD has no tile or
  resize switch, and striding is incoherent for a pipeline that differences consecutive
  frames) and the model ~1x (`yolov5s` at 640 is already the cheap end of what works).

  **The premise inverts on the measured numbers.** GLAD is the *second-fastest* of the five
  configurations this project has timed and carries 7-70x the recall of every one of them;
  the two fastest score exactly **zero** tiny-target recall. There is no speed/accuracy
  trade available because the fast end of the curve is empty.

  **Two premises in the prompt were corrected in the doc.** The 11 fps / 0.72 fps pair in
  `CLAUDE.md` is `src.baseline_detect` with **`yolov8n` at 720p** — a different CLI, model
  and resolution, not GLAD's number. And GLAD's own two figures disagree: **4.8 fps spot vs
  2.67 fps sustained**, with `exp005` observed at 3.16-3.92 — the mean-vs-worst-case rule
  applying to our own measurement, not just to vendors'.

  Recommendation: **Jetson Orin Nano Super 8 GB (~$249, 67 TOPS, 102 GB/s, 15 W)**, GLAD
  unchanged, `.pt` -> ONNX -> **TensorRT 10 FP16**, stopping short of INT8 (1.2x for a
  3-7 point mAP risk landing in exactly the smallest size bands, where 18,265 of ARD-MAV's
  28,160 targets live). Confirmed a **toolchain break** that makes this a port and not an
  export: the released `detector*_trt.py` deserialises **TensorRT 7.2** engines, and TRT
  engines do not load across JetPack versions — JetPack 6 ships TRT 10.x with a changed
  API, so the engines and the `libmyplugins.so` plugin must be rebuilt. Six explicit
  change-my-mind triggers recorded, including **multi-intruder, which architecturally
  disqualifies GLAD** (single target by construction) and promotes YOLOMG regardless of
  speed.

  Alternatives surveyed with board/precision/resolution/batch/content on every figure and
  the non-transferable ones marked: YOLOMG (133/35 FPS but on an **RTX 2080Ti**, different
  dataset *and* split from EXP-004, GPL-3.0, no pretrained weights), A2A-YOLO (15 FPS on
  RK3588 but Det-Fly's 4K large targets, appearance-only), TransVisDrone, and plain
  YOLO26n/YOLO11n at 3.80 ms INT8 on Orin — which is the "fast model" the question reaches
  for and which EXP-001-003 already measured at **AP@0.5 0.006-0.025**.

  **The whole document is marked as assumption where it is one.** Closing speed and
  persistence frames remain **the user's calls and are still unsupplied**, so §1 is worked
  with invented values and labelled as an illustration; §8 separates measured / published-
  under-their-conditions / assumed line by line. No edge hardware has been benchmarked and
  the doc says so at the top. Every accuracy claim about a swapped or quantised model is
  flagged unproven until re-scored through `src.evaluate` at the same criterion — that
  verdict is `algo-agent`'s property, not `deploy-agent`'s.

  **No tests: docs-only change, no behaviour and no parameters under `src/`.** Two
  follow-ups filed under Open — the per-stage profile (which could not be taken because
  `exp005` was holding the CPU) and the ONNX/OpenVINO export.

- [x] 2026-08-20 — [meta] [deploy] **Added `deploy-agent`, a third boundary beside data
  and model.** `.claude/agents/deploy-agent.md`, registered in `CLAUDE.md`. M6 and M7 were
  both open and **neither existing agent owned them**: `dataset-agent` opens with "models
  are somebody else's problem", and `algo-agent`'s only contact with hardware is one bullet
  telling it to quote GPU cost. Nobody owned *does it run on the target board at the target
  frame rate*. The new agent owns `docs/edge-budget.md`, latency/power budgets, export and
  quantisation, board selection, and renting the GPU that training runs on — while the
  accuracy ledger stays `algo-agent`'s property, so the M6/M7 split is stated in both
  agents' terms rather than left to be discovered.

  **Three project-specific traps written into it**, all of which would otherwise have been
  rediscovered on the hardware:

  1. **A mean frame rate is not a guarantee.** GLAD's 23.6 FPS on a Xavier NX is an
     *average* over a pipeline whose expensive motion path fires only when appearance
     detection fails — GMD alone is 5.1 FPS on that board. Throughput collapses ~4.6x
     exactly when the scene is hard, so the agent reports worst case and percentiles.
  2. **Export is a protocol change, not packaging.** FP32→FP16→INT8 is the `imgsz` rule
     `algo-agent` already enforces, wearing a deployment badge — and the damage lands in
     the smallest size bands, where a 10–30 px target is a handful of activations.
     Re-score through `src.evaluate`; the size cut is a re-cut of a persisted dump, not a
     new experiment.
  3. **Every number in the ledger came from our CPU shim, not from TensorRT.** EXP-004 ran
     `src/algo/glad/`; the released entry point deserialises TRT 7.2 engines through
     `detector*_trt.py`. A Jetson means returning to a code path **no measurement in this
     project was taken on** — a new implementation to validate, not an export.

  Also derives "real-time" from closing speed and persistence frames rather than inheriting
  30 FPS, and separates detection latency from inference latency. No tests: an agent
  definition is markdown config, with no behaviour under `src/` to cover.

- [x] 2026-08-20 — [algo] **Bin false alarms by distance from the nearest real drone.**
  `src/eval/alarms.py` + CLI `src/alarm_eval.py`, documented in
  [alarm_eval.md](alarm_eval.md); `nearest_gt_dist` / `nearest_gt_dist_rel` /
  `nearest_gt_size` added to the dump in `src/eval/records.py`, computed by the new
  `matching.nearest_target` (then in `metrics`). Separates the two failures `far` pools: a box on the drone
  versus a box on a rooftop. Zero cost — dumps written before the columns landed are
  re-derived from the frame's own rows, so no run needed re-scoring. Result in
  [experiments.md](experiments.md): **all 3,470 alarms IoU@0.50 adds over centre matching
  are inside one target size** (median 2.2 px), while GLAD's 188 real alarms have **zero**
  inside one target size and a median of 100 px. Tabled in both units (`--unit px` added
  to the ledger 2026-08-20, `runs/exp004_glad/alarm_distance_px.csv`); the pixel ladder
  smears the criterion boundary across three bins because the boundary is size-relative,
  which is why `rel` is the default. The baselines are 65–78% beyond 32 target
  sizes at a ~600–870 px median — their failure was never localisation.

- [x] 2026-08-20 — [algo] **Cross-cut the scoring dump by target size *and* capture
  condition at once.** `src/eval/crosscut.py` + CLI `src/cross_eval.py`, documented in
  [cross_eval.md](cross_eval.md). Answers the conjunction the metric block's one-way
  blocks cannot — the marginals are not multipliable, since 5,032 of ARD-MAV's 5,677
  sub-8 px targets are `small_mav`. Cells are built frame-by-frame so a false alarm
  inherits the band of the frame it fired in, which is what makes per-cell `far`
  comparable to the metric block's. Zero cost: a `GROUP BY` over persisted dumps, and
  pooling every cell reproduces EXP-004's headline exactly. Result in
  [experiments.md](experiments.md): the sub-8 px / `complex` cell is Pd 0.8625 at 0.0031
  FA/frame under centre matching and Pd 0.5844 at 0.2812 under IoU@0.50 — and 562 of its
  640 targets are one video (phantom58), so the background label is not what the number
  is about.

- [x] 2026-08-20 — [tools] [algo] **`src.render_video`: a scored run drawn back onto its source video.** Both boxes on every frame — ground truth and the detector's claim — coloured by the match outcome, with a nearest-neighbour magnified inset (a 9 px drone is otherwise a speck) and a caption strip carrying the frame key, the frame's TP/FP/missed tally, the matching rule and whatever the run recorded per frame (`branch`, for GLAD). **Verdicts come from `match_frame`**, the same function `records` and `curves` call, so the picture cannot disagree with the ledger; nothing is re-inferred, the boxes come from the persisted `detections.jsonl`. Rendered `phantom19` from EXP-004 to `runs/exp004_glad/examples/phantom19_overlay.mp4` (2,158 frames, 251 MB, not committed — `runs/` is gitignored). New `src/output/overlay.py` and `src/output/video.py` (`LazyVideoWriter`, now shared with `annotate.VideoSink`); `load_frames` gained a `key_filter` so one video does not pay 28,337 label reads. 32 new tests; reference in [render_video.md](render_video.md).

- [x] 2026-08-20 — [M5] [algo] **`src.evaluate` now reports FAR and a size-normalised localisation error per size bin.** `far` (FP/frame), `loc_err` + `loc_err_p90` (centre offset in multiples of the target's own size) and `loc_by_size` on `Metrics`, `far` on `ConditionScore`, `loc_error_by_size` in `src.eval.curves` (new `ErrorCurve`, written into the `plot_eval` sidecar). All appended after `criterion`, so the `--json-out` key order the ledger cites is unchanged. **No separate `Pd` field: Pd *is* recall** — ARD-MAV carries ~1 target per frame so the per-target and per-frame readings coincide, and a duplicate field would be two names for one number that could drift; the report labels it `recall (Pd)`. **Applied to EXP-004 (M4a) only** — see [M5-1] for the three baselines. Both criteria re-scored from the persisted JSONL with every headline number reproducing exactly. **The finding:** relative localisation error degrades 5x monotonically as targets shrink (0.051 target-widths at 32-48 px to 0.255 below 8 px), which is the measured mechanism behind the IoU/centre gap this project has been asserting from P/R since EXP-004 — at 0.255 no pair can clear IoU 0.50. FAR 0.0066/frame centre vs 0.1291 IoU, and **69% of the centre-matched false alarms are one video** (phantom63, 129 of 188), so the aggregate rate is not a property of the detector. 11 new tests; full write-up in docs/experiments.md.
- [x] 2026-08-20 — [meta] **Make the definition of done enforceable instead of repeatable.** The instruction "commit, push, test, document, tick the todo" was typed again on 2026-08-20 despite already being three separate bullets in `CLAUDE.md` — and tests were never in there at all. Now one `## Definition of done` section in `CLAUDE.md` (four numbered gates), plus `.claude/hooks/definition_of_done.py` wired in a new `.claude/settings.json`: `PostToolUse` on `Write|Edit` records which repo file the session touched (async, so it adds no latency to an edit), and `Stop` blocks the turn while any of those is uncommitted or unpushed. **Tracks the session's own edits, not `git status`** — parallel sessions each see the others' work in progress as a dirty tree, and a hook that cries wolf gets switched off. Blocks only on what is machine-checkable; tests/docs/todo ride in the message. Guarded against looping via `stop_hook_active`, and silent on gitignored paths, paths outside the repo, malformed stdin and a missing upstream. Verified by piping synthetic payloads through both modes. **Two things to know:** `.claude/settings.json` did not exist when the current sessions started, so the hook only takes effect in sessions opened *after* this commit; and the file carries hooks only — `/fewer-permission-prompts` still has the permission allowlist to write.
- [x] 2026-08-16 — [M4a] [algo] Run GLAD's released weights (MIT, no training) over the ARD-MAV 15 test videos and compare against their published P/R/F1 — 0.99/0.96/0.97 ordinary, 0.94/0.86/0.90 complex, 0.82/0.67/0.73 small MAV. **This is a test of our pipeline, not of GLAD.** GLAD's weights were trained on ARD-MAV's other 45 videos and its architecture was tuned against this very split, so the number is optimistic by construction and must never be reported as "GLAD scores X for us". Reproducing their per-category rows is the only independent confirmation we have that M1's harness and M2's conversion are correct; a large gap means we have a bug. **Needs contiguous frames** — GLAD's motion modules difference consecutive frames, so run it on the original `.mp4`s, not M3's stride-10 stills. **Two corrections from the 2026-08-16 model survey ([glad-model.md](glad-model.md)):** (a) the reproduction target is the *ablation* row **P 0.91 / R 0.81 / F1 0.86**, not full GLAD — the released code has no Kalman filter and no adaptive search region (fixed 320×320), so the per-condition figures above are an upper bound; (b) it **runs on this laptop** — `GLAD.py` wants TensorRT engines, but `yolov5s_GLAD.pt` / `-crop.pt` ship alongside and load on CPU via `third_party/yolov5` (v6.0, `nc=1`, `names=['Drone']`, verified). Needs a shim replacing `detector{1,2,3}_trt` with yolov5 CPU inference. Score on **P/R/F1 only** — GLAD emits no confidences, so any AP for it is not comparable. **Done 2026-08-16 — EXP-004.** CPU port at `src/algo/glad/` driven by `src.glad_detect`; all 28,337 frames in 2.9 h at 2.67 fps. **The harness is validated:** `ordinary` came back 0.987/0.965 against the published 0.99/0.96, and the aggregate 0.856/0.771/0.811 sits within 0.054 P and 0.039 R of the ablation target. The residual is concentrated entirely in `small_mav` and is a **scoring threshold, not a detector** — re-scored at IoU 0.40 our numbers bracket the published ones in all three categories. Three upstream defects found; the letterbox padding one is fixed behind `--pad` and measured to be worth nothing. Full write-up in docs/experiments.md.
- [x] 2026-08-16 — [M2a] [data] Wired `conditions.json` into `src.evaluate` via `--conditions`: per-scene-category precision/recall/F1/AP, which is the only form comparable to GLAD's published per-category figures. Re-scored EXP-001/002/003 for free from the persisted JSONL. Immediately paid off — it exposed that precision on *ordinary* backgrounds is 5x that on *complex* (the domain-shift signature), and that `small_mav` is a total wipeout (2 correct out of 935) rather than merely weak, both of which the aggregate hid. Empty categories report NaN, not 0.0.
- [x] 2026-08-16 — [M3] [algo] Baseline established: EXP-001/002/003, one set of off-the-shelf weights at three effective resolutions (6.7 / 13.3 / 20 px on target). AP@0.5 0.0058 / 0.0219 / 0.0251. Recall rises 10x with resolution but precision collapses to 3% (11,353 false positives when tiled), and 97% of tiny targets are missed even with no downscaling at all. Resolution is the binding constraint but is not sufficient — the ceiling is the model. False positives land on window recesses and AC units, and predicted boxes run 1.7x too large. Ran three configs rather than the planned two so the medium bucket could act as a control. Full write-up in docs/experiments.md.
- [x] 2026-08-13 — [M2] [data] Prepared the ARD-MAV test split: 28,337 frames / 28,160 boxes from the 15 official test videos, VOC XML → YOLO, with `data.yaml`, `conditions.json` and `MANIFEST.md`. Validation battery clean — no size mismatches, no out-of-range coordinates, no degenerate boxes, only `Drone` as a class name. Labels visually verified across three videos and two scene categories. **Two things I had wrong going in:** `CAP_PROP_FRAME_COUNT` overstates by 307 (header metadata, not decodable frames — every decodable frame *is* annotated), and a missing XML means *unannotated*, not *negative*; the 177 genuine negatives are frames whose XML contains zero objects, clustered in phantom47/08/09. Size distribution: 64.8% tiny, 28.2% small, 7.0% medium, **0% large**.
- [x] 2026-08-13 — [M1] [test] Verify the evaluation math. 50 tests over `src/eval/` covering IoU, greedy confidence-ordered matching, one-claim-per-target, class-awareness, AP endpoints and the CLI contract. All 8 deliberate mutations (threshold `>=`→`>`, ascending sort, repeated claims, dropped class check, IoU union, AP envelope, YOLO half-extent, corner/centre swap) were caught.

## Superseded

Not to be worked. Kept because each records why a plausible option was rejected —
deleting them invites re-proposing the same dataset next month.

- [~] 2026-08-18 — [M4b-1-alt] [algo] **Download NPS-Drones and score GLAD on its published 10-video test split.** *Superseded 2026-08-18: chosen against the wrong criterion.* The reasoning was that NPS is the only freely downloadable set carrying a published GLAD number in metrics we can compute (Table V: **P 0.92 / R 0.95 / F1 0.93**); Drone-vs-Bird was ruled out because GLAD published **AP only** there (0.701) and our port emits no confidence scores. But the anchor for M4b is EXP-004, our own run, so protocol identity is what matters and an external published number is not. **The fatal defect:** NPS targets are **delta-wing fixed-wing UAVs, not multirotors** — the source paper attributes its detection signature to "the delta wing shape of the UAVs", and the max target box is 65×21, a 3:1 planform. Useful as generic small-object-against-sky pretraining only. Also, GLAD trained on 40 of the 50 videos, so it says nothing about held-out performance. Detail retained if it is ever revived: direct HTTP, BSD-3, no registration (`curl -O https://engineering.purdue.edu/~bouman/UAV_Dataset/{Videos.zip,Video_Annotation-v2.zip}`), ≈1.5 h CPU for the 10 test videos (~14k frames at 2.67 fps); annotation **v2** only, the published last-10 split, P/R/F1 never AP, and NPS ships **two resolutions** (1920×1080 and 1280×960) against 1080p-tuned constants, so restrict to 1080p or rescale by frame diagonal and report which.
- [~] 2026-08-18 — [M4b-0] [algo] **Run GLAD over 15 of its own 45 training videos and compare against EXP-004's test numbers.** *Declined 2026-08-18.* The idea: all 60 raw ARD-MAV videos are on disk, so the train/test gap is measurable tonight for ~2.9 h CPU with zero new code, and it bounds how optimistic ARD100-extra will be — if train ≈ test (~0.86 F1) GLAD is not overfit to this campaign; if train ≫ test, the optimism is quantified in advance. Bucketing was to use `src.data.scene_stats`, which derives `lighting` and `relative_range` from imagery and labels directly. Declined in favour of spending the effort on acquiring genuinely unseen video instead.
- [~] 2026-08-17 — [M4b] [data] **MOT-FLY as a third test set.** *Withdrawn 2026-08-17: unobtainable.* Its Google Drive link returns "file does not exist", and repo issue #1 (link expired) has sat unanswered since 2024-12-27 with no mirror anywhere. Only the untested Baidu link (`pe53`) and an email to `3120210041@bit.edu.cn` remain, so it cannot be planned around.
- [~] 2026-08-16 — [M4b] [data] **Det-Fly and Anti-UAV300 as test sets.** *Rejected.* Det-Fly is sparse stills with no stated frame ordering, so GLAD's motion branch cannot run and it would benchmark the appearance path alone — breaks criteria 1 and 5 above. Anti-UAV300 is shot from a static ground camera, which makes motion compensation trivial and would score GLAD artificially high — breaks 2 and 4.
