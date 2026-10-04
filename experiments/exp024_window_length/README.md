# EXP-024 — window length: is a 10- or 15-frame persistence gate better than 5?

Forked from `exp017_motion_first/` so that experiment stays frozen at the numbers
EXP-021 and EXP-022 cite. Only `overlay_window.py` is copied here; everything it imports
(`window.py`, `geometry.py`, `criteria.py`, `masks.py`, `skyline.py`, `budget.py`) is
still EXP-017's and is reached through the `PYTHONPATH` stack.

## The two differences from EXP-017's renderer

**`--direction`, default OFF.** EXP-017's `overlay_window` *always* applied the epipolar
direction test, so its videos painted red rejections from a test EXP-021 measured as
rejecting at chance, and its headline "kept" figure was the direction-gated one rather
than the persistence-alone one the standing recommendation refers to. The flag matches
`overlay_stage2.py`'s spelling and default. With the test off the epipole/lift block is
skipped rather than printed as a 0.00 lift.

**Every operating point from one pass.** `Track.appearances` does not depend on
`min_appear` — `Track.verdict` only compares against it — so one render measures the whole
load-versus-recall curve for that window length:

```
  OPERATING POINTS (persistence alone, every threshold from this one pass)
    need  load/frame     drone kept  seeds held
     4/5        25.8     79 ( 42%)       19.4%
```

Sweeping `--min-appear` by re-rendering costs k times as much for the same table, and
invites reading thresholds off different spans. Only the **video** is drawn at the single
`--min-appear` passed; the table covers the rest.

**`--dump`, a per-seed CSV.** One row per seed per rendered frame: `frame, x, y, c,
appearances, steps, verdict, on_target`. `c` is the seed's contrast, carried out of
`budget.candidates`' column 0 — `build_tracks` returns one track per seed in seed order, so
the two line up by index. `on_target` is `overlay_video.on_drone`, the same grown-box
criterion every other experiment matches with. Like the table above it is
threshold-independent, so one pass answers *how many false alarms* and *what the drone
scored against them* at every `--min-appear`, not just the one rendered. Counting rows with
`appearances >= m` reproduces the OPERATING POINTS row for `m/k` exactly; that is the check
that the dump describes the same pass the report does.

## Running

Clip config first on the path, then this folder, then what it builds on.

```bash
# O4 first_catch -- outputs to runs/sofa_o4/exp024_window_length/
PYTHONPATH="experiments/exp024_window_length/o4_first_catch;experiments/exp024_window_length;experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
    py -3.13 -m overlay_window --start 650 --end 964 --appear-radius 14 --k 10 --min-appear 5

# SOFA-ANALOG catch_2 -- outputs to runs/sofa_analog/exp024_window_length/
PYTHONPATH="experiments/exp024_window_length/analog_catch_2;experiments/exp024_window_length;experiments/exp017_motion_first;experiments/exp015_normalised_motion;." \
    py -3.13 -m overlay_window --start 441 --end 800 --appear-radius 9 --k 10 --min-appear 5
```

`--appear-radius` is **14 px on O4 and 9 px on analog** — the same physical radius read on
a 1440 px and a 960 px picture. Everything else is left at EXP-017's defaults.

**The video is drawn in the clean look by default** (2026-10-01, at the user's request),
the same as EXP-023's `clean_catch_2_441_800.mp4`: stage-0 mask tints, each kept track as a
red circle 10 px across, and a two-line caption. `--diagnostic` brings back EXP-017's full
drawing (stage-1 tint and horizon, epipole, grey flash dots, green survivors with arrows,
the label box). Drawing only — the report and every number above are identical either
way. The analog `window10_` / `window15_catch_2_441_800.mp4` were re-rendered clean; the O4
EXP-024 videos predate this and were deleted with the O4 run folders.

## O4 result (first_catch 650-964, 315 frames, 190 with the drone as a candidate)

k=5 is at least as good at every load it can reach, and strictly better below ~40
candidates/frame. Drone recall out of those 190 frames:

| load/frame | k=5 | k=10 | k=15 |
| ---: | :--- | :--- | :--- |
| ~60-67 | 2/5 → **71%** | 2/10 → 72% | 2/15 → 72% |
| ~39-40 | 3/5 → **55%** | 3/10 → 56% (at 47.6) | 4/15 → 45% |
| ~26-29 | 4/5 → **42%** | 5/10 → 33% | 6/15 → 29% |
| ~14 | 5/5 → **25%** | 8/10 → 19% | 10/15 → 17% |
| ~9.8 | *unreachable* | 9/10 → **17%** | 11/15 → 13% |

The one thing a longer window offers is **reach**: 5/5 floors k=5 at 14.3 candidates/frame
and nothing shorter is available, so a downstream stage needing less than that has to use
k=10 at 9/10 — 17% recall at 9.8/frame. A granularity argument, not a motion one.

**`usable residual steps per seed: median 3` at k=5, k=10 and k=15 alike** (47% reach 4+ at
k=5, 37% reach 9+ at k=10, 32% reach 14+ at k=15). Lengthening the window does not lengthen
the LK tracks, so the accumulated residual a long window exists to compute is mostly
unavailable. Any motion-magnitude test has to chain candidate *peaks* by position, which
needs no texture; `window._appears` already does that for one frame without anything
chaining it.

**Span dependence.** On frames 900-964 alone — the easy end, target 70-107 px — the strict
settings look tied: 52% / 51% / 48% at load ~14. On the full span, where the target is
19 px at frame 708, the same comparison is 25% / 19% / 17%. EXP-021's headline 78% and
31.2/frame are that easy sub-span.

See `docs/experiments.md` EXP-024 for the full entry.

## Analog result (catch_2 441-800, 360 frames, 174 with the drone as a candidate)

Stage-0 coverage reproduces EXP-021's analog run exactly (hud 11.64%, prop 0.44%, ladder
3.21%, edge 7.63%, union 19.49%), which is what says this clip config is the same one.
167.6 seeds/frame.

| load/frame | k=5 | k=10 | k=15 |
| ---: | :--- | :--- | :--- |
| ~168 | 1/5 → 100% | 1/10 → 100% | 1/15 → 100% |
| ~18-23 | 2/5 → 34% | 2/10 → **37%** | 2/15 → **37%** |
| ~4-8 | 3/5 → 21% | 3/10 → 22% | 3/15 → 22% |
| ~1.1-3.6 | 4/5 → 13% | 4/10 → 17% | 4/15 → 17% |
| ~0.1-1.9 | 5/5 → 2% | 5/10 → 10% | 5/15 → 10% |

**The O4 conclusion does not transfer.** At matched load the three window lengths are
indistinguishable here (k=10 interpolated to k=5's 18.1/frame is ~33.5% against 34%), so
longer is not worse on analog — it is merely not better. What dominates is the collapse:
1/k keeps the whole target at 167.6 candidates/frame and 2/k keeps 37% at ~20. There is no
setting where persistence both filters and keeps the target.

**Short windows (2026-10-04).** At k=3 and k=4 the loose end of the curve is 2/3 → 33% at
13.8/frame and 2/4 → 34% at 16.4, against 2/5's 34% at 18.1. "Twice" is satisfied by the
same target frames at every k, so the shortest window is the cheapest version of that gate.
3/4 → 18% at 3.1 and 3/3 → 10% at 1.5. Videos for 2/3, 2/4, 2/5 and 3/5:
`window{k}_need{m}_catch_2_441_800.mp4`.

`rank_hist.py` (no arguments, reads the k=3 and k=4 dumps and EXP-023's candidate dump)
draws the drone's rank among each frame's survivors at 2/3, 2/4 and sky c>=6 into
`rank_hist_catch_2_441_800.png`. Median rank 5 for both windows, 2 for the sky branch.

### The fb gate, measured rather than assumed

`usable residual steps per seed: median 0` at every k, against median 3 on O4. The analog
gate is `px(1.0) = 0.67 px` on a 960 px frame, which EXP-017's own `--fb-max` help text
warns "rejects nearly every track" on CVBS. This is structural, not cosmetic:
`build_tracks` uses LK to decide *where* to test `_appears`, so a track that dies at step 0
has a NaN position and scores no appearance however reliably the detector fired.

| | seeds at full depth | load/frame | drone kept |
| --- | ---: | ---: | ---: |
| k=5, fb 0.67 | 13% got 4+ | 1.1 | 13% |
| k=5, fb 3.0 | 28% got 4+ | 1.9 | 16% |
| k=10, fb 0.67 | 6% got 9+ | 2.9 | 17% |
| k=10, fb 3.0 | **16%** got 9+ | 5.9 | **25%** |

The gate is throttling LK and is **not** the explanation: at matched load ~6/frame it is
25% against 22%, about three points, and median usable steps stays 0 at a 4.5x looser gate.
The analog collapse is mostly the detector not firing on the target consistently. Worth
having anyway, and `--fb-max` deserves deciding on its own terms rather than inside a
window-length question.

## 1 frame in 6 of catch_2 is a duplicate (2026-10-04)

148 of 890 frames repeat their predecessor — mean |diff| 0.32-0.94 against 8-26 for real
frames — at `f % 6 == 4` with no exception. 25 fps content in a 30 fps container. **A
duplicate pair produces zero candidates**: all 60 zero-seed frames in the k=5 dump are
exactly those frames.

Every `m/k` on the curves above is therefore stricter than it reads, because the effective
window is `k * 5/6`:

| k | live frames per window | strictest reachable |
| ---: | :--- | :--- |
| 5 | 4 live in 300 of 360, 5 live in 60 | 5/5 in **17%** of windows |
| 7 | 6 live in 300 of 360, 5 live in 60 | 7/7 in **0%** |

7-of-7 kept 0 of 60338 seeds, which is arithmetic rather than sampling. 4-of-5 is a
perfect-run requirement: 4 appearances out of 4 live chances in 83% of windows. Fix this
before reading any analog persistence number — deduplicate before windowing, or count the
threshold against live frames.

## Contrast of what survives (2026-10-04)

From `--dump`. `c` is median contrast, rank is the drone's median rank among its frame's
survivors.

| gate | kept/fr | false/fr | drone frames | drone `c` | FA p50 | FA p90 | rank | is #1 |
| ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| 4/5 | 1.09 | 1.03 | 22 (13%) | 4.44 | 2.77 | 5.05 | #1 | 50% |
| 5/7 | 0.84 | 0.79 | 18 (10%) | 4.48 | 2.79 | 4.93 | #1 | 56% |

At matched false-alarm load the two are within one frame of each other (k=7 interpolated to
1.03/frame is ~12.0% against 12.6%). The drone's contrast does not depend on the window —
the window selects seeds, it does not strengthen them — and sits at ~1.6x the false alarms'
median but just below their p90, so about one in ten outscores it. Keeping only the
strongest survivor per frame halves false alarms and halves recall, landing on 6%
everywhere; not an operating point.

## Against EXP-023's 2-frame sky branch (2026-10-04)

Recut from EXP-023's `candidates_catch_2_441_800.csv` and this experiment's `seeds_k5_` dump
— no re-run. **Both out of the same 224 labelled frames**; EXP-024's own reports divide by
174, which is the motion front-end's ceiling rather than a denominator.

| load/frame | 5-frame window | 2-frame sky branch | sky `c >=` |
| ---: | ---: | ---: | ---: |
| 167.6 | 1/5 -> 174 (77.7%) | **223 (99.6%)** | floor |
| 18.1 | 2/5 -> 60 (26.8%) | **99 (44.2%)** | 4.10 |
| 4.25 | 3/5 -> 36 (16.1%) | **63 (28.1%)** | 6.58 |
| 1.09 | 4/5 -> 22 (9.8%) | **38 (17.0%)** | 9.26 |

The sky branch wins at every matched load by ~1.7x. Its shipping point (c>=6) is 32.6% at
5.27 false alarms/frame; the window needs 3.2x that load to reach a lower 26.8%. The cause
is upstream of the gate: the motion front-end makes the target a candidate in 174 of 224
frames (77.7%) against the sky branch's 223 (99.6%).

The sky branch is also **immune to the duplicate frames** — 5.38 candidates/frame on them
against 5.63 on live frames, and it finds the target in 12 of the 60 where the motion branch
finds nothing — because `silhouette.detect` reads one frame and the pair only warps the
mask.

Different detectors, so this is a pipeline comparison at matched load, not a
5-frames-vs-2-frames ablation. The k=5/7/10/15 curves above are the ablation.

## Recommendation

**O4: keep k=5 at 4-of-5.** **Analog: the motion path loses to the 2-frame sky branch at every load — ask whether persistence helps *that* branch instead, and deduplicate before trusting any motion number** — it is the wrong
stage. Raise `--fb-max` off 0.67 px, and look upstream at why the target is not a candidate
in 186 of 360 frames. For both clips the durable next step is peak-chaining association,
which needs neither texture nor flow and would make persistence independent of LK.
