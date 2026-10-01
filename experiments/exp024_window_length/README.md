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

## Recommendation

**O4: keep k=5 at 4-of-5.** **Analog: do not tune the window at all** — it is the wrong
stage. Raise `--fb-max` off 0.67 px, and look upstream at why the target is not a candidate
in 186 of 360 frames. For both clips the durable next step is peak-chaining association,
which needs neither texture nor flow and would make persistence independent of LK.
