"""EXP-017 stage 2, step 1: a THRESHOLD candidate budget instead of a top-N one.

Why this is not a detail
-----------------------
Measured 2026-09-28 on `first_catch`, top-200 `win_b5_e4` peaks over 15 frames:

    warp border + 24 px margin + HUD              40.2% of peaks in the real scene
    ... + EXP-015's screen_fixed (4.36% of frame) 41.2%

Masking 4.36% of the frame bought **one point**, and all 200 requested peaks were still
available in every frame. That is not a bad mask. It is a rank-based budget: `top=200`
always returns 200, so removing clutter promotes the next clutter peak instead of
reducing load. Under top-N, no masking work can ever show a benefit.

With a threshold the count is free to fall. A frame whose clutter has been masked away
yields fewer candidates, which is the behaviour every later stage was assuming it had.

Calibration
-----------
The threshold is fitted on **empty frames only** and two-fold, the same discipline
EXP-016 used for z*: tune on one contiguous half, count on the other, both ways round,
and freeze the more conservative of the two before any drone frame is scored. The target
is a mean candidates-per-empty-frame, so the operating point is comparable to EXP-015's
"10 candidates per empty frame" convention and to EXP-016's seed budget of 200.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import common          # EXP-015's peaks(), so a candidate means what it meant there


@dataclass(frozen=True)
class Budget:
    """A frozen operating point: one threshold, and what it did on the calibration set."""
    tau: float
    target_per_frame: float
    achieved_per_frame: float
    fold_taus: tuple[float, float]
    n_empty_frames: int

    def describe(self) -> str:
        return (f"tau={self.tau:.4g} for {self.target_per_frame:g} cands/empty frame "
                f"(achieved {self.achieved_per_frame:.1f}; folds "
                f"{self.fold_taus[0]:.4g} / {self.fold_taus[1]:.4g}; "
                f"{self.n_empty_frames} empty frames)")


def candidates(score_map: np.ndarray, valid: np.ndarray, tau: float,
               hard_cap: int = 2000) -> np.ndarray:
    """Every local maximum at or above `tau`, strongest first. (N, 3) of (value, x, y).

    `hard_cap` is a memory guard, not a budget: it is far above any sane operating point,
    and a run that hits it is reporting that its threshold is wrong rather than quietly
    behaving like top-N again.
    """
    pk = np.asarray(common.peaks(score_map, valid, top=hard_cap), float)
    if not len(pk):
        return pk.reshape(0, 3)
    return pk[pk[:, 0] >= tau]


def count_at(score_maps, valids, tau: float) -> float:
    """Mean candidates per frame at `tau`."""
    n = [len(candidates(m, v, tau)) for m, v in zip(score_maps, valids)]
    return float(np.mean(n)) if n else 0.0


def _tau_for_target(score_maps, valids, target: float,
                    lo: float, hi: float, iters: int = 32) -> float:
    """Bisect for the threshold giving `target` candidates per frame.

    Monotone by construction -- raising tau can only drop candidates -- so bisection is
    exact to the tolerance, and no sweep grid has to be guessed in advance.
    """
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if count_at(score_maps, valids, mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def calibrate(score_maps, valids, target_per_frame: float = 200.0) -> Budget:
    """Two-fold threshold calibration on empty frames. Takes the more conservative fold.

    `score_maps` and `valids` must come from frames with **no target in them**. A
    threshold fitted on frames containing the drone would be tuned partly by the thing it
    is meant to find.
    """
    n = len(score_maps)
    if n < 4:
        raise ValueError(f"need at least 4 empty frames to fold, got {n}")
    mid = n // 2
    folds = ((slice(0, mid), slice(mid, n)), (slice(mid, n), slice(0, mid)))

    hi = max(float(np.max(m)) for m in score_maps) + 1.0
    taus = []
    for fit, _ in folds:
        taus.append(_tau_for_target(score_maps[fit], valids[fit], target_per_frame, 0.0, hi))

    tau = max(taus)      # the higher threshold is the fewer-candidates one
    achieved = count_at(score_maps, valids, tau)
    return Budget(tau, target_per_frame, achieved, (taus[0], taus[1]), n)
