"""EXP-029: the moving factor joined to the epipolar direction test.

Two statistics about the same displacement, and they answer different questions.

  * **The moving factor** (`src/algo/kinematics.py`) asks *how far* a track has travelled
    against the static scene over a short window. Measured on `catch_2` (EXP-024,
    2026-10-04): the target's ego-compensated travel over 5 frames is a median 28.4 px
    with a p10 of 13.9, against the background field's median 3.18 px and p90 5.60. The
    tails do not overlap, so a threshold between them exists.
  * **The direction test** (`criteria.epipolar_direction`, EXP-017) asks *which way* it
    points. For any static point the leftover displacement after plane compensation is
    `mu = gamma * (e - x)`, so it is collinear with the line to the epipole whatever its
    depth. Parallax therefore cannot explain motion across that line.

Why both, and why in this order
-------------------------------
The direction test is the sharper idea and the weaker measurement. EXP-021 measured it
rejecting at chance (30.7% agreement against a 22.8% floor) while costing 33 points of
recall, and `criteria.epipolar_direction` names the reason itself: an intercept target sits
near the focus of expansion, where every direction is nearly radial. It is degenerate
precisely on this project's geometry.

So magnitude leads and direction may only *veto*. A test that rejects at chance must never
be able to reject by default, which is what `require` below would let it do -- that mode
exists to measure the cost, not to be used. `MODES` is ordered by how much authority the
direction test is given:

  * `off`     -- magnitude alone. The control.
  * `veto`    -- moving, unless the direction test is usable *and* says the motion is
                 along the epipolar line. The default: direction can only take away, and
                 only where it has an answer.
  * `require` -- moving only if the direction test is usable *and* says off-epipolar.
                 Every unjudged track is rejected, which is the EXP-021 failure mode.

`unjudged` is kept distinct from `rejected` throughout, because a track near the epipole
and a track measured as parallax are different claims and collapsing them is how the
direction test's cost got hidden in EXP-017's headline numbers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import criteria

MODES = ("off", "veto", "require")

STILL = "still"          # below the moving factor: has not travelled far enough
PARALLAX = "parallax"    # travelled, but along the line to the epipole
UNJUDGED = "unjudged"    # travelled, and the direction test has no answer here
MOVING = "moving"        # travelled, and parallax cannot explain the direction

KEPT = (MOVING, UNJUDGED)   # what `veto` and `off` show; `require` shows MOVING only


@dataclass(frozen=True)
class MoveVerdict:
    """One track's verdict, with both measurements kept so the report can cut either way."""

    outcome: str             # STILL, PARALLAX, UNJUDGED or MOVING
    moved: float             # ego-compensated travel over the window, px
    span: int                # frames that travel was measured across
    sin_angle: float         # |sin| off the epipolar line, nan when not usable
    reason: str              # the direction test's own words, for the report

    @property
    def kept(self) -> bool:
        return self.outcome in KEPT


def judge(moved: float, span: int, move_vec, position, epipole, *,
          min_move: float, mode: str = "veto",
          epipole_reliable: bool = True, degenerate: bool = False,
          sin_threshold: float = 0.35, min_span: int = 1) -> MoveVerdict:
    """Judge one track's window of travel.

    `moved`/`move_vec` come from `KinematicTracker`'s `GateResult`; they are already
    ego-compensated, since the tracker carries every anchor through each frame's
    homography. `position` is where the track is now and `epipole` the window's focus of
    expansion, both in current-frame coordinates.

    A track whose history does not yet span `min_span` frames has no measurement, so it is
    `UNJUDGED` rather than `STILL`: refusing it would drop every track at birth, which is
    the mistake EXP-025b's hard persistence gate made.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if not np.isfinite(moved) or span < min_span:
        return MoveVerdict(UNJUDGED, float(moved), int(span), float("nan"),
                           "no span yet")
    if moved < min_move:
        return MoveVerdict(STILL, float(moved), int(span), float("nan"),
                           f"travelled {moved:.1f} px < {min_move:.1f}")
    if mode == "off":
        return MoveVerdict(MOVING, float(moved), int(span), float("nan"),
                           "magnitude only")

    v = criteria.epipolar_direction(
        np.asarray(move_vec, float).reshape(2), np.asarray(position, float).reshape(2),
        np.asarray(epipole, float).reshape(2),
        epipole_reliable=epipole_reliable, degenerate=degenerate,
        sin_threshold=sin_threshold,
        # The floor is already carried by `min_move`, which is far above the 2 px EXP-017
        # used; letting it apply twice would re-reject what the moving factor just passed.
        min_displacement=0.0)

    if not v.usable:
        # `require` has no answer here and refuses; `veto` may only take away, so it keeps.
        return MoveVerdict(UNJUDGED if mode == "veto" else PARALLAX,
                           float(moved), int(span), float("nan"), v.reason)
    if v.moving:
        return MoveVerdict(MOVING, float(moved), int(span), float(v.sin_angle), v.reason)
    return MoveVerdict(PARALLAX, float(moved), int(span), float(v.sin_angle), v.reason)


def _selfcheck() -> int:
    """Synthetic checks for the combination, run with the EXP-017 stack on PYTHONPATH.

    `py -3.13 -m moving`. What is checked is what would be wrong silently: that each mode
    gives the direction test exactly as much authority as it is supposed to have, and that
    a track with no span is never charged as still.
    """
    import selftest as sr

    print("moving.py self-check (synthetic, no video)")
    print()
    ok_all = True

    e = np.array([0.0, 0.0])        # epipole at the origin
    pos = np.array([100.0, 0.0])    # track out along the x axis, so radial = x
    radial = np.array([20.0, 0.0])          # straight away from the epipole: parallax
    across = np.array([0.0, 20.0])          # across the line: parallax cannot explain it

    ok_all &= sr.check(
        "below the factor is still, in every mode",
        all(judge(3.0, 5, across, pos, e, min_move=8.0, mode=m).outcome == STILL
            for m in MODES))

    ok_all &= sr.check(
        "no span is unjudged, not still",
        judge(float("nan"), 0, across, pos, e, min_move=8.0).outcome == UNJUDGED
        and judge(0.0, 0, across, pos, e, min_move=8.0, min_span=3).outcome == UNJUDGED)

    ok_all &= sr.check(
        "mode off ignores the direction entirely",
        judge(20.0, 5, radial, pos, e, min_move=8.0, mode="off").outcome == MOVING
        and judge(20.0, 5, across, pos, e, min_move=8.0, mode="off").outcome == MOVING)

    ok_all &= sr.check(
        "veto rejects radial travel as parallax",
        judge(20.0, 5, radial, pos, e, min_move=8.0, mode="veto").outcome == PARALLAX)

    ok_all &= sr.check(
        "veto keeps travel across the epipolar line",
        judge(20.0, 5, across, pos, e, min_move=8.0, mode="veto").outcome == MOVING)

    ok_all &= sr.check(
        "veto keeps what the direction test cannot judge",
        judge(20.0, 5, across, pos, e, min_move=8.0, mode="veto",
              epipole_reliable=False).kept
        and judge(20.0, 5, across, pos, e, min_move=8.0, mode="veto",
                  degenerate=True).kept)

    ok_all &= sr.check(
        "require drops what it cannot judge -- EXP-021's failure mode",
        not judge(20.0, 5, across, pos, e, min_move=8.0, mode="require",
                  epipole_reliable=False).kept)

    ok_all &= sr.check(
        "the measurement survives the verdict",
        judge(20.0, 5, across, pos, e, min_move=8.0).moved == 20.0
        and judge(20.0, 5, across, pos, e, min_move=8.0).span == 5)

    refused = False
    try:
        judge(20.0, 5, across, pos, e, min_move=8.0, mode="sometimes")
    except ValueError:
        refused = True
    ok_all &= sr.check("an unknown mode is refused rather than guessed", refused)

    print()
    print("ALL PASS" if ok_all else "FAILURES ABOVE")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(_selfcheck())
