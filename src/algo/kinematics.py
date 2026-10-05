"""Overruling candidates that move in a way no drone can.

A per-frame detector ranks every frame on its own, so its answer can land anywhere: on
the drone in one frame, on a cloud edge 400 px away in the next. A real airframe cannot
do that. It has a top speed, so between two frames its picture position can move only so
far. This module holds every candidate to that.

**Speed in the picture, not in metres.** One camera sees px/frame. Turning m/s into
px/frame needs the range, and the range is not observable: EXP-016 estimated it from
apparent size and the cap was inert, because 53% of the seeds sat at the 120 px size
clamp. Today's blobs are also 0.05-0.16x the airframe's true size, so a size-based range
would be wrong by 6-20x. `SpeedLimit` therefore converts the limit at an *assumed minimum
range* -- a target closer than that may move faster than allowed -- and caps the result at
an absolute px/frame ceiling, so the limit always bites.

**Hysteresis, not persistence.** A hard m-of-k gate costs recall on this detector:
EXP-025b's 4 of 5 took the drone in the top 3 from 61 to 27 of 224 frames, because its
strong firings come in short runs. Here a *strong* candidate births a track, which is
shown once it has `confirm` hits. A confirmed track may then be continued by any
candidate inside its speed gate, weak or strong, and it coasts through misses with a reach
that grows with the gap. A strong candidate far from every track starts a new track and is
overruled until that track earns its history. A weak candidate never starts a track.

**Raw picture coordinates, with the camera as an excuse.** On intercept footage the
camera follows the target, so the drone is steadier in the picture than in the scene:
EXP-025b kept it in 27 frames chaining raw coordinates against 3 camera-compensated. A
whip-turn of the host still moves everything at once, so a step passes when *either* its
raw displacement or its camera-compensated displacement is within reach.

**Evidence, for a ranking that does not flicker.** Two confirmed tracks may both move
plausibly, and ranking them by each frame's own score swaps #1 between them whenever their
scores cross. Each track therefore carries `evidence`, a decayed sum of its candidates'
scores (`decay` per frame, so a track missed for a frame loses some of it). Ranking by
evidence keeps #1 on the object with the stronger recent history.

What this does not do: it stops answers *jumping*. A piece of clutter that stays put moves
plausibly by definition, and this gate passes it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Reasons a candidate carries out of `KinematicTracker.step`.
OK = "ok"                    # on a confirmed track: shown
NEW = "new"                  # strong, no track within reach: started one, overruled
UNCONFIRMED = "unconfirmed"  # continued a track that has not yet earned `confirm` hits
WEAK_ORPHAN = "weak-orphan"  # weak, no confirmed track within reach: never starts one


@dataclass(frozen=True)
class SpeedLimit:
    """How far a drone may move in the picture per frame.

    The defaults are the analog intercept rig. They use a 40 m/s closing speed (the
    `docs/edge-budget.md` assumption) and the R1 Mini CCD's 130 degree lens. The ceiling
    is 25 px at 960 px wide, about 2x the labelled drone's p90 step of 12.67 px on
    `catch_2`. The slack absorbs blob-centre jitter, which is present even when the target
    is still.
    """

    width_px: int
    fps: float
    v_max_ms: float = 40.0
    min_range_m: float = 10.0
    hfov_deg: float = 130.0
    ceiling_px: float = 25.0
    slack_px: float = 4.0

    @property
    def focal_px(self) -> float:
        """Pinhole focal length in pixels for this width and horizontal field of view."""
        return (self.width_px / 2) / math.tan(math.radians(self.hfov_deg) / 2)

    @property
    def physical_px(self) -> float:
        """The px/frame a target at `min_range_m` crossing at `v_max_ms` would move."""
        return self.v_max_ms * self.focal_px / (self.min_range_m * self.fps)

    @property
    def px_per_frame(self) -> float:
        """The enforced limit: the physical one, never above the ceiling."""
        return min(self.physical_px, self.ceiling_px)

    def reach(self, gap: int) -> float:
        """How far a track last seen `gap` frames ago may be from its last position."""
        return self.slack_px + self.px_per_frame * gap


@dataclass(eq=False)
class _Track:
    """One object's history: where it was seen, and where the camera has carried that."""

    tid: int
    pos: np.ndarray    # last seen position, picture coordinates
    scene: np.ndarray  # the same point carried through every camera motion since
    last: int          # frame it was last seen in
    hits: int = 1
    confirmed: bool = False
    evidence: float = 0.0


@dataclass(frozen=True)
class GateResult:
    """One frame's verdict, aligned with the candidates `step` was given.

    `track_id` is -1 for a candidate that joined no track. `shown` is true exactly for
    candidates on a confirmed track, and `reason` says why each one was or was not shown.
    `evidence` is the track's decayed score sum after this frame, 0 off a track. The counts describe the tracks: how many were `born` this frame, how many weak
    candidates continued a confirmed track (`continued_weak`), how many confirmed tracks
    are alive but unseen (`coasting`), and how many `died`.
    """

    track_id: np.ndarray
    shown: np.ndarray
    reason: np.ndarray
    evidence: np.ndarray
    born: int = 0
    continued_weak: int = 0
    coasting: int = 0
    died: int = 0


@dataclass
class KinematicTracker:
    """Tracks candidates frame to frame and shows only those that move like a drone.

    Call `step` once per frame, in increasing frame order. A frame index that skips
    (striding, a dropped duplicate) is a longer gap, and the reach grows to match.
    """

    limit: SpeedLimit
    confirm: int = 2
    max_coast: int = 5
    decay: float = 0.8
    _tracks: list[_Track] = field(default_factory=list, init=False, repr=False)
    _next_id: int = field(default=0, init=False, repr=False)
    _frame: int | None = field(default=None, init=False, repr=False)

    def reset(self) -> None:
        """Forget every track, for a cut in the footage."""
        self._tracks, self._next_id, self._frame = [], 0, None

    def step(self, frame: int, xy: np.ndarray, score: np.ndarray, strong: np.ndarray,
             prev_to_cur: np.ndarray | None = None) -> GateResult:
        """Judge one frame's candidates against the tracks so far.

        `xy` is (N, 2) candidate centres, `score` (N,) feeds each track's evidence and breaks
        ties between equally near candidates, and `strong` (N,) marks those allowed to start a track.
        `prev_to_cur` is the 3x3 homography carrying the previous frame's points into this
        one. Pass None when the camera motion is unknown and only raw displacement is used.
        """
        if self._frame is not None and frame <= self._frame:
            raise ValueError(f"frame {frame} is not after frame {self._frame}")
        self._frame = frame
        xy = np.asarray(xy, float).reshape(-1, 2)
        score = np.asarray(score, float).reshape(-1)
        strong = np.asarray(strong, bool).reshape(-1)
        n = len(xy)
        track_id = np.full(n, -1)
        reason = np.full(n, WEAK_ORPHAN, dtype=object)
        evidence = np.zeros(n)
        if prev_to_cur is not None:
            for t in self._tracks:
                t.scene = _apply(prev_to_cur, t.scene)

        free = np.ones(n, bool)
        confirmed = [t for t in self._tracks if t.confirmed]
        tentative = [t for t in self._tracks if not t.confirmed]
        hit = self._match(confirmed, xy, score, free, frame) \
            | self._match(tentative, xy, score, free & strong, frame)

        continued_weak = 0
        for t, j in hit.items():
            t.pos = t.scene = xy[j].copy()
            t.evidence = t.evidence * self.decay ** (frame - t.last) + score[j]
            evidence[j] = t.evidence
            t.last, t.hits = frame, t.hits + 1
            t.confirmed = t.confirmed or t.hits >= self.confirm
            track_id[j], free[j] = t.tid, False
            reason[j] = OK if t.confirmed else UNCONFIRMED
            continued_weak += t.confirmed and not strong[j]

        before = len(self._tracks)
        self._tracks = [t for t in self._tracks if t in hit or (
            t.confirmed and frame - t.last <= self.max_coast)]
        died = before - len(self._tracks)
        coasting = sum(t.confirmed and t not in hit for t in self._tracks)

        born = 0
        for j in np.nonzero(free & strong)[0]:
            t = _Track(self._next_id, xy[j].copy(), xy[j].copy(), frame,
                       confirmed=self.confirm <= 1, evidence=float(score[j]))
            evidence[j] = t.evidence
            self._next_id += 1
            self._tracks.append(t)
            track_id[j], reason[j], born = t.tid, OK if t.confirmed else NEW, born + 1

        return GateResult(track_id=track_id, shown=reason == OK, reason=reason,
                          evidence=evidence, born=born,
                          continued_weak=int(continued_weak), coasting=int(coasting),
                          died=died)

    def _match(self, tracks: list[_Track], xy: np.ndarray, score: np.ndarray,
               eligible: np.ndarray, frame: int) -> dict[_Track, int]:
        """Each track's nearest admissible candidate, nearest pairs claimed first.

        A pair is admissible when the candidate is within the track's reach of where it
        was last seen or of where the camera has since carried that point. Ties go to the
        higher score. One candidate serves at most one track.
        """
        idx = np.nonzero(eligible)[0]
        pairs = []
        for t in tracks:
            d = np.minimum(np.hypot(*(xy[idx] - t.pos).T), np.hypot(*(xy[idx] - t.scene).T))
            near = d <= self.limit.reach(frame - t.last)
            pairs += [(float(dj), -float(score[j]), t.tid, t, int(j))
                      for dj, j in zip(d[near], idx[near])]
        out: dict[_Track, int] = {}
        taken: set[int] = set()
        for _, _, _, t, j in sorted(pairs, key=lambda p: p[:3]):
            if t not in out and j not in taken:
                out[t] = j
                taken.add(j)
        return out


def _apply(hmat: np.ndarray, p: np.ndarray) -> np.ndarray:
    """One point through a 3x3 homography."""
    v = np.asarray(hmat, float) @ np.array([p[0], p[1], 1.0])
    return v[:2] / v[2]
