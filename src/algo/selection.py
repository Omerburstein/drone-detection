"""Choosing which of the confirmed tracks is the drone.

The kinematic gate (`kinematics.KinematicTracker`) decides which candidates are *allowed*.
These are the ones on a track that moves like a drone. On FIELD and on the analog clips
that still leaves several per frame, and EXP-031 ranked them by one number, the track's
decayed sum of detector contrast. One number cannot tell a drone from a ridge-top tree
that is just as contrasty. EXP-031 FIELD had such a tree hold #1 for 71 frames.

`DroneSelector` ranks them on what a person watching the video uses instead. Each cue is
a number in [0, 1], and the score is their weighted mean:

  * **evidence** -- the track's decayed contrast sum, relative to the strongest this frame.
    It is the old ranking, kept as one cue among several.
  * **appearance** -- how close the candidate's colour is to the drone's colour, learned
    from previous #1 picks. The drone does not change colour between frames; the clutter
    that steals #1 is usually a different colour (green trees, grey rock, white structure).
  * **proximity** -- how close it is to where #1 was recently. Every past #1 contributes,
    weighted by `memory_decay` per frame of age, with a spread that grows with that age,
    because the drone has had longer to fly away from an older sighting.
  * **consistency** -- how often the track was seen since it was first shown, times how
    long it has existed. A track that flickers, or was born a frame ago, has earned less.
  * **smoothness** -- how well the track's latest position was predicted from its last two
    at constant velocity. A drone has inertia. Detector noise hopping between nearby
    clutter does not.
  * **motion** -- its speed against the static scene, from the tracker's ego-compensated
    displacement. A drone flies; a tree only parallaxes. This is the soft form of the
    moving factor's hard cut, which cost catch_5 13 drone frames in EXP-031.

A cue that cannot be measured yet reads `NEUTRAL` (0.5) for every candidate, so it shifts
no ranking. Examples are a track with fewer than three sightings (smoothness) and a
selector with no #1 history yet (appearance). Proximity with no history reads 0 for all.

**Lengths are quoted at a reference width.** Spreads and speeds are given at
`scale.ref_width_px` (default 1440) and read in this picture's pixels, so one setting
serves 960-wide analog and the 4128-wide `.raw` alike.

**The drone's colour is learned slowly and only from a stable #1.** The reference colour
moves `look_rate` of the way to #1's colour on frames where #1 is the same track as the
frame before. One wrong frame at #1 therefore cannot overwrite it. While #1 is away, the
reference's influence decays at `memory_decay` per frame toward neutral.

Nothing here touches the filesystem, and the image is read only by `look`.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, fields

import cv2
import numpy as np

from .scale import PixelScale

CUES = ("evidence", "appearance", "proximity", "consistency", "smoothness", "motion")
NEUTRAL = 0.5  # a cue that cannot be measured yet: favours no candidate over another

# A look is (L, a, b) of the candidate's core and its core's L minus its surrounding ring's.
# The ring term keeps the polarity: a dark drone on bright sky and a bright one on dark
# ground share no colour, but each keeps its own sign from frame to frame.
LOOK_SIZE = 4
# Spread of each look component at which similarity falls to exp(-0.5), in OpenCV's
# 8-bit Lab units (L 0-255, a and b offset by 128).
LOOK_SPREAD = (20.0, 8.0, 8.0, 15.0)


@dataclass(frozen=True)
class CueWeights:
    """How much each cue counts. Only the ratios matter; the score is normalised."""

    evidence: float = 1.0
    appearance: float = 1.0
    proximity: float = 1.0
    consistency: float = 0.5
    smoothness: float = 0.5
    motion: float = 0.5

    def __post_init__(self) -> None:
        if any(w < 0 for w in self.as_array()):
            raise ValueError(f"cue weights cannot be negative: {self}")
        if not self.as_array().sum() > 0:
            raise ValueError("at least one cue weight must be positive")

    def as_array(self) -> np.ndarray:
        """The weights in `CUES` order."""
        return np.array([getattr(self, c) for c in CUES], float)

    @classmethod
    def parse(cls, text: str) -> CueWeights:
        """`"appearance=2,motion=0"` -> those two changed, the rest at their defaults."""
        if not text.strip():
            return cls()
        names = {f.name for f in fields(cls)}
        out = {}
        for item in text.split(","):
            key, sep, value = item.partition("=")
            key = key.strip()
            if not sep or key not in names:
                raise ValueError(f"expected <cue>=<weight> with a cue from {CUES}, got {item!r}")
            out[key] = float(value)
        return cls(**out)

    def label(self) -> str:
        """Every weight, for a run's header line."""
        return ", ".join(f"{c} {getattr(self, c):g}" for c in CUES)


@dataclass(frozen=True)
class Selection:
    """One frame's ranking, aligned with the candidates `step` was given.

    `order` lists candidate indices best first. `cues` holds each cue's value per candidate,
    in [0, 1], and `score` is their weighted mean.
    """

    score: np.ndarray
    cues: dict[str, np.ndarray]
    order: np.ndarray


@dataclass(eq=False)
class _Seen:
    """One track as the selector has seen it while it was shown."""

    first: int
    last: int
    hits: int = 0
    # The last three sightings, (frame, raw position, position carried by the camera since).
    path: deque = field(default_factory=lambda: deque(maxlen=3))


@dataclass
class DroneSelector:
    """Ranks the gate's confirmed candidates by how much each one looks like the drone.

    Call `step` once per frame, in increasing frame order, with the candidates the gate
    showed. Lengths are quoted at `scale.ref_width_px`.
    """

    scale: PixelScale
    weights: CueWeights = field(default_factory=CueWeights)
    memory_decay: float = 0.9       # per frame: weight of an old #1, and of the drone's colour
    memory_frames: int = 60         # #1 sightings and unseen tracks are forgotten after this
    proximity_sigma: float = 30.0   # spread of the proximity prior 1 frame after a #1, px
    maturity: float = 10.0          # frames: a track this old has 63% of full consistency
    smooth_sigma: float = 12.0      # constant-velocity prediction error at exp(-0.5), px
    motion_speed: float = 3.0       # px/frame against the scene at which motion is 63%
    look_rate: float = 0.2          # how far the drone's colour moves toward a stable #1's
    use_colour: bool = True         # False: compare L only (CVBS chroma is crawl, not colour)
    look_min_radius: float = 3.0    # a candidate's colour is read from a disc at least ...
    look_max_radius: float = 24.0   # ... this wide and at most this wide, px
    _seen: dict[int, _Seen] = field(default_factory=dict, init=False, repr=False)
    _memory: list = field(default_factory=list, init=False, repr=False)
    _look: np.ndarray | None = field(default=None, init=False, repr=False)
    _look_frame: int = field(default=0, init=False, repr=False)
    _best: int | None = field(default=None, init=False, repr=False)
    _frame: int | None = field(default=None, init=False, repr=False)

    def reset(self) -> None:
        """Forget every track, every past #1 and the drone's colour, for a cut in the footage."""
        self._seen, self._memory, self._look, self._best, self._frame = {}, [], None, None, None

    @property
    def reference_look(self) -> np.ndarray | None:
        """The drone's colour as learned so far, or None before the first stable #1."""
        return None if self._look is None else self._look.copy()

    def look(self, img_bgr: np.ndarray, xy: np.ndarray, diameter: np.ndarray) -> np.ndarray:
        """(N, 4) looks of N candidates in a BGR frame; a row is NaN off the picture."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        diameter = np.asarray(diameter, float).reshape(-1)
        lo, hi = self.scale.px(self.look_min_radius), self.scale.px(self.look_max_radius)
        out = np.full((len(xy), LOOK_SIZE), np.nan)
        for i, ((x, y), d) in enumerate(zip(xy, diameter)):
            out[i] = colour_signature(img_bgr, x, y, float(np.clip(d / 2, lo, hi)))
        return out

    def step(self, frame: int, track_id: np.ndarray, xy: np.ndarray, evidence: np.ndarray,
             speed: np.ndarray | None = None, look: np.ndarray | None = None,
             prev_to_cur: np.ndarray | None = None) -> Selection:
        """Rank one frame's candidates and remember the winner as this frame's #1.

        `track_id` (N,) is each candidate's kinematic track, `xy` (N, 2) its centre,
        `evidence` (N,) its track's evidence. `speed` (N,) is its displacement against the
        scene per frame, in picture pixels, NaN where unmeasured. `look` (N, 4) comes from
        `look`. `prev_to_cur` carries the previous frame's points into this one.
        """
        if self._frame is not None and frame <= self._frame:
            raise ValueError(f"frame {frame} is not after frame {self._frame}")
        self._frame = frame
        track_id = np.asarray(track_id, int).reshape(-1)
        xy = np.asarray(xy, float).reshape(-1, 2)
        evidence = np.asarray(evidence, float).reshape(-1)
        n = len(xy)
        speed = np.full(n, np.nan) if speed is None else np.asarray(speed, float).reshape(-1)
        look = (np.full((n, LOOK_SIZE), np.nan) if look is None
                else np.asarray(look, float).reshape(n, LOOK_SIZE))
        if prev_to_cur is not None:
            self._carry(prev_to_cur)
        self._forget(frame)
        for tid, p in zip(track_id, xy):
            s = self._seen.setdefault(int(tid), _Seen(first=frame, last=frame))
            s.path.append((frame, p.copy(), p.copy()))
            s.last, s.hits = frame, s.hits + 1

        cues = {
            "evidence": self._evidence(evidence),
            "appearance": self._appearance(look, frame),
            "proximity": self._proximity(xy, frame),
            "consistency": self._consistency(track_id, frame),
            "smoothness": self._smoothness(track_id),
            "motion": self._motion(speed),
        }
        w = self.weights.as_array()
        stack = np.stack([cues[c] for c in CUES]) if n else np.zeros((len(CUES), 0))
        score = (w[:, None] * stack).sum(axis=0) / w.sum()
        # Best score first; ties to the stronger evidence, then to the earlier candidate.
        order = np.lexsort((np.arange(n), -evidence, -score)) if n else np.zeros(0, int)
        if n:
            self._remember(frame, int(order[0]), track_id, xy, look)
        else:
            self._best = None
        return Selection(score=score, cues=cues, order=order)

    # --- the cues ----------------------------------------------------------------------

    @staticmethod
    def _evidence(evidence: np.ndarray) -> np.ndarray:
        top = evidence.max() if len(evidence) else 0.0
        if top <= 0:
            return np.full(len(evidence), NEUTRAL)
        return np.clip(evidence / top, 0.0, 1.0)

    def _appearance(self, look: np.ndarray, frame: int) -> np.ndarray:
        out = np.full(len(look), NEUTRAL)
        if self._look is None:
            return out
        trust = self.memory_decay ** (frame - self._look_frame)
        for i, row in enumerate(look):
            if np.isfinite(row).all():
                sim = signature_similarity(row, self._look, self.use_colour)
                out[i] = NEUTRAL + (sim - NEUTRAL) * trust
        return out

    def _proximity(self, xy: np.ndarray, frame: int) -> np.ndarray:
        out = np.zeros(len(xy))
        sigma0 = self.scale.px(self.proximity_sigma)
        for seen, raw, scene in self._memory:
            age = frame - seen
            weight = self.memory_decay ** age
            sigma = sigma0 * math.sqrt(max(age, 1))
            d = np.minimum(np.hypot(*(xy - raw).T), np.hypot(*(xy - scene).T))
            out = np.maximum(out, weight * np.exp(-0.5 * (d / sigma) ** 2))
        return out

    def _consistency(self, track_id: np.ndarray, frame: int) -> np.ndarray:
        out = np.empty(len(track_id))
        for i, tid in enumerate(track_id):
            s = self._seen[int(tid)]
            age = frame - s.first + 1
            out[i] = (s.hits / age) * (1.0 - math.exp(-age / self.maturity))
        return out

    def _smoothness(self, track_id: np.ndarray) -> np.ndarray:
        out = np.full(len(track_id), NEUTRAL)
        sigma = self.scale.px(self.smooth_sigma)
        for i, tid in enumerate(track_id):
            path = self._seen[int(tid)].path
            if len(path) < 3:
                continue
            (f0, r0, s0), (f1, r1, s1), (f2, r2, s2) = path
            ratio = (f2 - f1) / (f1 - f0)
            # Raw or camera-compensated, whichever the drone kept steadier: on intercept
            # footage the camera follows it, so raw is often the smooth one (as `kinematics`).
            err = min(np.hypot(*(r2 - (r1 + (r1 - r0) * ratio))),
                      np.hypot(*(s2 - (s1 + (s1 - s0) * ratio))))
            out[i] = math.exp(-0.5 * (err / sigma) ** 2)
        return out

    def _motion(self, speed: np.ndarray) -> np.ndarray:
        v0 = self.scale.px(self.motion_speed)
        out = np.full(len(speed), NEUTRAL)
        ok = np.isfinite(speed)
        out[ok] = 1.0 - np.exp(-np.maximum(speed[ok], 0.0) / v0)
        return out

    # --- memory ------------------------------------------------------------------------

    def _carry(self, hmat: np.ndarray) -> None:
        """Every stored scene position through this frame's camera motion."""
        self._memory = [(f, raw, _apply(hmat, scene)) for f, raw, scene in self._memory]
        for s in self._seen.values():
            s.path = deque(((f, raw, _apply(hmat, scene)) for f, raw, scene in s.path),
                           maxlen=3)

    def _forget(self, frame: int) -> None:
        horizon = frame - self.memory_frames
        self._memory = [m for m in self._memory if m[0] > horizon]
        self._seen = {t: s for t, s in self._seen.items() if s.last > horizon}

    def _remember(self, frame: int, best: int, track_id: np.ndarray, xy: np.ndarray,
                  look: np.ndarray) -> None:
        """This frame's #1 joins the history; a stable #1 teaches the drone's colour."""
        tid = int(track_id[best])
        self._memory.append((frame, xy[best].copy(), xy[best].copy()))
        row = look[best]
        if np.isfinite(row).all():
            if self._look is None:
                self._look, self._look_frame = row.copy(), frame
            elif tid == self._best:
                self._look = self._look + self.look_rate * (row - self._look)
                self._look_frame = frame
        self._best = tid


def colour_signature(img_bgr: np.ndarray, x: float, y: float, radius: float) -> np.ndarray:
    """A candidate's look: (L, a, b) of the disc of `radius` around (x, y), and that disc's
    L minus the ring out to twice the radius. NaN when the disc is off the picture.

    Only the patch is converted to Lab, so a 4128-wide frame costs no more than a 960 one.
    """
    h, w = img_bgr.shape[:2]
    r = max(float(radius), 1.0)
    reach = int(math.ceil(2 * r))
    x0, x1 = max(int(round(x)) - reach, 0), min(int(round(x)) + reach + 1, w)
    y0, y1 = max(int(round(y)) - reach, 0), min(int(round(y)) + reach + 1, h)
    if x1 <= x0 or y1 <= y0:
        return np.full(LOOK_SIZE, np.nan)
    patch = img_bgr[y0:y1, x0:x1]
    if patch.ndim == 2:
        patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(np.ascontiguousarray(patch), cv2.COLOR_BGR2Lab).astype(float)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx - x, yy - y)
    core, ring = d <= r, (d > r) & (d <= 2 * r)
    if not core.any():
        return np.full(LOOK_SIZE, np.nan)
    mean = lab[core].mean(axis=0)
    contrast = mean[0] - lab[ring, 0].mean() if ring.any() else 0.0
    return np.array([mean[0], mean[1], mean[2], contrast])


def signature_similarity(a: np.ndarray, b: np.ndarray, use_colour: bool = True) -> float:
    """1 for identical looks, falling as a Gaussian of the spread-normalised difference.
    `use_colour=False` compares only L and the ring contrast."""
    z = (np.asarray(a, float) - np.asarray(b, float)) / np.asarray(LOOK_SPREAD)
    if not use_colour:
        z = z[[0, 3]]
    return float(math.exp(-0.5 * float(z @ z)))


def _apply(hmat: np.ndarray, p: np.ndarray) -> np.ndarray:
    """One point through a 3x3 homography."""
    v = np.asarray(hmat, float) @ np.array([p[0], p[1], 1.0])
    return v[:2] / v[2]
