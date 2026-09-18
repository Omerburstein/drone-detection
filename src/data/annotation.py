"""One hand-annotation session: a verdict per frame, and how verdicts spread.

`src.data.annotate` is the window; this is what it edits. Split out so the
rules that decide what lands in a label file are testable without a display.

**A verdict is one of two things** -- a box, or "confirmed: no target here".
A frame with no verdict was never adjudicated and is never scored. That is the
same three-way split `src.data.seed_track` draws between label files and
`verified.jsonl`, and for the same reason: counting an unlooked-at frame as a
negative invents precision nobody measured.

**How a verdict spreads when the user steps or plays forward:**

* from a box -> the follower proposes the next box (`TRACKED`, with its score);
* from "no target" -> "no target" carries (`CARRIED`), so a negative span is
  labelled by playing through it rather than frame by frame;
* from nothing -> nothing.

A `HUMAN` verdict is never overwritten by propagation. A `TRACKED` or `CARRIED`
one is overwritten when playing forward -- it came from an earlier pass, and if
the user has since corrected something upstream, the old proposal is stale.
Stepping *backward* fills only frames with no verdict at all, so reviewing a
span never rewrites it.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.data.box_edit import Box
from src.data.seed_track import (DEFAULT_MIN_SCORE, DEFAULT_SCALES,
                                 DEFAULT_SEARCH, TemplateTracker, to_yolo)

HUMAN = "human"      # drawn, moved, resized or declared by the user
TRACKED = "tracked"  # a box the follower proposed
CARRIED = "carried"  # "no target", carried forward from a frame that said so

SCHEMA = 1

# Normalised correlation is contrast-invariant, so a featureless patch is
# stretched to full contrast before it is compared -- and faint sky gradient then
# matches faint sky gradient anywhere. Measured on `first_catch`: a box drawn on
# empty sky (grey std 1.1) "followed" across open sky at scores of ~0.8, while
# the 80x34 drone at frame 962 has a std of 46.9. Both guards key off that gap.
MIN_TEMPLATE_STD = 4.0     # grey levels; below this there is nothing to follow
MIN_CONTRAST_RATIO = 0.35  # a match this much flatter than the template is sky


@dataclass(frozen=True)
class Verdict:
    """What one frame holds: a box, or `box=None` for a confirmed negative."""

    box: Box | None
    source: str
    score: float = 1.0

    @property
    def absent(self) -> bool:
        """True when the frame is a confirmed negative."""
        return self.box is None


# --- following -----------------------------------------------------------

def predict(box: Box, prior: Box | None) -> Box:
    """Where `box` should be one frame on, if it keeps its last displacement.

    The motion half of the follower. Correlation alone searches a window a few
    target sizes wide around the *last* position; a small target crossing the
    frame fast can leave that window in one frame. Shifting the window by the
    last observed velocity keeps it centred on where the target is going.
    """
    if prior is None:
        return box
    x, y, w, h = box
    px, py, pw, ph = prior
    dx = (x + w / 2) - (px + pw / 2)
    dy = (y + h / 2) - (py + ph / 2)
    return x + dx, y + dy, w, h


class Follower:
    """Carries the user's box to the next frame: correlation plus a motion prior.

    Deliberately **not** frame differencing. The camera is on a moving drone,
    and EXP-012a measured the background residual after motion compensation at
    ~10 px against a target moving 4.5 px/frame -- a motion detector on this
    footage fires on the terrain. Correlation against the box the user drew
    knows only what the target looks like, and the velocity prior (`predict`)
    supplies the motion.

    The template is the **last box the user placed**, never a tracked one: this
    is `seed_track`'s pinned-template finding (with blending, a track that
    slipped onto terrain matched it at 0.99). Every correction re-seeds it, so
    the appearance stays current exactly as fast as the user corrects.
    """

    def __init__(self, search: float = DEFAULT_SEARCH,
                 min_score: float = DEFAULT_MIN_SCORE,
                 scales: tuple[float, ...] = DEFAULT_SCALES) -> None:
        self.options = {"search": search, "min_score": min_score, "scales": scales}
        self._tracker: TemplateTracker | None = None
        self._contrast = 0.0
        self.seeded_from: int | None = None

    def seed(self, image: np.ndarray, box: Box, frame: int) -> bool:
        """Take a new template from a user-placed box.

        False -- and no template -- when the box is degenerate or featureless,
        so the next step reports the target lost instead of wandering.
        """
        self.reset()
        patch = _grey_patch(image, box)
        if patch is None or patch.std() < MIN_TEMPLATE_STD:
            return False
        self._tracker = TemplateTracker(image, box, update=0.0, **self.options)
        self._contrast = float(patch.std())
        self.seeded_from = frame
        return True

    def reset(self) -> None:
        """Drop the template, so the next step re-seeds from the box on screen."""
        self._tracker = None
        self.seeded_from = None

    @property
    def ready(self) -> bool:
        """True once a template exists."""
        return self._tracker is not None

    def follow(self, image: np.ndarray, box: Box,
               prior: Box | None) -> tuple[Box, float] | None:
        """The box in `image`, searched around where motion says it went."""
        if self._tracker is None:
            return None
        self._tracker.box = predict(box, prior)
        found = self._tracker.step(image)
        if found is None:
            return None
        patch = _grey_patch(image, found[0])
        if patch is None or patch.std() < MIN_CONTRAST_RATIO * self._contrast:
            return None
        return found


def _grey_patch(image: np.ndarray, box: Box) -> np.ndarray | None:
    """The greyscale pixels under a box, or None if under 2x2 inside the frame."""
    x, y, w, h = (int(round(v)) for v in box)
    patch = image[max(y, 0):y + h, max(x, 0):x + w]
    if patch.shape[0] < 2 or patch.shape[1] < 2:
        return None
    return cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch


# --- the session ---------------------------------------------------------

FollowFn = Callable[[Box, Box | None], tuple[Box, float] | None]


class Session:
    """Every verdict for one video, plus saving and resuming them."""

    def __init__(self, video: Path, stem: str, width: int, height: int) -> None:
        self.video = video
        self.stem = stem
        self.width = width
        self.height = height
        self.verdicts: dict[int, Verdict] = {}
        self.dirty = False

    # editing ------------------------------------------------------------

    def get(self, frame: int) -> Verdict | None:
        """The verdict for a 1-based frame, or None if nobody adjudicated it."""
        return self.verdicts.get(frame)

    def set_box(self, frame: int, box: Box) -> None:
        """Record a box the user placed."""
        self._put(frame, Verdict(box, HUMAN))

    def mark_absent(self, frame: int) -> None:
        """Record that the user confirmed there is no target in this frame."""
        self._put(frame, Verdict(None, HUMAN))

    def clear(self, frame: int) -> None:
        """Forget this frame's verdict: back to never-adjudicated."""
        if self.verdicts.pop(frame, None) is not None:
            self.dirty = True

    def _put(self, frame: int, verdict: Verdict) -> None:
        self.verdicts[frame] = verdict
        self.dirty = True

    # spreading ----------------------------------------------------------

    def propagate(self, src: int, dst: int,
                  follow: FollowFn) -> tuple[Verdict | None, bool]:
        """Carry `src`'s verdict into the adjacent frame `dst`.

        Returns `(verdict at dst, lost)`. `lost` is True when `src` held a box
        and the follower could not find it in `dst` -- the caller's cue to stop
        and hand control back to the user. Any stale proposal at `dst` is
        dropped then, so an old track is never displayed as if it still held.
        """
        forward = dst > src
        existing = self.verdicts.get(dst)
        if existing is not None and (existing.source == HUMAN or not forward):
            return existing, False

        source = self.verdicts.get(src)
        if source is None:
            return existing, False
        if source.absent:
            self._put(dst, Verdict(None, CARRIED))
            return self.verdicts[dst], False

        found = follow(source.box, self._prior(src, 1 if forward else -1))
        if found is None:
            if existing is not None:
                self.clear(dst)
            return None, True
        box, score = found
        self._put(dst, Verdict(box, TRACKED, round(float(score), 4)))
        return self.verdicts[dst], False

    def _prior(self, src: int, step: int) -> Box | None:
        """The box one frame *behind* `src`, for the velocity estimate.

        Withheld when `src` is a user correction and the frame behind it is
        not: the difference between a wrong proposal and its correction is the
        size of the mistake, not the target's velocity, and extrapolating it
        would throw the search window away from the target.
        """
        here = self.verdicts[src]
        behind = self.verdicts.get(src - step)
        if behind is None or behind.absent:
            return None
        if here.source == HUMAN and behind.source != HUMAN:
            return None
        return behind.box

    # summary ------------------------------------------------------------

    def counts(self) -> dict[str, int]:
        """How many frames hold a box, a confirmed negative, and a user verdict."""
        boxes = sum(1 for v in self.verdicts.values() if not v.absent)
        return {"boxes": boxes, "absent": len(self.verdicts) - boxes,
                "human": sum(1 for v in self.verdicts.values() if v.source == HUMAN)}

    # persistence --------------------------------------------------------

    def to_json(self) -> dict:
        """The session file's content. `source` and `score` ride along so a
        later review can find exactly the frames no human placed."""
        frames = {}
        for frame in sorted(self.verdicts):
            verdict = self.verdicts[frame]
            frames[str(frame)] = {
                "box": None if verdict.box is None else [round(v, 2) for v in verdict.box],
                "source": verdict.source,
                "score": verdict.score,
            }
        return {"schema": SCHEMA, "video": str(self.video), "stem": self.stem,
                "width": self.width, "height": self.height, "frames": frames}

    def save(self, path: Path) -> None:
        """Write the session atomically, so a crash mid-write loses nothing."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(self.to_json(), indent=1), encoding="utf-8")
        os.replace(temp, path)
        self.dirty = False

    @classmethod
    def load(cls, path: Path, video: Path, stem: str, width: int,
             height: int) -> "Session":
        """Resume a saved session, refusing one made on different geometry.

        A size mismatch means a different file -- the raw capture against its
        cropped copy, say -- and every box would land in the wrong place.
        """
        session = cls(video, stem, width, height)
        if not path.exists():
            return session
        data = json.loads(path.read_text(encoding="utf-8"))
        if (data["width"], data["height"]) != (width, height):
            raise ValueError(f"{path} was made on a {data['width']}x{data['height']} "
                             f"video; {video.name} is {width}x{height}")
        if data["stem"] != stem:
            raise ValueError(f"{path} belongs to stem {data['stem']!r}, not {stem!r}")
        for key, row in data["frames"].items():
            box = None if row["box"] is None else tuple(float(v) for v in row["box"])
            session.verdicts[int(key)] = Verdict(box, row["source"], row["score"])
        return session

    # export -------------------------------------------------------------

    def export(self, labels_out: Path, verified_out: Path, images_dir: Path) -> None:
        """Write YOLO labels and `verified.jsonl`, in `seed_track`'s formats.

        **The session is authoritative for its stem.** Label files for this
        stem that the session does not hold a box for are deleted, and this
        stem's rows in `verified.jsonl` are replaced wholesale -- otherwise a
        box the user deleted, or a frame they un-verified, would survive in the
        export and be scored. Other stems' rows are left untouched.
        """
        own = re.compile(rf"^{re.escape(self.stem)}_\d+$")
        labels_out.mkdir(parents=True, exist_ok=True)
        boxed = {f: v for f, v in self.verdicts.items() if not v.absent}
        for path in labels_out.glob(f"{self.stem}_*.txt"):
            if own.match(path.stem) and int(path.stem.rsplit("_", 1)[1]) not in boxed:
                path.unlink()
        for frame, verdict in boxed.items():
            cx, cy, w, h = to_yolo(verdict.box, self.width, self.height)
            (labels_out / f"{self.key(frame)}.txt").write_text(
                f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n", encoding="utf-8")

        rows: dict[str, dict] = {}
        if verified_out.exists():
            for line in verified_out.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    name = Path(record["image"]).stem
                    if not own.match(name):
                        rows[name] = record
        for frame in self.verdicts:
            name = self.key(frame)
            rows[name] = {"image": str(images_dir / f"{name}.jpg"), "detections": []}
        verified_out.parent.mkdir(parents=True, exist_ok=True)
        with verified_out.open("w", encoding="utf-8") as handle:
            for name in sorted(rows):
                handle.write(json.dumps(rows[name]) + "\n")

    def key(self, frame: int) -> str:
        """The frame key every run and label file uses: `<stem>_<frame:04d>`."""
        return f"{self.stem}_{frame:04d}"

