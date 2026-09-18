"""Annotate a video by hand, with the box following the target between corrections.

Play, pause and step through a clip; drag a box around the drone; move or
resize it by its body, edges or corners. From then on the box **follows the
target** as the video advances -- the user only steps in when it is wrong, and
the correction becomes the new template. When the follower loses the target,
playback stops and says so.

Frames confirmed to hold no drone are marked with `x` and the negative carries
forward as you play, so a target-free span is labelled by watching it.

Writes the same two things `src.data.seed_track` does -- YOLO labels and
`verified.jsonl` for `src.evaluate --keys-from` -- plus a session file that
lets you stop and resume. See docs/annotate.md for every key and parameter.

Example
-------
    py -3.13 -m src.data.annotate
        --video data/processed/SOFA-O4/videos/first_catch.avi
        --labels-out data/processed/SOFA-O4/labels/test
        --verified-out data/processed/SOFA-O4/verified.jsonl
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from src.data.annotation import CARRIED, HUMAN, Follower, Session, Verdict
from src.data.box_edit import (MIN_SIZE, Box, Viewport, clamp_box, drag,
                               from_corners, hit_test)
from src.data.seed_track import DEFAULT_MIN_SCORE, DEFAULT_SEARCH

WINDOW = "annotate"
BAR = 50                  # status strip under the picture, in window pixels
TIMELINE = 8              # coverage strip across the top of the status bar
GRAB_TOL = 8.0            # handle tolerance, in *window* pixels
SPEEDS = (1, 2, 5, 10, 15, 30, 60)
AUTOSAVE_SECONDS = 30.0

# Win32 `waitKeyEx` codes. The letters are the portable fallback.
KEY_LEFT, KEY_RIGHT = 2424832, 2555904
KEY_DELETE = 3014656

GREEN, YELLOW, RED, CYAN = (0, 220, 0), (0, 220, 255), (40, 40, 230), (255, 230, 0)
WHITE = (255, 255, 255)

HELP = (
    "space  play / pause          d or ->  next frame     a or <-  previous frame",
    "D / A  jump 30 frames (no tracking)   click the timeline to jump",
    "drag on empty picture: new box        drag box body / edge / corner: move / resize",
    "x or Del  no target here (carries forward while playing)",
    "c  clear this frame's verdict   + / -  playback speed",
    "wheel  zoom at cursor   right-drag  pan   f  zoom to box   0  whole frame",
    "s  save + export        q / Esc  save, export, quit       h  toggle this help",
)


class FrameCache:
    """Random access into a video that stays fast for the ways people browse.

    Sequential reads are cheap and seeks are not: on `first_catch.avi` a seek
    costs 1,109 ms against 26 ms for the next frame. So the cache keeps the
    last `size` decoded frames (stepping back is free), and a short jump
    forward decodes through rather than seeking.

    Reading through is also what keeps **long-GOP** sources exact: a seek on
    an mp4 can land a keyframe away from where it was asked, and a box drawn on
    the wrong frame is a wrong label. Only jumps further than `read_through`
    frames, or backward beyond the cache, seek at all.

    Going backward past the cache seeks **half a cache earlier** and decodes
    forward, so the next steps back are already held. Seeking to exactly the
    frame asked for cost 454 ms per step walking back from frame 962 of
    `first_catch` -- one seek per keypress.
    """

    def __init__(self, capture: cv2.VideoCapture, size: int = 90,
                 read_through: int = 60) -> None:
        self.capture = capture
        self.size = size
        self.read_through = read_through
        self._frames: OrderedDict[int, np.ndarray] = OrderedDict()
        self._next = 1  # the 1-based frame the decoder will return next
        self.seeks = 0

    def get(self, index: int) -> np.ndarray | None:
        """The 1-based frame `index`, or None past the end of the stream."""
        if index in self._frames:
            self._frames.move_to_end(index)
            return self._frames[index]
        if not (self._next <= index <= self._next + self.read_through):
            backward = index < self._next
            start = max(1, index - self.size // 2) if backward else index
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, start - 1)
            self._next = start
            self.seeks += 1
        frame = None
        while self._next <= index:
            ok, frame = self.capture.read()
            if not ok:
                return None
            self._store(self._next, frame)
            self._next += 1
        return frame

    def _store(self, index: int, frame: np.ndarray) -> None:
        self._frames[index] = frame
        self._frames.move_to_end(index)
        while len(self._frames) > self.size:
            self._frames.popitem(last=False)


class Annotator:
    """The interactive loop's state and its responses to keys and the mouse.

    `on_mouse`, `on_key` and `render` never touch a window, so the whole
    interaction is drivable from a test; `run` is the only part that does.
    """

    def __init__(self, frames: FrameCache, session: Session, follower: Follower,
                 total: int, display: tuple[int, int], save: Callable[[], None],
                 start: int = 1, fps: int = 10) -> None:
        self.frames = frames
        self.session = session
        self.follower = follower
        self.total = total
        self.save = save
        self.index = min(max(start, 1), total)
        self.playing = False
        self.speed = min(range(len(SPEEDS)), key=lambda i: abs(SPEEDS[i] - fps))
        self.view = Viewport((session.width, session.height), display)
        self.show_help = False
        self.quit = False
        self.message = ""
        self._drag: tuple | None = None  # (kind, press point, box at press)
        self.preview: Box | None = None
        self.image = self.frames.get(self.index)
        if self.image is None:
            raise ValueError(f"could not read frame {self.index}")

    # -- navigation --------------------------------------------------------

    def step(self, delta: int) -> None:
        """Move one frame, carrying the verdict with it."""
        dst = self.index + delta
        if not 1 <= dst <= self.total:
            self.playing = False
            return
        image = self.frames.get(dst)
        if image is None:  # the header overstated the frame count
            self.total = dst - 1
            self.playing = False
            self.say(f"end of stream at frame {self.total}")
            return

        self._ensure_seeded()
        src = self.index
        verdict, lost = self.session.propagate(
            src, dst, lambda box, prior: self.follower.follow(image, box, prior))
        self.index, self.image = dst, image

        if lost:
            self.playing = False
            self.say("LOST the target - draw the box, or press x if it is gone")
        elif verdict is not None and verdict.source == HUMAN and verdict.box:
            self.follower.seed(image, verdict.box, dst)
        if verdict is not None and verdict.box is not None:
            self.view.keep_in_view(verdict.box)

    def jump(self, target: int) -> None:
        """Go to a frame without carrying anything: skipped frames stay unjudged."""
        target = min(max(target, 1), self.total)
        image = self.frames.get(target)
        if image is None:
            self.say(f"could not read frame {target}")
            return
        self.index, self.image = target, image
        self.playing = False
        self.follower.reset()  # its template belongs to wherever we jumped from

    def _ensure_seeded(self) -> None:
        """Seed the follower from the current box if it holds no template yet.

        Happens after resuming a session or jumping onto a tracked span; the
        box shown is then the best appearance available.
        """
        verdict = self.session.get(self.index)
        if verdict is not None and verdict.box is not None and not self.follower.ready:
            self.follower.seed(self.image, verdict.box, self.index)

    # -- editing -----------------------------------------------------------

    def commit_box(self, box: Box) -> None:
        """The user placed a box: record it and make it the follower's template.

        A featureless box is still recorded -- the user may know better -- but
        the follower refuses it, and says so now rather than at the next step.
        """
        box = clamp_box(box, self.session.width, self.session.height)
        self.session.set_box(self.index, box)
        if self.follower.seed(self.image, box, self.index):
            self.say(f"box {box[2]:.0f}x{box[3]:.0f} px set on frame {self.index}")
        else:
            self.say("box set, but it is featureless - nothing to follow. "
                     "Is the drone inside it?")

    def mark_absent(self) -> None:
        """Confirm the current frame holds no target."""
        self.session.mark_absent(self.index)
        self.say("no target here - carries forward while playing")

    # -- input -------------------------------------------------------------

    def on_mouse(self, event: int, x: int, y: int, flags: int, *_) -> None:
        """Draw, move, resize, pan, zoom, and click-to-seek on the timeline."""
        width, height = self.view.display_size
        if event == cv2.EVENT_LBUTTONDOWN and y >= height:
            if y < height + TIMELINE + 6:
                self.jump(1 + int(x / width * self.total))
            return
        point = self.view.to_image(x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.playing = False
            verdict = self.session.get(self.index)
            box = verdict.box if verdict is not None else None
            handle = hit_test(box, point, GRAB_TOL / self.view.zoom) if box else None
            self._drag = (handle or "new", point, box)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._drag = ("pan", (x, y), None)
        elif event == cv2.EVENT_MOUSEMOVE and self._drag is not None:
            kind, start, box = self._drag
            if kind == "pan":
                self.view.pan(x - start[0], y - start[1])
                self._drag = ("pan", (x, y), None)
            elif kind == "new":
                self.preview = from_corners(*start, *point)
            else:
                self.preview = drag(box, kind, point[0] - start[0], point[1] - start[1])
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP) and self._drag:
            kind = self._drag[0]
            if (kind != "pan" and self.preview is not None
                    and min(self.preview[2], self.preview[3]) >= MIN_SIZE):
                self.commit_box(self.preview)
            self._drag, self.preview = None, None
        elif event == cv2.EVENT_MOUSEWHEEL:
            factor = 1.25 if cv2.getMouseWheelDelta(flags) > 0 else 0.8
            self.view.zoom_at(x, y, factor)

    def on_key(self, key: int) -> None:
        """Every keyboard command. `key` is a `waitKeyEx` code."""
        char = chr(key) if 0 <= key < 256 else ""
        if char == " ":
            self.playing = not self.playing
        elif char == "d" or key == KEY_RIGHT:
            self.playing = False
            self.step(1)
        elif char == "a" or key == KEY_LEFT:
            self.playing = False
            self.step(-1)
        elif char in ("D", "A"):
            self.jump(self.index + (30 if char == "D" else -30))
        elif char == "x" or key == KEY_DELETE:
            self.mark_absent()
        elif char == "c":
            self.session.clear(self.index)
            self.say(f"frame {self.index} is unjudged again")
        elif char in ("+", "="):
            self.speed = min(self.speed + 1, len(SPEEDS) - 1)
        elif char == "-":
            self.speed = max(self.speed - 1, 0)
        elif char == "f":
            verdict = self.session.get(self.index)
            if verdict is not None and verdict.box is not None:
                self.view.focus(verdict.box)
        elif char == "0":
            self.view.reset()
        elif char == "h":
            self.show_help = not self.show_help
        elif char == "s":
            self.save()
            self.say("saved and exported")
        elif char == "q" or key == 27:
            self.quit = True

    def tick(self) -> None:
        """Advance one frame if playing; called once per loop iteration."""
        if self.playing:
            self.step(1)

    def say(self, text: str) -> None:
        """Put a line on the status bar until the next one replaces it."""
        self.message = text

    # -- drawing -----------------------------------------------------------

    def render(self) -> np.ndarray:
        """The picture through the viewport, the box, and the status bar."""
        width, height = self.view.display_size
        interpolation = cv2.INTER_NEAREST if self.view.zoom >= 3 else cv2.INTER_LINEAR
        picture = cv2.warpAffine(self.image, self.view.matrix(), (width, height),
                                 flags=interpolation, borderValue=(40, 40, 40))

        verdict = self.session.get(self.index)
        if self.preview is not None:
            self._draw_box(picture, self.preview, CYAN, "")
        elif verdict is not None and verdict.box is not None:
            human = verdict.source == HUMAN
            self._draw_box(picture, verdict.box, GREEN if human else YELLOW,
                           "" if human else f"{verdict.score:.2f}")
        if verdict is not None and verdict.absent:
            cv2.rectangle(picture, (0, 0), (width, 26), RED, -1)
            label = "NO TARGET" + (" (carried)" if verdict.source == CARRIED else "")
            cv2.putText(picture, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
        if self.show_help:
            self._draw_help(picture)
        return np.vstack([picture, self._status_bar(verdict)])

    def _draw_box(self, canvas: np.ndarray, box: Box, colour: tuple, text: str) -> None:
        x, y, w, h = box
        p0 = tuple(int(round(v)) for v in self.view.to_display(x, y))
        p1 = tuple(int(round(v)) for v in self.view.to_display(x + w, y + h))
        cv2.rectangle(canvas, p0, p1, colour, 1)
        for corner in ((p0[0], p0[1]), (p1[0], p0[1]), (p0[0], p1[1]), (p1[0], p1[1])):
            cv2.rectangle(canvas, (corner[0] - 2, corner[1] - 2),
                          (corner[0] + 2, corner[1] + 2), colour, -1)
        if text:
            cv2.putText(canvas, text, (p0[0], max(p0[1] - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)

    def _draw_help(self, canvas: np.ndarray) -> None:
        top = 34
        shade = canvas[top:top + 20 * len(HELP) + 12, :].astype(np.float32) * 0.3
        canvas[top:top + 20 * len(HELP) + 12, :] = shade.astype(np.uint8)
        for row, line in enumerate(HELP):
            cv2.putText(canvas, line, (10, top + 20 * (row + 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)

    def _status_bar(self, verdict: Verdict | None) -> np.ndarray:
        width = self.view.display_size[0]
        bar = np.full((BAR, width, 3), 30, dtype=np.uint8)
        bar[:TIMELINE] = self._timeline(width)
        cursor = int((self.index - 1) / max(self.total - 1, 1) * (width - 1))
        cv2.line(bar, (cursor, 0), (cursor, TIMELINE + 2), WHITE, 2)

        if verdict is None:
            state = "unjudged"
        elif verdict.absent:
            state = f"no target ({verdict.source})"
        else:
            state = f"box ({verdict.source}" + (
                f" {verdict.score:.2f})" if verdict.source != HUMAN else ")")
        counts = self.session.counts()
        line = (f"frame {self.index}/{self.total}   "
                f"{'PLAYING' if self.playing else 'paused'} {SPEEDS[self.speed]} fps   "
                f"zoom {self.view.zoom:.1f}x   {state}   "
                f"[{counts['boxes']} boxes, {counts['absent']} empty]   h: help")
        cv2.putText(bar, line, (8, TIMELINE + 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, WHITE, 1)
        cv2.putText(bar, self.message, (8, TIMELINE + 36), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, YELLOW, 1)
        return bar

    def _timeline(self, width: int) -> np.ndarray:
        """One pixel column per slice of the video, coloured by what it holds.

        Green where boxes are, red where negatives are, dark where nothing was
        judged -- so the gaps still to annotate are visible at a glance.
        """
        strip = np.full((TIMELINE, width, 3), 60, dtype=np.uint8)
        if not self.session.verdicts:
            return strip
        frames = np.fromiter(self.session.verdicts, dtype=np.int64)
        absent = np.fromiter((v.absent for v in self.session.verdicts.values()),
                             dtype=bool)
        columns = ((frames - 1) / max(self.total - 1, 1) * (width - 1)).astype(int)
        columns = np.clip(columns, 0, width - 1)
        strip[:, columns[absent]] = RED
        strip[:, columns[~absent]] = GREEN  # a box outranks a negative in a shared column
        return strip

    # -- the window --------------------------------------------------------

    def run(self) -> None:
        """Show the window and loop until the user quits or closes it."""
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        last_save = time.monotonic()
        while not self.quit:
            started = time.monotonic()
            self.tick()
            cv2.imshow(WINDOW, self.render())
            budget = 1.0 / SPEEDS[self.speed] if self.playing else 0.03
            wait = max(1, int((budget - (time.monotonic() - started)) * 1000))
            key = cv2.waitKeyEx(wait)
            if key != -1:
                self.on_key(key)
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break  # closed with the window's X
            if self.session.dirty and time.monotonic() - last_save > AUTOSAVE_SECONDS:
                self.save()
                last_save = time.monotonic()
        cv2.destroyWindow(WINDOW)


# --- CLI -------------------------------------------------------------------

def display_size(width: int, height: int, limit: tuple[int, int]) -> tuple[int, int]:
    """The window's picture size: the frame scaled down to fit `limit`, never up."""
    scale = min(limit[0] / width, (limit[1] - BAR) / height, 1.0)
    return int(width * scale), int(height * scale)


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for the interactive annotator."""
    ap = argparse.ArgumentParser(prog="src.data.annotate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path,
                    help="Clip to annotate. Use the processed one runs are made on, "
                         "so the boxes share its coordinate system.")
    ap.add_argument("--labels-out", required=True, type=Path,
                    help="Label directory, e.g. data/processed/SOFA-O4/labels/test.")
    ap.add_argument("--verified-out", required=True, type=Path,
                    help="JSONL of every judged frame, for src.evaluate --keys-from.")
    ap.add_argument("--stem", default=None,
                    help="Frame-key prefix. Defaults to the video's stem.")
    ap.add_argument("--session", type=Path, default=None,
                    help="Session file for stop/resume. Defaults to "
                         "<dataset>/annotations/<stem>.json beside the labels tree.")
    ap.add_argument("--images-dir", type=Path, default=None,
                    help="Path the verified rows are keyed by. Defaults to the "
                         "labels tree's sibling images/<split>.")
    ap.add_argument("--start", type=int, default=None,
                    help="1-based frame to open on. Defaults to the last judged "
                         "frame of a resumed session, else 1.")
    ap.add_argument("--fps", type=int, default=10,
                    help="Initial playback speed (default 10). +/- change it live.")
    ap.add_argument("--max-window", type=int, nargs=2, default=(1400, 850),
                    metavar=("W", "H"),
                    help="Largest window, status bar included (default 1400 850).")
    ap.add_argument("--buffer", type=int, default=90, metavar="FRAMES",
                    help="Decoded frames kept for instant stepping back "
                         "(default 90; ~4.7 MB each at 1440x1080).")
    ap.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE,
                    help=f"Follower stops below this correlation (default "
                         f"{DEFAULT_MIN_SCORE}). Lower follows longer and drifts more.")
    ap.add_argument("--search", type=float, default=DEFAULT_SEARCH, metavar="SIZES",
                    help=f"Follower search window, in target sizes (default "
                         f"{DEFAULT_SEARCH}).")
    return ap


def main() -> None:
    """Open the video and session, run the window, save and export on exit."""
    args = build_parser().parse_args()
    stem = args.stem or args.video.stem
    dataset = args.labels_out.parent.parent
    session_path = args.session or dataset / "annotations" / f"{stem}.json"
    images_dir = args.images_dir or dataset / "images" / args.labels_out.name

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        sys.exit(f"Could not open {args.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        session = Session.load(session_path, args.video, stem, width, height)
    except ValueError as error:
        sys.exit(f"error: {error}")
    start = args.start or (max(session.verdicts) if session.verdicts else 1)
    print(f"{args.video.name}: {width}x{height}, {total} frames; "
          f"session {session_path} ({len(session.verdicts)} frames judged)")

    def save() -> None:
        session.save(session_path)
        session.export(args.labels_out, args.verified_out, images_dir)

    try:
        annotator = Annotator(FrameCache(capture, args.buffer), session,
                              Follower(search=args.search, min_score=args.min_score),
                              total, display_size(width, height, args.max_window),
                              save, start=start, fps=args.fps)
        annotator.say("h for help - drag a box around the drone, then space to play")
        annotator.run()
    finally:
        capture.release()
        save()
    counts = session.counts()
    print(f"{counts['boxes']} boxes ({counts['human']} placed by hand), "
          f"{counts['absent']} confirmed-empty frames")
    print(f"labels -> {args.labels_out}\nverified -> {args.verified_out}\n"
          f"session -> {session_path}")


if __name__ == "__main__":
    main()
