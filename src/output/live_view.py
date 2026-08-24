"""The live window: the frame, the box, a magnified inset, and the caveats.

Drawing a detection on screen without saying how it was produced invites the
viewer to read it as a full-rate result. On a duty-cycled feed it is not one, so
the heads-up display carries the policy, what that policy cost when it was
measured, and how stale the picture is — beside the picture, not in a log.

The inset is not decoration either. A 10-30 px drone on a 1920x1080 frame
scaled to fit a laptop screen is two or three pixels; without magnification the
window shows a green speck on grey and cannot be judged. `src.output.overlay`
solves the same problem for scored video and its zoom arithmetic is reused here.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..algo.deployment import PolicyChoice
from ..algo.detections import Detections
from .overlay import INSET_MARGIN, fit_zoom

WINDOW = "drone-detection - live"

BOX_COLOUR = (255, 190, 40)  # blue, as in the scored overlay
SEARCH_COLOUR = (60, 60, 255)  # red: nothing found
LOCK_COLOUR = (90, 220, 90)  # green: tracking
INK = (255, 255, 255)
MUTED = (170, 170, 170)
STRIP = (24, 24, 24)

FONT = cv2.FONT_HERSHEY_SIMPLEX
TITLE_SCALE = 0.9
LINE_SCALE = 0.5
HUD_HEIGHT = 104
BOX_PAD = 4  # push the rectangle off the target so it does not cover it
INSET_SPAN = 120  # side of the magnified source window, in original pixels
INSET_ZOOM = 4


class DisplayUnavailable(RuntimeError):
    """This OpenCV build cannot open a window."""


def require_display() -> None:
    """Fail now, with the fix, rather than at the first frame.

    `opencv-python-headless` and `opencv-python` install into the same `cv2`
    namespace, so having both means whichever wins the import decides whether
    there is a GUI at all. The headless build still *exposes* `imshow`; it
    raises when called. Checked before the model is loaded, because discovering
    it after a 40-second checkpoint load is a poor way to find out.
    """
    probe = np.zeros((2, 2, 3), dtype=np.uint8)
    try:
        cv2.imshow(WINDOW, probe)
        cv2.waitKey(1)
        cv2.destroyWindow(WINDOW)
    except cv2.error as exc:
        raise DisplayUnavailable(
            "This OpenCV build has no GUI support, so nothing can be shown on "
            "screen.\n"
            "  Cause: 'opencv-python-headless' is installed alongside "
            "'opencv-python' and shadows it.\n"
            "  Fix:   py -3.13 -m pip uninstall -y opencv-python-headless\n"
            "         py -3.13 -m pip install --force-reinstall opencv-python\n"
            f"  OpenCV said: {exc}") from exc


@dataclass(frozen=True)
class LiveStatus:
    """Everything the HUD reports about one processed pair."""

    policy: PolicyChoice
    branch: str
    detections: Detections
    processed: int
    dropped: int
    sustained_fps: float
    latency: float  # seconds between capture and this frame being drawn

    @property
    def locked(self) -> bool:
        """Whether this frame produced a box."""
        return len(self.detections) > 0

    @property
    def duty_cycle(self) -> float:
        """Share of delivered frames that reached the detector."""
        total = self.processed + self.dropped
        return self.processed / total if total else 0.0


def _draw_boxes(canvas: np.ndarray, dets: Detections, scale: float) -> None:
    """Outline each detection, pushed outward so the target stays visible."""
    for (x1, y1, x2, y2), _score, _cls in dets:
        p1 = (int(x1 * scale) - BOX_PAD, int(y1 * scale) - BOX_PAD)
        p2 = (int(x2 * scale) + BOX_PAD, int(y2 * scale) + BOX_PAD)
        cv2.rectangle(canvas, p1, p2, BOX_COLOUR, 2)


def _draw_inset(canvas: np.ndarray, source: np.ndarray, dets: Detections) -> None:
    """Paste a magnified crop around the detection into the top-right corner."""
    height, width = source.shape[:2]
    span = int(min(INSET_SPAN, width, height))
    zoom = fit_zoom(canvas.shape[:2], span, INSET_ZOOM)
    if zoom <= 0 or not len(dets):
        return

    box = dets.boxes[0]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    x0 = int(min(max(cx - span / 2, 0), width - span))
    y0 = int(min(max(cy - span / 2, 0), height - span))
    crop = source[y0:y0 + span, x0:x0 + span]
    if crop.size == 0:
        return

    view = cv2.resize(crop, (span * zoom, span * zoom),
                      interpolation=cv2.INTER_NEAREST)
    side = span * zoom
    top, left = INSET_MARGIN, canvas.shape[1] - side - INSET_MARGIN
    canvas[top:top + side, left:left + side] = view
    cv2.rectangle(canvas, (left - 1, top - 1), (left + side, top + side),
                  BOX_COLOUR, 1)
    cv2.putText(canvas, f"{zoom}x", (left + 6, top + 20), FONT, LINE_SCALE,
                BOX_COLOUR, 1, cv2.LINE_AA)


def _draw_hud(canvas: np.ndarray, status: LiveStatus) -> None:
    """The strip along the bottom: state, policy, and what the policy costs."""
    height, width = canvas.shape[:2]
    top = height - HUD_HEIGHT
    cv2.rectangle(canvas, (0, top), (width, height), STRIP, cv2.FILLED)

    state = "TRACKING" if status.locked else "SEARCHING"
    colour = LOCK_COLOUR if status.locked else SEARCH_COLOUR
    cv2.putText(canvas, state, (16, top + 34), FONT, TITLE_SCALE, colour, 2,
                cv2.LINE_AA)
    cv2.putText(canvas, f"branch: {status.branch}", (200, top + 34), FONT,
                LINE_SCALE, INK, 1, cv2.LINE_AA)

    policy = status.policy
    cv2.putText(canvas,
                f"policy: {policy.name}  |  {status.sustained_fps:.2f} fps  |  "
                f"duty {status.duty_cycle:.1%}  |  lag {status.latency * 1000:.0f} ms",
                (16, top + 62), FONT, LINE_SCALE, INK, 1, cv2.LINE_AA)

    # The caveat line. A duty-cycled detection is not a full-rate detection and
    # the screen has to say so, or the picture overstates what was measured.
    caveat = (f"retains {policy.retention:.0%} of full-rate recall "
              f"(measured, ARD100)")
    delay = policy.expected_detection_delay
    if delay is not None:
        caveat += f"  |  mean {delay:.1f}s to first detection"
    cv2.putText(canvas, caveat, (16, top + 88), FONT, LINE_SCALE, MUTED, 1,
                cv2.LINE_AA)


def render(frame: np.ndarray, status: LiveStatus,
           max_width: int = 1280) -> np.ndarray:
    """One display frame: the picture scaled to fit, boxed, insetted and captioned."""
    height, width = frame.shape[:2]
    scale = min(1.0, max_width / width) if width else 1.0
    canvas = (cv2.resize(frame, (int(width * scale), int(height * scale)))
              if scale < 1.0 else frame.copy())
    canvas = np.vstack([canvas, np.full((HUD_HEIGHT, canvas.shape[1], 3), STRIP,
                                        dtype=canvas.dtype)])

    _draw_boxes(canvas, status.detections, scale)
    _draw_inset(canvas, frame, status.detections)
    _draw_hud(canvas, status)
    return canvas


def show(canvas: np.ndarray) -> bool:
    """Draw one frame; return False when the operator asks to quit."""
    cv2.imshow(WINDOW, canvas)
    key = cv2.waitKey(1) & 0xFF
    return key not in (27, ord("q"))  # Esc or q


def close() -> None:
    """Tear the window down."""
    cv2.destroyAllWindows()
