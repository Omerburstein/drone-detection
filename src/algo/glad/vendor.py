"""Importing GLAD's released code and yolov5 v6.0 out of `third_party/`.

The upstream trees make three assumptions this project cannot satisfy. All three
are handled here so nothing else has to know about them:

* **Flat imports.** `MOD2.py` does `from Functions import ...`, and the
  checkpoints unpickle as `models.yolo.Model`, so both clones have to sit on
  `sys.path` as top-level import roots.
* **Checkpoints are trusted.** `torch.load` has defaulted to `weights_only=True`
  since torch 2.6 and refuses these files. They are the MIT-licensed release
  from the paper's own authors, so they are loaded in full — but only inside a
  context manager, never by mutating the global default.
* **Paths are relative to the caller's cwd.** Upstream opens
  `./weights/Net_best.pth`. Everything here resolves from the repo root instead,
  so a run works from any directory.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterator

import torch

# src/algo/glad/vendor.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
GLAD_DIR = REPO_ROOT / "third_party" / "GLAD"
YOLOV5_DIR = REPO_ROOT / "third_party" / "yolov5"

GLOBAL_WEIGHTS = "yolov5s_GLAD.pt"  # GAD, trained on full 1920x1080 frames
LOCAL_WEIGHTS = "yolov5s_GLAD-crop.pt"  # LAD, trained on 320x320 target crops
CLASSIFIER_WEIGHTS = "Net_best.pth"  # the LeNet gate ending both motion branches


def weights_dir(glad_dir: Path = GLAD_DIR) -> Path:
    """Directory holding the three released checkpoints."""
    return glad_dir / "weights"


_roots_added = False


def add_import_roots(glad_dir: Path = GLAD_DIR,
                     yolov5_dir: Path = YOLOV5_DIR) -> None:
    """Put both vendored clones on `sys.path`, once per process.

    yolov5 goes first: unpickling either detector checkpoint needs
    `models.yolo` importable before `Functions`/`MOD2` are touched.

    The first call wins and every later one is a no-op, so a caller that has
    already pointed this at a non-default clone is not overridden by the
    defaulted calls scattered through the loaders below it.
    """
    global _roots_added
    if _roots_added:
        return

    for directory in (yolov5_dir, glad_dir):
        if not directory.is_dir():
            raise FileNotFoundError(
                f"{directory} is missing. The GLAD and yolov5 clones live under "
                f"third_party/ and are gitignored; see docs/glad-model.md.")
        path = str(directory)
        if path not in sys.path:
            sys.path.insert(0, path)
    _roots_added = True


@contextmanager
def trusted_torch_load() -> Iterator[None]:
    """Temporarily restore `torch.load`'s pre-2.6 `weights_only=False` default.

    Scoped rather than global so an unpickle anywhere else in the process keeps
    the safe default.
    """
    original = torch.load

    def _load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = _load
    try:
        yield
    finally:
        torch.load = original


def import_motion(gate: Callable[[Any], int] | None = None,
                  glad_dir: Path = GLAD_DIR,
                  yolov5_dir: Path = YOLOV5_DIR) -> ModuleType:
    """Import upstream `MOD2`, optionally swapping in a cached appearance gate.

    `MOD2` binds `Mynet_infer` by value at import time, so the substitution has
    to happen on `Functions` *before* `MOD2` is first imported — and on `MOD2`
    itself if something already imported it. Both are rebound.
    """
    add_import_roots(glad_dir, yolov5_dir)

    import Functions  # noqa: PLC0415 -- only importable once sys.path is set up

    if gate is not None:
        Functions.Mynet_infer = gate

    import MOD2  # noqa: PLC0415

    if gate is not None:
        MOD2.Mynet_infer = gate
    return MOD2


def import_enlarge_region() -> Callable[..., tuple[int, int, int, int]]:
    """Upstream's `enlarge_region2`, used verbatim to place the search region.

    Imported rather than reimplemented: it takes the box *top-left* as the
    region centre (not the box centre) and clamps by shifting the region rather
    than shrinking it. Both are load-bearing, and both look like typos.
    """
    add_import_roots()
    from Functions import enlarge_region2  # noqa: PLC0415

    return enlarge_region2
