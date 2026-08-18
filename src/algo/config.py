"""Detector settings for one run."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InferenceConfig:
    """Everything the detector needs to know, decoupled from argparse.

    Exists so inference can be driven from a test or a notebook without
    constructing a Namespace, and so a run's settings can be passed around as
    one immutable value.
    """

    #: Letterbox size for the *whole-frame* path only. Ignored when `tile` is set --
    #: a crop is fed at its own edge length. See `tile_predict_kwargs`.
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.5
    classes: list[int] | None = None
    #: Tiled inference is the default: air-to-air targets are 10-30 px, and the
    #: whole-frame letterbox destroys them before the detector runs.
    tile: bool = True
    tile_size: int = 640
    tile_overlap: float = 0.2
    tile_batch: int = 8

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> InferenceConfig:
        """Build a config from parsed CLI arguments."""
        return cls(
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            classes=args.classes,
            tile=args.tile,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            tile_batch=args.tile_batch,
        )

    def _predict_kwargs(self, imgsz: int) -> dict[str, Any]:
        """Keyword arguments for `YOLO.predict` at a given letterbox size."""
        kw: dict[str, Any] = {"imgsz": imgsz, "conf": self.conf, "iou": self.iou,
                              "verbose": False}
        if self.classes:
            kw["classes"] = self.classes
        return kw

    def predict_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for predicting on a whole frame."""
        return self._predict_kwargs(self.imgsz)

    def tile_predict_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for predicting on one native-resolution crop.

        A crop is fed at `tile_size`, never at `imgsz`. The two are separate
        arguments with separate defaults, so `--tile-size 1280 --imgsz 640` would
        otherwise letterbox every crop down by half -- reintroducing exactly the
        rescaling tiling exists to avoid, while paying the tiling cost to do it.
        """
        return self._predict_kwargs(self.tile_size)
