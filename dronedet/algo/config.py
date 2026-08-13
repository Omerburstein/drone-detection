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

    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.5
    classes: list[int] | None = None
    tile: bool = False
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

    def predict_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for `YOLO.predict`."""
        kw: dict[str, Any] = {"imgsz": self.imgsz, "conf": self.conf, "iou": self.iou,
                              "verbose": False}
        if self.classes:
            kw["classes"] = self.classes
        return kw
