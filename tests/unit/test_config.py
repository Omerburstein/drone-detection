"""`InferenceConfig`: which letterbox size reaches the model on each path.

The tiled path exists to feed native pixels. `imgsz` and `tile_size` are separate
arguments with separate defaults, so nothing but these tests stops a future edit
from letterboxing every crop back down and silently undoing the whole point.
"""

from __future__ import annotations

import pytest

from src.algo.config import InferenceConfig
from src.baseline_detect import build_parser

BASE_ARGS = ["--weights", "w.pt", "--source", "s.mp4"]


def parse(*extra: str) -> InferenceConfig:
    """Build the config the CLI would produce for these extra arguments."""
    return InferenceConfig.from_args(build_parser().parse_args(BASE_ARGS + list(extra)))


class TestPredictSize:
    """Which `imgsz` each path hands to `YOLO.predict`."""

    def test_whole_frame_uses_imgsz(self):
        cfg = InferenceConfig(imgsz=1280, tile_size=640)
        assert cfg.predict_kwargs()["imgsz"] == 1280

    def test_tiled_uses_tile_size_not_imgsz(self):
        # The bug this guards: a 1280 crop letterboxed to 640 is a 2x downscale,
        # paid for at tiling's price.
        cfg = InferenceConfig(imgsz=640, tile_size=1280)
        assert cfg.tile_predict_kwargs()["imgsz"] == 1280

    def test_tiled_default_is_lossless(self):
        cfg = InferenceConfig()
        assert cfg.tile_predict_kwargs()["imgsz"] == cfg.tile_size

    @pytest.mark.parametrize("kwargs_name", ["predict_kwargs", "tile_predict_kwargs"])
    def test_both_paths_share_thresholds(self, kwargs_name):
        cfg = InferenceConfig(conf=0.15, iou=0.4, classes=[0, 2])
        kw = getattr(cfg, kwargs_name)()
        assert (kw["conf"], kw["iou"], kw["classes"]) == (0.15, 0.4, [0, 2])

    def test_no_class_filter_is_omitted(self):
        # An empty `classes` list must not reach predict as a filter matching nothing.
        assert "classes" not in InferenceConfig(classes=None).predict_kwargs()
        assert "classes" not in InferenceConfig(classes=[]).predict_kwargs()


class TestTileDefault:
    """Tiling is on unless explicitly refused."""

    def test_default_config_tiles(self):
        assert InferenceConfig().tile is True

    def test_cli_default_tiles(self):
        assert parse().tile is True

    def test_no_tile_opts_out(self):
        assert parse("--no-tile").tile is False

    def test_explicit_tile_flag_still_accepted(self):
        assert parse("--tile").tile is True

    def test_last_flag_wins(self):
        assert parse("--no-tile", "--tile").tile is True
        assert parse("--tile", "--no-tile").tile is False
