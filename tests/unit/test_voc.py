"""Tests for VOC parsing and the VOC -> YOLO conversion.

This is the conversion that turns the raw download into the labels every metric
is computed against. A corner/centre or width/height mix-up here produces
labels that pass every numeric check while being silently wrong, and would read
downstream as a bad model rather than a bad pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.prepare_ardmav import (
    OFFICIAL_TEST_VIDEOS,
    SCENE_CATEGORIES,
    annotation_path,
)
from src.data.voc import parse_voc, to_yolo
from src.eval.labels import yolo_to_xyxy

XML_TEMPLATE = """<annotation>
    <filename>{stem}.jpg</filename>
    <size><width>{w}</width><height>{h}</height><depth>3</depth></size>
    {objects}
</annotation>
"""

OBJECT_TEMPLATE = """<object>
        <name>{name}</name>
        <bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin><xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox>
    </object>"""


def write_xml(tmp_path, boxes, width=1920, height=1080, name="Drone", stem="f_0001"):
    """Write a VOC XML with the given absolute xyxy boxes."""
    objects = "\n    ".join(
        OBJECT_TEMPLATE.format(name=name, x1=x1, y1=y1, x2=x2, y2=y2)
        for x1, y1, x2, y2 in boxes)
    path = tmp_path / f"{stem}.xml"
    path.write_text(XML_TEMPLATE.format(stem=stem, w=width, h=height, objects=objects))
    return path


class TestParseVoc:

    def test_reads_size_and_box(self, tmp_path):
        annotation = parse_voc(write_xml(tmp_path, [(821, 281, 840, 292)]))
        assert (annotation.width, annotation.height) == (1920, 1080)
        assert len(annotation.objects) == 1

        obj = annotation.objects[0]
        assert (obj.xmin, obj.ymin, obj.xmax, obj.ymax) == (821, 281, 840, 292)
        assert (obj.width, obj.height) == (19, 11)
        assert obj.name == "Drone"

    def test_reads_multiple_objects(self, tmp_path):
        annotation = parse_voc(write_xml(tmp_path, [(0, 0, 10, 10), (50, 50, 70, 90)]))
        assert len(annotation.objects) == 2

    def test_frame_with_no_object_is_valid(self, tmp_path):
        assert parse_voc(write_xml(tmp_path, [])).objects == []

    def test_float_coordinates_are_accepted(self, tmp_path):
        obj = parse_voc(write_xml(tmp_path, [(10.5, 20.25, 30.5, 40.25)])).objects[0]
        assert obj.width == pytest.approx(20.0)

    def test_missing_size_block_fails_loudly(self, tmp_path):
        path = tmp_path / "bad.xml"
        path.write_text("<annotation><filename>x.jpg</filename></annotation>")
        with pytest.raises(ValueError, match="size"):
            parse_voc(path)

    def test_missing_bndbox_fails_loudly(self, tmp_path):
        path = tmp_path / "bad.xml"
        path.write_text(XML_TEMPLATE.format(
            stem="x", w=100, h=100, objects="<object><name>Drone</name></object>"))
        with pytest.raises(ValueError, match="bndbox"):
            parse_voc(path)

    def test_empty_coordinate_fails_loudly(self, tmp_path):
        path = tmp_path / "bad.xml"
        path.write_text(XML_TEMPLATE.format(stem="x", w=100, h=100, objects="""<object>
            <name>Drone</name>
            <bndbox><xmin></xmin><ymin>1</ymin><xmax>2</xmax><ymax>3</ymax></bndbox>
        </object>"""))
        with pytest.raises(ValueError):
            parse_voc(path)


class TestToYolo:

    def test_known_conversion(self, tmp_path):
        """A 20x10 box at (100,50)-(120,60) in a 200x100 frame."""
        obj = parse_voc(write_xml(tmp_path, [(100, 50, 120, 60)],
                                  width=200, height=100)).objects[0]
        cx, cy, w, h = to_yolo(obj, 200, 100)
        assert (cx, cy) == pytest.approx((110 / 200, 55 / 100))
        assert (w, h) == pytest.approx((20 / 200, 10 / 100))

    def test_round_trips_through_yolo_to_xyxy(self, tmp_path):
        """to_yolo and yolo_to_xyxy must be exact inverses.

        These two live in different modules -- conversion writes labels with one,
        evaluation reads them with the other -- so a convention drift between
        them would corrupt every metric while both looked correct alone.
        """
        original = (821.0, 281.0, 840.0, 292.0)
        obj = parse_voc(write_xml(tmp_path, [original])).objects[0]
        rows = np.array([to_yolo(obj, 1920, 1080)])
        assert yolo_to_xyxy(rows, 1920, 1080)[0] == pytest.approx(original)

    def test_non_square_frame_does_not_swap_axes(self, tmp_path):
        """A box spanning the full width but not the height catches a w/h swap."""
        obj = parse_voc(write_xml(tmp_path, [(0, 400, 1920, 500)])).objects[0]
        _, _, w, h = to_yolo(obj, 1920, 1080)
        assert w == pytest.approx(1.0)
        assert h == pytest.approx(100 / 1080)


class TestOfficialSplit:
    """The published split is load-bearing: it is what makes our numbers
    comparable to GLAD's. Pinned so an edit cannot silently change it."""

    def test_fifteen_test_videos(self):
        assert len(OFFICIAL_TEST_VIDEOS) == 15
        assert len(set(OFFICIAL_TEST_VIDEOS)) == 15

    def test_categories_partition_the_test_set(self):
        categorised = [v for members in SCENE_CATEGORIES.values() for v in members]
        assert sorted(categorised) == sorted(OFFICIAL_TEST_VIDEOS)
        assert len(categorised) == len(set(categorised)), "a video is in two categories"

    def test_each_category_has_five_videos(self):
        assert {k: len(v) for k, v in SCENE_CATEGORIES.items()} == {
            "ordinary": 5, "complex": 5, "small_mav": 5}


class TestAnnotationPath:
    """Frame numbering is one-based and four-digit; everything depends on it."""

    @pytest.mark.parametrize("number, expected", [
        (1, "phantom05_0001.xml"),
        (42, "phantom05_0042.xml"),
        (1799, "phantom05_1799.xml"),
        (12345, "phantom05_12345.xml"),  # padding is a minimum, not a truncation
    ])
    def test_naming_convention(self, tmp_path, number, expected):
        assert annotation_path(tmp_path, "phantom05", number).name == expected

    def test_lives_under_a_per_video_folder(self, tmp_path):
        path = annotation_path(tmp_path, "phantom05", 1)
        assert path.parent.name == "phantom05"
        assert path.parent.parent.name == "Annotations"
