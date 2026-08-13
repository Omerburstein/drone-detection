"""Reading Pascal VOC XML annotations.

Kept separate from any one dataset: VOC XML is the interchange format several
drone datasets ship in, and the parsing is identical even where the directory
conventions differ.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VocObject:
    """One annotated object: a class name and absolute xyxy corners."""

    name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin


@dataclass(frozen=True)
class VocAnnotation:
    """One frame's annotation: the declared frame size and its objects.

    `width`/`height` come from the XML's own `<size>` block, which is what box
    coordinates must be normalised against. It is checked against the decoded
    frame rather than trusted, because a mismatch would rescale every box.
    """

    width: int
    height: int
    objects: list[VocObject]


def _require(node: ET.Element, tag: str, path: Path) -> ET.Element:
    """Fetch a child element, failing loudly rather than returning None."""
    found = node.find(tag)
    if found is None:
        raise ValueError(f"{path}: missing <{tag}>")
    return found


def _number(node: ET.Element, tag: str, path: Path) -> float:
    """Read a numeric child element.

    VOC box coordinates are nominally integers but some tools emit floats, so
    parse as float and let the caller round.
    """
    text = _require(node, tag, path).text
    if text is None or not text.strip():
        raise ValueError(f"{path}: empty <{tag}>")
    return float(text.strip())


def parse_voc(path: Path) -> VocAnnotation:
    """Parse one VOC XML file.

    Raises ValueError on anything malformed. A silently-skipped bad annotation
    would show up later as an unexplained model failure, so this never guesses.
    """
    root = ET.parse(path).getroot()
    size = _require(root, "size", path)

    objects = []
    for obj in root.findall("object"):
        name_node = _require(obj, "name", path)
        box = _require(obj, "bndbox", path)
        objects.append(VocObject(
            name=(name_node.text or "").strip(),
            xmin=_number(box, "xmin", path),
            ymin=_number(box, "ymin", path),
            xmax=_number(box, "xmax", path),
            ymax=_number(box, "ymax", path),
        ))

    return VocAnnotation(
        width=int(_number(size, "width", path)),
        height=int(_number(size, "height", path)),
        objects=objects,
    )


def to_yolo(obj: VocObject, width: int, height: int) -> tuple[float, float, float, float]:
    """Absolute VOC corners -> normalised YOLO centre/size.

    The inverse of `src.eval.labels.yolo_to_xyxy`; the two are pinned against
    each other by a round-trip test.
    """
    return (
        ((obj.xmin + obj.xmax) / 2) / width,
        ((obj.ymin + obj.ymax) / 2) / height,
        obj.width / width,
        obj.height / height,
    )
