"""DocLayNet loading utilities.

Two access paths are supported:

- **raw COCO**: read a full ``DocLayNet_core/COCO/<split>.json`` from the
  official IBM zip release. Used by ``prepare_subset.py`` to filter and copy
  out a small working set.
- **local subset**: read a slimmed ``annotations.json`` + images written by
  ``prepare_subset.py``. This is what the benchmark consumes.

The category id↔name map is read from the data itself rather than hardcoded,
so this module keeps working if upstream renumbers the classes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

TABLE_LABEL = "Table"

# Default location for the extracted DocLayNet_core archive. Extract the zip
# into <repo>/data/raw/ so its top-level DocLayNet_core/ folder lands here.
DEFAULT_SOURCE = Path("data/raw/DocLayNet_core")

# DocLayNet ships split JSON under different filenames than HF's split names.
SPLIT_FILES: dict[str, str] = {
    "train": "train.json",
    "validation": "val.json",
    "val": "val.json",
    "test": "test.json",
}


@dataclass(frozen=True)
class Annotation:
    category_id: int
    category_name: str
    bbox: tuple[float, float, float, float]  # [x, y, w, h] in COCO order


@dataclass
class Page:
    image_id: int
    file_name: str
    width: int
    height: int
    annotations: list[Annotation]
    doc_category: str | None = None
    page_no: int | None = None

    def has_category(self, name: str) -> bool:
        return any(a.category_name == name for a in self.annotations)


def has_table(page: Page) -> bool:
    return page.has_category(TABLE_LABEL)


def _coco_to_pages(coco: dict) -> list[Page]:
    id2name = {c["id"]: c["name"] for c in coco["categories"]}
    by_image: dict[int, list[Annotation]] = {}
    for a in coco["annotations"]:
        by_image.setdefault(a["image_id"], []).append(
            Annotation(
                category_id=a["category_id"],
                category_name=id2name[a["category_id"]],
                bbox=tuple(a["bbox"]),
            )
        )
    return [
        Page(
            image_id=img["id"],
            file_name=img["file_name"],
            width=img["width"],
            height=img["height"],
            annotations=by_image.get(img["id"], []),
            doc_category=img.get("doc_category"),
            page_no=img.get("page_no"),
        )
        for img in coco["images"]
    ]


def iter_coco_pages(
    source: Path | str = DEFAULT_SOURCE,
    split: str = "validation",
) -> Iterator[Page]:
    """Yield Page records from ``<source>/COCO/<split>.json``.

    Image bytes aren't loaded; PNGs live in ``<source>/PNG/<file_name>`` and
    are copied (or symlinked) lazily by the caller.
    """
    source = Path(source)
    coco_path = source / "COCO" / SPLIT_FILES[split]
    with coco_path.open() as f:
        coco = json.load(f)
    yield from _coco_to_pages(coco)


def load_local_subset(data_dir: Path | str) -> list[Page]:
    """Read a COCO-style subset written by ``prepare_subset.py``."""
    data_dir = Path(data_dir)
    with (data_dir / "annotations.json").open() as f:
        coco = json.load(f)
    return _coco_to_pages(coco)
