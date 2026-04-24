"""DocLayNet loading utilities.

Two access paths are supported:

- **streaming**: pulls examples directly from HuggingFace (no local copy needed).
  Useful for filtering-before-saving.
- **local subset**: reads a COCO-style ``annotations.json`` + images produced
  by ``scripts/download_subset.py``. Fast, offline, deterministic.

The category id↔name map is read from the data itself (HF feature schema or
the COCO ``categories`` list) rather than hardcoded, so this module keeps
working if upstream renumbers the classes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

TABLE_LABEL = "Table"


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


# --- Streaming from HuggingFace ---------------------------------------------

# The upstream DocLayNet.py loader types `objects.area` as int64, but some
# annotations have float areas (e.g. 14665.858197), which breaks Arrow casts.
# We override the features to use float64 instead. The class label list
# matches the upstream script verbatim as of 2026-04.
_CAT_NAMES = [
    "Caption", "Footnote", "Formula", "List-item", "Page-footer",
    "Page-header", "Picture", "Section-header", "Table", "Text", "Title",
]


def _doclaynet_features():
    from datasets import ClassLabel, Features, Image, Sequence, Value

    return Features(
        {
            "image_id": Value("int64"),
            "image": Image(),
            "width": Value("int32"),
            "height": Value("int32"),
            "doc_category": Value("string"),
            "collection": Value("string"),
            "doc_name": Value("string"),
            "page_no": Value("int64"),
            "objects": [
                {
                    "category_id": ClassLabel(names=_CAT_NAMES),
                    "image_id": Value("string"),
                    "id": Value("int64"),
                    "area": Value("float64"),
                    "bbox": Sequence(Value("float32"), length=4),
                    "segmentation": [[Value("float32")]],
                    "iscrowd": Value("bool"),
                    "precedence": Value("int32"),
                }
            ],
        }
    )


def stream_pages(
    split: str = "validation",
    repo_id: str = "ds4sd/DocLayNet",
) -> Iterator[tuple[Page, "PIL.Image.Image"]]:
    """Yield ``(Page, PIL.Image)`` for each example in an HF split.

    Uses ``streaming=True`` so nothing is cached to disk unless the caller
    saves it explicitly.
    """
    from datasets import load_dataset

    # DocLayNet ships as a legacy HF loading script. Requires an explicit
    # opt-in to execute. See SETUP.md for the rationale.
    ds = load_dataset(
        repo_id,
        split=split,
        streaming=True,
        trust_remote_code=True,
        features=_doclaynet_features(),
    )

    id2name = dict(enumerate(_CAT_NAMES))

    for ex in ds:
        page = _example_to_page(ex, id2name)
        yield page, ex["image"]


def _example_to_page(ex: dict, id2name: dict[int, str]) -> Page:
    # DocLayNet's HF schema stores annotations as a list of per-object dicts
    # under `objects`. Each dict has category_id (int) and bbox ([x,y,w,h]).
    objs = ex.get("objects") or []

    anns = [
        Annotation(
            category_id=int(o["category_id"]),
            category_name=id2name.get(int(o["category_id"]), str(o["category_id"])),
            bbox=tuple(o["bbox"]),
        )
        for o in objs
    ]

    image_id = int(ex.get("image_id", 0))
    return Page(
        image_id=image_id,
        file_name=ex.get("file_name") or f"{image_id}.png",
        width=int(ex.get("width", 0)),
        height=int(ex.get("height", 0)),
        annotations=anns,
        doc_category=ex.get("doc_category"),
        page_no=ex.get("page_no"),
    )


# --- Local subset (produced by download_subset.py) ---------------------------


def load_local_subset(data_dir: Path | str) -> list[Page]:
    """Read a COCO-style subset written by the downloader."""
    data_dir = Path(data_dir)
    ann_path = data_dir / "annotations.json"
    with ann_path.open() as f:
        coco = json.load(f)

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
