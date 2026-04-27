"""Build a filtered subset from a locally-extracted DocLayNet_core.

Reads ``<source>/COCO/<split>.json``, applies a filter (default: drop pages
whose annotations contain a Table), and links the matching PNGs from
``<source>/PNG/`` into ``<out>/images/``. Writes a slimmed COCO file at
``<out>/annotations.json`` so the benchmark runner can read the subset back.

Run via the console script defined in pyproject.toml::

    scanning-prepare --split validation --max-pages 200 --no-tables

The dev machine has no direct HuggingFace access, so the data has to be
fetched manually from https://github.com/DS4SD/DocLayNet#downloads —
``DocLayNet_core.zip`` is what we need (PNG + COCO). ``DocLayNet_extra.zip``
(per-page JSON + PDFs) is unused so far.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

from doclaynet import (
    DEFAULT_SOURCE,
    SPLIT_FILES,
    TABLE_LABEL,
    has_table,
    iter_coco_pages,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"extracted DocLayNet_core directory (default: {DEFAULT_SOURCE})",
    )
    p.add_argument("--split", default="validation", choices=sorted(SPLIT_FILES))
    p.add_argument("--max-pages", type=int, default=200)
    p.add_argument(
        "--no-tables", action="store_true", help=f"drop pages containing '{TABLE_LABEL}'"
    )
    p.add_argument("--out", type=Path, default=Path("data"))
    return p.parse_args(argv)


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copyfile(src, dst)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    coco_dir = args.source / "COCO"
    png_dir = args.source / "PNG"
    if not coco_dir.is_dir() or not png_dir.is_dir():
        print(
            f"Expected DocLayNet_core layout at {args.source} "
            f"(missing COCO/ or PNG/). Extract DocLayNet_core.zip there.",
            file=sys.stderr,
        )
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    img_out = args.out / "images"
    img_out.mkdir(exist_ok=True)

    kept_images: list[dict] = []
    kept_anns: list[dict] = []
    categories: dict[int, str] = {}
    ann_uid = 1

    pbar = tqdm(total=args.max_pages, desc="pages kept", unit="page")
    scanned = 0
    for page in iter_coco_pages(args.source, split=args.split):
        scanned += 1
        if args.no_tables and has_table(page):
            continue

        src_img = png_dir / page.file_name
        if not src_img.exists():
            tqdm.write(f"missing image, skipping: {src_img}")
            continue
        _link_or_copy(src_img, img_out / page.file_name)

        kept_images.append(
            {
                "id": page.image_id,
                "file_name": page.file_name,
                "width": page.width,
                "height": page.height,
                "doc_category": page.doc_category,
                "page_no": page.page_no,
            }
        )
        for ann in page.annotations:
            categories.setdefault(ann.category_id, ann.category_name)
            kept_anns.append(
                {
                    "id": ann_uid,
                    "image_id": page.image_id,
                    "category_id": ann.category_id,
                    "bbox": list(ann.bbox),
                }
            )
            ann_uid += 1

        pbar.update(1)
        if len(kept_images) >= args.max_pages:
            break
    pbar.close()

    coco = {
        "info": {
            "source": str(args.source),
            "split": args.split,
            "filter": "no_tables" if args.no_tables else "none",
            "scanned": scanned,
            "kept": len(kept_images),
        },
        "categories": [{"id": cid, "name": name} for cid, name in sorted(categories.items())],
        "images": kept_images,
        "annotations": kept_anns,
    }
    (args.out / "annotations.json").write_text(json.dumps(coco, indent=2))

    print(
        f"kept {len(kept_images)} pages "
        f"(scanned {scanned}, wrote {len(kept_anns)} annotations) "
        f"→ {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
