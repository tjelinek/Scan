"""Download a filtered DocLayNet subset from HuggingFace.

Streams the requested split, applies a filter (default: drop any page whose
annotations contain a Table), and saves the first ``--max-pages`` that pass
under ``--out``:

    out/
      images/<file_name>.png
      annotations.json        # COCO-style

Run via the console script defined in pyproject.toml::

    scanning-download --split validation --max-pages 200 --no-tables
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

from doclaynet import TABLE_LABEL, has_table, stream_pages


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    p.add_argument("--max-pages", type=int, default=200)
    p.add_argument("--no-tables", action="store_true", help=f"drop pages containing '{TABLE_LABEL}'")
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--repo-id", default="ds4sd/DocLayNet")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    img_dir = args.out / "images"
    img_dir.mkdir(exist_ok=True)

    kept_images: list[dict] = []
    kept_anns: list[dict] = []
    categories: dict[int, str] = {}
    ann_uid = 1

    pbar = tqdm(total=args.max_pages, desc="pages kept", unit="page")
    scanned = 0
    for page, image in stream_pages(split=args.split, repo_id=args.repo_id):
        scanned += 1
        if args.no_tables and has_table(page):
            continue

        img_path = img_dir / page.file_name
        if not img_path.exists():
            image.save(img_path)

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
            "source": args.repo_id,
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
