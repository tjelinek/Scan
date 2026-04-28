"""Score a benchmark run against the local DocLayNet subset.

Loads ``<data-dir>/annotations.json`` (ground truth) and a results JSONL
written by ``scanning-benchmark``, maps PP-DocLayoutV3's native classes
onto DocLayNet's, and reports COCO-style mAP via ``pycocotools``.

DocLayNet upstream ships no evaluation tooling, so this is the project's
own scorer. Predictions don't carry per-region confidence yet (the SDK
strips it after threshold-filtering), so every detection is scored
**1.0** uniformly. Under that assumption mAP collapses toward
precision/recall at the model's fixed threshold — useful for relative
A/B between runs, **not directly comparable to leaderboard mAP numbers**.

Run via the console script::

    scanning-evaluate                        # newest results/*.jsonl vs data/
    scanning-evaluate --results results/run-20260428-1530.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# PP-DocLayoutV3 native label → DocLayNet class name. PP-DocLayoutV3 has 25
# classes; we collapse onto DocLayNet's 11. DocLayNet's "List-item" has no
# direct counterpart in PP-DocLayoutV3, so its recall will always be 0
# under this mapping — known class-mismatch limitation.
DEFAULT_CLASS_MAP: dict[str, str] = {
    "abstract": "Text",
    "algorithm": "Text",
    "aside_text": "Text",
    "chart": "Picture",
    "content": "Text",
    "display_formula": "Formula",
    "doc_title": "Title",
    "figure_title": "Caption",
    "footer": "Page-footer",
    "footnote": "Footnote",
    "formula_number": "Formula",
    "header": "Page-header",
    "image": "Picture",
    "inline_formula": "Formula",
    "number": "Page-footer",
    "paragraph_title": "Section-header",
    "reference": "Text",
    "reference_content": "Text",
    "seal": "Picture",
    "table": "Table",
    "text": "Text",
    "vertical_text": "Text",
    "vision_footnote": "Footnote",
}


def newest_run(results_dir: Path) -> Path | None:
    runs = sorted(results_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def load_results(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def predictions_to_coco_results(
    records: list[dict],
    name2cat: dict[str, int],
    class_map: dict[str, str] = DEFAULT_CLASS_MAP,
    valid_image_ids: set[int] | None = None,
) -> tuple[list[dict], dict[str, int], dict[str, int], int]:
    """Flatten benchmark predictions into the COCO results format.

    Returns ``(coco_results, skipped_unmapped, skipped_unknown, pages_used)``.
    Every detection is scored 1.0 — see module docstring for why.
    """
    preds: list[dict] = []
    skipped_unmapped: dict[str, int] = {}
    skipped_unknown: dict[str, int] = {}
    pages_used = 0
    for rec in records:
        if "error" in rec or not rec.get("layout"):
            continue
        if valid_image_ids is not None and rec["image_id"] not in valid_image_ids:
            continue
        # SDK wraps regions in a per-page list; results.layout is [[regions]].
        layout = rec["layout"]
        regions = layout[0] if layout and isinstance(layout[0], list) else layout
        if not regions:
            continue
        pages_used += 1
        for r in regions:
            native = r.get("native_label") or r.get("label")
            mapped = class_map.get(native)
            if mapped is None:
                skipped_unmapped[native] = skipped_unmapped.get(native, 0) + 1
                continue
            cat_id = name2cat.get(mapped)
            if cat_id is None:
                skipped_unknown[mapped] = skipped_unknown.get(mapped, 0) + 1
                continue
            x1, y1, x2, y2 = r["bbox_2d"]
            preds.append(
                {
                    "image_id": rec["image_id"],
                    "category_id": cat_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": 1.0,
                }
            )
    return preds, skipped_unmapped, skipped_unknown, pages_used


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument(
        "--results",
        type=Path,
        default=None,
        help="results JSONL path (default: newest under results/)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    gt_path = args.data_dir / "annotations.json"
    if not gt_path.exists():
        print(f"missing {gt_path}; run scanning-prepare first", file=sys.stderr)
        return 1

    results_path = args.results or newest_run(Path("results"))
    if results_path is None or not results_path.exists():
        print("no results jsonl found; run scanning-benchmark first", file=sys.stderr)
        return 1

    # pycocotools pulls in numpy + a C extension — keep the import lazy.
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = json.loads(gt_path.read_text())
    name2cat = {c["name"]: c["id"] for c in gt["categories"]}
    cat2name = {v: k for k, v in name2cat.items()}
    gt_image_ids = {img["id"] for img in gt["images"]}

    # pycocotools requires iscrowd and area on every GT annotation. Backfill
    # for subsets built by prepare_subset.py before iscrowd/area were emitted.
    for ann in gt["annotations"]:
        ann.setdefault("iscrowd", 0)
        if "area" not in ann:
            x, y, w, h = ann["bbox"]
            ann["area"] = float(w) * float(h)

    records = load_results(results_path)
    preds, unmapped, unknown, pages_used = predictions_to_coco_results(
        records, name2cat, valid_image_ids=gt_image_ids
    )
    if not preds:
        print("no usable predictions after class mapping; aborting", file=sys.stderr)
        return 1

    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(preds)

    e = COCOeval(coco_gt, coco_dt, iouType="bbox")
    # Restrict to images we actually scored, so missing pages don't zero
    # out the mean.
    e.params.imgIds = sorted({p["image_id"] for p in preds})
    e.evaluate()
    e.accumulate()
    e.summarize()

    # Per-class AP @[.5:.95]. precision shape: [T, R, K, A, M]
    # (T=10 IoU thresholds, K=cats, A=areas (all,small,med,large), M=maxDets).
    print()
    print("Per-class AP @[.5:.95]")
    print("-" * 36)
    cat_ids_sorted = sorted(name2cat.values())
    precision = e.eval["precision"]
    for k_idx, cat_id in enumerate(cat_ids_sorted):
        p_k = precision[:, :, k_idx, 0, -1]
        ap = float(p_k[p_k > -1].mean()) if (p_k > -1).any() else float("nan")
        ap_str = "  n/a" if ap != ap else f"{ap:.3f}"  # NaN check
        print(f"  {cat2name[cat_id]:<18s} {ap_str}")

    if unmapped:
        n = sum(unmapped.values())
        print(
            f"\nskipped {n} preds with unmapped native_labels: "
            f"{dict(sorted(unmapped.items()))}",
            file=sys.stderr,
        )
    if unknown:
        n = sum(unknown.values())
        print(
            f"skipped {n} preds whose mapped class is absent in GT: "
            f"{dict(sorted(unknown.items()))}",
            file=sys.stderr,
        )
    print(f"\nresults: {results_path}", file=sys.stderr)
    print(f"pages with preds: {pages_used}", file=sys.stderr)
    print(f"predictions used: {len(preds)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
