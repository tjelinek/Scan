"""Run the GLM-OCR adapter over a local DocLayNet subset and write results.

Defaults work out-of-the-box once ``scanning-prepare`` has populated
``data/`` — just run ``scanning-benchmark`` and a timestamped JSONL lands
under ``results/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from doclaynet import load_local_subset
from glm_ocr import GLMOCRAdapter


def _default_out() -> Path:
    return Path("results") / f"run-{datetime.now():%Y%m%d-%H%M%S}.jsonl"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output JSONL path (default: results/run-<timestamp>.jsonl)",
    )
    p.add_argument("--limit", type=int, default=None, help="process at most N pages")
    p.add_argument("--config", type=Path, default=None, help="glmocr config.yaml override")
    args = p.parse_args(argv)
    if args.out is None:
        args.out = _default_out()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pages = load_local_subset(args.data_dir)
    if args.limit is not None:
        pages = pages[: args.limit]
    if not pages:
        print(f"No pages found under {args.data_dir}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    img_dir = args.data_dir / "images"

    adapter_kwargs = {"config_path": args.config} if args.config else {}
    errors = 0

    with GLMOCRAdapter(**adapter_kwargs) as ocr, args.out.open("w") as f:
        for page in tqdm(pages, desc="benchmark", unit="page"):
            img_path = img_dir / page.file_name
            t0 = time.perf_counter()
            try:
                pred = ocr.infer(img_path)
                record = {
                    "image_id": page.image_id,
                    "file_name": page.file_name,
                    "elapsed_s": round(time.perf_counter() - t0, 3),
                    "markdown": pred.markdown,
                    "layout": pred.layout,
                }
            except Exception as e:
                errors += 1
                record = {
                    "image_id": page.image_id,
                    "file_name": page.file_name,
                    "elapsed_s": round(time.perf_counter() - t0, 3),
                    "error": f"{type(e).__name__}: {e}",
                }
            f.write(json.dumps(record) + "\n")

    print(f"wrote {args.out} ({len(pages)} pages, {errors} errors)")
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
