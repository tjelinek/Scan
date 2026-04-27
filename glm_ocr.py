"""Thin adapter around ``glmocr.GlmOcr`` for benchmarking.

Normalises the model's output into a small dict that the benchmark runner
writes to JSONL, and isolates the SDK wiring (config path, context-manager
lifecycle) so the rest of the code doesn't care which backend hosts the
OCR model.

Locks the SDK to fully offline operation: the layout model
(``PaddlePaddle/PP-DocLayoutV3_safetensors``) must already be staged on
disk under ``pipeline.layout.model_dir`` from ``config.yaml``. We set
``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` before importing ``glmocr``
so any accidental HuggingFace fetch errors out fast instead of hanging on
the network.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


@dataclass
class OCRPrediction:
    markdown: str | None
    layout: Any  # json_result is str | dict | list depending on backend


def _resolve_layout_model_dir(config_path: Path) -> Path:
    """Read ``pipeline.layout.model_dir`` from the SDK config and resolve it.

    Relative paths are resolved against the config file's parent so the
    benchmark works regardless of CWD.
    """
    import yaml

    cfg = yaml.safe_load(config_path.read_text()) or {}
    raw = cfg.get("pipeline", {}).get("layout", {}).get("model_dir")
    if not raw:
        raise RuntimeError(f"{config_path} is missing pipeline.layout.model_dir")
    p = Path(raw)
    return p if p.is_absolute() else (config_path.parent / p).resolve()


class GLMOCRAdapter:
    """Context-manager wrapper: ``with GLMOCRAdapter() as ocr: ocr.infer(...)``."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG):
        self._config_path = Path(config_path)
        self._parser = None

    def __enter__(self) -> "GLMOCRAdapter":
        # Block any HF network call before glmocr/transformers get a chance.
        # setdefault so the operator can still flip these off explicitly.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        layout_dir = _resolve_layout_model_dir(self._config_path)
        if not (layout_dir / "config.json").is_file():
            raise FileNotFoundError(
                f"Layout model not found at {layout_dir}. Stage it on a "
                f"machine with HuggingFace access (see SETUP.md → 'Stage the "
                f"layout model') and copy the directory across so "
                f"'config.json' lives at {layout_dir / 'config.json'}."
            )

        from glmocr import GlmOcr

        self._parser = GlmOcr(config_path=str(self._config_path))
        self._parser.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._parser is not None:
            self._parser.__exit__(exc_type, exc, tb)
            self._parser = None

    def infer(self, image_path: str | Path) -> OCRPrediction:
        if self._parser is None:
            raise RuntimeError("GLMOCRAdapter must be used as a context manager")
        result = self._parser.parse(str(image_path))
        return OCRPrediction(
            markdown=getattr(result, "markdown_result", None),
            layout=getattr(result, "json_result", None),
        )
