"""Thin adapter around ``glmocr.GlmOcr`` for benchmarking.

Normalises the model's output into a small dict that the benchmark runner
writes to JSONL, and isolates the SDK wiring (config path, context-manager
lifecycle) so the rest of the code doesn't care which backend hosts the
OCR model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


@dataclass
class OCRPrediction:
    markdown: str | None
    layout: Any  # json_result is str | dict | list depending on backend


class GLMOCRAdapter:
    """Context-manager wrapper: ``with GLMOCRAdapter() as ocr: ocr.infer(...)``."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG):
        self._config_path = str(config_path)
        self._parser = None

    def __enter__(self) -> "GLMOCRAdapter":
        from glmocr import GlmOcr

        self._parser = GlmOcr(config_path=self._config_path)
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
