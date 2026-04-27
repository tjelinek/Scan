# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo does

Benchmark harness for running [GLM-OCR](https://github.com/zai-org/GLM-OCR) on
a filtered subset of [DocLayNet](https://github.com/DS4SD/DocLayNet). Two
moving parts: a DocLayNet loader/filter, and an adapter that wraps the
`glmocr` SDK so the benchmark runner doesn't touch SDK internals.

Full install + fresh-machine instructions (and a dated change log) live in
[SETUP.md](SETUP.md). This file is for orienting future Claude sessions, not
for end users.

## Layout

Flat — top-level modules, no package. The repo doubles as a small experiment
harness, so simplicity beats packaging ceremony.

```
doclaynet.py             — DocLayNet loader (raw COCO + local subset) and filters
glm_ocr.py               — GLMOCRAdapter: context-manager wrapper around glmocr.GlmOcr
prepare_subset.py        — reads extracted DocLayNet_core, filters, writes data/ subset
run_benchmark.py         — iterates local subset, calls adapter, writes JSONL
config.yaml              — glmocr runtime config; points OCR backend at Ollama
data/raw/                — gitignored; extracted DocLayNet_core/ (PNG + COCO)
data/                    — gitignored; subset output of prepare_subset.py
results/                 — gitignored; JSONL benchmark runs
```

Console scripts (declared in `pyproject.toml` under `py-modules`):
`scanning-prepare`, `scanning-benchmark`.

## Architecture decisions worth knowing

- **OCR backend is Ollama, not vLLM.** `glmocr[selfhosted]` expects an
  external HTTP service for the OCR model. The dev machine has 4 GB VRAM, so
  vLLM is a non-starter; Ollama's quantized `glm-ocr:latest` fits. The SDK
  has native Ollama support via `api_mode: ollama_generate` — wired up in
  `config.yaml`.
- **Layout model runs on CPU.** Set in `config.yaml`
  (`pipeline.layout.device: cpu`). On machines with more VRAM, change to
  `cuda`.
- **DocLayNet is loaded from a locally-extracted `DocLayNet_core.zip`**, not
  HuggingFace. The dev environment can't reach HF, so the prep script reads
  `<source>/COCO/<split>.json` + `<source>/PNG/` and symlinks the kept PNGs
  into `data/images/`. The 28 GB raw archive sits under `data/raw/` (or
  wherever `--source` points); the working subset is just symlinks +
  `annotations.json`.
- **Category mapping is read at runtime** from the COCO `categories` list.
  Do not hardcode numeric ids — upstream ordering has changed before.

## Common commands

```bash
# setup (once)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ollama pull glm-ocr

# extract DocLayNet_core.zip into data/raw/ first (see SETUP.md)
# build a 200-page no-tables subset (val split)
scanning-prepare --split validation --max-pages 200 --no-tables

# run the benchmark (needs Ollama serving on :11435)
scanning-benchmark        # --out defaults to results/run-<timestamp>.jsonl

# dev
pytest                    # tests live alongside modules as they're added
ruff check .              # lint
```

## When extending this repo

- **Adding a new OCR backend**: drop another adapter module at the repo
  root with the same `__enter__`/`__exit__`/`infer(path) -> OCRPrediction`
  interface. The benchmark runner can stay generic.
- **Adding a metric**: the benchmark writes raw predictions to JSONL. A
  separate evaluation step (not yet implemented) should consume that JSONL
  plus the subset's `annotations.json`. Keep benchmarking and scoring
  separate so scoring can be iterated without re-running the model.
- **Schema surprises**: COCO files from `DocLayNet_core` are flattened by
  `_coco_to_pages` in `doclaynet.py`. If a future release changes the field
  shape, fix it there, not at call sites.

## Maintenance

Update this file when the architecture changes (new adapter, new backend,
new stage in the pipeline). For install/environment changes, update
`SETUP.md` and add a dated entry to its Changelog.
