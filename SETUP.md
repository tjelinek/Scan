# Setup

Fresh-machine instructions for running this project end-to-end. Keep in sync
with `pyproject.toml`. The Changelog at the bottom records notable setup
decisions and changes — append entries instead of rewriting history.

## Prerequisites

- **Python 3.12+** (the venv in this repo uses 3.12.3).
- **Ollama 0.21.2+** for hosting the GLM-OCR model locally. Older builds
  (e.g. 0.3.x) fail to pull `glm-ocr`. Install options below — prefer the
  user-level install unless you're setting up a shared machine.
- **GPU is optional.** With a small GPU (≤ 4 GB VRAM) plan to run layout
  detection on CPU and rely on Ollama to offload OCR weights as needed. With
  an 8 GB+ GPU you can move layout to CUDA for speed.
- **Disk:** budget ~6–8 GB for Python deps + layout model, ~2 GB for the
  Ollama OCR model, ~30 GB for the extracted `DocLayNet_core` (PNG + COCO).
  The working subset under `data/images/` is symlinked into the extracted
  PNGs and adds essentially no disk.

## One-time setup

```bash
# 1. Create venv and install the project (editable)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"

# 2. Install Ollama — pick A or B
```

### A. User-level Ollama (no sudo required) — recommended

Downloads the Ollama tarball to `~/.local`. Runs on **port 11435** to avoid
clashing with any system Ollama on :11434. `config.yaml` already points at
11435.

```bash
mkdir -p ~/.local ~/.ollama
curl -fSL https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst \
  -o /tmp/ollama.tar.zst
tar --zstd -xf /tmp/ollama.tar.zst -C ~/.local
rm /tmp/ollama.tar.zst
export PATH=$HOME/.local/bin:$PATH   # add to ~/.bashrc

# Start server on :11435 (leave running in another terminal, or launch once):
OLLAMA_HOST=127.0.0.1:11435 ollama serve &

# Pull the model (tell client to target :11435):
OLLAMA_HOST=127.0.0.1:11435 ollama pull glm-ocr
```

Heads up: the Linux tarball is ~2 GB (bundles CUDA runtime libs). Model is
another ~2 GB.

### B. System-wide Ollama (needs sudo)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull glm-ocr       # uses default :11434
```

Then change `config.yaml`: set `ocr_api.api_port: 11434`.

## Get DocLayNet locally

The dev machine has no direct HuggingFace access, so we use the official IBM
zip release rather than streaming. Download once on a machine with network
access and copy it across.

1. Grab `DocLayNet_core.zip` (~28 GB) from the [DocLayNet README's Download
   section](https://github.com/DS4SD/DocLayNet#downloads). Only `_core` is
   used; `DocLayNet_extra.zip` (per-page JSON + PDFs) isn't needed for the
   benchmark.
2. Extract it so the layout is:
   ```
   data/raw/DocLayNet_core/
     COCO/{train,val,test}.json
     PNG/*.png
   ```
   `data/` is gitignored. Any other location works too — pass it via
   `--source`.

## Stage the layout model

`glmocr` runs `PaddlePaddle/PP-DocLayoutV3_safetensors` to crop regions
before each OCR call. This dev machine has no HuggingFace access at
runtime, so the model files are staged on disk and loaded with
`HF_HUB_OFFLINE=1` (set automatically by `glm_ocr.py`). On a machine that
*does* have HF access, fetch the snapshot and copy the resulting
directory across:

```bash
# On a connected machine — model is ~80 MB.
hf download PaddlePaddle/PP-DocLayoutV3_safetensors \
  --local-dir models/PP-DocLayoutV3
# Then scp/rsync the whole models/ folder onto this machine.
```

`config.yaml` already points `pipeline.layout.model_dir` at
`models/PP-DocLayoutV3` (relative to the repo root). Override the path
there if you stage it elsewhere — absolute paths are honoured.

The benchmark errors out with a clear "Layout model not found at …"
message if `models/PP-DocLayoutV3/config.json` is missing, instead of
hanging on a network call.

## Build the working subset

```bash
# Reads <source>/COCO/val.json, drops pages containing tables, links the
# first 200 PNGs into data/images/, and writes data/annotations.json.
scanning-prepare --split validation --max-pages 200 --no-tables
# or with a non-default source:
scanning-prepare --source ~/datasets/DocLayNet_core --max-pages 200 --no-tables
```

PNGs are symlinked rather than copied (falls back to copy if the filesystem
disallows symlinks), so the subset stays small.

## Run the benchmark

```bash
# Assumes Ollama is up and `scanning-prepare` has populated data/.
scanning-benchmark             # --out defaults to results/run-<timestamp>.jsonl
```

Each output record contains the page id, elapsed time, the model's markdown,
and the predicted layout JSON.

## Troubleshooting

- **`ollama pull glm-ocr` fails with HTTP 412** — Ollama is too old
  (0.3.x). Install 0.21.2+ via path A or B above.
- **Layout model OOM on GPU** — set `GLMOCR_LAYOUT_DEVICE=cpu` before
  running. `config.yaml` already defaults to CPU.
- **Ollama not reachable** — check `curl http://127.0.0.1:11435/api/tags`
  (or :11434 for the system install). Start with
  `OLLAMA_HOST=127.0.0.1:11435 ollama serve`.
- **GPU not detected by user-level Ollama** — the tarball ships its own
  CUDA libs but may miss the NVML shim. If inference falls back to CPU
  unexpectedly, check `tail /tmp/ollama-user.log` for GPU discovery
  messages and ensure the NVIDIA driver matches the bundled CUDA major
  version.

## Changelog

- **2026-04-24** — Initial setup. Python 3.12 venv. Chose Ollama over vLLM
  for the OCR backend because the reference machine has only 4 GB VRAM
  (vLLM's KV cache + CUDA graphs push past that). `glmocr[selfhosted]`
  extra covers layout detection locally; no cloud MaaS (`ZHIPU_API_KEY`
  not required). DocLayNet accessed via `datasets` streaming instead of
  the 28 GB S3 zip.
- **2026-04-24** — `datasets` pinned to `<4.0`. The `ds4sd/DocLayNet` HF
  repo uses a legacy script loader (`DocLayNet.py`); `datasets>=4.0`
  refuses to execute scripts. `datasets` 3.x still supports them behind
  `trust_remote_code=True`. Acceptable because DS4SD is a trusted IBM
  publisher; revisit if a parquet mirror becomes available.
- **2026-04-24** — Feature override in `doclaynet._doclaynet_features()`:
  the upstream loader types `objects.area` as `int64` but some annotations
  carry float areas (e.g. `14665.858197`), triggering an Arrow cast error
  on streaming. We cast to `float64`. Class names list is mirrored
  verbatim; if upstream renumbers, update that constant.
- **2026-04-24** — User-level Ollama install (no sudo). Tarball extracted
  to `~/.local`, server runs on **port 11435** so it coexists with any
  system Ollama on :11434. `config.yaml` defaults to :11435. Ollama 0.21.2
  is the version used for development; upgrade the tarball opportunistically.
- **2026-04-27** — Dropped HuggingFace streaming. The dev environment has
  no direct HF access, so the loader now reads the official IBM
  `DocLayNet_core.zip` (PNG + COCO) from disk. `datasets` and `requests`
  removed from dependencies. `download_subset.py` renamed to
  `prepare_subset.py`; console script `scanning-download` →
  `scanning-prepare`. Default `--source` is `data/raw/DocLayNet_core`.
- **2026-04-27** — Layout model is now loaded from a local snapshot
  instead of HF on first use. `config.yaml`'s
  `pipeline.layout.model_dir` defaults to `models/PP-DocLayoutV3`
  (gitignored). `glm_ocr.GLMOCRAdapter` sets `HF_HUB_OFFLINE=1` and
  `TRANSFORMERS_OFFLINE=1` before importing `glmocr`, and validates that
  `<model_dir>/config.json` exists up front so a missing snapshot fails
  fast with a pointer to this section instead of hanging on a network
  call.
