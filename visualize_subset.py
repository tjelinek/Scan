"""Generate a clickable HTML viewer for the local DocLayNet subset.

Reads ``<data-dir>/annotations.json`` and writes a single self-contained
HTML page that opens directly via ``file://`` — no server needed. Pages
are listed in COCO order; navigate with the on-screen Prev/Next buttons,
the arrow keys, or by typing a page number. Bounding boxes are color-coded
by category and can be toggled per-class via the legend.

If ``--predictions <results.jsonl>`` is given, predictions from a
``scanning-benchmark`` run are overlaid on the same page (dashed strokes,
same category colors) and the predicted markdown is shown in a side
panel. PP-DocLayoutV3's native classes are mapped onto DocLayNet's via
``evaluate_run.DEFAULT_CLASS_MAP`` so predictions and GT share a legend.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser
from pathlib import Path

from evaluate_run import DEFAULT_CLASS_MAP, load_results

# Browser-side template. ``__DATA__``, ``__PRED__``, ``__IMG_PREFIX__`` are
# substituted in by main(). Keep curly braces literal — we use str.replace,
# not format.
HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DocLayNet subset viewer</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; background: #1e1e1e; color: #ddd; }
  header { padding: 8px 12px; background: #2a2a2a; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; position: sticky; top: 0; z-index: 1; }
  header button { padding: 4px 12px; cursor: pointer; background: #444; color: #ddd; border: 1px solid #666; border-radius: 3px; }
  header button:hover { background: #555; }
  header input[type=number] { width: 70px; background: #1e1e1e; color: #ddd; border: 1px solid #666; padding: 3px; }
  #meta { margin-left: auto; font-size: 12px; opacity: 0.8; }
  main { display: flex; gap: 12px; padding: 12px; align-items: flex-start; }
  #stage { flex: 1; position: relative; max-width: 1100px; background: #000; }
  #stage img { width: 100%; display: block; }
  #stage svg { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
  #stage svg rect { fill: none; stroke-width: 2; }
  #stage svg g.pred rect { stroke-width: 2.5; stroke-dasharray: 6 4; }
  #side { width: 260px; display: flex; flex-direction: column; gap: 12px; position: sticky; top: 56px; max-height: calc(100vh - 80px); }
  .panel { background: #2a2a2a; padding: 10px; border-radius: 4px; font-size: 13px; }
  .panel h4 { margin: 0 0 8px 0; }
  .panel label { display: flex; align-items: center; gap: 6px; padding: 2px 0; cursor: pointer; }
  .panel .swatch { display: inline-block; width: 18px; height: 12px; border-radius: 2px; flex-shrink: 0; }
  .panel .swatch.dashed { border: 2px dashed; height: 0; padding: 0; background: transparent !important; border-radius: 0; }
  .panel .count { margin-left: auto; opacity: 0.6; font-variant-numeric: tabular-nums; }
  #md { flex: 1; min-height: 100px; overflow: auto; white-space: pre-wrap; font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; line-height: 1.4; background: #181818; }
  #md:empty::before { content: "(no markdown for this page)"; opacity: 0.5; font-style: italic; }
</style>
</head>
<body>
<header>
  <button id="prev">&#9664; Prev</button>
  <span><input id="idx" type="number" min="1"> / <span id="total"></span></span>
  <button id="next">Next &#9654;</button>
  <span id="meta"></span>
</header>
<main>
  <div id="stage">
    <img id="img" alt="">
    <svg id="boxes" preserveAspectRatio="none">
      <g class="gt" id="gt-layer"></g>
      <g class="pred" id="pred-layer"></g>
    </svg>
  </div>
  <aside id="side">
    <div class="panel" id="legend-gt"><h4>Ground truth</h4><div id="cats-gt"></div></div>
    <div class="panel" id="legend-pred" style="display:none"><h4>Predictions</h4><div id="cats-pred"></div></div>
    <div class="panel" id="md-panel" style="display:none"><h4>Predicted markdown</h4><div id="md"></div></div>
  </aside>
</main>
<script id="data" type="application/json">__DATA__</script>
<script id="pred" type="application/json">__PRED__</script>
<script>
(() => {
  const IMG_PREFIX = "__IMG_PREFIX__";
  const data = JSON.parse(document.getElementById('data').textContent);
  const predRaw = JSON.parse(document.getElementById('pred').textContent);
  const imgs = data.images;
  const cats = new Map(data.categories.map(c => [c.id, c.name]));

  const annsByImage = new Map();
  for (const a of data.annotations) {
    if (!annsByImage.has(a.image_id)) annsByImage.set(a.image_id, []);
    annsByImage.get(a.image_id).push(a);
  }

  // Pred annotations and per-page markdown. predRaw is null when the user
  // didn't pass --predictions.
  const predByImage = new Map();
  const mdByImage = new Map();
  if (predRaw) {
    for (const a of predRaw.annotations) {
      if (!predByImage.has(a.image_id)) predByImage.set(a.image_id, []);
      predByImage.get(a.image_id).push(a);
    }
    for (const [k, v] of Object.entries(predRaw.markdown)) {
      mdByImage.set(+k, v);
    }
  }

  const ids = [...cats.keys()].sort((a, b) => a - b);
  const colors = new Map(ids.map((id, i) => [id, `hsl(${Math.floor(i * 360 / ids.length)} 80% 60%)`]));
  const hiddenGt = new Set();
  const hiddenPred = new Set();

  function buildLegend(containerId, entries, hiddenSet, kind) {
    const el = document.getElementById(containerId);
    for (const id of ids) {
      const row = document.createElement('label');
      const swatchClass = kind === 'pred' ? 'swatch dashed' : 'swatch';
      const swatchStyle = kind === 'pred'
        ? `border-color:${colors.get(id)}`
        : `background:${colors.get(id)}`;
      row.innerHTML =
        `<input type="checkbox" checked data-cat="${id}">` +
        `<span class="${swatchClass}" style="${swatchStyle}"></span>` +
        `<span>${cats.get(id)}</span>` +
        `<span class="count">${entries.get(id) || 0}</span>`;
      el.appendChild(row);
    }
    el.addEventListener('change', e => {
      const id = +e.target.dataset.cat;
      if (e.target.checked) hiddenSet.delete(id); else hiddenSet.add(id);
      render();
    });
  }

  const gtCounts = new Map();
  for (const a of data.annotations) gtCounts.set(a.category_id, (gtCounts.get(a.category_id) || 0) + 1);
  buildLegend('cats-gt', gtCounts, hiddenGt, 'gt');

  if (predRaw) {
    const predCounts = new Map();
    for (const a of predRaw.annotations) predCounts.set(a.category_id, (predCounts.get(a.category_id) || 0) + 1);
    buildLegend('cats-pred', predCounts, hiddenPred, 'pred');
    document.getElementById('legend-pred').style.display = '';
    document.getElementById('md-panel').style.display = '';
  }

  let cursor = 0;
  const totalEl = document.getElementById('total');
  const idxEl = document.getElementById('idx');
  const metaEl = document.getElementById('meta');
  const imgEl = document.getElementById('img');
  const svgEl = document.getElementById('boxes');
  const gtLayer = document.getElementById('gt-layer');
  const predLayer = document.getElementById('pred-layer');
  const mdEl = document.getElementById('md');
  totalEl.textContent = imgs.length;

  function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

  function rectsHtml(anns, hiddenSet) {
    return anns
      .filter(a => !hiddenSet.has(a.category_id))
      .map(a => {
        const [x, y, w, h] = a.bbox;
        const c = colors.get(a.category_id);
        const tip = a.native_label
          ? `${cats.get(a.category_id) || ''} (${a.native_label})`
          : (cats.get(a.category_id) || '');
        return `<rect x="${x}" y="${y}" width="${w}" height="${h}" stroke="${c}"><title>${escapeHtml(tip)}</title></rect>`;
      })
      .join('');
  }

  function render() {
    const img = imgs[cursor];
    idxEl.value = cursor + 1;
    metaEl.textContent = `${img.file_name}  ·  ${img.width}×${img.height}  ·  ${img.doc_category ?? ''}  ·  page ${img.page_no ?? '?'}`;
    imgEl.src = `${IMG_PREFIX}/${img.file_name}`;
    svgEl.setAttribute('viewBox', `0 0 ${img.width} ${img.height}`);

    const gtAnns = annsByImage.get(img.id) || [];
    gtLayer.innerHTML = rectsHtml(gtAnns, hiddenGt);

    if (predRaw) {
      const predAnns = predByImage.get(img.id) || [];
      predLayer.innerHTML = rectsHtml(predAnns, hiddenPred);
      mdEl.textContent = mdByImage.get(img.id) || '';
    }
  }

  function go(delta) { cursor = (cursor + delta + imgs.length) % imgs.length; render(); }
  document.getElementById('prev').onclick = () => go(-1);
  document.getElementById('next').onclick = () => go(1);
  idxEl.addEventListener('change', () => {
    cursor = Math.max(1, Math.min(imgs.length, +idxEl.value || 1)) - 1;
    render();
  });
  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT') return;
    if (e.key === 'ArrowLeft') go(-1);
    else if (e.key === 'ArrowRight') go(1);
  });

  render();
})();
</script>
</body>
</html>
"""


def _build_pred_payload(
    results_path: Path,
    name2cat: dict[str, int],
    valid_image_ids: set[int],
) -> dict:
    """Read a results JSONL and return ``{annotations, markdown}`` for the viewer.

    Annotations are flattened to GT's COCO shape (bbox in xywh, category_id
    looked up via DEFAULT_CLASS_MAP). Markdown is keyed by image_id (str
    keys for JSON-friendliness — JS converts back to int).
    """
    records = load_results(results_path)
    annotations: list[dict] = []
    markdown: dict[str, str] = {}
    for rec in records:
        img_id = rec.get("image_id")
        if img_id not in valid_image_ids:
            continue
        if rec.get("markdown"):
            markdown[str(img_id)] = rec["markdown"]
        layout = rec.get("layout") or []
        regions = layout[0] if layout and isinstance(layout[0], list) else layout
        for r in regions or []:
            native = r.get("native_label") or r.get("label")
            mapped = DEFAULT_CLASS_MAP.get(native)
            if mapped is None:
                continue
            cat_id = name2cat.get(mapped)
            if cat_id is None:
                continue
            x1, y1, x2, y2 = r["bbox_2d"]
            annotations.append(
                {
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "native_label": native,
                }
            )
    return {"annotations": annotations, "markdown": markdown}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output HTML path (default: <data-dir>/index.html)",
    )
    p.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="overlay a results JSONL written by scanning-benchmark",
    )
    p.add_argument("--open", action="store_true", help="open the result in a browser")
    args = p.parse_args(argv)

    ann_path = args.data_dir / "annotations.json"
    if not ann_path.exists():
        print(f"missing {ann_path}; run scanning-prepare first", file=sys.stderr)
        return 1

    out = args.out or (args.data_dir / "index.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    coco = json.loads(ann_path.read_text())

    img_prefix = os.path.relpath(args.data_dir / "images", out.parent).replace(os.sep, "/")

    pred_payload: dict | None = None
    if args.predictions is not None:
        if not args.predictions.exists():
            print(f"missing {args.predictions}", file=sys.stderr)
            return 1
        name2cat = {c["name"]: c["id"] for c in coco["categories"]}
        valid_ids = {img["id"] for img in coco["images"]}
        pred_payload = _build_pred_payload(args.predictions, name2cat, valid_ids)

    # Inline JSON inside <script>: a literal "</" would close the tag early.
    def _embed(obj) -> str:
        return json.dumps(obj).replace("</", "<\\/")

    html = (
        HTML_TEMPLATE
        .replace("__DATA__", _embed(coco))
        .replace("__PRED__", _embed(pred_payload))
        .replace("__IMG_PREFIX__", img_prefix)
    )
    out.write_text(html)

    n_pages = len(coco.get("images", []))
    if pred_payload is None:
        print(f"wrote {out} ({n_pages} pages)")
    else:
        print(
            f"wrote {out} ({n_pages} pages, "
            f"{len(pred_payload['annotations'])} predictions, "
            f"{len(pred_payload['markdown'])} markdown blobs)"
        )
    if args.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
