"""Generate a clickable HTML viewer for the local DocLayNet subset.

Reads ``<data-dir>/annotations.json`` and writes a single self-contained
HTML page that opens directly via ``file://`` — no server needed. Pages
are listed in COCO order; navigate with the on-screen Prev/Next buttons,
the arrow keys, or by typing a page number. Bounding boxes are color-coded
by category and can be toggled per-class via the legend.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser
from pathlib import Path

# Browser-side template. ``__DATA__`` and ``__IMG_PREFIX__`` are substituted
# in by main(). Keep curly braces literal — we use str.replace, not format.
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
  #legend { width: 220px; font-size: 13px; background: #2a2a2a; padding: 10px; border-radius: 4px; position: sticky; top: 56px; }
  #legend h4 { margin: 0 0 8px 0; }
  #legend label { display: flex; align-items: center; gap: 6px; padding: 2px 0; cursor: pointer; }
  #legend .swatch { display: inline-block; width: 12px; height: 12px; border-radius: 2px; }
  #legend .count { margin-left: auto; opacity: 0.6; font-variant-numeric: tabular-nums; }
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
    <svg id="boxes" preserveAspectRatio="none"></svg>
  </div>
  <aside id="legend"><h4>Categories</h4><div id="cats"></div></aside>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
(() => {
  const IMG_PREFIX = "__IMG_PREFIX__";
  const data = JSON.parse(document.getElementById('data').textContent);
  const imgs = data.images;
  const cats = new Map(data.categories.map(c => [c.id, c.name]));
  const annsByImage = new Map();
  for (const a of data.annotations) {
    if (!annsByImage.has(a.image_id)) annsByImage.set(a.image_id, []);
    annsByImage.get(a.image_id).push(a);
  }

  const ids = [...cats.keys()].sort((a, b) => a - b);
  const colors = new Map(ids.map((id, i) => [id, `hsl(${Math.floor(i * 360 / ids.length)} 80% 60%)`]));
  const hidden = new Set();

  const totalCounts = new Map();
  for (const a of data.annotations) totalCounts.set(a.category_id, (totalCounts.get(a.category_id) || 0) + 1);

  const legend = document.getElementById('cats');
  for (const id of ids) {
    const row = document.createElement('label');
    row.innerHTML = `<input type="checkbox" checked data-cat="${id}"><span class="swatch" style="background:${colors.get(id)}"></span><span>${cats.get(id)}</span><span class="count">${totalCounts.get(id) || 0}</span>`;
    legend.appendChild(row);
  }
  legend.addEventListener('change', e => {
    const id = +e.target.dataset.cat;
    if (e.target.checked) hidden.delete(id); else hidden.add(id);
    render();
  });

  let cursor = 0;
  const totalEl = document.getElementById('total');
  const idxEl = document.getElementById('idx');
  const metaEl = document.getElementById('meta');
  const imgEl = document.getElementById('img');
  const svgEl = document.getElementById('boxes');
  totalEl.textContent = imgs.length;

  function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

  function render() {
    const img = imgs[cursor];
    idxEl.value = cursor + 1;
    metaEl.textContent = `${img.file_name}  ·  ${img.width}×${img.height}  ·  ${img.doc_category ?? ''}  ·  page ${img.page_no ?? '?'}`;
    imgEl.src = `${IMG_PREFIX}/${img.file_name}`;
    svgEl.setAttribute('viewBox', `0 0 ${img.width} ${img.height}`);
    const anns = annsByImage.get(img.id) || [];
    svgEl.innerHTML = anns
      .filter(a => !hidden.has(a.category_id))
      .map(a => {
        const [x, y, w, h] = a.bbox;
        const c = colors.get(a.category_id);
        return `<rect x="${x}" y="${y}" width="${w}" height="${h}" stroke="${c}"><title>${escapeHtml(cats.get(a.category_id) || '')}</title></rect>`;
      })
      .join('');
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output HTML path (default: <data-dir>/index.html)",
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

    # Inline JSON inside <script>: a literal "</" would close the tag early.
    data_json = json.dumps(coco).replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("__DATA__", data_json).replace("__IMG_PREFIX__", img_prefix)
    out.write_text(html)

    print(f"wrote {out} ({len(coco.get('images', []))} pages)")
    if args.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
