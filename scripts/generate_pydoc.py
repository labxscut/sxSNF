#!/usr/bin/env python3
"""
Generate PyDoc HTML files and lightweight Markdown API references.
"""

from __future__ import annotations

import html as html_lib
import importlib
import inspect
import os
import pkgutil
import pydoc
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
PYDOC_DIR = DOCS_DIR / "pydoc"
PACKAGE = "sxsnf"

# Section title -> submodule names (without the sxsnf. prefix) for docs/index.html
API_GROUPS: list[tuple[str, list[str]]] = [
    ("Data & configuration", ["config", "data"]),
    ("Graphs & SNF", ["graph"]),
    ("Models & training", ["models", "training"]),
    ("Pipeline", ["pipeline"]),
    ("Clustering & diagnostics", ["clustering", "diagnostics"]),
    ("Utilities", ["utils"]),
]


def module_names():
    """Return sxSNF module names that should be documented."""
    import sxsnf

    names = []
    for module_info in pkgutil.iter_modules(sxsnf.__path__):
        if not module_info.name.startswith("_"):
            names.append(f"{PACKAGE}.{module_info.name}")
    return sorted(names)


def _grouped_modules(modules: list[str]) -> list[tuple[str, list[str]]]:
    """Order modules into API_GROUPS; append 'Other' for anything not listed."""
    ms = set(modules)
    sections: list[tuple[str, list[str]]] = []
    for title, suffixes in API_GROUPS:
        items = [f"{PACKAGE}.{s}" for s in suffixes if f"{PACKAGE}.{s}" in ms]
        if items:
            sections.append((title, items))
    covered = {m for _, lst in sections for m in lst}
    leftover = sorted(m for m in modules if m not in covered)
    if leftover:
        sections.append(("Other", leftover))
    return sections


def _css_href_for(html_path: Path) -> str:
    """Relative URL from an HTML file's directory to ``docs/assets/docs.css``."""
    return Path(
        os.path.relpath(DOCS_DIR / "assets" / "docs.css", html_path.parent)
    ).as_posix()


def _nav_html_fragment(html_path: Path) -> str:
    """Shared top nav; links are correct from ``html_path`` (any depth under ``docs/``)."""
    here = html_path.parent
    root = DOCS_DIR

    def href(target: Path) -> str:
        return html_lib.escape(Path(os.path.relpath(target, here)).as_posix())

    return (
        "  <nav class=\"doc-nav\">\n"
        f"    <a href=\"{href(root / 'index.html')}\"><strong>sxSNF docs</strong></a>\n"
        f"    <a href=\"{href(root / 'API_REFERENCE.html')}\">API reference</a>\n"
        f"    <a href=\"{href(root / 'WORKFLOW.html')}\">Workflow</a>\n"
        "    <a href=\"https://github.com/labxscut/sxSNF\">Repository</a>\n"
        "  </nav>"
    )


def _write_full_doc_page(html_path: Path, title: str, body_below_nav: str) -> None:
    """Write a complete HTML page with shared CSS and nav."""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    css = html_lib.escape(_css_href_for(html_path))
    nav = _nav_html_fragment(html_path)
    esc_title = html_lib.escape(title)
    html_path.write_text(
        f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc_title}</title>
  <link rel="stylesheet" href="{css}">
</head>
<body>
{nav}

{body_below_nav}
</body>
</html>
""",
        encoding="utf-8",
    )


def _is_html_document_fragment(raw: str) -> bool:
    """True if content is not a full HTML document (e.g. raw ``pydoc`` output)."""
    head = raw.lstrip()[:900].lower()
    return not (head.startswith("<!doctype") or head.startswith("<html"))


def _wrap_html_fragment(html_path: Path, fragment: str) -> str:
    """Wrap legacy body-only HTML in a shell with ``docs.css`` and nav."""
    css = html_lib.escape(_css_href_for(html_path))
    nav = _nav_html_fragment(html_path)
    title = html_lib.escape(f"sxSNF — {html_path.stem}")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <link rel="stylesheet" href="{css}">
</head>
<body>
{nav}

<main class="pydoc-page">
{fragment}
</main>
</body>
</html>
"""


def _patch_full_html_document(html_path: Path, text: str) -> str:
    """Normalize stylesheet href and nav for any depth under ``docs/``."""
    css = html_lib.escape(_css_href_for(html_path))
    nav = _nav_html_fragment(html_path)
    s = text
    if re.search(r'href=["\'][^"\']*docs\.css["\']', s, re.I):
        s = re.sub(
            r'<link\s+rel=["\']stylesheet["\']\s+href=["\'][^"\']*docs\.css["\']\s*/?>',
            f'<link rel="stylesheet" href="{css}">',
            s,
            count=1,
            flags=re.IGNORECASE,
        )
    else:
        m = re.search(r"<head[^>]*>", s, flags=re.IGNORECASE)
        if m:
            ins = m.end()
            s = s[:ins] + f"\n  <link rel=\"stylesheet\" href=\"{css}\">\n" + s[ins:]

    if re.search(r"<body[^>]*>\s*<nav\s+class=['\"]doc-nav['\"]", s, re.I):
        s = re.sub(
            r"<body([^>]*)>\s*<nav\s+class=['\"]doc-nav['\"][^>]*>.*?</nav>",
            r"<body\1>\n" + nav,
            s,
            count=1,
            flags=re.IGNORECASE | re.DOTALL,
        )
    elif re.search(r"<body[^>]*>\s*<div\s+class=['\"]doc-nav['\"]", s, re.I):
        s = re.sub(
            r"<body([^>]*)>\s*<div\s+class=['\"]doc-nav['\"][^>]*>.*?</div>",
            r"<body\1>\n" + nav,
            s,
            count=1,
            flags=re.IGNORECASE | re.DOTALL,
        )
    elif re.search(r'<nav\s+class=["\']doc-nav["\']', s, re.I):
        s = re.sub(
            r"\n[ \t]*<nav\s+class=['\"]doc-nav['\"][^>]*>.*?</nav>",
            "\n" + nav,
            s,
            count=1,
            flags=re.IGNORECASE | re.DOTALL,
        )
    elif re.search(r"<body[^>]*>", s, re.I):

        def _inject_nav(m: re.Match[str]) -> str:
            return m.group(0) + "\n" + nav + "\n"

        s = re.sub(r"<body[^>]*>", _inject_nav, s, count=1, flags=re.IGNORECASE)

    return s


def apply_docs_shell_recursively() -> None:
    """
    Ensure every ``docs/**/*.html`` uses ``assets/docs.css`` and the shared nav.

    - Full documents: patch ``docs.css`` link and shared top nav (including legacy
      ``<div class="doc-nav">`` shells).
    - Fragments (typical ``pydoc`` output): wrap in a minimal HTML shell.
    """
    css_file = DOCS_DIR / "assets" / "docs.css"
    if not css_file.is_file():
        print("[warn] docs/assets/docs.css missing; skip docs shell pass")
        return

    for path in sorted(DOCS_DIR.rglob("*.html")):
        if not path.is_file():
            continue
        try:
            path.relative_to(DOCS_DIR)
        except ValueError:
            continue
        raw = path.read_text(encoding="utf-8")
        if _is_html_document_fragment(raw):
            wrapped = _wrap_html_fragment(path, raw)
            if wrapped != raw:
                path.write_text(wrapped, encoding="utf-8")
                print(f"[skin] wrapped {path.relative_to(PROJECT_ROOT)}")
            continue
        patched = _patch_full_html_document(path, raw)
        if patched != raw:
            path.write_text(patched, encoding="utf-8")
            print(f"[skin] patched {path.relative_to(PROJECT_ROOT)}")


def generate_html(modules):
    """Generate PyDoc HTML pages into docs/pydoc without spawning subprocesses."""
    PYDOC_DIR.mkdir(parents=True, exist_ok=True)
    html_doc = pydoc.HTMLDoc()

    for module_name in modules:
        print(f"[pydoc] {module_name}")
        module = importlib.import_module(module_name)
        html_text = html_doc.docmodule(module)
        (PYDOC_DIR / f"{module_name}.html").write_text(html_text, encoding="utf-8")


def generate_api_reference(modules):
    """Generate a compact Markdown API index."""
    lines = [
        "# API Reference",
        "",
        "This file is generated from module docstrings and public functions/classes.",
        "",
    ]
    for module_name in modules:
        module = importlib.import_module(module_name)
        lines.append(f"## `{module_name}`")
        lines.append("")
        doc = inspect.getdoc(module) or "No module docstring provided."
        lines.append(doc.splitlines()[0])
        lines.append("")
        lines.append(f"- **HTML:** [`pydoc/{module_name}.html`](pydoc/{module_name}.html)")
        lines.append("")

        public = []
        for name, obj in inspect.getmembers(module):
            if name.startswith("_"):
                continue
            if inspect.isfunction(obj) and obj.__module__ == module_name:
                public.append((name, "function", inspect.signature(obj)))
            elif inspect.isclass(obj) and obj.__module__ == module_name:
                public.append((name, "class", None))

        if public:
            lines.append("| Name | Type | Signature |")
            lines.append("|---|---|---|")
            for name, typ, sig in public:
                sig_txt = f"`{html_lib.escape(str(sig))}`" if sig is not None else ""
                lines.append(f"| `{name}` | {typ} | {sig_txt} |")
            lines.append("")

    (DOCS_DIR / "API_REFERENCE.md").write_text("\n".join(lines), encoding="utf-8")


def generate_workflow():
    """Generate a concise workflow document."""
    text = """# sxSNF Workflow

```text
Chen-2019 RNA h5ad           Chen-2019 ATAC h5ad
        |                            |
        v                            v
RNA preprocessing              ATAC preprocessing
HVG -> normalize -> log1p      LSI with scGLUE
scale -> PCA                   neighbors / UMAP
        |                            |
        v                            v
RNA PCA matrix                 ATAC LSI matrix
        |                            |
        +------------+---------------+
                     |
                     v
       Local-scaling kNN affinity graphs
                     |
                     v
          Geometry-anchored SNF fusion
                     |
                     v
        Fused cell-cell similarity graph
                     |
                     v
  Masked-edge self-supervised DeepGCNII encoder
                     |
                     v
      Cell embeddings + Leiden/KMeans evaluation
```

Core command:

```bash
python main.py \\
  --rna datasets/Chen-2019-RNA.h5ad \\
  --atac datasets/Chen-2019-ATAC.h5ad \\
  --outdir results/chen2019
```
"""
    (DOCS_DIR / "WORKFLOW.md").write_text(text, encoding="utf-8")


def _markdown_to_html_pages():
    """Emit browsable HTML from Markdown (requires optional ``markdown``)."""
    try:
        import markdown
    except ImportError:
        print(
            "[warn] package 'markdown' not installed; "
            "install with `pip install -e \".[docs]\"` to build API_REFERENCE.html "
            "and WORKFLOW.html"
        )
        return

    md = markdown.Markdown(extensions=["tables", "fenced_code"])
    for stem, page_title in (
        ("API_REFERENCE", "API reference"),
        ("WORKFLOW", "Workflow"),
    ):
        src = DOCS_DIR / f"{stem}.md"
        if not src.exists():
            continue
        body = md.convert(src.read_text(encoding="utf-8"))
        md.reset()
        out = DOCS_DIR / f"{stem}.html"
        inner = f"""  <article class="md-body">
{body}
  </article>"""
        _write_full_doc_page(out, f"sxSNF — {page_title}", inner)
        print(f"[html] {out.relative_to(PROJECT_ROOT)}")


def generate_index(modules):
    """Generate docs/index.html with grouped API navigation."""
    sections_html = []
    for title, names in _grouped_modules(modules):
        items = "".join(
            f'<li><a href="pydoc/{m}.html"><code>{html_lib.escape(m)}</code></a></li>'
            for m in names
        )
        sections_html.append(
            f'  <div class="doc-card"><h2>{html_lib.escape(title)}</h2><ul>{items}</ul></div>'
        )

    blocks = "\n".join(sections_html)
    body = f"""  <h1>sxSNF API documentation</h1>
  <p class="muted">Generated with Python <code>pydoc</code> for the <code>sxsnf</code> package.</p>

  <div class="doc-card">
    <h2>Guides</h2>
    <ul>
      <li><a href="API_REFERENCE.html">API reference</a> (HTML) · <a href="API_REFERENCE.md">Markdown source</a></li>
      <li><a href="WORKFLOW.html">Workflow overview</a> (HTML) · <a href="WORKFLOW.md">Markdown source</a></li>
    </ul>
  </div>

{blocks}"""
    _write_full_doc_page(DOCS_DIR / "index.html", "sxSNF API Documentation", body)


def main():
    sys.path.insert(0, str(PROJECT_ROOT))
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    modules = module_names()
    generate_html(modules)
    generate_api_reference(modules)
    generate_workflow()
    _markdown_to_html_pages()
    generate_index(modules)
    apply_docs_shell_recursively()
    print(f"[done] Documentation generated under {DOCS_DIR}")


if __name__ == "__main__":
    main()
