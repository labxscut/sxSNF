#!/usr/bin/env python3
"""
Generate PyDoc HTML files and lightweight Markdown API references.
"""

from __future__ import annotations

import html as html_lib
import importlib
import inspect
import pkgutil
import pydoc
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
PYDOC_DIR = DOCS_DIR / "pydoc"
PACKAGE = "sxsnf"


def module_names():
    """Return sxSNF module names that should be documented."""
    import sxsnf

    names = []
    for module_info in pkgutil.iter_modules(sxsnf.__path__):
        if not module_info.name.startswith("_"):
            names.append(f"{PACKAGE}.{module_info.name}")
    return sorted(names)


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
        lines.append(f"- HTML: [`docs/pydoc/{module_name}.html`](pydoc/{module_name}.html)")
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


def generate_index(modules):
    """Generate docs/index.html."""
    links = []
    for module in modules:
        links.append(
            f'<li><a href="pydoc/{module}.html"><code>{module}</code></a></li>'
        )

    index = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>sxSNF API Documentation</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 960px; margin: 40px auto; line-height: 1.6; }}
    h1 {{ border-bottom: 2px solid #333; padding-bottom: 10px; }}
    code {{ background: #f6f8fa; padding: 2px 4px; border-radius: 4px; }}
    .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin: 14px 0; }}
  </style>
</head>
<body>
  <h1>sxSNF API Documentation</h1>
  <p>This documentation was generated with Python <code>pydoc</code>.</p>

  <div class="card">
    <h2>Project Documents</h2>
    <ul>
      <li><a href="API_REFERENCE.md">API_REFERENCE.md</a></li>
      <li><a href="WORKFLOW.md">WORKFLOW.md</a></li>
    </ul>
  </div>

  <div class="card">
    <h2>PyDoc Modules</h2>
    <ul>
      {''.join(links)}
    </ul>
  </div>
</body>
</html>
"""
    (DOCS_DIR / "index.html").write_text(index, encoding="utf-8")


def main():
    sys.path.insert(0, str(PROJECT_ROOT))
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    modules = module_names()
    generate_html(modules)
    generate_api_reference(modules)
    generate_workflow()
    generate_index(modules)
    print(f"[done] Documentation generated under {DOCS_DIR}")


if __name__ == "__main__":
    main()
