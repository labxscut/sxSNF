#!/usr/bin/env python3
"""Generate pydoc HTML docs and a markdown index for top-level modules."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
PYDOC_DIR = DOCS_DIR / "pydoc"
API_INDEX_PATH = DOCS_DIR / "API_REFERENCE.md"
DOCS_CSS_PATH = DOCS_DIR / "assets" / "docs.css"

DOCS_CSS = """
:root {
    color-scheme: light dark;
    --bg: #f7f8fb;
    --panel: #ffffff;
    --text: #1f2937;
    --muted: #6b7280;
    --accent: #2563eb;
    --border: #e5e7eb;
}

@media (prefers-color-scheme: dark) {
    :root {
        --bg: #0f172a;
        --panel: #111827;
        --text: #e5e7eb;
        --muted: #94a3b8;
        --accent: #60a5fa;
        --border: #334155;
    }
}

html, body {
    margin: 0;
    padding: 0;
    background: var(--bg);
    color: var(--text);
    font-family: Inter, Segoe UI, Arial, sans-serif;
}

body {
    max-width: 1100px;
    margin: 0 auto;
    padding: 2rem 1.2rem;
    line-height: 1.6;
}

a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

table {
    width: 100%;
    border-collapse: collapse;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    margin: 1rem 0;
}

td, th {
    padding: 0.6rem 0.8rem;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
}

.heading .title strong.title,
strong.bigsection {
    font-size: 1.15rem;
}

.code {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}

p, li, dt, dd {
    color: var(--text);
}

.doc-nav {
    display: flex;
    flex-wrap: wrap;
    gap: 0.8rem;
    margin-bottom: 1rem;
    padding: 0.7rem 0.9rem;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--panel);
}

.search-input {
    margin: 0.4rem 0 1rem;
    padding: 0.45rem 0.55rem;
    width: 100%;
    max-width: 520px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--panel);
    color: var(--text);
}
""".strip()

STUB_MODULES = [
    "numpy",
    "scipy",
    "scipy.sparse",
    "scipy.spatial",
    "scipy.spatial.distance",
    "scipy.io",
    "sklearn",
    "sklearn.cluster",
    "sklearn.decomposition",
    "sklearn.manifold",
    "sklearn.metrics",
    "sklearn.metrics.pairwise",
    "sklearn.model_selection",
    "sklearn.neighbors",
    "sklearn.preprocessing",
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.nn.parameter",
    "torch.optim",
    "torch.utils",
    "torch.utils.data",
    "torch_geometric",
    "torch_geometric.nn",
    "matplotlib",
    "matplotlib.pyplot",
    "seaborn",
]

STUB_CONTT = '''"""Auto-generated pydoc stub module."""

class _StubMeta(type):
    def __getattr__(cls, name):
        return _StubType


class _StubType(metaclass=_StubMeta):
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return _StubType

    def __call__(self, *args, **kwargs):
        return _StubType()

    def __iter__(self):
        return iter(())


def __getattr__(name):
    return _StubType
'''


def get_top_level_modules(root: Path) -> list[Path]:
    modules = []
    for path in root.glob("*.py"):
        if path.name.startswith("_"):
            continue
        modules.append(path)
    return sorted(modules, key=lambda p: p.name.lower())


def module_description(module_path: Path) -> str:
    try:
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        doc = ast.get_docstring(tree)
        if doc:
            first_line = doc.strip().splitlines()[0].strip()
            if first_line:
                return first_line
    except Exception:
        pass
    return "No module docstring provided."


def build_stub_packages(stub_root: Path) -> None:
    for dotted in STUB_MODULES:
        parts = dotted.split(".")
        if len(parts) == 1:
            init_file = stub_root / parts[0] / "__init__.py"
        else:
            pkg_dir = stub_root.joinpath(*parts[:-1])
            pkg_init = pkg_dir / "__init__.py"
            pkg_init.parent.mkdir(parents=True, exist_ok=True)
            pkg_init.write_text(STUB_CONTT, encoding="utf-8")
            init_file = stub_root.joinpath(*parts) / "__init__.py"
        init_file.parent.mkdir(parents=True, exist_ok=True)
        init_file.write_text(STUB_CONTT, encoding="utf-8")


def run_pydoc_for_module(module_name: str, stub_root: Path) -> tuple[bool, str]:
    cmd = [sys.executable, "-m", "pydoc", "-w", module_name]
    env = dict(os.environ)
    prior = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(stub_root) if not prior else f"{stub_root}{os.pathsep}{prior}"
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=env)
    if proc.returncode == 0:
        return True, ""
    stderr = (proc.stderr or proc.stdout or "").strip()
    return False, stderr


def write_docs_css() -> None:
    DOCS_CSS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCS_CSS_PATH.write_text(DOCS_CSS + "\n", encoding="utf-8")


def style_pydoc_html(html_path: Path) -> None:
    html = html_path.read_text(encoding="utf-8")

    if "../assets/docs.css" not in html:
        injection = (
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            '<link rel="stylesheet" href="../assets/docs.css">'
        )
        if "</head>" in html:
            html = html.replace("</head>", f"{injection}\n</head>", 1)
        else:
            html = f"<head>{injection}</head>\n{html}"

    if 'class="doc-nav"' not in html:
        nav = (
            '<div class="doc-nav">'
            '<a href="../index.html">Docs Home</a>'
            '<a href="../WORKFLOW.md">Workflow</a>'
            '<a href="../API_REFERENCE.md">API Reference</a>'
            '<a href="../../README.md">README</a>'
            "</div>"
        )
        if "<body>" in html:
            html = html.replace("<body>", f"<body>\n{nav}", 1)
        else:
            html = f"{nav}\n{html}"

    html_path.write_text(html, encoding="utf-8")


def write_markdown_index(entries: list[dict[str, str]], failed: list[dict[str, str]]) -> None:
    lines = [
        "# API Reference",
        "",
        "This file is generated by `scripts/generate_pydoc.py`.",
        "",
        "## Documentation Entry Points",
        "",
        "- Workflow chart: [`WORKFLOW.md`](WORKFLOW.md)",
        "- API landing page: [`index.html`](index.html)",
        "- Repository guide: [`../README.md`](../README.md)",
        "",
        "Search keywords: sxSNF, workflow, SNF, GNN, clustering, embeddings, pydoc.",
        "",
        "## Modules",
        "",
    ]
    for entry in entries:
        lines.append(f"- [`{entry['module']}`]({entry['html_rel_path']}) - {entry['description']}")

    if failed:
        lines.extend(["", "## Generation Warnings", ""])
        for item in failed:
            lines.append(f"- `{item['module']}`: {item['error']}")

    lines.append("")
    API_INDEX_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PYDOC_DIR.mkdir(parents=True, exist_ok=True)
    write_docs_css()

    modules = get_top_level_modules(REPO_ROOT)
    if not modules:
        print("No top-level Python modules found.")
        return 0

    entries: list[dict[str, str]] = []
    failed: list[dict[str, str]] = []

    with tempfile.TemporaryDirectory(prefix="pydoc_stubs_") as temp_dir:
        stub_root = Path(temp_dir)
        build_stub_packages(stub_root)

        for module_path in modules:
            module_name = module_path.stem
            ok, err = run_pydoc_for_module(module_name, stub_root)

            generated_html = REPO_ROOT / f"{module_name}.html"
            target_html = PYDOC_DIR / f"{module_name}.html"

            if generated_html.exists():
                if target_html.exists():
                    target_html.unlink()
                generated_html.replace(target_html)

            if not target_html.exists():
                message = err or "pydoc did not produce an HTML file."
                failed.append({"module": module_name, "error": message})
                continue

            style_pydoc_html(target_html)

            if not ok and err:
                failed.append({"module": module_name, "error": err})

            entries.append(
                {
                    "module": module_name,
                    "html_rel_path": f"pydoc/{module_name}.html",
                    "description": module_description(module_path),
                }
            )

    write_markdown_index(entries, failed)

    print(f"Generated docs for {len(entries)} module(s).")
    if failed:
        print(f"Warnings for {len(failed)} module(s). See {API_INDEX_PATH}.")
    print(f"API index written to {API_INDEX_PATH}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
