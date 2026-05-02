# sxSNF

**sxSNF** is a single-cell multi-omics integration workflow that combines modality-specific similarity graphs, geometry-anchored Similarity Network Fusion (SNF), and self-supervised graph representation learning.

This repository version reorganizes the original Chen-2019 notebook into a standard GitHub/Python package layout.

## Repository structure

```text
sxSNF/
├── sxsnf/                    # Reusable Python package
│   ├── config.py             # Runtime configuration
│   ├── data.py               # Chen-2019 RNA/ATAC loading and preprocessing
│   ├── graph.py              # kNN graph, local scaling, anchored SNF
│   ├── clustering.py         # Leiden/KMeans/Spectral clustering and metrics
│   ├── models.py             # DeepGCNII encoder
│   ├── training.py           # Masked-edge self-supervised training
│   ├── diagnostics.py        # RNA/ATAC kNN overlap diagnostics
│   └── pipeline.py           # End-to-end sxSNF workflow
├── scripts/
│   ├── run_chen2019.py       # Command-line entry point
│   ├── generate_pydoc.py     # Generate PyDoc API pages
│   └── sync_to_github.sh     # Pull, regenerate docs, commit and push
├── notebooks/
│   └── sxSNF2.0_Chen.ipynb   # Original notebook kept for traceability
├── docs/
│   ├── index.html            # Documentation home
│   ├── API_REFERENCE.md      # Markdown API index
│   ├── WORKFLOW.md           # Workflow description
│   └── pydoc/                # PyDoc HTML pages
├── tests/
│   └── test_imports.py
├── main.py                   # Root-level compatibility runner
├── requirements.txt
└── pyproject.toml
```

## Installation

Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For editable development:

```bash
pip install -e .
```

## Run Chen-2019 workflow

```bash
python main.py \
  --rna datasets/Chen-2019-RNA.h5ad \
  --atac datasets/Chen-2019-ATAC.h5ad \
  --outdir results/chen2019
```

Equivalent command:

```bash
python scripts/run_chen2019.py \
  --rna datasets/Chen-2019-RNA.h5ad \
  --atac datasets/Chen-2019-ATAC.h5ad \
  --outdir results/chen2019
```

## Core workflow

```text
RNA h5ad + ATAC h5ad
        ↓
RNA PCA + ATAC LSI
        ↓
Local-scaling kNN graphs
        ↓
Geometry-anchored SNF
        ↓
Fused cell-cell graph
        ↓
Masked-edge self-supervised DeepGCNII
        ↓
Cell embeddings and clustering metrics
```

## Output files

The run directory contains:

```text
metrics_all_stages.txt
metrics_all_stages.json
config.json
sxsnf_embedding.npy
snf_fused_network.npy
training_process/loss_history.npy
```

## Generate PyDoc

After modifying code or docstrings:

```bash
python scripts/generate_pydoc.py
```

Open the documentation locally:

```bash
explorer.exe docs/index.html   # WSL on Windows
# or
xdg-open docs/index.html       # Linux
```

## GitHub update workflow

After copying this organized project into your local repository:

```bash
cd /mnt/e/Research/sxSNF/sxSNF
bash scripts/sync_to_github.sh "Reorganize sxSNF into standard package and update PyDoc"
```

Manual equivalent:

```bash
git pull origin main
python scripts/generate_pydoc.py
git add .
git commit -m "Reorganize sxSNF into standard package and update PyDoc"
git push origin main
```

## Notes

- The original notebook is preserved under `notebooks/` for reproducibility.
- Large datasets and generated outputs are ignored by `.gitignore`.
- `scanpy`, `scglue`, `leidenalg`, and `igraph` are needed for the complete Chen-2019 workflow.
