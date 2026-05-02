# sxSNF Workflow

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
python main.py \
  --rna datasets/Chen-2019-RNA.h5ad \
  --atac datasets/Chen-2019-ATAC.h5ad \
  --outdir results/chen2019
```
