# API Reference

This file is generated from module docstrings and public functions/classes.

## `sxsnf.clustering`

Clustering and evaluation utilities for sxSNF.

- HTML: [`docs/pydoc/sxsnf.clustering.html`](pydoc/sxsnf.clustering.html)

| Name | Type | Signature |
|---|---|---|
| `atac_only_eval_exact_scanpy` | function | `(atac_adata, label_key: &#x27;str&#x27; = &#x27;cell_type&#x27;, n_neighbors: &#x27;int&#x27; = 15, resolution: &#x27;float&#x27; = 1.0) -&gt; &#x27;Tuple[np.ndarray, Dict[str, float]]&#x27;` |
| `atac_only_eval_from_array_scanpy` | function | `(atac_lsi: &#x27;np.ndarray&#x27;, labels, n_neighbors: &#x27;int&#x27; = 15, resolution: &#x27;float&#x27; = 1.0, seed: &#x27;int&#x27; = 42) -&gt; &#x27;Tuple[np.ndarray, Dict[str, float]]&#x27;` |
| `clustering_metrics` | function | `(true_labels, pred_labels) -&gt; &#x27;Dict[str, float]&#x27;` |
| `labels_to_int` | function | `(labels) -&gt; &#x27;np.ndarray&#x27;` |
| `leiden_cluster_from_affinity` | function | `(W: &#x27;np.ndarray&#x27;, resolution: &#x27;float&#x27; = 1.0, seed: &#x27;int&#x27; = 42, fallback_n_clusters: &#x27;Optional[int]&#x27; = None) -&gt; &#x27;np.ndarray&#x27;` |
| `leiden_cluster_from_embedding` | function | `(Z: &#x27;np.ndarray&#x27;, n_neighbors: &#x27;int&#x27; = 15, resolution: &#x27;float&#x27; = 1.0, seed: &#x27;int&#x27; = 42, fallback_n_clusters: &#x27;int&#x27; = 10) -&gt; &#x27;np.ndarray&#x27;` |

## `sxsnf.config`

Configuration objects for sxSNF workflows.

- HTML: [`docs/pydoc/sxsnf.config.html`](pydoc/sxsnf.config.html)

| Name | Type | Signature |
|---|---|---|
| `SxSNFConfig` | class |  |

## `sxsnf.data`

Data loading and preprocessing utilities for the Chen-2019 RNA/ATAC workflow.

- HTML: [`docs/pydoc/sxsnf.data.html`](pydoc/sxsnf.data.html)

| Name | Type | Signature |
|---|---|---|
| `check_cell_alignment` | function | `(rna, atac) -&gt; &#x27;bool&#x27;` |
| `load_h5ad` | function | `(path: &#x27;str&#x27;)` |
| `make_chen2019_representations` | function | `(rna, atac, label_key: &#x27;str&#x27; = &#x27;cell_type&#x27;) -&gt; &#x27;Tuple[np.ndarray, np.ndarray, np.ndarray]&#x27;` |
| `preprocess_atac` | function | `(atac, n_components: &#x27;int&#x27; = 100, n_iter: &#x27;int&#x27; = 15)` |
| `preprocess_rna` | function | `(rna, n_top_genes: &#x27;int&#x27; = 2000, n_pcs: &#x27;int&#x27; = 100, cell_type_key: &#x27;str&#x27; = &#x27;cell_type&#x27;)` |

## `sxsnf.diagnostics`

Diagnostic utilities for checking modality-specific neighborhood consistency.

- HTML: [`docs/pydoc/sxsnf.diagnostics.html`](pydoc/sxsnf.diagnostics.html)

| Name | Type | Signature |
|---|---|---|
| `diagnose_knn_overlap` | function | `(rna_pca: &#x27;np.ndarray&#x27;, atac_lsi: &#x27;np.ndarray&#x27;, k: &#x27;int&#x27; = 20, metric: &#x27;str&#x27; = &#x27;cosine&#x27;, normalize: &#x27;bool&#x27; = True, verbose: &#x27;bool&#x27; = True)` |
| `knn_indices` | function | `(X: &#x27;np.ndarray&#x27;, k: &#x27;int&#x27; = 20, metric: &#x27;str&#x27; = &#x27;cosine&#x27;) -&gt; &#x27;np.ndarray&#x27;` |
| `knn_jaccard_overlap` | function | `(idx_a: &#x27;np.ndarray&#x27;, idx_b: &#x27;np.ndarray&#x27;, k: &#x27;Optional[int]&#x27; = None) -&gt; &#x27;Dict[str, object]&#x27;` |

## `sxsnf.graph`

Graph construction, geometry-anchored SNF, and sparse PyTorch graph helpers.

- HTML: [`docs/pydoc/sxsnf.graph.html`](pydoc/sxsnf.graph.html)

| Name | Type | Signature |
|---|---|---|
| `dense_to_torch_sparse` | function | `(A: &#x27;np.ndarray&#x27;, device)` |
| `knn_affinity` | function | `(X, k: &#x27;int&#x27; = 20, metric: &#x27;str&#x27; = &#x27;cosine&#x27;, mode: &#x27;str&#x27; = &#x27;cos&#x27;, sigma: &#x27;Optional[float]&#x27; = None, sym: &#x27;bool&#x27; = True) -&gt; &#x27;np.ndarray&#x27;` |
| `knn_affinity_local_scaling` | function | `(X, k: &#x27;int&#x27; = 20, metric: &#x27;str&#x27; = &#x27;cosine&#x27;, sym: &#x27;bool&#x27; = True, eps: &#x27;float&#x27; = 1e-12) -&gt; &#x27;np.ndarray&#x27;` |
| `knn_sparsify_dense` | function | `(W: &#x27;np.ndarray&#x27;, k: &#x27;int&#x27; = 20) -&gt; &#x27;np.ndarray&#x27;` |
| `normalize_adj_torch_sparse` | function | `(adj)` |
| `row_normalize` | function | `(W: &#x27;np.ndarray&#x27;, eps: &#x27;float&#x27; = 1e-12) -&gt; &#x27;np.ndarray&#x27;` |
| `snf_fusion_dense_anchored` | function | `(W_list: &#x27;Iterable[np.ndarray]&#x27;, k: &#x27;int&#x27; = 20, t: &#x27;int&#x27; = 20, alpha: &#x27;float&#x27; = 0.2, eps: &#x27;float&#x27; = 1e-12) -&gt; &#x27;np.ndarray&#x27;` |

## `sxsnf.models`

PyTorch neural network modules used by sxSNF.

- HTML: [`docs/pydoc/sxsnf.models.html`](pydoc/sxsnf.models.html)

| Name | Type | Signature |
|---|---|---|
| `DeepGCNIIEncoder` | class |  |
| `GCNIIBlock` | class |  |
| `MLP` | class |  |
| `edge_logits` | function | `(z, edges_uv)` |

## `sxsnf.pipeline`

End-to-end sxSNF multimodal integration pipeline.

- HTML: [`docs/pydoc/sxsnf.pipeline.html`](pydoc/sxsnf.pipeline.html)

| Name | Type | Signature |
|---|---|---|
| `run_chen2019_from_h5ad` | function | `(rna_path: &#x27;str&#x27;, atac_path: &#x27;str&#x27;, config: &#x27;Optional[SxSNFConfig]&#x27; = None, label_key: &#x27;str&#x27; = &#x27;cell_type&#x27;, n_top_genes: &#x27;int&#x27; = 2000, n_pcs: &#x27;int&#x27; = 100, lsi_components: &#x27;int&#x27; = 100, lsi_iter: &#x27;int&#x27; = 15, save_results: &#x27;bool&#x27; = True)` |
| `run_multimodal_flow` | function | `(rna_pca, atac_lsi, labels, config: &#x27;SxSNFConfig&#x27;, atac_adata=None, atac_label_key: &#x27;str&#x27; = &#x27;cell_type&#x27;, save_results: &#x27;bool&#x27; = True) -&gt; &#x27;Dict[str, Dict[str, float]]&#x27;` |
| `save_metrics` | function | `(metrics: &#x27;Dict[str, Dict[str, float]]&#x27;, outdir: &#x27;str&#x27;) -&gt; &#x27;None&#x27;` |

## `sxsnf.training`

Self-supervised masked-edge training for sxSNF graph encoders.

- HTML: [`docs/pydoc/sxsnf.training.html`](pydoc/sxsnf.training.html)

| Name | Type | Signature |
|---|---|---|
| `negative_sampling_undirected` | function | `(n_nodes: &#x27;int&#x27;, num_samples: &#x27;int&#x27;, existing_undirected_set, device, seed: &#x27;int&#x27; = 42)` |
| `sample_masked_edges_from_sparse_undirected` | function | `(adj_sparse, mask_ratio: &#x27;float&#x27; = 0.1, seed: &#x27;int&#x27; = 42)` |
| `train_masked_edge_ssl` | function | `(features, adj_fused_sparse, hidden_dim: &#x27;int&#x27; = 128, emb_dim: &#x27;int&#x27; = 64, dropout: &#x27;float&#x27; = 0.5, lr: &#x27;float&#x27; = 0.001, weight_decay: &#x27;float&#x27; = 0.0005, epochs: &#x27;int&#x27; = 200, mask_ratio: &#x27;float&#x27; = 0.1, neg_ratio: &#x27;float&#x27; = 1.0, seed: &#x27;int&#x27; = 42, outdir: &#x27;str&#x27; = &#x27;training_process&#x27;, gnn_layers: &#x27;int&#x27; = 6, gcn2_alpha: &#x27;float&#x27; = 0.1, gcn2_beta: &#x27;float&#x27; = 0.5, use_norm: &#x27;bool&#x27; = True, log_every: &#x27;int&#x27; = 10)` |
| `undirected_edge_set_from_sparse` | function | `(adj_sparse, drop_self_loops: &#x27;bool&#x27; = True) -&gt; &#x27;Set[Tuple[int, int]]&#x27;` |

## `sxsnf.utils`

General utility functions shared across sxSNF modules.

- HTML: [`docs/pydoc/sxsnf.utils.html`](pydoc/sxsnf.utils.html)

| Name | Type | Signature |
|---|---|---|
| `ensure_dir` | function | `(path: &#x27;Union[str, Path]&#x27;) -&gt; &#x27;Path&#x27;` |
| `l2norm_rows` | function | `(X, eps: &#x27;float&#x27; = 1e-12)` |
| `set_seed` | function | `(seed: &#x27;int&#x27; = 42) -&gt; &#x27;None&#x27;` |
