"""
End-to-end sxSNF multimodal integration pipeline.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np
import torch

from .clustering import (
    atac_only_eval_exact_scanpy,
    atac_only_eval_from_array_scanpy,
    clustering_metrics,
    labels_to_int,
    leiden_cluster_from_affinity,
    leiden_cluster_from_embedding,
)
from .config import SxSNFConfig
from .data import (
    load_h5ad,
    make_chen2019_representations,
    preprocess_atac,
    preprocess_rna,
)
from .graph import (
    dense_to_torch_sparse,
    knn_affinity_local_scaling,
    knn_sparsify_dense,
    snf_fusion_dense_anchored,
)
from .training import train_masked_edge_ssl
from .utils import ensure_dir, l2norm_rows, set_seed


def save_metrics(metrics: Dict[str, Dict[str, float]], outdir: str) -> None:
    """
    Save stage-wise clustering metrics as text and JSON.

    Parameters
    ----------
    metrics : dict
        Nested mapping from stage name to metric dictionary.
    outdir : str
        Output directory.
    """
    ensure_dir(outdir)

    with open(os.path.join(outdir, "metrics_all_stages.txt"), "w", encoding="utf-8") as f:
        for stage, values in metrics.items():
            f.write(f"[{stage}]\n")
            for key in ["ARI", "NMI", "AMI"]:
                f.write(f"{key}: {values[key]:.6f}\n")
            f.write("\n")

    with open(os.path.join(outdir, "metrics_all_stages.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def run_multimodal_flow(
    rna_pca,
    atac_lsi,
    labels,
    config: SxSNFConfig,
    atac_adata=None,
    atac_label_key: str = "cell_type",
    save_results: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Run the complete sxSNF workflow from low-dimensional RNA/ATAC features.

    The workflow includes:
    1. RNA and ATAC local-scaling graph construction.
    2. RNA-only and ATAC-only baseline clustering.
    3. Geometry-anchored SNF graph fusion.
    4. SNF-network clustering.
    5. Masked-edge self-supervised graph learning with DeepGCNII.
    6. GNN-embedding clustering.

    Parameters
    ----------
    rna_pca : array-like, shape (n_cells, n_rna_features)
        RNA PCA representation.
    atac_lsi : array-like, shape (n_cells, n_atac_features)
        ATAC LSI representation.
    labels : array-like
        Ground-truth cell labels.
    config : SxSNFConfig
        Pipeline configuration.
    atac_adata : anndata.AnnData, optional
        Processed ATAC AnnData object. If provided, the ATAC-only baseline
        exactly follows the Scanpy workflow used in the notebook.
    atac_label_key : str, default="cell_type"
        Label column for ATAC-only evaluation when ``atac_adata`` is provided.
    save_results : bool, default=True
        Whether to save stage-wise metrics.

    Returns
    -------
    dict
        Stage-wise ARI/NMI/AMI metrics.
    """
    set_seed(config.seed)
    device = config.resolved_device()
    ensure_dir(config.outdir)

    labels_eval = labels_to_int(labels)
    n_true = len(np.unique(labels_eval))
    stage_metrics = {}

    print("[1] Build modality-specific similarity networks...")
    x_rna = l2norm_rows(np.asarray(rna_pca, dtype=np.float32))
    x_atac = l2norm_rows(np.asarray(atac_lsi, dtype=np.float32))

    w_rna = knn_affinity_local_scaling(x_rna, k=config.k, metric="cosine", sym=True)
    w_atac = knn_affinity_local_scaling(x_atac, k=config.k, metric="cosine", sym=True)

    print("[1b] Clustering on RNA-only network...")
    pred_rna = leiden_cluster_from_affinity(
        w_rna,
        resolution=config.pre_snf_resolution,
        seed=config.seed,
        fallback_n_clusters=n_true,
    )
    m_rna = clustering_metrics(labels_eval, pred_rna)
    stage_metrics["RNA_network"] = m_rna
    print(f"    RNA metrics:  ARI={m_rna['ARI']:.4f}, NMI={m_rna['NMI']:.4f}, AMI={m_rna['AMI']:.4f}")

    print("[1c] Clustering on ATAC-only representation...")
    if atac_adata is not None:
        _, m_atac = atac_only_eval_exact_scanpy(
            atac_adata=atac_adata,
            label_key=atac_label_key,
            n_neighbors=config.atac_leiden_k,
            resolution=config.pre_snf_resolution,
        )
    else:
        _, m_atac = atac_only_eval_from_array_scanpy(
            atac_lsi=atac_lsi,
            labels=labels,
            n_neighbors=config.atac_leiden_k,
            resolution=config.pre_snf_resolution,
            seed=config.seed,
        )
    stage_metrics["ATAC_only_scanpy"] = m_atac
    print(f"    ATAC metrics: ARI={m_atac['ARI']:.4f}, NMI={m_atac['NMI']:.4f}, AMI={m_atac['AMI']:.4f}")

    print(f"[2] Geometry-anchored SNF fusion: k={config.k}, t={config.t}, alpha={config.snf_alpha}")
    p_fused = snf_fusion_dense_anchored(
        [w_rna, w_atac],
        k=config.k,
        t=config.t,
        alpha=config.snf_alpha,
    )

    if config.fused_topk is not None:
        p_fused = knn_sparsify_dense(p_fused, k=config.fused_topk)

    print("[2b] Clustering on SNF-fused network...")
    pred_snf = leiden_cluster_from_affinity(
        p_fused,
        resolution=config.snf_resolution,
        seed=config.seed,
        fallback_n_clusters=n_true,
    )
    m_snf = clustering_metrics(labels_eval, pred_snf)
    stage_metrics["SNF_network"] = m_snf
    print(f"    SNF metrics:  ARI={m_snf['ARI']:.4f}, NMI={m_snf['NMI']:.4f}, AMI={m_snf['AMI']:.4f}")

    print("[3] Self-supervised masked-edge learning on fused network...")
    features_np = np.concatenate([x_rna, x_atac], axis=1).astype(np.float32)
    features = torch.from_numpy(features_np).to(device)
    adj_sparse = dense_to_torch_sparse(p_fused, device=device)

    model, adj_norm = train_masked_edge_ssl(
        features=features,
        adj_fused_sparse=adj_sparse,
        hidden_dim=config.hidden_dim,
        emb_dim=config.embedding_dim,
        dropout=config.dropout,
        lr=config.lr,
        weight_decay=config.weight_decay,
        epochs=config.epochs,
        mask_ratio=config.mask_ratio,
        neg_ratio=config.neg_ratio,
        seed=config.seed,
        outdir=os.path.join(config.outdir, "training_process"),
        gnn_layers=config.gnn_layers,
        gcn2_alpha=config.gcn2_alpha,
        gcn2_beta=config.gcn2_beta,
        use_norm=config.use_norm,
    )

    model.eval()
    with torch.no_grad():
        z = model(features, adj_norm).detach().cpu().numpy()

    np.save(os.path.join(config.outdir, "sxsnf_embedding.npy"), z)
    np.save(os.path.join(config.outdir, "snf_fused_network.npy"), p_fused)

    print("[3b] Clustering on GNN embeddings...")
    pred_gnn = leiden_cluster_from_embedding(
        z,
        n_neighbors=config.leiden_k,
        resolution=config.leiden_resolution,
        seed=config.seed,
        fallback_n_clusters=n_true,
    )
    m_gnn = clustering_metrics(labels_eval, pred_gnn)
    stage_metrics["GNN_embedding"] = m_gnn
    print(f"    GNN metrics:  ARI={m_gnn['ARI']:.4f}, NMI={m_gnn['NMI']:.4f}, AMI={m_gnn['AMI']:.4f}")

    if save_results:
        save_metrics(stage_metrics, config.outdir)
        with open(os.path.join(config.outdir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2)

    return stage_metrics


def run_chen2019_from_h5ad(
    rna_path: str,
    atac_path: str,
    config: Optional[SxSNFConfig] = None,
    label_key: str = "cell_type",
    n_top_genes: int = 2000,
    n_pcs: int = 100,
    lsi_components: int = 100,
    lsi_iter: int = 15,
    save_results: bool = True,
):
    """
    Run sxSNF directly from Chen-2019 RNA and ATAC ``.h5ad`` files.

    Parameters
    ----------
    rna_path : str
        Path to ``Chen-2019-RNA.h5ad``.
    atac_path : str
        Path to ``Chen-2019-ATAC.h5ad``.
    config : SxSNFConfig, optional
        Pipeline configuration. If omitted, defaults are used.
    label_key : str, default="cell_type"
        Cell-type label column.
    n_top_genes : int, default=2000
        Highly variable genes for RNA preprocessing.
    n_pcs : int, default=100
        PCA components for RNA.
    lsi_components : int, default=100
        LSI components for ATAC.
    lsi_iter : int, default=15
        Number of LSI iterations.
    save_results : bool, default=True
        Whether to save metrics and embeddings.

    Returns
    -------
    dict
        Stage-wise ARI/NMI/AMI metrics.
    """
    if config is None:
        config = SxSNFConfig()

    print(f"[Data] Load RNA:  {rna_path}")
    rna = load_h5ad(rna_path)
    print(f"[Data] Load ATAC: {atac_path}")
    atac = load_h5ad(atac_path)

    print("[Data] Preprocess RNA...")
    rna = preprocess_rna(rna, n_top_genes=n_top_genes, n_pcs=n_pcs, cell_type_key=label_key)

    print("[Data] Preprocess ATAC...")
    atac = preprocess_atac(atac, n_components=lsi_components, n_iter=lsi_iter)

    rna_pca, atac_lsi, labels = make_chen2019_representations(rna, atac, label_key=label_key)
    print("Device:", config.resolved_device())
    print("RNA PCA:", rna_pca.shape, "ATAC LSI:", atac_lsi.shape, "labels:", np.asarray(labels).shape)

    return run_multimodal_flow(
        rna_pca=rna_pca,
        atac_lsi=atac_lsi,
        labels=labels,
        config=config,
        atac_adata=atac,
        atac_label_key=label_key,
        save_results=save_results,
    )
