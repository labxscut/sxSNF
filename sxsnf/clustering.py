"""
Clustering and evaluation utilities for sxSNF.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


def labels_to_int(labels) -> np.ndarray:
    """
    Convert arbitrary labels to integer category codes.

    Parameters
    ----------
    labels : array-like
        Ground-truth or predicted labels.

    Returns
    -------
    np.ndarray
        Integer labels with dtype int32.
    """
    lab = np.asarray(labels)
    if np.issubdtype(lab.dtype, np.number):
        return lab.astype(np.int32)
    return pd.Categorical(lab).codes.astype(np.int32)


def clustering_metrics(true_labels, pred_labels) -> Dict[str, float]:
    """
    Compute standard clustering metrics.

    Parameters
    ----------
    true_labels : array-like
        Ground-truth labels.
    pred_labels : array-like
        Predicted cluster labels.

    Returns
    -------
    dict
        Dictionary with ARI, NMI and AMI.
    """
    true_int = labels_to_int(true_labels)
    pred_int = labels_to_int(pred_labels)
    return {
        "ARI": float(adjusted_rand_score(true_int, pred_int)),
        "NMI": float(normalized_mutual_info_score(true_int, pred_int)),
        "AMI": float(adjusted_mutual_info_score(true_int, pred_int)),
    }


def leiden_cluster_from_embedding(Z: np.ndarray, n_neighbors: int = 15,
                                  resolution: float = 1.0, seed: int = 42,
                                  fallback_n_clusters: int = 10) -> np.ndarray:
    """
    Cluster low-dimensional cell embeddings using Leiden.

    Parameters
    ----------
    Z : np.ndarray, shape (n_cells, n_features)
        Embedding matrix.
    n_neighbors : int, default=15
        Number of neighbors for Scanpy graph construction.
    resolution : float, default=1.0
        Leiden resolution.
    seed : int, default=42
        Random seed.
    fallback_n_clusters : int, default=10
        KMeans cluster number used when Scanpy/Leiden is unavailable.

    Returns
    -------
    np.ndarray
        Integer cluster labels.
    """
    try:
        import anndata as ad
        import scanpy as sc

        adata = ad.AnnData(X=Z.astype(np.float32))
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X", random_state=seed)
        sc.tl.leiden(adata, resolution=resolution, random_state=seed, key_added="leiden")
        return adata.obs["leiden"].astype("category").cat.codes.to_numpy().astype(np.int32)
    except Exception as e:
        print(f"[WARN] Leiden(embedding) failed ({e}). Fallback KMeans.")
        km = KMeans(n_clusters=int(fallback_n_clusters), n_init=20, random_state=seed)
        return km.fit_predict(Z).astype(np.int32)


def leiden_cluster_from_affinity(W: np.ndarray, resolution: float = 1.0,
                                 seed: int = 42,
                                 fallback_n_clusters: Optional[int] = None) -> np.ndarray:
    """
    Cluster a precomputed affinity matrix using Leiden.

    Parameters
    ----------
    W : np.ndarray, shape (n_cells, n_cells)
        Cell-cell affinity matrix.
    resolution : float, default=1.0
        Leiden resolution.
    seed : int, default=42
        Random seed.
    fallback_n_clusters : int, optional
        Number of spectral clusters used if Scanpy/Leiden is unavailable.

    Returns
    -------
    np.ndarray
        Integer cluster labels.
    """
    W = np.asarray(W).astype(np.float32)
    n = W.shape[0]
    np.fill_diagonal(W, 0.0)
    try:
        import anndata as ad
        import scanpy as sc
        import scipy.sparse as sp

        adata = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32))
        adata.obsp["connectivities"] = sp.csr_matrix(W)
        adata.uns["neighbors"] = {
            "connectivities_key": "connectivities",
            "distances_key": None,
            "params": {},
        }
        sc.tl.leiden(adata, resolution=resolution, random_state=seed, key_added="leiden")
        return adata.obs["leiden"].astype("category").cat.codes.to_numpy().astype(np.int32)
    except Exception as e:
        print(f"[WARN] Leiden(affinity) failed ({e}). Fallback SpectralClustering.")
        if fallback_n_clusters is None:
            fallback_n_clusters = 10
        Wsym = 0.5 * (W + W.T)
        model = SpectralClustering(
            n_clusters=int(fallback_n_clusters),
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=seed,
        )
        return model.fit_predict(Wsym).astype(np.int32)


def atac_only_eval_exact_scanpy(atac_adata, label_key: str = "cell_type",
                                n_neighbors: int = 15,
                                resolution: float = 1.0) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Evaluate ATAC-only clustering using the original notebook's Scanpy procedure.

    Parameters
    ----------
    atac_adata : anndata.AnnData
        ATAC AnnData object containing ``obsm["X_lsi"]``.
    label_key : str, default="cell_type"
        Ground-truth label column in ``atac_adata.obs``.
    n_neighbors : int, default=15
        Number of neighbors for Scanpy graph construction.
    resolution : float, default=1.0
        Leiden resolution.

    Returns
    -------
    tuple
        Predicted integer labels and metrics dictionary.
    """
    import scanpy as sc

    adata = atac_adata.copy()
    sc.pp.neighbors(adata, use_rep="X_lsi", key_added="leiden_neighbors", n_neighbors=n_neighbors)
    sc.tl.leiden(adata, neighbors_key="leiden_neighbors", key_added="leiden", resolution=resolution)

    true_labels = adata.obs[label_key].to_numpy()
    pred_labels = adata.obs["leiden"].to_numpy()
    pred_int = labels_to_int(pred_labels)
    return pred_int, clustering_metrics(true_labels, pred_int)


def atac_only_eval_from_array_scanpy(atac_lsi: np.ndarray, labels,
                                     n_neighbors: int = 15,
                                     resolution: float = 1.0,
                                     seed: int = 42) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Evaluate ATAC-only clustering from a precomputed LSI matrix.

    Parameters
    ----------
    atac_lsi : np.ndarray
        ATAC LSI matrix.
    labels : array-like
        Ground-truth cell labels.
    n_neighbors : int, default=15
        Number of neighbors for Scanpy graph construction.
    resolution : float, default=1.0
        Leiden resolution.
    seed : int, default=42
        Random seed.

    Returns
    -------
    tuple
        Predicted integer labels and metrics dictionary.
    """
    import anndata as ad
    import scanpy as sc

    X = np.asarray(atac_lsi).astype(np.float32)
    adata = ad.AnnData(X=X)
    sc.pp.neighbors(
        adata,
        use_rep="X",
        key_added="leiden_neighbors",
        random_state=seed,
        n_neighbors=n_neighbors,
    )
    sc.tl.leiden(
        adata,
        neighbors_key="leiden_neighbors",
        key_added="leiden",
        random_state=seed,
        resolution=resolution,
    )

    pred_int = adata.obs["leiden"].astype("category").cat.codes.to_numpy().astype(np.int32)
    return pred_int, clustering_metrics(labels, pred_int)
