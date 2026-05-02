"""
Diagnostic utilities for checking modality-specific neighborhood consistency.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .utils import l2norm_rows


def knn_indices(X: np.ndarray, k: int = 20, metric: str = "cosine") -> np.ndarray:
    """
    Return k-nearest-neighbor indices excluding each cell itself.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    k : int, default=20
        Number of neighbors.
    metric : str, default="cosine"
        Distance metric for neighbor search.

    Returns
    -------
    np.ndarray
        Neighbor indices with shape ``(n_cells, k)`` or fewer when ``n_cells`` is small.
    """
    X = np.asarray(X)
    n = X.shape[0]
    k_eff = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=k_eff, metric=metric)
    nn.fit(X)
    _, idx = nn.kneighbors(X, return_distance=True)
    return idx[:, 1:]


def knn_jaccard_overlap(idx_a: np.ndarray, idx_b: np.ndarray,
                        k: Optional[int] = None) -> Dict[str, object]:
    """
    Compute per-cell Jaccard overlap between two kNN neighborhoods.

    Parameters
    ----------
    idx_a : np.ndarray
        First neighbor index matrix.
    idx_b : np.ndarray
        Second neighbor index matrix.
    k : int, optional
        Use only the first k neighbors from each matrix.

    Returns
    -------
    dict
        Summary statistics and per-cell overlap arrays.
    """
    n = idx_a.shape[0]
    if k is not None:
        idx_a = idx_a[:, :k]
        idx_b = idx_b[:, :k]

    k_a = idx_a.shape[1]
    k_b = idx_b.shape[1]
    jac = np.zeros(n, dtype=np.float32)
    inter = np.zeros(n, dtype=np.int32)

    for i in range(n):
        a = set(idx_a[i].tolist())
        b = set(idx_b[i].tolist())
        inter_i = len(a & b)
        union_i = len(a | b)
        inter[i] = inter_i
        jac[i] = inter_i / max(union_i, 1)

    return {
        "jaccard_per_cell": jac,
        "intersect_per_cell": inter,
        "mean_jaccard": float(jac.mean()),
        "median_jaccard": float(np.median(jac)),
        "p10_jaccard": float(np.quantile(jac, 0.10)),
        "p90_jaccard": float(np.quantile(jac, 0.90)),
        "mean_intersection": float(inter.mean()),
        "kA": int(k_a),
        "kB": int(k_b),
    }


def diagnose_knn_overlap(rna_pca: np.ndarray, atac_lsi: np.ndarray,
                         k: int = 20, metric: str = "cosine",
                         normalize: bool = True,
                         verbose: bool = True):
    """
    Compare RNA and ATAC nearest-neighbor neighborhoods.

    Parameters
    ----------
    rna_pca : np.ndarray
        RNA PCA matrix.
    atac_lsi : np.ndarray
        ATAC LSI matrix.
    k : int, default=20
        Number of neighbors.
    metric : str, default="cosine"
        Distance metric.
    normalize : bool, default=True
        Whether to L2-normalize rows before neighbor search.
    verbose : bool, default=True
        Whether to print a compact summary.

    Returns
    -------
    tuple
        ``(idx_rna, idx_atac, summary_dict)``.
    """
    x_rna = l2norm_rows(rna_pca) if normalize else np.asarray(rna_pca)
    x_atac = l2norm_rows(atac_lsi) if normalize else np.asarray(atac_lsi)

    idx_rna = knn_indices(x_rna, k=k, metric=metric)
    idx_atac = knn_indices(x_atac, k=k, metric=metric)

    summary = knn_jaccard_overlap(idx_rna, idx_atac, k=k)
    if verbose:
        print(f"[KNN overlap] k={k}, metric={metric}, normalize={normalize}")
        print(f"  mean Jaccard   : {summary['mean_jaccard']:.4f}")
        print(f"  median Jaccard : {summary['median_jaccard']:.4f}")
        print(f"  p10 / p90      : {summary['p10_jaccard']:.4f} / {summary['p90_jaccard']:.4f}")
        print(f"  mean |Nr∩Na|   : {summary['mean_intersection']:.2f} (out of k={k})")

    return idx_rna, idx_atac, summary
