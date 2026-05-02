"""
Graph construction, geometry-anchored SNF, and sparse PyTorch graph helpers.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors


def knn_affinity(X, k: int = 20, metric: str = "cosine", mode: str = "cos",
                 sigma: Optional[float] = None, sym: bool = True) -> np.ndarray:
    """
    Construct a dense k-nearest-neighbor affinity matrix.

    Parameters
    ----------
    X : np.ndarray, shape (n_cells, n_features)
        Cell-level feature matrix.
    k : int, default=20
        Number of nearest neighbors retained for each cell.
    metric : str, default="cosine"
        Distance metric passed to ``sklearn.neighbors.NearestNeighbors``.
    mode : {"cos", "heat"}, default="cos"
        Weighting mode. ``"cos"`` converts cosine distance to similarity using
        ``1 - distance``. ``"heat"`` uses a Gaussian heat kernel.
    sigma : float, optional
        Heat-kernel bandwidth. If omitted, the median neighbor distance is used.
    sym : bool, default=True
        Whether to symmetrize the graph by ``max(W, W.T)``.

    Returns
    -------
    np.ndarray
        Dense affinity matrix of shape ``(n_cells, n_cells)``.
    """
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    nnbrs = NearestNeighbors(n_neighbors=min(k + 1, n), metric=metric)
    nnbrs.fit(X)
    dists, idx = nnbrs.kneighbors(X, return_distance=True)

    dists = dists[:, 1:]
    idx = idx[:, 1:]

    W = np.zeros((n, n), dtype=np.float32)

    if mode == "heat":
        if sigma is None:
            sigma = np.median(dists)
            sigma = max(float(sigma), 1e-6)
        denom = 2.0 * (sigma ** 2)
        weights = np.exp(-(dists ** 2) / denom).astype(np.float32)
    elif mode == "cos":
        weights = (1.0 - dists).astype(np.float32)
        weights = np.clip(weights, 0.0, 1.0)
    else:
        raise ValueError("mode must be 'heat' or 'cos'")

    for i in range(n):
        W[i, idx[i]] = weights[i]

    if sym:
        W = np.maximum(W, W.T)

    np.fill_diagonal(W, 0.0)
    return W


def knn_affinity_local_scaling(X, k: int = 20, metric: str = "cosine",
                               sym: bool = True, eps: float = 1e-12) -> np.ndarray:
    """
    Construct a self-tuning kNN affinity matrix with local scaling.

    The affinity uses

    ``W_ij = exp(-d_ij^2 / (sigma_i * sigma_j + eps))``,

    where ``sigma_i`` is the distance from cell ``i`` to its k-th nearest
    neighbor. This is the graph construction used in the Chen-2019 notebook.

    Parameters
    ----------
    X : np.ndarray, shape (n_cells, n_features)
        Feature matrix, usually L2-normalized PCA/LSI features.
    k : int, default=20
        Number of nearest neighbors.
    metric : str, default="cosine"
        Neighbor search distance metric.
    sym : bool, default=True
        Whether to symmetrize the graph.
    eps : float, default=1e-12
        Numerical constant to avoid division by zero.

    Returns
    -------
    np.ndarray
        Dense local-scaling affinity matrix.
    """
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    k_eff = min(k + 1, n)
    nnbrs = NearestNeighbors(n_neighbors=k_eff, metric=metric)
    nnbrs.fit(X)
    dists, idx = nnbrs.kneighbors(X, return_distance=True)

    dists = dists[:, 1:]
    idx = idx[:, 1:]

    sigma = dists[:, -1].astype(np.float32)
    sigma = np.maximum(sigma, 1e-6)

    W = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        j = idx[i]
        dij = dists[i].astype(np.float32)
        denom = sigma[i] * sigma[j] + eps
        W[i, j] = np.exp(-(dij ** 2) / denom).astype(np.float32)

    if sym:
        W = np.maximum(W, W.T)
    np.fill_diagonal(W, 0.0)
    return W


def knn_sparsify_dense(W: np.ndarray, k: int = 20) -> np.ndarray:
    """
    Keep the top-k neighbors per row in a dense affinity matrix.

    Parameters
    ----------
    W : np.ndarray, shape (n_cells, n_cells)
        Dense affinity matrix.
    k : int, default=20
        Number of strongest row-wise edges retained.

    Returns
    -------
    np.ndarray
        Symmetric sparsified dense matrix.
    """
    W = np.asarray(W, dtype=np.float32)
    n = W.shape[0]
    W2 = np.zeros_like(W)
    for i in range(n):
        row = W[i]
        nn_idx = np.argpartition(row, -k)[-k:] if k < n else np.arange(n)
        W2[i, nn_idx] = row[nn_idx]
    W2 = np.maximum(W2, W2.T)
    np.fill_diagonal(W2, 0.0)
    return W2


def row_normalize(W: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Row-normalize a nonnegative matrix.

    Parameters
    ----------
    W : np.ndarray
        Input matrix.
    eps : float, default=1e-12
        Numerical constant to avoid division by zero.

    Returns
    -------
    np.ndarray
        Row-normalized matrix with dtype float32.
    """
    W = W.astype(np.float32, copy=False)
    rs = W.sum(axis=1, keepdims=True)
    return (W / (rs + eps)).astype(np.float32)


def snf_fusion_dense_anchored(W_list: Iterable[np.ndarray], k: int = 20,
                              t: int = 20, alpha: float = 0.2,
                              eps: float = 1e-12) -> np.ndarray:
    """
    Fuse multiple modality-specific graphs using geometry-anchored SNF.

    Compared with standard SNF, this variant preserves part of each original
    modality graph at every iteration:

    ``P_i <- (1 - alpha) * Normalize(S_i @ P_others @ S_i.T) + alpha * P0_i``

    Parameters
    ----------
    W_list : iterable of np.ndarray
        Modality-specific dense affinity matrices. Each matrix must have shape
        ``(n_cells, n_cells)``.
    k : int, default=20
        Number of neighbors used to construct the sparse diffusion operators.
    t : int, default=20
        Number of SNF cross-diffusion iterations.
    alpha : float, default=0.2
        Anchor strength in ``[0, 1]``. Larger values preserve original modality
        geometry more strongly.
    eps : float, default=1e-12
        Numerical constant for row normalization.

    Returns
    -------
    np.ndarray
        Symmetric fused similarity matrix with shape ``(n_cells, n_cells)``.
    """
    W_list = [np.asarray(W, dtype=np.float32) for W in W_list]
    if len(W_list) < 2:
        raise ValueError("W_list must contain at least two modality graphs.")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1].")

    m = len(W_list)
    n = W_list[0].shape[0]
    for W in W_list:
        if W.shape != (n, n):
            raise ValueError("All affinity matrices must have shape (n_cells, n_cells).")

    P0_list: List[np.ndarray] = []
    S_list: List[np.ndarray] = []

    for W in W_list:
        W0 = W.copy()
        np.fill_diagonal(W0, 0.0)
        W0 = np.maximum(W0, 0.0)

        P0 = row_normalize(W0, eps=eps)
        P0_list.append(P0)

        S = knn_sparsify_dense(W0, k=k)
        S = row_normalize(S, eps=eps)
        S_list.append(S)

    P_list = [p.copy() for p in P0_list]

    for _ in range(t):
        P_new = []
        P_bar = sum(P_list) / m
        for i in range(m):
            P_others = (m * P_bar - P_list[i]) / (m - 1)
            Pi = S_list[i] @ P_others @ S_list[i].T
            Pi = row_normalize(Pi, eps=eps)
            Pi = (1.0 - alpha) * Pi + alpha * P0_list[i]
            Pi = row_normalize(Pi, eps=eps)
            P_new.append(Pi.astype(np.float32))
        P_list = P_new

    P_fused = sum(P_list) / m
    P_fused = 0.5 * (P_fused + P_fused.T)
    np.fill_diagonal(P_fused, 0.0)
    return P_fused.astype(np.float32)


def dense_to_torch_sparse(A: np.ndarray, device):
    """
    Convert a dense NumPy matrix to a PyTorch sparse COO tensor.

    Parameters
    ----------
    A : np.ndarray, shape (n_cells, n_cells)
        Dense matrix.
    device : torch.device or str
        Target device.

    Returns
    -------
    torch.Tensor
        Sparse COO tensor.
    """
    import torch

    A = np.asarray(A)
    rows, cols = np.nonzero(A)
    vals = A[rows, cols].astype(np.float32)
    idx = torch.tensor(np.vstack([rows, cols]), dtype=torch.long, device=device)
    val = torch.tensor(vals, dtype=torch.float32, device=device)
    n = A.shape[0]
    return torch.sparse_coo_tensor(idx, val, (n, n), device=device).coalesce()


def normalize_adj_torch_sparse(adj):
    """
    Symmetrically normalize a sparse adjacency matrix with self-loops.

    Computes ``D^{-1/2} (A + I) D^{-1/2}``.

    Parameters
    ----------
    adj : torch.Tensor
        Sparse COO adjacency tensor with shape ``(n_cells, n_cells)``.

    Returns
    -------
    torch.Tensor
        Normalized sparse COO adjacency tensor.
    """
    import torch

    n = adj.shape[0]
    eye_idx = torch.arange(n, device=adj.device)
    eye = torch.sparse_coo_tensor(
        torch.stack([eye_idx, eye_idx], dim=0),
        torch.ones(n, device=adj.device),
        (n, n),
        device=adj.device,
    ).coalesce()

    A = (adj + eye).coalesce()
    deg = torch.sparse.sum(A, dim=1).to_dense()
    deg_inv_sqrt = torch.pow(deg + 1e-12, -0.5)

    idx = A.indices()
    val = A.values()
    val_norm = val * deg_inv_sqrt[idx[0]] * deg_inv_sqrt[idx[1]]
    return torch.sparse_coo_tensor(idx, val_norm, A.shape, device=A.device).coalesce()
