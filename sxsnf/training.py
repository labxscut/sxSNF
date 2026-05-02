"""
Self-supervised masked-edge training for sxSNF graph encoders.
"""

from __future__ import annotations

import os
import random
from typing import Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .graph import normalize_adj_torch_sparse
from .models import DeepGCNIIEncoder, edge_logits
from .utils import set_seed


def undirected_edge_set_from_sparse(adj_sparse, drop_self_loops: bool = True) -> Set[Tuple[int, int]]:
    """
    Extract an undirected edge set from a sparse adjacency tensor.

    Parameters
    ----------
    adj_sparse : torch.Tensor
        Sparse COO adjacency matrix.
    drop_self_loops : bool, default=True
        Whether to remove self-loops.

    Returns
    -------
    set of tuple
        Undirected edge set represented as ``(min(u, v), max(u, v))``.
    """
    idx = adj_sparse.indices().T
    if drop_self_loops:
        idx = idx[idx[:, 0] != idx[:, 1]]

    undirected = set()
    for a, b in idx.tolist():
        a = int(a)
        b = int(b)
        undirected.add((a, b) if a < b else (b, a))
    return undirected


def sample_masked_edges_from_sparse_undirected(adj_sparse, mask_ratio: float = 0.1,
                                               seed: int = 42):
    """
    Mask a fraction of undirected graph edges for reconstruction.

    Parameters
    ----------
    adj_sparse : torch.Tensor
        Sparse COO adjacency tensor.
    mask_ratio : float, default=0.1
        Fraction of undirected edges to mask.
    seed : int, default=42
        Random seed.

    Returns
    -------
    tuple
        ``(adj_masked_sparse, pos_edges_masked)`` where the first element is the
        remaining graph and the second is a directed positive edge list.
    """
    set_seed(seed)
    device = adj_sparse.device
    n = adj_sparse.shape[0]

    und_edges = sorted(list(undirected_edge_set_from_sparse(adj_sparse, drop_self_loops=True)))
    edge_count = len(und_edges)
    if edge_count == 0:
        raise ValueError("No undirected edges found in adj_sparse.")

    n_mask = max(1, int(edge_count * mask_ratio))
    if n_mask >= edge_count:
        raise ValueError("mask_ratio too large: it would mask all edges.")

    perm = torch.randperm(edge_count, device=device)
    masked_und = [und_edges[i] for i in perm[:n_mask].tolist()]

    pos_masked = []
    for u, v in masked_und:
        pos_masked.append((u, v))
        pos_masked.append((v, u))
    pos_edges_masked = torch.tensor(pos_masked, dtype=torch.long, device=device)

    idx = adj_sparse.indices().T
    val = adj_sparse.values()
    masked_set = set(masked_und)

    keep_pairs, keep_vals = [], []
    for (a, b), w in zip(idx.tolist(), val.tolist()):
        a = int(a)
        b = int(b)
        if a == b:
            continue
        uu, vv = (a, b) if a < b else (b, a)
        if (uu, vv) in masked_set:
            continue
        keep_pairs.append((a, b))
        keep_vals.append(float(w))

    if len(keep_pairs) == 0:
        raise ValueError("All edges removed after masking; reduce mask_ratio.")

    keep_idx = torch.tensor(keep_pairs, dtype=torch.long, device=device).T
    keep_val = torch.tensor(keep_vals, dtype=torch.float32, device=device)
    adj_masked = torch.sparse_coo_tensor(keep_idx, keep_val, (n, n), device=device).coalesce()
    return adj_masked, pos_edges_masked


def negative_sampling_undirected(n_nodes: int, num_samples: int,
                                 existing_undirected_set, device,
                                 seed: int = 42):
    """
    Sample undirected non-edges and return them as directed edge pairs.

    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    num_samples : int
        Number of undirected negative samples.
    existing_undirected_set : set of tuple
        Existing undirected graph edges.
    device : torch.device
        Target device.
    seed : int, default=42
        Random seed.

    Returns
    -------
    torch.Tensor
        Directed negative edge list with shape ``(2 * num_samples, 2)``.
    """
    set_seed(seed)
    neg_und = []
    while len(neg_und) < num_samples:
        u = random.randrange(n_nodes)
        v = random.randrange(n_nodes)
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) not in existing_undirected_set:
            neg_und.append((a, b))

    neg_dir = []
    for a, b in neg_und:
        neg_dir.append((a, b))
        neg_dir.append((b, a))
    return torch.tensor(neg_dir, dtype=torch.long, device=device)


def train_masked_edge_ssl(
    features,
    adj_fused_sparse,
    hidden_dim: int = 128,
    emb_dim: int = 64,
    dropout: float = 0.5,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    epochs: int = 200,
    mask_ratio: float = 0.1,
    neg_ratio: float = 1.0,
    seed: int = 42,
    outdir: str = "training_process",
    gnn_layers: int = 6,
    gcn2_alpha: float = 0.1,
    gcn2_beta: float = 0.5,
    use_norm: bool = True,
    log_every: int = 10,
):
    """
    Train a DeepGCNII encoder with masked-edge reconstruction.

    Parameters
    ----------
    features : torch.Tensor
        Node features with shape ``(n_cells, n_features)``.
    adj_fused_sparse : torch.Tensor
        Sparse fused graph adjacency.
    hidden_dim : int, default=128
        Encoder hidden dimension.
    emb_dim : int, default=64
        Output embedding dimension.
    dropout : float, default=0.5
        Dropout probability.
    lr : float, default=1e-3
        Adam learning rate.
    weight_decay : float, default=5e-4
        Adam weight decay.
    epochs : int, default=200
        Number of training epochs.
    mask_ratio : float, default=0.1
        Fraction of graph edges masked as positive examples.
    neg_ratio : float, default=1.0
        Negative-to-positive sampling ratio.
    seed : int, default=42
        Random seed.
    outdir : str, default="training_process"
        Directory where ``loss_history.npy`` is saved.
    gnn_layers : int, default=6
        Number of GCNII-style graph blocks.
    gcn2_alpha : float, default=0.1
        Initial residual strength.
    gcn2_beta : float, default=0.5
        Identity mapping strength.
    use_norm : bool, default=True
        Whether to use LayerNorm.
    log_every : int, default=10
        Print loss every N epochs. Set to 0 to disable.

    Returns
    -------
    tuple
        Trained model and normalized masked adjacency.
    """
    os.makedirs(outdir, exist_ok=True)
    device = features.device
    set_seed(seed)

    adj_masked_sparse, pos_masked = sample_masked_edges_from_sparse_undirected(
        adj_fused_sparse, mask_ratio=mask_ratio, seed=seed
    )
    adj_norm = normalize_adj_torch_sparse(adj_masked_sparse)
    existing_und = undirected_edge_set_from_sparse(adj_fused_sparse, drop_self_loops=True)

    model = DeepGCNIIEncoder(
        in_dim=features.shape[1],
        hidden_dim=hidden_dim,
        emb_dim=emb_dim,
        num_layers=gnn_layers,
        dropout=dropout,
        alpha=gcn2_alpha,
        beta=gcn2_beta,
        use_norm=use_norm,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_hist = []

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        z = model(features, adj_norm)

        pos_logits = edge_logits(z, pos_masked)
        pos_y = torch.ones_like(pos_logits)

        num_pos_und = pos_masked.shape[0] // 2
        num_neg_und = max(1, int(num_pos_und * neg_ratio))
        neg_edges = negative_sampling_undirected(
            features.shape[0], num_neg_und, existing_und, device, seed + epoch
        )

        neg_logits = edge_logits(z, neg_edges)
        neg_y = torch.zeros_like(neg_logits)

        logits = torch.cat([pos_logits, neg_logits], dim=0)
        y = torch.cat([pos_y, neg_y], dim=0)

        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        opt.step()

        loss_hist.append(float(loss.item()))
        if log_every and epoch % log_every == 0:
            print(f"Epoch {epoch}/{epochs}  loss={loss.item():.6f}")

    np.save(os.path.join(outdir, "loss_history.npy"), np.array(loss_hist, dtype=np.float32))
    return model, adj_norm
