"""
PyTorch neural network modules used by sxSNF.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Two-layer MLP used as the graph-embedding projection head.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    hidden_dim : int
        Hidden feature dimension.
    out_dim : int
        Output feature dimension.
    dropout : float, default=0.0
        Dropout probability.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        """
        Project input features.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Projected tensor.
        """
        return self.net(x)


class GCNIIBlock(nn.Module):
    """
    GCNII-style residual graph convolution block.

    The block applies sparse graph propagation, initial residual injection,
    identity mapping, GELU activation, dropout, residual connection and optional
    LayerNorm.

    Parameters
    ----------
    dim : int
        Hidden feature dimension.
    dropout : float, default=0.0
        Dropout probability.
    alpha : float, default=0.1
        Initial residual strength.
    beta : float, default=0.5
        Identity mapping strength.
    use_norm : bool, default=True
        Whether to use LayerNorm.
    """

    def __init__(self, dim: int, dropout: float = 0.0,
                 alpha: float = 0.1, beta: float = 0.5,
                 use_norm: bool = True):
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=True)
        self.dropout = dropout
        self.alpha = alpha
        self.beta = beta
        self.norm = nn.LayerNorm(dim) if use_norm else nn.Identity()

    def forward(self, h, h0, adj_norm_sparse):
        """
        Apply one GCNII-style graph block.

        Parameters
        ----------
        h : torch.Tensor
            Current hidden representation.
        h0 : torch.Tensor
            Initial hidden representation after input projection.
        adj_norm_sparse : torch.Tensor
            Normalized sparse adjacency matrix.

        Returns
        -------
        torch.Tensor
            Updated hidden representation.
        """
        m = torch.sparse.mm(adj_norm_sparse, h)
        m = (1.0 - self.alpha) * m + self.alpha * h0

        m_lin = self.lin(m)
        out = (1.0 - self.beta) * m + self.beta * m_lin

        out = F.gelu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)

        out = out + h
        out = self.norm(out)
        return out


class DeepGCNIIEncoder(nn.Module):
    """
    Deep GCNII-style encoder for fused cell-cell graphs.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    hidden_dim : int
        Hidden feature dimension.
    emb_dim : int
        Output embedding dimension.
    num_layers : int, default=6
        Number of GCNII-style blocks.
    dropout : float, default=0.5
        Dropout probability.
    alpha : float, default=0.1
        Initial residual strength.
    beta : float, default=0.5
        Identity mapping strength.
    use_norm : bool, default=True
        Whether to use LayerNorm.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        emb_dim: int,
        num_layers: int = 6,
        dropout: float = 0.5,
        alpha: float = 0.1,
        beta: float = 0.5,
        use_norm: bool = True,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [
                GCNIIBlock(
                    dim=hidden_dim,
                    dropout=dropout,
                    alpha=alpha,
                    beta=beta,
                    use_norm=use_norm,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_proj = MLP(hidden_dim, hidden_dim, emb_dim, dropout=dropout)

    def forward(self, x, adj_norm_sparse):
        """
        Encode cell features on the fused graph.

        Parameters
        ----------
        x : torch.Tensor
            Input cell feature matrix.
        adj_norm_sparse : torch.Tensor
            Normalized sparse adjacency matrix.

        Returns
        -------
        torch.Tensor
            Cell embeddings.
        """
        h = self.input_proj(x)
        h0 = h
        for block in self.blocks:
            h = block(h, h0, adj_norm_sparse)
        return self.out_proj(h)


def edge_logits(z, edges_uv):
    """
    Dot-product edge decoder.

    Parameters
    ----------
    z : torch.Tensor, shape (n_cells, emb_dim)
        Cell embeddings.
    edges_uv : torch.Tensor, shape (n_edges, 2)
        Directed edge list.

    Returns
    -------
    torch.Tensor
        Edge logits.
    """
    return (z[edges_uv[:, 0]] * z[edges_uv[:, 1]]).sum(dim=1)
