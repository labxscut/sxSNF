"""
Configuration objects for sxSNF workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class SxSNFConfig:
    """
    Runtime configuration for the sxSNF multimodal integration pipeline.

    Parameters
    ----------
    seed : int
        Random seed used by NumPy, Python, PyTorch, Leiden and fallback models.
    device : str
        PyTorch device string. Use ``"auto"`` to select CUDA when available.
    k : int
        Number of nearest neighbors used to build modality-specific graphs.
    t : int
        Number of geometry-anchored SNF iterations.
    fused_topk : Optional[int]
        If provided, sparsify the fused dense SNF network by retaining this many
        strongest neighbors per cell.
    snf_alpha : float
        Anchor strength in geometry-anchored SNF. Larger values preserve more
        modality-specific geometry; smaller values allow stronger cross-diffusion.
    hidden_dim : int
        Hidden dimension of the DeepGCNII encoder.
    embedding_dim : int
        Output embedding dimension of the graph encoder.
    dropout : float
        Dropout probability in the encoder.
    lr : float
        Learning rate for masked-edge self-supervised training.
    weight_decay : float
        Weight decay for Adam optimization.
    epochs : int
        Number of self-supervised training epochs.
    mask_ratio : float
        Fraction of undirected graph edges masked as reconstruction positives.
    neg_ratio : float
        Number of negative undirected edges sampled per positive edge.
    gnn_layers : int
        Number of GCNII-style residual graph blocks.
    gcn2_alpha : float
        Initial residual injection strength in each GCNII-style block.
    gcn2_beta : float
        Identity mapping strength in each GCNII-style block.
    use_norm : bool
        Whether to use LayerNorm after each GCNII-style residual block.
    leiden_k : int
        Number of neighbors for Leiden clustering on GNN embeddings.
    leiden_resolution : float
        Resolution for Leiden clustering on GNN embeddings.
    pre_snf_resolution : float
        Resolution for RNA-only and ATAC-only baseline clustering.
    snf_resolution : float
        Resolution for clustering the fused SNF network.
    atac_leiden_k : int
        Number of neighbors for ATAC-only Leiden clustering.
    outdir : str
        Output directory used for metrics and training curves.
    """

    seed: int = 42
    device: str = "auto"

    # SNF / graph
    k: int = 20
    t: int = 30
    fused_topk: Optional[int] = 30
    snf_alpha: float = 0.1

    # GNN
    hidden_dim: int = 128
    embedding_dim: int = 64
    dropout: float = 0.3
    lr: float = 1e-4
    weight_decay: float = 5e-4
    epochs: int = 1500
    mask_ratio: float = 0.5
    neg_ratio: float = 1.0

    # Deep GCNII knobs
    gnn_layers: int = 6
    gcn2_alpha: float = 0.1
    gcn2_beta: float = 0.5
    use_norm: bool = True

    # Clustering
    leiden_k: int = 15
    leiden_resolution: float = 0.3
    pre_snf_resolution: float = 1.0
    snf_resolution: float = 0.3
    atac_leiden_k: int = 15

    outdir: str = "sxsnf_multimodal_run_anchored_deepgcn2"

    def resolved_device(self):
        """
        Return a concrete ``torch.device`` instance.

        Returns
        -------
        torch.device
            CUDA device when ``device="auto"`` and CUDA is available; otherwise CPU.
        """
        import torch

        if self.device == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a plain dictionary.
        """
        return asdict(self)
