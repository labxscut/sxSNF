"""
sxSNF: Similarity Network Fusion with self-supervised graph learning.

This package reorganizes the Chen-2019 notebook workflow into reusable modules:
data preprocessing, modality-specific graph construction, geometry-anchored SNF,
masked-edge self-supervised graph learning, clustering evaluation, and diagnostics.
"""

from .config import SxSNFConfig
from .pipeline import run_multimodal_flow

__all__ = ["SxSNFConfig", "run_multimodal_flow"]
__version__ = "0.2.0"
