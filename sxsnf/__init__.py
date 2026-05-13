"""
sxSNF: Similarity Network Fusion with self-supervised graph learning.

This package reorganizes the Chen-2019 notebook workflow into reusable modules:
data preprocessing, modality-specific graph construction, geometry-anchored SNF,
masked-edge self-supervised graph learning, clustering evaluation, and diagnostics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["SxSNFConfig", "run_multimodal_flow"]
__version__ = "0.2.0"

if TYPE_CHECKING:
    from .config import SxSNFConfig
    from .pipeline import run_multimodal_flow


def __getattr__(name: str) -> Any:
    if name == "SxSNFConfig":
        from .config import SxSNFConfig

        return SxSNFConfig
    if name == "run_multimodal_flow":
        from .pipeline import run_multimodal_flow

        return run_multimodal_flow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
