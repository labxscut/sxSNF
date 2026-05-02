"""
General utility functions shared across sxSNF modules.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Union

import numpy as np


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Create an output directory if it does not exist.

    Parameters
    ----------
    path : str or pathlib.Path
        Directory path to create.

    Returns
    -------
    pathlib.Path
        Normalized path object.
    """
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def set_seed(seed: int = 42) -> None:
    """
    Set Python, NumPy and PyTorch random seeds.

    Parameters
    ----------
    seed : int, default=42
        Random seed for reproducible experiments.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # Allow documentation generation and CPU-only utility usage without torch.
        pass


def l2norm_rows(X, eps: float = 1e-12):
    """
    L2-normalize a matrix row by row.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input feature matrix.
    eps : float, default=1e-12
        Small numerical constant to avoid division by zero.

    Returns
    -------
    np.ndarray
        Row-normalized feature matrix with dtype float32.
    """
    X = np.asarray(X, dtype=np.float32)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)
