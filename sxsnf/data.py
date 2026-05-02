"""
Data loading and preprocessing utilities for the Chen-2019 RNA/ATAC workflow.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def load_h5ad(path: str):
    """
    Load an AnnData object from an ``.h5ad`` file.

    Parameters
    ----------
    path : str
        Path to the input ``.h5ad`` file.

    Returns
    -------
    anndata.AnnData
        Loaded AnnData object.
    """
    import anndata as ad

    return ad.read_h5ad(path)


def preprocess_rna(rna, n_top_genes: int = 2000, n_pcs: int = 100,
                   cell_type_key: str = "cell_type"):
    """
    Preprocess RNA AnnData using the notebook workflow.

    Steps include count-layer preservation, highly variable gene selection,
    library-size normalization, log transformation, scaling, PCA, neighbors
    and UMAP.

    Parameters
    ----------
    rna : anndata.AnnData
        RNA AnnData object.
    n_top_genes : int, default=2000
        Number of highly variable genes.
    n_pcs : int, default=100
        Number of PCA components.
    cell_type_key : str, default="cell_type"
        Cell-type label used for optional UMAP plotting.

    Returns
    -------
    anndata.AnnData
        Processed RNA AnnData object.
    """
    import scanpy as sc

    rna = rna.copy()
    rna.layers["counts"] = rna.X.copy()
    sc.pp.highly_variable_genes(rna, n_top_genes=n_top_genes, flavor="seurat_v3")
    sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)
    sc.pp.scale(rna)
    sc.tl.pca(rna, n_comps=n_pcs, svd_solver="auto")
    sc.pp.neighbors(rna, metric="cosine")
    sc.tl.umap(rna)
    return rna


def preprocess_atac(atac, n_components: int = 100, n_iter: int = 15):
    """
    Preprocess ATAC AnnData using scGLUE LSI and Scanpy neighbors.

    Parameters
    ----------
    atac : anndata.AnnData
        ATAC AnnData object.
    n_components : int, default=100
        Number of LSI components.
    n_iter : int, default=15
        Number of randomized SVD iterations used by scGLUE LSI.

    Returns
    -------
    anndata.AnnData
        Processed ATAC AnnData object containing ``obsm["X_lsi"]``.
    """
    import scanpy as sc
    import scglue

    atac = atac.copy()
    scglue.data.lsi(atac, n_components=n_components, n_iter=n_iter)
    sc.pp.neighbors(atac, use_rep="X_lsi", metric="cosine")
    sc.tl.umap(atac)
    return atac


def check_cell_alignment(rna, atac) -> bool:
    """
    Check that RNA and ATAC objects are in the same cell order.

    Parameters
    ----------
    rna : anndata.AnnData
        RNA object.
    atac : anndata.AnnData
        ATAC object.

    Returns
    -------
    bool
        ``True`` if ``obs_names`` are aligned.
    """
    return bool((rna.obs_names == atac.obs_names).all())


def make_chen2019_representations(rna, atac, label_key: str = "cell_type") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract RNA PCA, ATAC LSI and integer labels from processed AnnData objects.

    Parameters
    ----------
    rna : anndata.AnnData
        Processed RNA object with ``obsm["X_pca"]``.
    atac : anndata.AnnData
        Processed ATAC object with ``obsm["X_lsi"]``.
    label_key : str, default="cell_type"
        Label column in ``rna.obs``.

    Returns
    -------
    tuple of np.ndarray
        ``(rna_pca, atac_lsi, labels)``.
    """
    if not check_cell_alignment(rna, atac):
        raise ValueError("RNA/ATAC cell order mismatch: obs_names are not aligned.")

    rna_pca = rna.obsm["X_pca"]
    atac_lsi = atac.obsm["X_lsi"]
    labels = rna.obs[label_key].astype("category").cat.codes.values
    return rna_pca, atac_lsi, labels
