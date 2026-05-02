#!/usr/bin/env python3
"""
Command-line entry point for the Chen-2019 sxSNF workflow.
"""

from __future__ import annotations

import argparse

from sxsnf.config import SxSNFConfig
from sxsnf.pipeline import run_chen2019_from_h5ad


def parse_args():
    parser = argparse.ArgumentParser(description="Run sxSNF on Chen-2019 RNA/ATAC h5ad files.")
    parser.add_argument("--rna", required=True, help="Path to Chen-2019-RNA.h5ad")
    parser.add_argument("--atac", required=True, help="Path to Chen-2019-ATAC.h5ad")
    parser.add_argument("--outdir", default="sxsnf_multimodal_run_anchored_deepgcn2")
    parser.add_argument("--label_key", default="cell_type")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")

    parser.add_argument("--n_top_genes", type=int, default=2000)
    parser.add_argument("--n_pcs", type=int, default=100)
    parser.add_argument("--lsi_components", type=int, default=100)
    parser.add_argument("--lsi_iter", type=int, default=15)

    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--t", type=int, default=30)
    parser.add_argument("--fused_topk", type=int, default=30)
    parser.add_argument("--snf_alpha", type=float, default=0.1)

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--neg_ratio", type=float, default=1.0)

    parser.add_argument("--gnn_layers", type=int, default=6)
    parser.add_argument("--gcn2_alpha", type=float, default=0.1)
    parser.add_argument("--gcn2_beta", type=float, default=0.5)
    parser.add_argument("--no_norm", action="store_true")

    parser.add_argument("--leiden_k", type=int, default=15)
    parser.add_argument("--leiden_resolution", type=float, default=0.3)
    parser.add_argument("--pre_snf_resolution", type=float, default=1.0)
    parser.add_argument("--snf_resolution", type=float, default=0.3)
    parser.add_argument("--atac_leiden_k", type=int, default=15)
    return parser.parse_args()


def main():
    args = parse_args()
    config = SxSNFConfig(
        seed=args.seed,
        device=args.device,
        k=args.k,
        t=args.t,
        fused_topk=args.fused_topk,
        snf_alpha=args.snf_alpha,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        mask_ratio=args.mask_ratio,
        neg_ratio=args.neg_ratio,
        gnn_layers=args.gnn_layers,
        gcn2_alpha=args.gcn2_alpha,
        gcn2_beta=args.gcn2_beta,
        use_norm=not args.no_norm,
        leiden_k=args.leiden_k,
        leiden_resolution=args.leiden_resolution,
        pre_snf_resolution=args.pre_snf_resolution,
        snf_resolution=args.snf_resolution,
        atac_leiden_k=args.atac_leiden_k,
        outdir=args.outdir,
    )

    metrics = run_chen2019_from_h5ad(
        rna_path=args.rna,
        atac_path=args.atac,
        config=config,
        label_key=args.label_key,
        n_top_genes=args.n_top_genes,
        n_pcs=args.n_pcs,
        lsi_components=args.lsi_components,
        lsi_iter=args.lsi_iter,
    )

    print("Done. Stage metrics:")
    for stage, values in metrics.items():
        print(stage, "=>", f"ARI={values['ARI']:.4f}, NMI={values['NMI']:.4f}, AMI={values['AMI']:.4f}")


if __name__ == "__main__":
    main()
