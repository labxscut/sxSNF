#!/usr/bin/env python3
"""
Compatibility wrapper for running sxSNF from the repository root.

Example
-------
python main.py --rna datasets/Chen-2019-RNA.h5ad --atac datasets/Chen-2019-ATAC.h5ad
"""

from scripts.run_chen2019 import main


if __name__ == "__main__":
    main()
