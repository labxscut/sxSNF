"""
Smoke tests for sxSNF package imports.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_import_package():
    import sxsnf

    assert sxsnf.__version__


def test_lazy_package_import_avoids_heavy_deps():
    """Importing ``sxsnf`` alone should not load PyTorch (lazy re-exports)."""
    code = (
        "import sys; import sxsnf; assert sxsnf.__version__; "
        "assert 'torch' not in sys.modules"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    r = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr


def test_config_defaults():
    from sxsnf import SxSNFConfig

    config = SxSNFConfig()
    assert config.k == 20
    assert config.t == 30
