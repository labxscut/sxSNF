"""
Smoke tests for sxSNF package imports.
"""

def test_import_package():
    import sxsnf
    assert sxsnf.__version__


def test_config_defaults():
    from sxsnf import SxSNFConfig
    config = SxSNFConfig()
    assert config.k == 20
    assert config.t == 30
