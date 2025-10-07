# tests/test_smoke.py
import os
from pathlib import Path

def test_repo_layout():
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "train.py").is_file(), "train.py missing"
    assert (root / "configs").is_dir(), "configs/ missing"

def test_imports():
    __import__("TR-ADE")
    __import__("TR-ADE_pipeline")
    __import__("subnetworks")
    __import__("subclasses")
    __import__("utils")
    __import__("model_args")

def test_data_exists_or_is_created():
    # CI creates data/heart.csv earlier; locally, just allow missing gracefully.
    root = Path(__file__).resolve().parents[1]
    f = root / "data" / "heart.csv"
    if not f.exists():
        # not a failure—just a gentle guard for local dev
        import pytest
        pytest.skip("data/heart.csv not present (CI step will create it)")
    assert f.is_file()
