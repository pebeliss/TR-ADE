import subprocess, sys, os
from pathlib import Path
import pytest

@pytest.mark.timeout(120)
def test_train_runs_one_epoch():
    root = Path(__file__).resolve().parents[1]
    data = root / "data" / "heart.csv"
    if not data.exists():
        pytest.skip("no data")
    cmd = [sys.executable, str(root / "scripts" / "train.py"), "--data_path", "data", "--epochs", "1"]
    # Keep it lightweight; rely on your train.py to be deterministic enough
    subprocess.check_call(cmd, cwd=root)