import pandas as pd
import pandas.api.types as ptypes
from pathlib import Path
import pytest

def test_heart_schema_basic():
    f = Path(__file__).resolve().parents[1] / "data" / "heart.csv"
    if not f.exists():
        pytest.skip("no heart.csv")
    df = pd.read_csv(f)
    assert "HeartDisease" in df.columns
    assert ptypes.is_integer_dtype(df["HeartDisease"]) or ptypes.is_bool_dtype(df["HeartDisease"])