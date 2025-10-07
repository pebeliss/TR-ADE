import pandas as pd
from tests.schema import heart_schema

def test_heart_csv_schema():
    df = pd.read_csv("data/heart.csv")
    heart_schema.validate(df.sample(min(200, len(df))), lazy=True)