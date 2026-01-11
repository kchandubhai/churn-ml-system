import pandas as pd
from src.data.load_data import load_csv

def test_load_csv_returns_dataframe():
    df = load_csv("data/raw/churn.csv")
    assert isinstance(df, pd.DataFrame)

def test_load_csv_has_rows():
    df = load_csv("data/raw/churn.csv")
    assert len(df) > 0
