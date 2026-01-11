import pandas as pd
import logging

def load_csv(path: str) -> pd.DataFrame:
    """
    Loads a CSV file from the given path and returns a DataFrame.
    """
    logging.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logging.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    return df

# NOTE: CSV loading utility for churn pipeline


