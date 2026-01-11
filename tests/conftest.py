import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


import pandas as pd
import pytest

@pytest.fixture
def valid_churn_dataframe():
    return pd.DataFrame({
        "customer_id": [1, 2],
        "age": [30, 40],
        "tenure": [12, 24],
        "monthly_charges": [100, 200],
        "churn": [0, 1],
    })

