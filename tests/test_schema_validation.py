import pandas as pd
import pytest
from src.data.validate_schema import validate_schema

REQUIRED_COLUMNS = [
    "customer_id",
    "age",
    "tenure",
    "monthly_charges",
    "churn",
]

@pytest.mark.parametrize("missing_column", REQUIRED_COLUMNS)
def test_each_required_column_is_enforced(missing_column):
    data = {
        "customer_id": [1, 2],
        "age": [30, 40],
        "tenure": [12, 24],
        "monthly_charges": [100, 200],
        "churn": [0, 1],
    }

    data.pop(missing_column)
    df = pd.DataFrame(data)

    with pytest.raises(ValueError):
        validate_schema(df)

def test_multiple_missing_columns_reported():
    df = pd.DataFrame({
        "customer_id": [1, 2],
        "age": [30, 40],
        # missing tenure
        # missing monthly_charges
        "churn": [0, 1],
    })

    with pytest.raises(ValueError, match="Missing columns"):
        validate_schema(df)
