
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
def test_each_required_column_is_enforced(valid_churn_dataframe, missing_column):
    df = valid_churn_dataframe.drop(columns=[missing_column])

    with pytest.raises(ValueError):
        validate_schema(df)

def test_multiple_missing_columns_reported(valid_churn_dataframe):
    df = valid_churn_dataframe.drop(columns=["tenure", "monthly_charges"])

    with pytest.raises(ValueError, match="Missing columns"):
        validate_schema(df)
