import logging

EXPECTED_COLUMNS = {
    "customer_id": str,
    "age": int,
    "tenure_months": int,
    "monthly_charges": float,
    "total_charges": float,
    "contract_type": str,
    "churn": int
}

def validate_schema(df):
    """
    Validates that the dataframe has expected columns.
    """
    logging.info("Validating data schema")

    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    logging.info("Schema validation passed")
    return True
