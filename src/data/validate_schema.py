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

    for column in EXPECTED_COLUMNS:
        if column not in df.columns:
            raise ValueError(f"Schema validation failed. Missing column: {column}")

    logging.info("Schema validation passed")
    return True
