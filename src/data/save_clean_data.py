import logging

def save_data(df, path: str):
    """
    Saves the cleaned dataframe to disk.
    """
    logging.info(f"Saving cleaned data to {path}")
    df.to_csv(path, index=False)
