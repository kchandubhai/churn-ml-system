import os
import yaml
import logging
from src.logger import setup_logging
from src.data.load_data import load_csv
from src.data.validate_schema import validate_schema
from src.data.save_clean_data import save_data

def main():
    try:
        with open("configs/config.yaml") as f:
            config = yaml.safe_load(f)

        env = os.getenv("ENV", config["app"]["env"])

        if env == "ci":
            setup_logging("ERROR")
        elif env == "prod":
            setup_logging("INFO")
        else:
            setup_logging(config["logging"]["level"])

        logging.info(f"Running in environment: {env}")

        df = load_csv(config["data"]["raw_data_path"])
        validate_schema(df)
        save_data(df, config["data"]["processed_data_path"])

        logging.info("Pipeline completed successfully")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
