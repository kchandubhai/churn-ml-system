import mlflow
import mlflow.sklearn
import pandas as pd

MODEL_NAME = "ChurnClassifierPipeline"
MODEL_ALIAS = "prod"


def load_pipeline():
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    pipeline = mlflow.sklearn.load_model(model_uri)

    if pipeline is None:
        raise RuntimeError(
            f"Failed to load model {MODEL_NAME}@{MODEL_ALIAS} from MLflow registry"
        )

    return pipeline


def predict(raw_input: pd.DataFrame):
    pipeline = load_pipeline()
    return pipeline.predict(raw_input)


if __name__ == "__main__":
    # RAW input — no preprocessing
    sample_input = pd.DataFrame([{
    "customer_id": "C999",
    "age": 45,
    "tenure_months": 12,
    "monthly_charges": 80,
    "total_charges": 960,
    "contract_type": "yearly"
}])


    prediction = predict(sample_input)
    print("Prediction:", prediction)
