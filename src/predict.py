import mlflow
import mlflow.sklearn
import pandas as pd

MODEL_NAME = "ChurnClassifier"
MODEL_STAGE = "Production"

def load_model():
    model_uri = f"models:/ChurnClassifier@prod"
    model = mlflow.sklearn.load_model(model_uri)

    if model is None:
        raise RuntimeError(
            "Failed to load model from MLflow registry. "
            "Check that alias 'prod' exists and model artifacts are available."
        )

    return model

def predict(input_df: pd.DataFrame):
    model = load_model()
    return model.predict(input_df)

if __name__ == "__main__":
    # Example input (must match training features)
    sample_input = pd.DataFrame([{
        "age": 45,
        "tenure_months": 12,
        "monthly_charges": 80,
        "total_charges" : 150,
        "contract_type_yearly": 0
    }])

    predictions = predict(sample_input)
    print("Prediction:", predictions)
