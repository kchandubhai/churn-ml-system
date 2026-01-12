from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd

MODEL_NAME = "ChurnClassifierPipeline"
MODEL_ALIAS = "prod"

app = FastAPI(title="Churn Prediction API")

# ---- Load model ONCE at startup ----
model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
pipeline = mlflow.sklearn.load_model(model_uri)


# ---- Input schema ----
class ChurnRequest(BaseModel):
    customer_id: str
    age: int
    tenure_months: int
    monthly_charges: float
    total_charges: float
    contract_type: str


# ---- Output schema ----
class ChurnResponse(BaseModel):
    churn: int


@app.post("/predict", response_model=ChurnResponse)
def predict(request: ChurnRequest):
    df = pd.DataFrame([request.dict()])
    prediction = pipeline.predict(df)[0]
    return {"churn": int(prediction)}
