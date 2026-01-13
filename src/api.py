from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import logging
import uuid
from datetime import datetime


MODEL_NAME = "ChurnClassifierPipeline"
MODEL_ALIAS = "prod"

app = FastAPI(title="Churn Prediction API")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger("inference")


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
    request_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()

    df = pd.DataFrame([request.dict()])
    prediction = int(pipeline.predict(df)[0])

    log_record = {
        "event": "inference",
        "timestamp": timestamp,
        "request_id": request_id,
        "model": f"{MODEL_NAME}@{MODEL_ALIAS}",
        "features": {
            "age": request.age,
            "tenure_months": request.tenure_months,
            "monthly_charges": request.monthly_charges,
            "total_charges": request.total_charges,
            "contract_type": request.contract_type,
        },
        "prediction": prediction,
    }

    logger.info(log_record)

    return {"churn": prediction}

