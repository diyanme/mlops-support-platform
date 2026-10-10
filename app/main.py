from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI
from prometheus_client import make_asgi_app

from app.metrics import (
    PREDICTION_ERRORS,
    PREDICTION_LATENCY,
    PREDICTION_REQUESTS,
    PREDICTIONS_TOTAL,
)
from app.model import ModelLoader
from app.schemas import PredictionRequest


model_loader = ModelLoader()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Application startup
    model_loader.load()

    yield

    # Application shutdown
    # Cleanup can be added here later if needed.


app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


@app.get("/ready")
def ready():
    if model_loader.model is None:
        return {
            "status": "not_ready",
        }

    return {
        "status": "ready",
    }


@app.get("/model")
def model_info():
    return {
        "model_name": model_loader.model_name,
        "model_alias": model_loader.model_alias,
        "model_uri": model_loader.model_uri,
    }

@app.post("/predict")
def predict(request: PredictionRequest):
    PREDICTION_REQUESTS.inc()

    with PREDICTION_LATENCY.time():
        try:
            data = pd.DataFrame(
                [
                    {
                        "tenure_months": request.tenure_months,
                        "monthly_charges": request.monthly_charges,
                        "support_tickets": request.support_tickets,
                        "contract_type": request.contract_type,
                    }
                ]
            )

            prediction = model_loader.predict(data)
            probabilities = model_loader.predict_proba(data)

            predicted_class = int(prediction[0])
            churn_probability = probabilities[0][1]

            PREDICTIONS_TOTAL.labels(
                prediction=str(predicted_class)
            ).inc()

            return {
                "prediction": predicted_class,
                "churn_probability": round(
                    float(churn_probability),
                    4,
                ),
            }

        except Exception:
            PREDICTION_ERRORS.inc()
            raise
