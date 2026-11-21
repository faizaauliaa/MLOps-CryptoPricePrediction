import os
import mlflow
import pandas as pd

from fastapi import FastAPI

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000"
)

MODEL_NAME = "BTCUSDT-LightGBM-Best"
MODEL_ALIAS = "staging"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(title="BTCUSDT Prediction API")

model = None


@app.on_event("startup")
def load_model():
    global model

    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

    print(f"Loading model from MLflow: {model_uri}")

    model = mlflow.pyfunc.load_model(model_uri)

    print("Model berhasil dimuat.")


@app.get("/")
def home():
    return {
        "message": "BTCUSDT Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(data: dict):
    input_data = pd.DataFrame([data])

    prediction = model.predict(input_data)

    return {
        "prediction": float(prediction[0])
    }