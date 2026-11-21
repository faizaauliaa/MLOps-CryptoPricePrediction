"""
API inferensi BTC/USDT.

Model tidak dibundel di dalam image. Saat start, API menarik model
`models:/BTCUSDT-LightGBM-Best@production` dari MLflow Tracking Server
(http://mlflow-server:5000 di dalam jaringan docker-compose).
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from mlflow import MlflowClient
from pydantic import BaseModel, Field


TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "BTCUSDT-LightGBM-Best")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")
LOAD_RETRIES = int(os.getenv("MODEL_LOAD_RETRIES", "10"))
LOAD_DELAY = int(os.getenv("MODEL_LOAD_DELAY", "5"))

FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "return_1h",
    "lag_close_1",
    "lag_close_3",
    "lag_close_6",
    "lag_close_24",
    "rolling_mean_6",
    "rolling_std_6",
    "price_range",
    "volume_change",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("api-service")

state = {"model": None, "version": None, "run_id": None, "loaded_at": None, "error": None}


def load_model_from_registry():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

    for attempt in range(1, LOAD_RETRIES + 1):
        try:
            version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
            model = mlflow.pyfunc.load_model(model_uri)

            state.update(
                model=model,
                version=version.version,
                run_id=version.run_id,
                loaded_at=datetime.now(timezone.utc).isoformat(),
                error=None,
            )
            logger.info(
                "Model %s versi %s berhasil ditarik dari %s",
                model_uri, version.version, TRACKING_URI,
            )
            return
        except Exception as error:
            state["error"] = str(error)
            logger.warning("[%d/%d] Gagal memuat model: %s", attempt, LOAD_RETRIES, error)
            time.sleep(LOAD_DELAY)

    logger.error("Model tidak bisa dimuat dari %s", TRACKING_URI)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_from_registry()
    yield


app = FastAPI(
    title="BTC/USDT Price Prediction API",
    description="Prediksi harga penutupan BTC/USDT 1 jam berikutnya. Model diambil dari MLflow Model Registry.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    instances: list[dict[str, float]] = Field(
        ..., min_length=1, description="Satu atau lebih baris fitur."
    )


@app.get("/")
def root():
    return {"service": "api-service", "docs": "/docs", "health": "/health", "model_info": "/model-info"}


@app.get("/health")
def health():
    body = {
        "status": "ok" if state["model"] is not None else "model_not_loaded",
        "mlflow_tracking_uri": TRACKING_URI,
        "model": f"{MODEL_NAME}@{MODEL_ALIAS}",
        "model_version": state["version"],
        "loaded_at": state["loaded_at"],
        "error": state["error"],
    }
    return JSONResponse(body, status_code=200 if state["model"] is not None else 503)


@app.get("/model-info")
def model_info():
    """Mengambil metadata model langsung dari MLflow server (bukti komunikasi antar kontainer)."""
    try:
        client = MlflowClient(tracking_uri=TRACKING_URI)
        version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        run = client.get_run(version.run_id)
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Gagal menghubungi MLflow server: {error}")

    return {
        "mlflow_tracking_uri": TRACKING_URI,
        "registered_model": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "version": version.version,
        "aliases": list(version.aliases),
        "run_id": version.run_id,
        "run_name": run.info.run_name,
        "params": run.data.params,
        "metrics": run.data.metrics,
        "loaded_version_in_api": state["version"],
    }


@app.post("/predict")
def predict(request: PredictRequest):
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model belum dimuat dari MLflow server.")

    df = pd.DataFrame(request.instances)
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Fitur tidak lengkap: {missing}")

    predictions = state["model"].predict(df[FEATURE_COLUMNS].astype("float64"))

    return {
        "model": f"{MODEL_NAME}@{MODEL_ALIAS}",
        "model_version": state["version"],
        "predictions": [round(float(p), 2) for p in predictions],
        "unit": "USDT",
    }


@app.post("/reload")
def reload_model():
    """Tarik ulang model dari registry, misalnya setelah alias production dipindah."""
    load_model_from_registry()
    if state["model"] is None:
        raise HTTPException(status_code=503, detail=state["error"])
    return {"status": "reloaded", "model_version": state["version"]}
