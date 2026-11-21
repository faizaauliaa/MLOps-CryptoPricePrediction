"""
Training + registrasi model BTC/USDT ke MLflow Tracking Server.

Dijalankan oleh service `model-trainer` di docker-compose. Script ini:
1. Menunggu MLflow server siap.
2. Kalau model dengan alias production sudah ada di registry, training dilewati
   (data di volume masih ada, jadi tidak perlu training ulang).
3. Kalau belum ada, melatih beberapa konfigurasi LightGBM, memilih yang terbaik,
   lalu mendaftarkannya ke Model Registry dengan alias `production` dan `champion`.

Set FORCE_RETRAIN=true untuk memaksa training ulang.
"""

import os
import sys
import time

import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error


TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "BTCUSDT-LightGBM-Regression")
MODEL_NAME = os.getenv("MODEL_NAME", "BTCUSDT-LightGBM-Best")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")
DATA_PATH = os.getenv("DATA_PATH", "data/processed/btc_usdt_1h_features.csv")
FORCE_RETRAIN = os.getenv("FORCE_RETRAIN", "false").lower() in ("1", "true", "yes")

TARGET_COLUMN = "target_close_next"
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

# Konfigurasi yang dibandingkan (versi 1 & 4 dari eksperimen sebelumnya + satu tambahan)
CONFIGS = [
    {"n_estimators": 100, "learning_rate": 0.05, "num_leaves": 15},
    {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31},
    {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 15},
]


def wait_for_server(client, retries=30, delay=3):
    for attempt in range(1, retries + 1):
        try:
            client.search_experiments(max_results=1)
            print(f"MLflow server siap di {TRACKING_URI}")
            return
        except Exception as error:
            print(f"[{attempt}/{retries}] Menunggu MLflow server... ({error.__class__.__name__})")
            time.sleep(delay)
    sys.exit("MLflow server tidak bisa dihubungi.")


def production_model_exists(client):
    try:
        version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        return version
    except MlflowException:
        return None


def load_dataset():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS].astype("float64")
    y = df[TARGET_COLUMN].astype("float64")

    # Data time series, jadi split berdasarkan urutan waktu (tanpa shuffle)
    split = int(len(df) * 0.8)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def evaluate(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return {"rmse": rmse, "mae": mae, "mape_percent": mape}


def main():
    print("MODEL TRAINING - BTC/USDT (Docker Compose)")
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    wait_for_server(client)

    existing = production_model_exists(client)
    if existing and not FORCE_RETRAIN:
        print(
            f"Model {MODEL_NAME}@{MODEL_ALIAS} sudah ada (versi {existing.version}). "
            "Training dilewati."
        )
        return

    mlflow.set_experiment(EXPERIMENT_NAME)
    X_train, X_test, y_train, y_test = load_dataset()
    print(f"Data train: {len(X_train)} baris | Data test: {len(X_test)} baris")

    results = []
    for i, params in enumerate(CONFIGS, start=1):
        run_params = {**params, "random_state": 42, "verbosity": -1}

        with mlflow.start_run(run_name=f"LightGBM-Compose-{i}") as run:
            model = LGBMRegressor(**run_params)
            model.fit(X_train, y_train)
            metrics = evaluate(y_test.values, model.predict(X_test))

            mlflow.log_params(run_params)
            mlflow.log_param("dataset", DATA_PATH)
            mlflow.log_metrics(metrics)
            mlflow.set_tag("trained_by", "docker-compose/model-trainer")

            model_info = mlflow.lightgbm.log_model(
                model,
                name="model",
                signature=infer_signature(X_train, model.predict(X_train)),
                input_example=X_train.head(2),
            )

        print(
            f"Run {i}: n_estimators={params['n_estimators']}, "
            f"learning_rate={params['learning_rate']}, num_leaves={params['num_leaves']} "
            f"-> RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, "
            f"MAPE={metrics['mape_percent']:.4f}%"
        )
        results.append({"run_id": run.info.run_id, "model_uri": model_info.model_uri, **metrics})

    best = min(results, key=lambda r: r["rmse"])
    print(f"\nModel terbaik: run {best['run_id']} (RMSE={best['rmse']:.4f})")

    registered = mlflow.register_model(best["model_uri"], MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, registered.version)
    client.set_registered_model_alias(MODEL_NAME, "champion", registered.version)
    client.set_model_version_tag(MODEL_NAME, registered.version, "selection", "lowest_rmse")
    client.update_registered_model(
        MODEL_NAME,
        description="Prediksi harga penutupan BTC/USDT satu jam berikutnya (LightGBM).",
    )

    print(
        f"Terdaftar sebagai {MODEL_NAME} versi {registered.version} "
        f"dengan alias '{MODEL_ALIAS}' dan 'champion'."
    )


if __name__ == "__main__":
    main()
