import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.lightgbm

from mlflow import MlflowClient
from lightgbm import LGBMRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)

DATA_PATH = "data/processed/btc_usdt_1h_features.csv"
TARGET = "target_close_next"
EXPERIMENT_NAME = "BTCUSDT-LightGBM-Regression"

EXPERIMENTS = [
    {"n_estimators": 100, "learning_rate": 0.05, "num_leaves": 15},
    {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31},
    {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 31},
]


BASELINE_METRICS = {
    "rmse": 521.2710,
    "mae": 354.3749,
    "mape": 0.4393,
}


def select_best_model(results):
    return min(
        results,
        key=lambda result: (
            result["rmse"],
            result["mae"],
            result["mape"],
        ),
    )

def validate_model(candidate_metrics):
    checks = {
        "rmse": round(candidate_metrics["rmse"], 4)
        <= round(BASELINE_METRICS["rmse"], 4),

        "mae": round(candidate_metrics["mae"], 4)
        <= round(BASELINE_METRICS["mae"], 4),

        "mape": round(candidate_metrics["mape"], 4)
        <= round(BASELINE_METRICS["mape"], 4),
    }

    print("\nHasil Validasi Model:")
    for metric, passed in checks.items():
        status = "LULUS" if passed else "GAGAL"
        print(f"{metric.upper():<6}: {status}")

    return all(checks.values())

def register_model_if_valid(best_result, is_valid):
    if not is_valid:
        print("\nModel tidak lolos validasi.")
        print("Registrasi ke MLflow dibatalkan.")
        return None

    model_name = "BTCUSDT-LightGBM-Best"
    run_id = best_result["run_id"]
    model_uri = f"runs:/{run_id}/model"

    client = MlflowClient()

    print("\nMendaftarkan model ke MLflow Model Registry...")

    model_version = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id,
        description=(
            f"Model hasil eksperimen {best_result['experiment']} "
            "yang lolos validasi baseline."
        ),
    )

    client.set_registered_model_alias(
        name=model_name,
        alias="staging",
        version=model_version.version,
    )

    print("\nRegistrasi model berhasil.")
    print(f"Nama model : {model_name}")
    print(f"Versi      : {model_version.version}")
    print("Alias      : staging")

    return model_version.version

def load_data(data_path=DATA_PATH):
    df = pd.read_csv(data_path)

    if TARGET not in df.columns:
        raise ValueError(f"Kolom target '{TARGET}' tidak ditemukan.")

    df = df.dropna(subset=[TARGET]).copy()

    exclude_cols = [TARGET, "open_time", "close_time"]
    X = df.drop(columns=exclude_cols, errors="ignore")
    X = X.select_dtypes(include=[np.number])
    y = df[TARGET]

    valid = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    if X.empty:
        raise ValueError("Dataset tidak memiliki data valid.")

    return X, y


def train_and_evaluate(X, y, params, experiment_number):
    split_index = int(len(X) * 0.8)

    if split_index == 0 or split_index >= len(X):
        raise ValueError("Dataset tidak cukup untuk pembagian train/test.")

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    model = LGBMRegressor(
        **params,
        random_state=42,
        verbosity=-1,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "mape": float(
            mean_absolute_percentage_error(y_test, predictions) * 100
        ),
    }

    with mlflow.start_run(
        run_name=f"LightGBM-Experiment-{experiment_number}"
    ) as run:
        mlflow.log_params(params)
        mlflow.log_param("model_type", "LightGBMRegressor")
        mlflow.log_param("dataset", DATA_PATH)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("features", ",".join(X.columns))

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        mlflow.lightgbm.log_model(model, name="model")

    return model, metrics, run.info.run_id


def run_training(data_path=DATA_PATH):
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_data(data_path)
    results = []
    trained_models = []

    os.makedirs("models", exist_ok=True)

    for i, params in enumerate(EXPERIMENTS, start=1):
        model, metrics, run_id = train_and_evaluate(X, y, params, i)

        result = {
        "experiment": i,
        "parameters": params,
        "run_id": run_id,
        **metrics,
    }

        results.append(result)
        trained_models.append(model)

        print(f"\nEksperimen {i}")
        print(f"Parameter : {params}")
        print(f"RMSE      : {metrics['rmse']:.4f}")
        print(f"MAE       : {metrics['mae']:.4f}")
        print(f"MAPE      : {metrics['mape']:.4f}%")

    best_result = select_best_model(results)
    best_model = trained_models[best_result["experiment"] - 1]

    print("\nModel Terbaik:")
    print(f"Eksperimen: {best_result['experiment']}")
    print(f"RMSE      : {best_result['rmse']:.4f}")
    print(f"MAE       : {best_result['mae']:.4f}")
    print(f"MAPE      : {best_result['mape']:.4f}%")

    is_valid = validate_model(best_result)

    print("\nStatus Model:", "VALID" if is_valid else "TIDAK VALID")

    registered_version = register_model_if_valid(best_result, is_valid)

    return {
        "results": results,
        "best_result": best_result,
        "best_model": best_model,
        "is_valid": is_valid,
        "registered_version": registered_version,
    }


if __name__ == "__main__":
    run_training()
