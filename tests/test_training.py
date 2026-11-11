
import pandas as pd
import pytest

from train import load_data, train_and_evaluate


def test_load_data_returns_valid_features_and_target(tmp_path):
    data = pd.DataFrame({
        "open_time": [1, 2, 3, 4, 5],
        "close_time": [2, 3, 4, 5, 6],
        "feature_a": [10, 20, 30, 40, 50],
        "feature_b": [1, 2, 3, 4, 5],
        "target_close_next": [11, 21, 31, 41, 51],
    })

    file_path = tmp_path / "sample.csv"
    data.to_csv(file_path, index=False)

    X, y = load_data(file_path)

    assert len(X) == 5
    assert len(y) == 5
    assert "feature_a" in X.columns
    assert "feature_b" in X.columns
    assert "target_close_next" not in X.columns
    assert "open_time" not in X.columns


def test_load_data_rejects_missing_target(tmp_path):
    data = pd.DataFrame({
        "feature_a": [10, 20, 30],
    })

    file_path = tmp_path / "invalid.csv"
    data.to_csv(file_path, index=False)

    with pytest.raises(ValueError, match="tidak ditemukan"):
        load_data(file_path)


def test_train_and_evaluate_returns_metrics():
    X = pd.DataFrame({
        "feature_a": list(range(1, 51)),
        "feature_b": [value * 2 for value in range(1, 51)],
    })

    y = pd.Series([value * 3 + 5 for value in range(1, 51)])

    params = {
        "n_estimators": 10,
        "learning_rate": 0.05,
        "num_leaves": 5,
    }

    model, metrics, run_id = train_and_evaluate(
        X,
        y,
        params,
        experiment_number=99,
    )

    assert model is not None
    assert run_id is not None
    assert len(run_id) > 0
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "mape" in metrics

    assert metrics["rmse"] >= 0
    assert metrics["mae"] >= 0
    assert metrics["mape"] >= 0
