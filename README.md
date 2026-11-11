A continuous training system for cryptocurrency price prediction using dynamic market data, automated preprocessing, experiment tracking, and model versioning.

## Project Overview

This project aims to build a machine learning system for predicting BTC/USDT prices that can adapt to changing market data through a continuous training workflow.

The project establishes an MLOps foundation covering data ingestion, preprocessing, feature engineering, model development, experiment tracking, model registration, and inference. Git is used to manage source code and documentation, while DVC tracks dataset and model metadata.

## Project Goals

- Retrieve cryptocurrency market data dynamically from the Binance Public Market Data API.
- Automate data preprocessing and feature engineering.
- Develop and evaluate machine learning models for price prediction.
- Track experiments, parameters, metrics, and model artifacts using MLflow.
- Manage model versions and identify the model currently used for inference.
- Track dataset and model metadata with DVC to support reproducibility and data lineage.
- Prepare the project for future continuous training and monitoring.

## Repository Structure

```text
MLOps-CryptoPricePrediction/
├── data/
│   ├── raw/
│   │   ├── btc_usdt_1h_raw.csv
│   │   └── btc_usdt_1h_YYYYMMDD_HHMMSS.csv
│   └── processed/
│       ├── btc_usdt_1h_clean.csv
│       ├── btc_usdt_1h_features.csv
│       └── btc_usdt_1h_preprocessed.csv
├── models/
├── notebooks/
│   └── 01_initial_eda.ipynb
├── src/
│   ├── data/
│   │   ├── ingestion.py
│   │   ├── cleaning.py
│   │   └── feature_engineering.py
│   ├── ingest_data.py
│   ├── preprocess.py
│   ├── features/
│   ├── training/
│   ├── inference/
│   └── monitoring/
├── config/
│   └── data_config.yaml
├── .devcontainer/
│   └── devcontainer.json
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

Additional experiment and model-management files may be generated during development, including MLflow tracking files, DVC metadata files, and model evaluation scripts.

## Data Sources

The project uses BTC/USDT market data retrieved from the Binance Public Market Data API.

| Configuration | Value |
|---|---|
| Trading pair | BTCUSDT |
| Data interval | 1 hour |
| Initial ingestion size | 100 records |
| Data format | CSV |
| Raw data directory | `data/raw/` |
| Processed data directory | `data/processed/` |

The dataset contains OHLCV market information and derived features used for model training and prediction.

## Data Ingestion

Dynamic data ingestion is implemented in:

`src/ingest_data.py`

The script retrieves the latest BTC/USDT market data and saves it in the `data/raw/` directory. Each execution creates a timestamped file to preserve previously collected data.

Filename pattern:

```text
btc_usdt_1h_YYYYMMDD_HHMMSS.csv
```

Example:

```text
data/raw/btc_usdt_1h_20260926_231644.csv
```

Run the ingestion script:

```bash
python src/ingest_data.py
```

## Data Preprocessing

Automated preprocessing is implemented in:

`src/preprocess.py`

The script identifies the latest raw dataset and performs the following operations:

1. Reads the latest raw dataset.
2. Removes duplicate records.
3. Converts `open_time` and `close_time` from Unix timestamp milliseconds into datetime values.
4. Converts numerical columns to appropriate numeric types.
5. Removes rows containing missing values.
6. Sorts records by `open_time`.
7. Saves the processed dataset to `data/processed/`.

Run preprocessing:

```bash
python src/preprocess.py
```

The resulting dataset is saved as:

```text
data/processed/btc_usdt_1h_preprocessed.csv
```

The latest reported preprocessing output contains 100 rows, 12 columns, and no missing values.

## Data Processing Workflow

```text
Binance Public Market Data API
             |
             v
     Dynamic Ingestion
      src/ingest_data.py
             |
             v
         data/raw/
             |
             v
     Latest Raw Dataset
             |
             v
       Preprocessing
       src/preprocess.py
             |
             v
      data/processed/
             |
             v
     Feature Engineering
             |
             v
     Training Dataset
             |
             v
     Model Training
             |
             v
   MLflow Tracking and Registry
             |
             v
       Model Inference
```

## Data Versioning with DVC

DVC is used alongside Git to track changes to datasets and model metadata.

Git manages source code, configuration, and documentation. DVC tracks data artifacts and their metadata, allowing dataset versions to be connected to the corresponding development and model history.

### Dataset Versioning Workflow

1. Run `python src/ingest_data.py` to retrieve new market data.
2. Run `dvc add data/raw` to update the raw dataset metadata.
3. Stage the DVC metadata and relevant `.gitignore` changes.
4. Commit the metadata changes to Git.
5. Run `dvc status` to check the dataset tracking status.
6. Use `dvc diff <commit_before> <commit_after>` to compare dataset versions.

During a previous versioning test, DVC detected the addition of a timestamped dataset file:

`btc_usdt_1h_20261011_160749.csv`

A DVC remote storage location has not yet been configured.

## Exploratory Data Analysis

The notebook `notebooks/01_initial_eda.ipynb` contains the initial exploratory data analysis.

The analysis focuses on:

- Understanding the structure of cryptocurrency market data.
- Inspecting data quality and completeness.
- Identifying patterns in OHLCV data.
- Preparing considerations for feature engineering and model development.

## Model Development and Experiment Tracking

The project uses MLflow to track model experiments, parameters, evaluation metrics, and model artifacts.

The current regression model is a LightGBM Regressor trained to predict the next-hour BTC/USDT closing price.

The experiment is registered under:

`BTCUSDT-LightGBM-Regression`

The model registry name is:

`BTCUSDT-LightGBM-Best`

### Model Version Comparison

Multiple model configurations were evaluated and registered to maintain a version history.

| Model Version | `n_estimators` | `learning_rate` | `num_leaves` | RMSE | MAE | MAPE |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 0.05 | 15 | 521.2710 | 354.3749 | 0.4393% |
| 4 | 200 | 0.05 | 31 | 596.3409 | 451.8453 | 0.5596% |

Version 1 was selected for production inference because it achieved lower RMSE, MAE, and MAPE than the alternative configuration.

## Model Registry and Production Alias

The selected model is registered in MLflow Model Registry as:

| Property | Value |
|---|---|
| Model name | `BTCUSDT-LightGBM-Best` |
| Selected version | `1` |
| Production alias | `production` |
| Champion alias | `champion` |
| Model framework | LightGBM |

MLflow 3.17.1 uses model aliases to identify model versions for deployment workflows. The `production` alias points to version 1 and is used by the inference code to load the active model.

Model URI:

```text
models:/BTCUSDT-LightGBM-Best@production
```

The model metadata is documented in `model_metadata.yaml` and tracked through DVC using `model_metadata.yaml.dvc`.

## Model Inference

The production model can be loaded programmatically using MLflow's PyFunc interface:

```python
import mlflow

model = mlflow.pyfunc.load_model(
    "models:/BTCUSDT-LightGBM-Best@production"
)
```

The inference test uses a feature row from:

`data/processed/btc_usdt_1h_features.csv`

The target column and timestamp columns are excluded from the input features.

The tested model successfully generated the following next-hour closing price prediction:

**81,244.34 USDT**

This result verifies that the registered production model can be loaded and used for inference.

## Development Workflow

The project follows the GitHub Flow branching strategy.

The `main` branch contains the stable project version. New development is performed in feature branches and merged through Pull Requests after validation.

The workflow consists of:

1. Creating a feature branch.
2. Implementing and testing changes.
3. Committing changes to the feature branch.
4. Pushing the branch to GitHub.
5. Opening a Pull Request.
6. Reviewing and merging validated changes into `main`.

## Future Improvements

Potential next steps include:

- Configuring a DVC remote for shared artifact storage.
- Automating model retraining when new data becomes available.
- Expanding model evaluation and comparison.
- Implementing inference monitoring and data drift detection.
- Integrating the pipeline with a CI/CD workflow.

## LK-8 — Automated End-to-End MLOps Pipeline

This project implements an automated MLOps pipeline using GitHub Actions. The workflow is triggered automatically when code is pushed to the configured branches or when a pull request targets `main`.

### Pipeline Workflow

The pipeline consists of the following stages:

1. Install project dependencies using `requirements.txt`.
2. Run automated tests using `pytest`.
3. Execute model training and evaluation through `train.py`.
4. Compare model performance against predefined baseline metrics (RMSE, MAE, and MAPE).
5. Register the selected model in MLflow Model Registry if it passes validation.

### Model Validation and Registration

The training pipeline evaluates three LightGBM configurations and selects the best-performing model based on RMSE, MAE, and MAPE.

The baseline metrics are:

| Metric | Baseline |
|---|---:|
| RMSE | 521.2710 |
| MAE | 354.3749 |
| MAPE | 0.4393% |

A model that passes all validation criteria is registered under `BTCUSDT-LightGBM-Best` with the `staging` alias. Models that fail validation are not registered.

### Workflow Configuration

The GitHub Actions workflow is defined in:

`.github/workflows/mlops-pipeline.yml`

Automated tests are located in `tests/test_training.py`, while the training, evaluation, and registration processes are implemented in `train.py`.

To trigger the pipeline manually through a code change:

```bash
git add .
git commit -m "docs: update README with LK08 automation"
git push
```

The workflow execution logs can be inspected through the **Actions** tab in GitHub.

### Automation Benefits

This automation integrates testing, training, evaluation, and model registration into one workflow. Automated tests detect implementation errors, baseline validation prevents underperforming models from being registered, and GitHub Actions logs help identify failures at each stage.
