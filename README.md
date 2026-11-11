## Automated End-to-End MLOps Pipeline (LK-8)

The project implements an automated end-to-end MLOps pipeline using GitHub Actions. This pipeline integrates automated testing, model training, evaluation, validation, and model registration into a single workflow.

The objective is to ensure that changes pushed to the repository can be tested and evaluated automatically, reducing the need for manual execution and helping prevent undetected errors during model development.

### Pipeline Components

The automated pipeline consists of the following components:

1. **Continuous Integration Trigger** — GitHub Actions automatically starts the workflow when changes are pushed to the configured branches or when a pull request is opened against `main`.
2. **Automated Testing** — `pytest` executes the project's automated tests to verify the training pipeline and data-loading functionality.
3. **Automated Model Training** — `train.py` trains and evaluates multiple LightGBM configurations.
4. **Model Evaluation** — The pipeline compares evaluation metrics against predefined baseline metrics.
5. **Model Registration** — A model that passes validation is registered in MLflow Model Registry and assigned the `staging` alias.

### GitHub Actions Workflow

The workflow is defined in:

`.github/workflows/mlops-pipeline.yml`

The pipeline is configured to run on pushes to the `main` and `feature/lk08-automation` branches, as well as pull requests targeting `main`.

The workflow performs the following steps:

```yaml
- name: Checkout repository
  uses: actions/checkout@v4

- name: Set up Python
  uses: actions/setup-python@v5
  with:
    python-version: "3.11"
    cache: "pip"

- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    pip install -r requirements.txt

- name: Run automated tests
  run: |
    python -m pytest -v

- name: Run model training and evaluation
  run: |
    python train.py
```

The workflow uses Python 3.11 and installs the dependencies listed in `requirements.txt` before executing the tests and training process.

### Automated Testing with Pytest

Automated tests are implemented in:

`tests/test_training.py`

The test suite verifies important parts of the training pipeline, including:

- Whether the data-loading function returns valid features and target values.
- Whether the system handles a dataset with a missing target column correctly.
- Whether the training and evaluation function returns a trained model, evaluation metrics, and an MLflow run ID.

The tests are executed using:

```bash
python -m pytest -v
```

The test suite successfully completed with **3 passed tests**.

### Automated Model Training

Model training and evaluation are implemented in:

`train.py`

The pipeline evaluates three LightGBM configurations:

| Configuration | `n_estimators` | `learning_rate` | `num_leaves` |
|---|---:|---:|---:|
| Experiment 1 | 100 | 0.05 | 15 |
| Experiment 2 | 200 | 0.05 | 31 |
| Experiment 3 | 300 | 0.03 | 31 |

Each experiment uses a chronological 80:20 train-test split to preserve the time order of the cryptocurrency data.

The pipeline calculates three evaluation metrics:

- **RMSE (Root Mean Squared Error)** — Measures the square root of the average squared prediction error.
- **MAE (Mean Absolute Error)** — Measures the average absolute difference between actual and predicted values.
- **MAPE (Mean Absolute Percentage Error)** — Measures the average absolute percentage error.

All experiment parameters, evaluation metrics, and model artifacts are logged to MLflow.

### Automated Model Validation

Before model registration, the pipeline compares the selected model's evaluation metrics against the predefined baseline metrics.

The baseline values are:

| Metric | Baseline |
|---|---:|
| RMSE | 521.2710 |
| MAE | 354.3749 |
| MAPE | 0.4393% |

The model is considered valid when its RMSE, MAE, and MAPE do not exceed the corresponding baseline values.

The `validate_model()` function performs this comparison automatically. If the selected model does not satisfy the validation criteria, the pipeline reports that validation has failed and skips model registration.

This validation mechanism helps prevent an underperforming model from being registered automatically.

### Automated Model Registration

A model that passes validation is automatically registered in MLflow Model Registry under the name:

`BTCUSDT-LightGBM-Best`

The registration process is implemented in the `register_model_if_valid()` function in `train.py`.

The function performs the following operations:

1. Checks whether the registered model already exists.
2. Creates the registered model if it does not exist.
3. Creates a new model version using the selected MLflow run.
4. Assigns the `staging` alias to the newly registered version.

The registry configuration is:

| Property | Value |
|---|---|
| Model name | `BTCUSDT-LightGBM-Best` |
| Experiment | `BTCUSDT-LightGBM-Regression` |
| Registration condition | Model passes baseline validation |
| Model alias | `staging` |

The `staging` alias identifies the latest model version that has passed the automated validation process. It is separate from the existing `production` alias used by the inference workflow.

### End-to-End Pipeline Execution

The complete automated workflow can be represented as follows:

```text
       Code Change
            |
            v
      Git Push / PR
            |
            v
     GitHub Actions
            |
            v
    Install Dependencies
            |
            v
    Automated Testing
       (pytest -v)
            |
            v
   Model Training
       (train.py)
            |
            v
   Model Evaluation
    (RMSE, MAE, MAPE)
            |
            v
   Baseline Validation
            |
       +----+----+
       |         |
       v         v
     PASS      FAIL
       |         |
       v         v
 Register     Skip Model
   Model      Registration
       |
       v
 Assign staging Alias
       |
       v
   Pipeline Complete
```

If testing fails, the workflow stops before model training. If model validation fails, the model is not registered. This ensures that each stage must complete successfully before the pipeline proceeds to the next relevant stage.

### Simulating an Automated Pipeline Trigger

The automation mechanism can be tested by making a small code change, committing the change, and pushing it to the configured feature branch.

Example commands:

```bash
git add .
git commit -m "test: trigger automated MLOps pipeline"
git push
```

After the push, GitHub Actions automatically starts a new workflow run. The execution results can be inspected through the **Actions** tab in the GitHub repository.

A successful run should show that the dependency installation, automated tests, and model training and evaluation steps have completed successfully.

### Automation and Prevention of Silent Failures

Automation helps reduce the risk of silent failures by validating the pipeline at multiple stages.

First, automated tests detect problems in data loading and training functionality before the model training process proceeds. Second, model evaluation checks whether the trained model meets the predefined performance baseline. Third, conditional model registration prevents models that fail validation from entering the model registry.

The workflow also provides execution logs through GitHub Actions, making it easier to identify which stage has failed and inspect the corresponding error messages.

Therefore, automation improves the consistency, reproducibility, and reliability of the model development process while reducing dependence on manual execution.

### LK-8 Implementation Summary

The LK-8 implementation integrates the following components:

| Component | Implementation |
|---|---|
| Automation platform | GitHub Actions |
| Workflow configuration | `.github/workflows/mlops-pipeline.yml` |
| Automated testing | `pytest` |
| Training and evaluation | `train.py` |
| Model framework | LightGBM Regressor |
| Experiment tracking | MLflow |
| Model validation | Baseline comparison using RMSE, MAE, and MAPE |
| Model registry | MLflow Model Registry |
| Registration alias | `staging` |

The implementation establishes an automated training and evaluation workflow triggered by repository changes. It connects software testing, model development, performance validation, and model versioning into a single pipeline.

## Future Improvements

Potential next steps include:

- Configuring a DVC remote for shared dataset and artifact storage.
- Automating data ingestion and preprocessing as part of the scheduled or event-driven pipeline.
- Expanding model evaluation with additional metrics and validation strategies.
- Implementing inference monitoring and data drift detection.
- Extending the GitHub Actions workflow with deployment and monitoring stages.
- Improving model promotion from `staging` to `production` through additional validation and approval mechanisms.
