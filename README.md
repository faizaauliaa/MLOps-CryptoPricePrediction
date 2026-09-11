# MLOps-CryptoPricePrediction

Continuous training system for cryptocurrency price prediction using dynamic market data.

## Project Goal

This project aims to develop a machine learning system for cryptocurrency price prediction that can adapt to dynamic market data through a continuous training approach.

The project focuses on building a basic MLOps infrastructure to support data processing, model development, and future model retraining.

## Directory Structure

```text
MLOps-CryptoPricePrediction/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   └── 01_initial_eda.ipynb
├── src/
│   ├── data/
│   ├── features/
│   ├── training/
│   ├── inference/
│   └── monitoring/
├── config/
├── .devcontainer/
│   └── devcontainer.json
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## How to Run in GitHub Codespaces

1. Open this repository on GitHub.
2. Click **Code** and select **Codespaces**.
3. Create a new Codespace or open an existing Codespace.
4. The project uses Python 3.11.
5. Project dependencies are installed automatically from `requirements.txt`.

To verify the Python environment, run:

    python --version

The expected output is:

    Python 3.11.16

## Initial EDA

The `notebooks/01_initial_eda.ipynb` notebook contains the initial exploratory data analysis preparation for the cryptocurrency price prediction project.

The notebook focuses on:

- Understanding the structure of cryptocurrency market data.
- Inspecting data quality and completeness.
- Identifying initial patterns in OHLCV data.
- Preparing considerations for further feature engineering and model development.

## Development Workflow

This project follows the GitHub Flow branching strategy.

The `main` branch contains the stable version of the project. New development is performed in feature branches and merged into `main` through Pull Requests after validation.
