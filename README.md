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

```
## Data Versioning Plan

Data pada project ini direncanakan menggunakan DVC (Data Version Control)
untuk melakukan versioning terhadap dataset yang bersifat dinamis.

Pembagian versioning:
- Git digunakan untuk source code, konfigurasi, dan dokumentasi.
- DVC digunakan untuk melacak versi dataset pada `data/raw/` dan
  `data/processed/`.

Dataset yang akan dikelola:
- `data/raw/btc_usdt_1h_raw.csv`
- `data/processed/btc_usdt_1h_clean.csv`
- `data/processed/btc_usdt_1h_features.csv`

Pola pembaruan data:
1. Data terbaru diambil dari Binance Public Market Data API.
2. Data disimpan pada `data/raw/`.
3. Data melalui proses cleaning dan feature engineering.
4. Hasil pemrosesan disimpan pada `data/processed/`.
5. Perubahan dataset akan dilacak menggunakan DVC pada tahap implementasi
   berikutnya.

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

## Dataset Version

Contoh versi dataset:
- v1.0: pengambilan data awal.
- v1.1: pembaruan data berikutnya.
- v1.2: pembaruan data berikutnya.
