import os
from datetime import datetime, timezone

import pandas as pd
import requests


# Konfigurasi Binance API
API_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
LIMIT = 100

# Folder penyimpanan data mentah
RAW_DIR = "data/raw"


def fetch_data():
    """Mengambil data Kline terbaru dari Binance API."""
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "limit": LIMIT
    }

    response = requests.get(API_URL, params=params, timeout=10)
    response.raise_for_status()

    return response.json()


def save_raw_data(data):
    """Menyimpan data mentah ke file CSV dengan timestamp."""
    os.makedirs(RAW_DIR, exist_ok=True)

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore"
    ]

    df = pd.DataFrame(data, columns=columns)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"btc_usdt_1h_{timestamp}.csv"
    filepath = os.path.join(RAW_DIR, filename)

    df.to_csv(filepath, index=False)

    return filepath, df


def main():
    print("DATA INGESTION - BTC/USDT")

    print("\nMengambil data terbaru dari Binance API...")

    try:
        data = fetch_data()

        filepath, df = save_raw_data(data)

        print("Data berhasil diambil.")
        print(f"Symbol       : {SYMBOL}")
        print(f"Interval     : {INTERVAL}")
        print(f"Jumlah data  : {len(df)}")
        print(f"File disimpan: {filepath}")

    except requests.RequestException as error:
        print(f"Terjadi error saat mengakses Binance API: {error}")

    except Exception as error:
        print(f"Terjadi error: {error}")


if __name__ == "__main__":
    main()