import requests
import pandas as pd
from pathlib import Path


API_URL = "https://api.binance.com/api/v3/klines"

PARAMS = {
    "symbol": "BTCUSDT",
    "interval": "1h",
    "limit": 100
}


def fetch_data():
    response = requests.get(
        API_URL,
        params=PARAMS,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

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

    return df


def save_raw_data(df):
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "btc_usdt_1h_raw.csv"

    df.to_csv(output_file, index=False)

    print(f"Data berhasil disimpan ke: {output_file}")
    print(f"Jumlah data: {len(df)}")


if __name__ == "__main__":
    df = fetch_data()

    print("Data berhasil diambil dari Binance API")
    print()
    print(df.head())

    save_raw_data(df)