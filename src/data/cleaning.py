import pandas as pd
from pathlib import Path


INPUT_FILE = "data/raw/btc_usdt_1h_raw.csv"
OUTPUT_FILE = "data/processed/btc_usdt_1h_clean.csv"


def clean_data(df):
    # Mengubah timestamp menjadi datetime
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    # Mengubah kolom harga dan volume menjadi numerik
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Menghapus data duplikat berdasarkan waktu pembukaan candle
    df = df.drop_duplicates(subset=["open_time"])

    # Mengurutkan data berdasarkan waktu
    df = df.sort_values("open_time")

    # Menghapus data dengan nilai penting yang kosong
    df = df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )

    # Memastikan nilai OHLC valid
    df = df[
        (df["high"] >= df["low"]) &
        (df["high"] >= df["open"]) &
        (df["high"] >= df["close"]) &
        (df["low"] <= df["open"]) &
        (df["low"] <= df["close"])
    ]

    return df


def main():
    print("Membaca data raw...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Jumlah data sebelum cleaning: {len(df)}")

    df_clean = clean_data(df)

    print(f"Jumlah data setelah cleaning: {len(df_clean)}")

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    df_clean.to_csv(OUTPUT_FILE, index=False)

    print(f"Data hasil cleaning disimpan ke: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()