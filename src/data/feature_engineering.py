import pandas as pd
from pathlib import Path


INPUT_FILE = "data/processed/btc_usdt_1h_clean.csv"
OUTPUT_FILE = "data/processed/btc_usdt_1h_features.csv"


def create_features(df):
    # Return harga 1 jam
    df["return_1h"] = df["close"].pct_change()

    # Lag harga penutupan
    df["lag_close_1"] = df["close"].shift(1)
    df["lag_close_3"] = df["close"].shift(3)
    df["lag_close_6"] = df["close"].shift(6)
    df["lag_close_24"] = df["close"].shift(24)

    # Rolling statistics
    df["rolling_mean_6"] = df["close"].rolling(window=6).mean()
    df["rolling_std_6"] = df["close"].rolling(window=6).std()

    # Rentang harga
    df["price_range"] = df["high"] - df["low"]

    # Perubahan volume
    df["volume_change"] = df["volume"].pct_change()

    # Target: harga penutupan candle berikutnya
    df["target_close_next"] = df["close"].shift(-1)

    # Hapus baris yang memiliki nilai kosong akibat lag/rolling
    df = df.dropna()

    return df


def main():
    print("Membaca data hasil cleaning...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Jumlah data sebelum feature engineering: {len(df)}")

    df_features = create_features(df)

    print(f"Jumlah data setelah feature engineering: {len(df_features)}")

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    df_features.to_csv(OUTPUT_FILE, index=False)

    print(f"Data dengan fitur disimpan ke: {OUTPUT_FILE}")
    print()
    print("Fitur yang dibuat:")
    print([
        "return_1h",
        "lag_close_1",
        "lag_close_3",
        "lag_close_6",
        "lag_close_24",
        "rolling_mean_6",
        "rolling_std_6",
        "price_range",
        "volume_change",
        "target_close_next"
    ])


if __name__ == "__main__":
    main()