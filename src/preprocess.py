import os
import glob
import pandas as pd

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"


def get_latest_raw_file():
    """
    Mengambil file raw hasil ingestion terbaru.
    File lama seperti btc_usdt_1h_raw.csv tidak dipilih.
    """
    pattern = os.path.join(RAW_DIR, "btc_usdt_1h_*.csv")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError("Tidak ditemukan file raw hasil ingestion.")

    return max(files, key=os.path.getmtime)


def preprocess_data(input_file):
    """
    Membersihkan dan melakukan preprocessing data.
    """
    df = pd.read_csv(input_file)

    print(f"File input       : {input_file}")
    print(f"Data sebelum     : {len(df)} baris")

    # Menghapus data duplikat
    df = df.drop_duplicates()

    # Mengubah kolom waktu menjadi datetime
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    # Mengubah kolom numerik
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

    # Menghapus baris yang memiliki nilai kosong
    df = df.dropna()

    # Mengurutkan berdasarkan waktu
    df = df.sort_values("open_time").reset_index(drop=True)

    print(f"Data sesudah     : {len(df)} baris")

    return df


def save_processed_data(df):
    """
    Menyimpan data hasil preprocessing.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    output_file = os.path.join(
        PROCESSED_DIR,
        "btc_usdt_1h_preprocessed.csv"
    )

    df.to_csv(output_file, index=False)

    print(f"File output      : {output_file}")

    return output_file


def main():
    print("DATA PREPROCESSING - BTC/USDT")

    try:
        input_file = get_latest_raw_file()
        df = preprocess_data(input_file)
        save_processed_data(df)

        print("\nPreprocessing berhasil.")

    except Exception as error:
        print(f"\nTerjadi error: {error}")


if __name__ == "__main__":
    main()