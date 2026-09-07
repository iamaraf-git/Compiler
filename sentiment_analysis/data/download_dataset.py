"""
Dataset Downloader & Loader for Sentiment Analysis
Fetches the 50K IMDb dataset or falls back to sample reviews.
"""
import os
import urllib.request
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_FILE = os.path.join(DATA_DIR, "sample_reviews.csv")
FULL_IMDB_FILE = os.path.join(DATA_DIR, "imdb_50k.csv")

# Direct raw URL for standard IMDb 50k dataset (mirror)
IMDB_URL = "https://raw.githubusercontent.com/datasets/imdb-reviews/master/data/imdb_dataset.csv"

def get_dataset(prefer_full: bool = True) -> pd.DataFrame:
    """
    Returns a pandas DataFrame with columns ['review', 'sentiment'].
    If prefer_full is True and the 50k dataset is not present, it attempts to download it.
    If download fails or prefer_full is False, it loads the included sample dataset.
    """
    if os.path.exists(FULL_IMDB_FILE):
        print(f"[+] Loading full IMDb 50K dataset from {FULL_IMDB_FILE}...")
        df = pd.read_csv(FULL_IMDB_FILE)
        return df

    if prefer_full:
        print("[*] Checking for online 50k dataset download...")
        try:
            # We can download a verified IMDb subset/dataset mirror
            print(f"[*] Downloading IMDb dataset to {FULL_IMDB_FILE}...")
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv",
                FULL_IMDB_FILE
            )
            print("[+] Download completed successfully!")
            return pd.read_csv(FULL_IMDB_FILE)
        except Exception as e:
            print(f"[!] Online download failed ({e}). Defaulting to built-in sample dataset.")

    print(f"[+] Loading sample dataset from {SAMPLE_FILE}...")
    return pd.read_csv(SAMPLE_FILE)

if __name__ == "__main__":
    df = get_dataset(prefer_full=True)
    print(f"Dataset Shape: {df.shape}")
    print(df.head())
