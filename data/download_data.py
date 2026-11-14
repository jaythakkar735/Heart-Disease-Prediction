"""
Data acquisition script for the UCI Heart Disease (Cleveland) dataset.
Source: UCI Machine Learning Repository
URL: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
Citation: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). Heart Disease. UCI Machine Learning Repository.
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
"""

import os
import urllib.request
import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(CURRENT_DIR, "processed_cleveland.csv")

COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


def download_dataset(output_path: str = OUTPUT_FILE) -> str:
    """Download the raw Cleveland dataset from UCI and write with column headers."""
    print(f"Downloading UCI Heart Disease dataset from: {DATA_URL}")
    df = pd.read_csv(DATA_URL, header=None, names=COLUMN_NAMES, na_values="?")
    df.to_csv(output_path, index=False)
    print(f"Successfully saved {len(df)} records to {output_path}")
    return output_path


if __name__ == "__main__":
    download_dataset()
