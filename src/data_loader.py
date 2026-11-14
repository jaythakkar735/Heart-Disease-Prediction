"""
Data loading and splitting module for Heart Disease classification.
Ensures zero data leakage between train and test splits.
"""

import os
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "processed_cleveland.csv",
)

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET_COL = "target"


def load_heart_data(data_path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV data, converting missing value representations and binarizing target."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset file not found at {data_path}. Please run download_data.py first."
        )

    df = pd.read_csv(data_path)

    # Coerce any string '?' representations to NaN
    for col in ["ca", "thal"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Binarize angiographic diagnosis target: 0 -> 0 (absence), 1-4 -> 1 (presence)
    df["target"] = (df["target"] > 0).astype(int)

    return df


def get_train_test_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and holdout test sets with stratification.
    Stratification ensures positive class ratio is preserved across both splits.
    """
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test
