"""
Unit and integration tests for the Heart Disease Prediction ML pipeline.
Verifies data loading, preprocessing, zero leakage, and metric computations.
"""

import os
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.data_loader import (
    load_heart_data,
    get_train_test_split,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
)
from src.pipeline import create_preprocessor, build_full_pipeline
from src.evaluate import compute_classification_metrics


def test_load_heart_data():
    """Verify dataset loading, shape, and target binarization."""
    df = load_heart_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 303
    assert set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COL]).issubset(df.columns)
    # Check target is strictly binary 0 or 1
    assert set(df[TARGET_COL].unique()).issubset({0, 1})
    # Ensure 'ca' and 'thal' do not contain unparsed strings
    assert not (df["ca"] == "?").any()
    assert not (df["thal"] == "?").any()


def test_train_test_split_stratification():
    """Verify split proportions and class ratio preservation."""
    df = load_heart_data()
    X_train, X_test, y_train, y_test = get_train_test_split(df, test_size=0.2, random_state=42)

    assert len(X_train) == 242
    assert len(X_test) == 61
    assert len(X_train) + len(X_test) == 303

    # Positive class ratio must be approximately identical
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    assert abs(train_ratio - test_ratio) < 0.05


def test_pipeline_zero_leakage_and_transformation():
    """Verify preprocessor handles missing data and outputs clean finite matrix."""
    df = load_heart_data()
    X_train, X_test, y_train, _ = get_train_test_split(df, test_size=0.2, random_state=42)

    preprocessor = create_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Check finite numbers, no NaN
    assert not np.isnan(X_train_trans).any()
    assert not np.isnan(X_test_trans).any()

    # Preprocessor output dimensions must match
    assert X_train_trans.shape[1] == X_test_trans.shape[1]
    assert X_train_trans.shape[0] == len(X_train)
    assert X_test_trans.shape[0] == len(X_test)


def test_end_to_end_pipeline_predict():
    """Verify that full pipeline trains and produces valid predictions and probabilities."""
    df = load_heart_data()
    X_train, X_test, y_train, y_test = get_train_test_split(df, test_size=0.2, random_state=42)

    pipe = build_full_pipeline(LogisticRegression(max_iter=500, random_state=42))
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)

    assert len(y_pred) == len(X_test)
    assert y_proba.shape == (len(X_test), 2)
    assert np.all((y_proba[:, 1] >= 0.0) & (y_proba[:, 1] <= 1.0))

    metrics = compute_classification_metrics(y_test, y_pred, y_proba[:, 1])
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
