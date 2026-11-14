"""
Evaluation, metrics computation, and visualization utilities for Heart Disease ML.
Generates publication-quality charts into reports/figures.
"""

import os
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None
) -> Dict[str, float]:
    """Calculate comprehensive classification and probability calibration metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        metrics["brier_score"] = float(brier_score_loss(y_true, y_proba))

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, output_path: str, model_name: str = "Model"
) -> None:
    """Plot an annotated confusion matrix with counts and percentages."""
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm / cm.sum() * 100

    labels = [
        f"{count}\n({pct:.1f}%)"
        for count, pct in zip(cm.flatten(), cm_percent.flatten())
    ]
    labels = np.asarray(labels).reshape(2, 2)

    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted No Disease", "Predicted Disease"],
        yticklabels=["Actual No Disease", "Actual Disease"],
    )
    plt.title(f"Confusion Matrix — {model_name}", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_roc_curves(
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: str,
) -> None:
    """Plot comparative ROC curves for multiple models."""
    plt.figure(figsize=(7, 5.5))
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Chance (AUC = 0.50)")

    for name, pipeline in models_dict.items():
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", lw=2)

    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Comparison", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_calibration_curves(
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: str,
) -> None:
    """Plot calibration curve (reliability diagram) to assess probability credibility."""
    plt.figure(figsize=(7, 5.5))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    for name, pipeline in models_dict.items():
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=5, strategy="uniform")
            brier = brier_score_loss(y_test, y_proba)
            plt.plot(prob_pred, prob_true, marker="o", lw=1.8, label=f"{name} (Brier = {brier:.3f})")

    plt.xlabel("Mean Predicted Probability", fontsize=11)
    plt.ylabel("Fraction of Positives (Empirical)", fontsize=11)
    plt.title("Calibration Curves (Reliability Diagram)", fontsize=12, fontweight="bold")
    plt.legend(loc="upper left", frameon=True)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_permutation_importance(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: str,
    n_repeats: int = 15,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute and plot permutation feature importance on held-out test data."""
    result = permutation_importance(
        pipeline, X_test, y_test, n_repeats=n_repeats, random_state=random_state, scoring="roc_auc"
    )

    perm_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values(by="importance_mean", ascending=True)

    plt.figure(figsize=(7, 5.5))
    plt.barh(
        perm_df["feature"],
        perm_df["importance_mean"],
        xerr=perm_df["importance_std"],
        color="#2b5c8f",
        edgecolor="black",
        alpha=0.85,
        capsize=4,
    )
    plt.xlabel("Permutation Importance (Mean Decrease in ROC-AUC)", fontsize=11)
    plt.title("Feature Importance via Permutation (Held-out Test)", fontsize=12, fontweight="bold")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return perm_df
