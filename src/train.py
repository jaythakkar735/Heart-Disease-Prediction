"""
Main model training, cross-validation, holdout evaluation, and report generation
for Heart Disease Prediction.
"""

import os
import json
import warnings
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", message="Found unknown categories.*")

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

import sys
from pathlib import Path

# Ensure project root is in sys.path for direct script execution
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from .data_loader import (
        load_heart_data,
        get_train_test_split,
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES,
    )
    from .pipeline import build_full_pipeline
    from .evaluate import (
        compute_classification_metrics,
        plot_confusion_matrix,
        plot_roc_curves,
        plot_calibration_curves,
        plot_permutation_importance,
    )
except (ImportError, ValueError):
    from src.data_loader import (
        load_heart_data,
        get_train_test_split,
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES,
    )
    from src.pipeline import build_full_pipeline
    from src.evaluate import (
        compute_classification_metrics,
        plot_confusion_matrix,
        plot_roc_curves,
        plot_calibration_curves,
        plot_permutation_importance,
    )

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


def generate_eda_figures(df: pd.DataFrame, output_dir: str = FIGURES_DIR) -> None:
    """Generate and save essential EDA figures."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Distributions of continuous variables by target
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    for i, col in enumerate(NUMERIC_FEATURES):
        sns.histplot(
            data=df,
            x=col,
            hue="target",
            kde=True,
            ax=axes[i],
            palette=["#3470a3", "#e05a47"],
            alpha=0.6,
        )
        axes[i].set_title(f"Distribution of {col}", fontsize=10, fontweight="bold")
    # Hide the unused 6th subplot
    fig.delaxes(axes[5])
    plt.suptitle("Feature Distributions by Heart Disease Diagnosis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_distributions.png"), dpi=300)
    plt.close()

    # 2. Correlation heatmap of numeric variables
    plt.figure(figsize=(6.5, 5))
    corr = df[NUMERIC_FEATURES + ["target"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
    plt.title("Correlation Matrix (Numeric Features + Target)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=300)
    plt.close()


def run_training_and_evaluation() -> Dict[str, Any]:
    """Execute end-to-end model evaluation workflow."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1. Load data
    df = load_heart_data()
    print(f"Loaded dataset: {df.shape[0]} samples, {df.shape[1]} columns")
    print(f"Target distribution:\n{df['target'].value_counts(normalize=True).round(3)}")

    # 2. Generate EDA plots
    generate_eda_figures(df, FIGURES_DIR)

    # 3. Train/Test Split (80/20 stratified)
    X_train, X_test, y_train, y_test = get_train_test_split(df, test_size=0.2, random_state=42)
    print(f"Training split: {X_train.shape[0]} samples | Test split: {X_test.shape[0]} samples")

    # 4. Define candidate models
    models = {
        "Dummy Baseline": DummyClassifier(strategy="stratified", random_state=42),
        "Logistic Regression": LogisticRegression(C=0.5, solver="lbfgs", max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_split=4, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=80, learning_rate=0.05, max_depth=3, random_state=42),
    }

    # 5. Stratified 5-Fold Cross-Validation on Training Set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    fitted_pipelines = {}

    scoring = ["roc_auc", "f1", "recall", "precision", "accuracy"]

    print("\n--- 5-Fold Stratified Cross-Validation on Training Data ---")
    for name, clf in models.items():
        pipe = build_full_pipeline(clf)
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        cv_results[name] = {
            f"cv_{metric}_mean": float(np.mean(scores[f"test_{metric}"]))
            for metric in scoring
        }
        for metric in scoring:
            cv_results[name][f"cv_{metric}_std"] = float(np.std(scores[f"test_{metric}"]))

        print(
            f"{name:20s} | ROC-AUC: {cv_results[name]['cv_roc_auc_mean']:.3f} (+/- {cv_results[name]['cv_roc_auc_std']:.3f}) "
            f"| F1: {cv_results[name]['cv_f1_mean']:.3f} | Recall: {cv_results[name]['cv_recall_mean']:.3f}"
        )

        # Fit on entire training set
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

    # 6. Evaluate on Holdout Test Set
    print("\n--- Holdout Test Set Evaluation (Unseen Data) ---")
    test_results = {}
    for name, pipe in fitted_pipelines.items():
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None
        metrics = compute_classification_metrics(y_test, y_pred, y_proba)
        test_results[name] = metrics
        print(
            f"{name:20s} | Accuracy: {metrics['accuracy']:.3f} | Precision: {metrics['precision']:.3f} "
            f"| Recall: {metrics['recall_sensitivity']:.3f} | F1: {metrics['f1_score']:.3f} | ROC-AUC: {metrics.get('roc_auc', 0.0):.3f}"
        )

    # 7. Generate diagnostic plots
    # Champion model selection: Logistic Regression (or best CV ROC-AUC)
    champion_name = "Logistic Regression"
    champion_pipe = fitted_pipelines[champion_name]

    # Confusion matrix for champion
    plot_confusion_matrix(
        y_test,
        champion_pipe.predict(X_test),
        os.path.join(FIGURES_DIR, "confusion_matrix.png"),
        model_name=champion_name,
    )

    # ROC Curves comparison
    plot_roc_curves(
        fitted_pipelines,
        X_test,
        y_test,
        os.path.join(FIGURES_DIR, "roc_curves.png"),
    )

    # Calibration curves
    plot_calibration_curves(
        {k: v for k, v in fitted_pipelines.items() if k != "Dummy Baseline"},
        X_test,
        y_test,
        os.path.join(FIGURES_DIR, "calibration_curve.png"),
    )

    # Permutation importance for champion
    plot_permutation_importance(
        champion_pipe,
        X_test,
        y_test,
        os.path.join(FIGURES_DIR, "feature_importance.png"),
    )

    # 8. Save metrics summary
    summary = {
        "dataset": "UCI Cleveland Heart Disease",
        "sample_size": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "cross_validation_results": cv_results,
        "holdout_test_results": test_results,
    }

    with open(os.path.join(REPORTS_DIR, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll artifacts generated successfully in {FIGURES_DIR} and {REPORTS_DIR}")
    return summary


if __name__ == "__main__":
    run_training_and_evaluation()
