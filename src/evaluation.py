"""Evaluation and visualisation functions for the Marketing Analysis project."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def _ensure_parent_dir(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_confusion_matrix(y_true, y_pred, output_path: str | Path) -> None:
    """Save a confusion matrix chart."""
    output_path = _ensure_parent_dir(output_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    ax.set_title("Confusion Matrix - XGBoost Marketing Model")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_true, y_proba, output_path: str | Path) -> None:
    """Save a ROC curve chart."""
    output_path = _ensure_parent_dir(output_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title("ROC Curve - XGBoost Marketing Model")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def get_feature_names(model_pipeline, X_train: pd.DataFrame) -> list[str]:
    """Extract post-preprocessing feature names from the fitted pipeline."""
    preprocessor = model_pipeline.named_steps["preprocessor"]
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X_train.select_dtypes(exclude=["object", "category"]).columns.tolist()

    categorical_names = []
    if categorical_cols:
        encoder = preprocessor.named_transformers_["categorical"]
        categorical_names = encoder.get_feature_names_out(categorical_cols).tolist()

    return categorical_names + numeric_cols


def plot_feature_importance(model_pipeline, X_train: pd.DataFrame, output_path: str | Path, top_n: int = 15) -> None:
    """Save a top-N feature importance chart from the XGBoost model."""
    output_path = _ensure_parent_dir(output_path)
    model = model_pipeline.named_steps["model"]
    feature_names = get_feature_names(model_pipeline, X_train)
    importances = model.feature_importances_

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .sort_values("importance", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance_df["feature"], importance_df["importance"])
    ax.set_title("Top Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
