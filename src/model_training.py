"""Model training script for Marketing Analysis using XGBoost.

Run from the project root:
    python src/model_training.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from data_preprocessing import (
    RANDOM_STATE,
    build_preprocessor,
    generate_synthetic_marketing_data,
    save_processed_data,
    split_features_target,
)
from evaluation import plot_confusion_matrix, plot_feature_importance, plot_roc_curve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "synthetic_marketing_campaigns.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "marketing_campaigns_processed.csv"
MODEL_PATH = PROJECT_ROOT / "outputs" / "xgboost_marketing_model.joblib"
METRICS_PATH = PROJECT_ROOT / "outputs" / "model_metrics.json"


def train_model() -> dict:
    """Generate data, train an XGBoost model, tune it with GridSearchCV and save outputs."""
    df = generate_synthetic_marketing_data(n_samples=800, output_path=RAW_DATA_PATH)
    save_processed_data(df, PROCESSED_DATA_PATH)

    X_train, X_test, y_train, y_test = split_features_target(df)
    preprocessor = build_preprocessor(df)

    xgb_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
        tree_method="hist",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", xgb_model),
        ]
    )

    param_grid = {
        "model__n_estimators": [30, 60],
        "model__max_depth": [2, 3],
        "model__learning_rate": [0.1],
        "model__subsample": [0.9],
        "model__colsample_bytree": [0.9],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="roc_auc",
        n_jobs=1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "best_params": grid_search.best_params_,
        "best_cv_roc_auc": round(float(grid_search.best_score_), 4),
        "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "test_precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "test_recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "test_f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "test_roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    plot_confusion_matrix(y_test, y_pred, PROJECT_ROOT / "outputs" / "confusion_matrix.png")
    plot_roc_curve(y_test, y_proba, PROJECT_ROOT / "outputs" / "roc_curve.png")
    plot_feature_importance(best_model, X_train, PROJECT_ROOT / "outputs" / "feature_importance.png")

    return metrics


if __name__ == "__main__":
    results = train_model()
    print("Model training complete. Results:")
    print(json.dumps(results, indent=4))
