from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ml.features import (
    ARTIFACT_DIR,
    CATEGORICAL_FEATURES,
    DATASET_PATH,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    build_feature_frame,
    generate_synthetic_retail_data,
    split_xy,
)

MODEL_PATH = ARTIFACT_DIR / "next_best_product_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def _classification_metrics(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.35).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "average_precision": round(float(average_precision_score(y_test, probabilities)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
    }


def _dcg(relevance: np.ndarray) -> float:
    discounts = 1 / np.log2(np.arange(2, relevance.size + 2))
    return float(np.sum(relevance * discounts))


def _ranking_metrics(
    model: Pipeline,
    eval_frame: pd.DataFrame,
    k: int = 5,
) -> dict[str, float]:
    scored = eval_frame[["customer_id", "product_id", "target_purchase"]].copy()
    scored["score"] = model.predict_proba(eval_frame[FEATURE_COLUMNS])[:, 1]
    precision_values: list[float] = []
    recall_values: list[float] = []
    ndcg_values: list[float] = []

    for _, group in scored.groupby("customer_id"):
        positives = int(group["target_purchase"].sum())
        if positives == 0:
            continue
        top = group.sort_values("score", ascending=False).head(k)
        hits = int(top["target_purchase"].sum())
        precision_values.append(hits / k)
        recall_values.append(hits / positives)
        relevance = top["target_purchase"].to_numpy(dtype=float)
        ideal = np.sort(group["target_purchase"].to_numpy(dtype=float))[::-1][:k]
        ideal_dcg = _dcg(ideal)
        ndcg_values.append(_dcg(relevance) / ideal_dcg if ideal_dcg else 0.0)

    return {
        f"precision_at_{k}": round(float(np.mean(precision_values)), 4),
        f"recall_at_{k}": round(float(np.mean(recall_values)), 4),
        f"ndcg_at_{k}": round(float(np.mean(ndcg_values)), 4),
    }


def train_and_save(seed: int = 42) -> dict[str, Any]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    retail_data = generate_synthetic_retail_data(seed=seed)
    data_dict = retail_data.as_dict()
    joblib.dump(data_dict, DATASET_PATH)

    feature_frame = build_feature_frame(data_dict)
    x, y = split_xy(feature_frame)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=y,
    )

    models: dict[str, Pipeline] = {
        "RandomForestClassifier baseline": Pipeline(
            [
                ("preprocess", _preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=220,
                        max_depth=12,
                        min_samples_leaf=4,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "GradientBoostingClassifier stronger_model": Pipeline(
            [
                ("preprocess", _preprocessor()),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=180,
                        learning_rate=0.045,
                        max_depth=3,
                        subsample=0.82,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }

    metrics: dict[str, Any] = {
        "dataset": {
            "rows": int(len(feature_frame)),
            "positive_rate": round(float(y.mean()), 4),
            "customers": int(retail_data.customers["customer_id"].nunique()),
            "products": int(retail_data.products["product_id"].nunique()),
            "target": "target_purchase equals 1 when a customer buys the candidate product in the 45-day label window.",
        },
        "models": {},
    }

    best_name = ""
    best_score = -1.0
    best_model: Pipeline | None = None
    eval_frame = feature_frame.loc[x_test.index].copy()

    for name, model in models.items():
        model.fit(x_train, y_train)
        model_metrics = _classification_metrics(model, x_test, y_test)
        model_metrics.update(_ranking_metrics(model, eval_frame, k=5))
        metrics["models"][name] = model_metrics
        selection_score = model_metrics["average_precision"] + model_metrics["roc_auc"]
        if selection_score > best_score:
            best_name = name
            best_score = selection_score
            best_model = model

    assert best_model is not None
    metrics["selected_model"] = best_name
    metrics["ranking_formula"] = "0.75 * supervised ML probability + 0.15 * similarity support + 0.10 * normalized business value"
    metrics["feature_columns"] = FEATURE_COLUMNS
    model_bundle = {
        "model_name": best_name,
        "model": best_model,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }
    joblib.dump(model_bundle, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    metrics = train_and_save()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
