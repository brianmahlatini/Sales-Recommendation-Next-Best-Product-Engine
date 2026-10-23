from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.ml.features import (
    ARTIFACT_DIR,
    DATASET_PATH,
    FEATURE_COLUMNS,
    build_feature_frame,
    describe_features,
    generate_synthetic_retail_data,
)

MODEL_PATH = ARTIFACT_DIR / "next_best_product_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"


def _load_or_create_dataset() -> dict[str, pd.DataFrame]:
    if DATASET_PATH.exists():
        return joblib.load(DATASET_PATH)
    retail_data = generate_synthetic_retail_data()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(retail_data.as_dict(), DATASET_PATH)
    return retail_data.as_dict()


@lru_cache(maxsize=1)
def load_model_bundle() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        try:
            from scripts.train_model import train_and_save
        except ModuleNotFoundError:
            from backend.scripts.train_model import train_and_save

        train_and_save()
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_metrics() -> dict[str, Any]:
    if not METRICS_PATH.exists():
        try:
            from scripts.train_model import train_and_save
        except ModuleNotFoundError:
            from backend.scripts.train_model import train_and_save

        train_and_save()
    return json.loads(METRICS_PATH.read_text())


@lru_cache(maxsize=1)
def load_retail_data() -> dict[str, pd.DataFrame]:
    _ = load_model_bundle()
    return _load_or_create_dataset()


def get_customers() -> list[dict[str, Any]]:
    customers = load_retail_data()["customers"].copy()
    return customers.sort_values("customer_id").to_dict(orient="records")


def get_products() -> list[dict[str, Any]]:
    products = load_retail_data()["products"].copy()
    return products.sort_values("product_id").to_dict(orient="records")


def recommend_products(customer_id: str, top_k: int = 8) -> dict[str, Any]:
    model_bundle = load_model_bundle()
    data = load_retail_data()
    customers = data["customers"]
    if customer_id not in set(customers["customer_id"]):
        raise ValueError(f"Unknown customer_id: {customer_id}")

    feature_frame = build_feature_frame(data, include_target=False)
    customer_frame = feature_frame[feature_frame["customer_id"] == customer_id].copy()
    model = model_bundle["model"]
    probabilities = model.predict_proba(customer_frame[FEATURE_COLUMNS])[:, 1]
    customer_frame["model_purchase_probability"] = probabilities

    value = customer_frame["business_value_score"].to_numpy(dtype=float)
    value_norm = (value - value.min()) / (value.max() - value.min() + 1e-9)
    customer_frame["final_ranking_score"] = (
        0.75 * customer_frame["model_purchase_probability"]
        + 0.15 * customer_frame["item_similarity_score"]
        + 0.10 * value_norm
    )
    customer_frame["expected_incremental_revenue"] = (
        customer_frame["model_purchase_probability"]
        * customer_frame["product_price"]
        * (1 + customer_frame["product_margin_pct"])
    )

    bought_products = set(
        data["transactions"].query("customer_id == @customer_id")["product_id"].unique().tolist()
    )
    customer_frame["already_purchased"] = customer_frame["product_id"].isin(bought_products)
    ranked = customer_frame.sort_values("final_ranking_score", ascending=False).head(top_k)
    customer_row = customers[customers["customer_id"] == customer_id].iloc[0].to_dict()

    recommendations = []
    for row in ranked.to_dict(orient="records"):
        recommendations.append(
            {
                "product_id": row["product_id"],
                "product_name": row["product_name"],
                "category": row["product_category"],
                "brand": row["product_brand"],
                "price": round(float(row["product_price"]), 2),
                "margin_pct": round(float(row["product_margin_pct"]), 3),
                "model_purchase_probability": round(float(row["model_purchase_probability"]), 4),
                "item_similarity_score": round(float(row["item_similarity_score"]), 4),
                "final_ranking_score": round(float(row["final_ranking_score"]), 4),
                "expected_incremental_revenue": round(float(row["expected_incremental_revenue"]), 2),
                "already_purchased": bool(row["already_purchased"]),
                "why": [
                    f"{row['product_category']} affinity {float(row['category_affinity']):.2f}",
                    f"{int(row['browse_views'])} recent browse events",
                    f"{float(row['model_purchase_probability']) * 100:.1f}% ML conversion probability",
                ],
            }
        )

    return {
        "customer": customer_row,
        "model": {
            "name": model_bundle["model_name"],
            "target": "Probability that this customer purchases the candidate product in the next 45 days.",
            "features": describe_features(),
            "ranking_formula": "0.75 * supervised ML purchase probability + 0.15 * similarity support + 0.10 * normalized business value",
        },
        "recommendations": recommendations,
        "portfolio_impact": {
            "total_expected_incremental_revenue": round(float(ranked["expected_incremental_revenue"].sum()), 2),
            "average_model_probability": round(float(np.mean(probabilities)), 4),
            "top_k": int(top_k),
        },
    }
