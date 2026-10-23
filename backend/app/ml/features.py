from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
DATASET_PATH = ARTIFACT_DIR / "synthetic_retail_data.joblib"

CATEGORIES = [
    "Electronics",
    "Home Office",
    "Fitness",
    "Beauty",
    "Grocery",
    "Outdoor",
    "Fashion",
    "Pet Care",
]

SEGMENT_CATEGORY_WEIGHTS = {
    "Premium": {"Electronics": 1.6, "Home Office": 1.4, "Beauty": 1.1, "Fashion": 1.2},
    "Value": {"Grocery": 1.7, "Pet Care": 1.3, "Fashion": 1.2, "Fitness": 1.1},
    "Family": {"Grocery": 1.6, "Home Office": 1.2, "Outdoor": 1.3, "Pet Care": 1.4},
    "New": {"Beauty": 1.2, "Fashion": 1.3, "Fitness": 1.2, "Electronics": 1.1},
    "At Risk": {"Grocery": 1.3, "Beauty": 1.1, "Pet Care": 1.2},
}

NUMERIC_FEATURES = [
    "customer_age",
    "customer_ltv",
    "customer_order_count",
    "customer_avg_order_value",
    "customer_discount_sensitivity",
    "customer_email_engagement",
    "product_price",
    "product_margin_pct",
    "product_rating",
    "product_popularity",
    "product_repeat_rate",
    "product_revenue_share",
    "prior_product_purchases",
    "prior_product_quantity",
    "prior_product_revenue",
    "days_since_product_purchase",
    "category_purchase_count",
    "category_revenue_share",
    "category_affinity",
    "browse_views",
    "browse_add_to_cart",
    "browse_dwell_seconds",
    "days_since_browse",
    "price_fit_score",
    "item_similarity_score",
    "business_value_score",
]

CATEGORICAL_FEATURES = [
    "customer_segment",
    "customer_region",
    "loyalty_tier",
    "product_category",
    "product_brand",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class RetailData:
    customers: pd.DataFrame
    products: pd.DataFrame
    transactions: pd.DataFrame
    browsing_events: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {
            "customers": self.customers,
            "products": self.products,
            "transactions": self.transactions,
            "browsing_events": self.browsing_events,
        }


def _category_preference(segment: str, category: str) -> float:
    return SEGMENT_CATEGORY_WEIGHTS.get(segment, {}).get(category, 1.0)


def generate_synthetic_retail_data(
    n_customers: int = 180,
    n_products: int = 72,
    seed: int = 42,
) -> RetailData:
    """Create deterministic retail data with enough signal for supervised ranking."""
    rng = np.random.default_rng(seed)
    base_date = pd.Timestamp("2026-01-01")
    end_date = pd.Timestamp("2026-09-30")

    product_rows: list[dict[str, Any]] = []
    for idx in range(n_products):
        category = CATEGORIES[idx % len(CATEGORIES)]
        price_anchor = {
            "Electronics": 420,
            "Home Office": 185,
            "Fitness": 95,
            "Beauty": 55,
            "Grocery": 24,
            "Outdoor": 145,
            "Fashion": 75,
            "Pet Care": 42,
        }[category]
        product_rows.append(
            {
                "product_id": f"P{idx + 1:03d}",
                "product_name": f"{category} Product {idx + 1:02d}",
                "product_category": category,
                "product_brand": f"Brand {chr(65 + idx % 6)}",
                "product_price": round(max(6, rng.normal(price_anchor, price_anchor * 0.22)), 2),
                "product_margin_pct": round(float(rng.uniform(0.18, 0.58)), 3),
                "product_rating": round(float(rng.uniform(3.6, 4.9)), 2),
            }
        )
    products = pd.DataFrame(product_rows)

    segment_choices = np.array(["Premium", "Value", "Family", "New", "At Risk"])
    segment_probs = np.array([0.2, 0.28, 0.24, 0.18, 0.1])
    regions = np.array(["North", "South", "East", "West", "Online"])
    loyalty = np.array(["Bronze", "Silver", "Gold", "Platinum"])
    customer_rows: list[dict[str, Any]] = []
    for idx in range(n_customers):
        segment = str(rng.choice(segment_choices, p=segment_probs))
        tier_probs = [0.42, 0.34, 0.18, 0.06] if segment != "Premium" else [0.12, 0.28, 0.36, 0.24]
        customer_rows.append(
            {
                "customer_id": f"C{idx + 1:03d}",
                "customer_name": f"Customer {idx + 1:03d}",
                "customer_segment": segment,
                "customer_region": str(rng.choice(regions)),
                "loyalty_tier": str(rng.choice(loyalty, p=tier_probs)),
                "customer_age": int(rng.integers(20, 67)),
                "customer_discount_sensitivity": round(float(rng.beta(2.2, 3.0)), 3),
                "customer_email_engagement": round(float(rng.beta(2.5, 2.2)), 3),
            }
        )
    customers = pd.DataFrame(customer_rows)

    transaction_rows: list[dict[str, Any]] = []
    browsing_rows: list[dict[str, Any]] = []
    date_range_days = int((end_date - base_date).days)
    for customer in customers.itertuples(index=False):
        segment_multiplier = {"Premium": 1.25, "Value": 1.0, "Family": 1.12, "New": 0.78, "At Risk": 0.62}[customer.customer_segment]
        n_orders = int(max(4, rng.poisson(14 * segment_multiplier)))
        product_scores = []
        for product in products.itertuples(index=False):
            score = (
                _category_preference(customer.customer_segment, product.product_category)
                * (0.6 + product.product_rating / 5)
                * (0.7 + product.product_margin_pct)
            )
            if customer.customer_discount_sensitivity > 0.62 and product.product_price < products["product_price"].median():
                score *= 1.25
            if customer.customer_segment == "Premium" and product.product_price > products["product_price"].quantile(0.65):
                score *= 1.2
            product_scores.append(score)
        probabilities = np.array(product_scores) / np.sum(product_scores)

        for order_idx in range(n_orders):
            product_idx = int(rng.choice(np.arange(len(products)), p=probabilities))
            product = products.iloc[product_idx]
            event_date = base_date + pd.Timedelta(days=int(rng.integers(0, date_range_days)))
            quantity = int(rng.choice([1, 1, 1, 2, 2, 3], p=[0.42, 0.22, 0.12, 0.14, 0.07, 0.03]))
            discount = float(np.clip(rng.normal(customer.customer_discount_sensitivity * 0.18, 0.04), 0, 0.35))
            transaction_rows.append(
                {
                    "transaction_id": f"T{customer.customer_id}-{order_idx + 1:03d}",
                    "customer_id": customer.customer_id,
                    "product_id": product.product_id,
                    "transaction_date": event_date,
                    "quantity": quantity,
                    "unit_price": float(product.product_price),
                    "discount_pct": round(discount, 3),
                    "revenue": round(float(product.product_price) * quantity * (1 - discount), 2),
                }
            )

        n_browses = int(max(12, rng.poisson(34 * segment_multiplier)))
        for browse_idx in range(n_browses):
            product_idx = int(rng.choice(np.arange(len(products)), p=probabilities))
            product = products.iloc[product_idx]
            event_date = base_date + pd.Timedelta(days=int(rng.integers(0, date_range_days)))
            action = str(rng.choice(["view", "view", "view", "add_to_cart", "wishlist"], p=[0.58, 0.18, 0.08, 0.11, 0.05]))
            browsing_rows.append(
                {
                    "event_id": f"B{customer.customer_id}-{browse_idx + 1:03d}",
                    "customer_id": customer.customer_id,
                    "product_id": product.product_id,
                    "event_date": event_date,
                    "event_type": action,
                    "dwell_seconds": int(rng.integers(12, 220)),
                }
            )

    transactions = pd.DataFrame(transaction_rows)
    browsing_events = pd.DataFrame(browsing_rows)
    cutoff = pd.Timestamp("2026-08-16")
    label_end = cutoff + pd.Timedelta(days=45)
    intent_window = browsing_events[
        (browsing_events["event_date"] >= cutoff - pd.Timedelta(days=45))
        & (browsing_events["event_date"] < cutoff)
    ]
    intent = intent_window.groupby(["customer_id", "product_id"]).agg(
        intent_events=("event_id", "count"),
        add_to_cart_events=("event_type", lambda s: int((s == "add_to_cart").sum())),
        intent_dwell=("dwell_seconds", "sum"),
    )
    extra_rows: list[dict[str, Any]] = []
    tx_counter = len(transactions) + 1
    product_lookup = products.set_index("product_id")
    customer_lookup = customers.set_index("customer_id")
    for (customer_id, product_id), row in intent.iterrows():
        customer = customer_lookup.loc[customer_id]
        product = product_lookup.loc[product_id]
        category_pref = _category_preference(str(customer["customer_segment"]), str(product["product_category"]))
        intent_score = (
            0.04
            + 0.055 * min(float(row["intent_events"]), 6)
            + 0.24 * min(float(row["add_to_cart_events"]), 2)
            + 0.00045 * min(float(row["intent_dwell"]), 900)
            + 0.055 * category_pref
        )
        if rng.random() < min(intent_score, 0.72):
            event_date = cutoff + pd.Timedelta(days=int(rng.integers(1, (label_end - cutoff).days)))
            quantity = int(rng.choice([1, 1, 2], p=[0.68, 0.18, 0.14]))
            discount = float(np.clip(rng.normal(float(customer["customer_discount_sensitivity"]) * 0.16, 0.035), 0, 0.32))
            extra_rows.append(
                {
                    "transaction_id": f"TML-{tx_counter:05d}",
                    "customer_id": customer_id,
                    "product_id": product_id,
                    "transaction_date": event_date,
                    "quantity": quantity,
                    "unit_price": float(product["product_price"]),
                    "discount_pct": round(discount, 3),
                    "revenue": round(float(product["product_price"]) * quantity * (1 - discount), 2),
                }
            )
            tx_counter += 1
    if extra_rows:
        transactions = pd.concat([transactions, pd.DataFrame(extra_rows)], ignore_index=True)
    transactions = transactions.sort_values("transaction_date").reset_index(drop=True)
    return RetailData(customers, products, transactions, browsing_events)


def build_feature_frame(
    retail_data: RetailData | dict[str, pd.DataFrame],
    cutoff_date: str | pd.Timestamp = "2026-08-16",
    label_window_days: int = 45,
    include_target: bool = True,
) -> pd.DataFrame:
    """Build customer-product features X and optional target y for purchase conversion."""
    data = retail_data.as_dict() if isinstance(retail_data, RetailData) else retail_data
    customers = data["customers"].copy()
    products = data["products"].copy()
    transactions = data["transactions"].copy()
    browsing = data["browsing_events"].copy()
    cutoff = pd.Timestamp(cutoff_date)
    label_end = cutoff + pd.Timedelta(days=label_window_days)

    transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])
    browsing["event_date"] = pd.to_datetime(browsing["event_date"])

    pre_tx = transactions[transactions["transaction_date"] < cutoff].copy()
    label_tx = transactions[(transactions["transaction_date"] >= cutoff) & (transactions["transaction_date"] <= label_end)].copy()
    pre_browse = browsing[browsing["event_date"] < cutoff].copy()

    pairs = customers[["customer_id"]].merge(products[["product_id"]], how="cross")
    frame = pairs.merge(customers, on="customer_id", how="left").merge(products, on="product_id", how="left")

    customer_agg = pre_tx.groupby("customer_id").agg(
        customer_ltv=("revenue", "sum"),
        customer_order_count=("transaction_id", "nunique"),
        customer_avg_order_value=("revenue", "mean"),
    )
    frame = frame.merge(customer_agg, on="customer_id", how="left")

    product_agg = pre_tx.groupby("product_id").agg(
        product_popularity=("customer_id", "nunique"),
        product_total_orders=("transaction_id", "nunique"),
        product_revenue=("revenue", "sum"),
    )
    product_agg["product_repeat_rate"] = product_agg["product_total_orders"] / product_agg["product_popularity"].clip(lower=1)
    product_agg["product_revenue_share"] = product_agg["product_revenue"] / product_agg["product_revenue"].sum()
    frame = frame.merge(product_agg[["product_popularity", "product_repeat_rate", "product_revenue_share"]], on="product_id", how="left")

    product_pair_agg = pre_tx.groupby(["customer_id", "product_id"]).agg(
        prior_product_purchases=("transaction_id", "nunique"),
        prior_product_quantity=("quantity", "sum"),
        prior_product_revenue=("revenue", "sum"),
        last_product_purchase=("transaction_date", "max"),
    )
    frame = frame.merge(product_pair_agg, on=["customer_id", "product_id"], how="left")
    frame["days_since_product_purchase"] = (cutoff - frame["last_product_purchase"]).dt.days

    tx_with_category = pre_tx.merge(products[["product_id", "product_category"]], on="product_id", how="left")
    category_agg = tx_with_category.groupby(["customer_id", "product_category"]).agg(
        category_purchase_count=("transaction_id", "nunique"),
        category_revenue=("revenue", "sum"),
    )
    frame = frame.merge(category_agg, on=["customer_id", "product_category"], how="left")
    frame["category_revenue_share"] = frame["category_revenue"] / frame["customer_ltv"].replace(0, np.nan)
    frame["category_affinity"] = frame["category_purchase_count"] / frame["customer_order_count"].replace(0, np.nan)

    browse_agg = pre_browse.groupby(["customer_id", "product_id"]).agg(
        browse_views=("event_id", "count"),
        browse_add_to_cart=("event_type", lambda s: int((s == "add_to_cart").sum())),
        browse_dwell_seconds=("dwell_seconds", "sum"),
        last_browse=("event_date", "max"),
    )
    frame = frame.merge(browse_agg, on=["customer_id", "product_id"], how="left")
    frame["days_since_browse"] = (cutoff - frame["last_browse"]).dt.days

    avg_price = pre_tx.groupby("customer_id")["unit_price"].mean().rename("customer_avg_purchase_price")
    frame = frame.merge(avg_price, on="customer_id", how="left")
    global_median_price = float(products["product_price"].median())
    frame["customer_avg_purchase_price"] = frame["customer_avg_purchase_price"].fillna(global_median_price)
    frame["price_fit_score"] = 1 - (
        (frame["customer_avg_purchase_price"] - frame["product_price"]).abs()
        / (frame["customer_avg_purchase_price"] + frame["product_price"]).clip(lower=1)
    )
    max_browse_views = max(float(frame["browse_views"].fillna(0).max()), 1.0)
    frame["item_similarity_score"] = (
        0.45 * frame["category_affinity"].fillna(0)
        + 0.25 * (frame["browse_views"].fillna(0) / max_browse_views)
        + 0.20 * frame["price_fit_score"].fillna(0)
        + 0.10 * (frame["product_rating"] / 5)
    )
    frame["business_value_score"] = frame["product_price"] * frame["product_margin_pct"] * (1 + frame["product_rating"] / 5)

    if include_target:
        labels = label_tx.groupby(["customer_id", "product_id"]).size().rename("purchased_in_label_window")
        frame = frame.merge(labels, on=["customer_id", "product_id"], how="left")
        frame["target_purchase"] = frame["purchased_in_label_window"].fillna(0).clip(upper=1).astype(int)

    fill_zero = [
        "customer_ltv",
        "customer_order_count",
        "customer_avg_order_value",
        "product_popularity",
        "product_repeat_rate",
        "product_revenue_share",
        "prior_product_purchases",
        "prior_product_quantity",
        "prior_product_revenue",
        "category_purchase_count",
        "category_revenue",
        "category_revenue_share",
        "category_affinity",
        "browse_views",
        "browse_add_to_cart",
        "browse_dwell_seconds",
    ]
    for column in fill_zero:
        if column in frame:
            frame[column] = frame[column].fillna(0)
    frame["days_since_product_purchase"] = frame["days_since_product_purchase"].fillna(999)
    frame["days_since_browse"] = frame["days_since_browse"].fillna(999)
    frame["customer_avg_order_value"] = frame["customer_avg_order_value"].fillna(0)
    frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].replace([np.inf, -np.inf], 0).fillna(0)
    frame[CATEGORICAL_FEATURES] = frame[CATEGORICAL_FEATURES].fillna("Unknown")
    return frame


def split_xy(feature_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return feature_frame[FEATURE_COLUMNS].copy(), feature_frame["target_purchase"].astype(int).copy()


def describe_features() -> list[dict[str, str]]:
    return [
        {"name": "customer signals", "description": "Segment, region, loyalty tier, age, LTV, order count, discount sensitivity, email engagement."},
        {"name": "product signals", "description": "Category, brand, price, margin, rating, popularity, repeat purchase rate, revenue share."},
        {"name": "history signals", "description": "Prior customer-product purchases, quantity, revenue, recency and category affinity."},
        {"name": "browse signals", "description": "Product views, add-to-cart events, dwell time and browsing recency."},
        {"name": "ranking signals", "description": "Price fit, item similarity support score and expected business value."},
    ]
