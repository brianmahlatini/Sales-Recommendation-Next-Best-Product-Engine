from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.ml.features import describe_features
from app.ml.outcome_model import get_customers, get_products, load_metrics, recommend_products

app = FastAPI(
    title="Sales Recommendation & Next-Best-Product Engine",
    description="Supervised ML recommendation API for customer-product ranking.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/api/customers")
def customers() -> list[dict]:
    return get_customers()


@app.get("/api/products")
def products() -> list[dict]:
    return get_products()


@app.get("/api/ml/metrics")
def metrics() -> dict:
    return load_metrics()


@app.get("/api/ml/features")
def features() -> dict:
    return {
        "target": "target_purchase: 1 when the customer buys the product during the 45-day label window, else 0.",
        "feature_groups": describe_features(),
    }


@app.get("/api/recommendations/{customer_id}")
def recommendations(
    customer_id: str,
    top_k: int = Query(default=8, ge=1, le=20),
) -> dict:
    try:
        return recommend_products(customer_id=customer_id, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
