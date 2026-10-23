from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_recommendations_include_model_prediction():
    response = client.get("/api/recommendations/C001?top_k=5")
    assert response.status_code == 200
    payload = response.json()

    assert payload["model"]["target"].startswith("Probability")
    assert len(payload["recommendations"]) == 5
    first = payload["recommendations"][0]
    assert 0 <= first["model_purchase_probability"] <= 1
    assert "expected_incremental_revenue" in first
