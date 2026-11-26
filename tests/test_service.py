import pytest
from fastapi.testclient import TestClient
from entity_embed_eval.service import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["service"] == "entity-embed-eval"


def test_embed_endpoint():
    payload = {
        "id": "item_123",
        "text": "Scalable streaming architectures with Kafka and PyTorch",
        "entity_type": "post",
        "tags": ["ml", "streaming"],
    }
    res = client.post("/embed", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["id"] == "item_123"
    assert data["dim"] == 64
    assert len(data["vector"]) == 64


def test_similarity_endpoint():
    payload = {
        "entity_a": {
            "id": "post_1",
            "text": "Low latency gaming monitors with 144Hz display",
            "entity_type": "post",
            "tags": ["gaming", "monitors"],
        },
        "entity_b": {
            "id": "ad_1",
            "text": "High refresh rate gaming screens with FreeSync",
            "entity_type": "ad",
            "tags": ["gaming", "hardware"],
        },
    }
    res = client.post("/similarity", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["similarity"] > 0.35
    assert data["is_match"] is True


def test_evaluate_endpoint():
    res = client.post("/evaluate", json={"similarity_threshold": 0.35})
    assert res.status_code == 200
    data = res.json()
    assert data["total_pairs"] == 10
    assert data["precision"] >= 0.80
    assert data["improvement_pct"] >= 20.0
