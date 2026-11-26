import math
import pytest
from entity_embed_eval.core import Entity, RepresentationEngine, cosine_similarity, normalize_text


def test_normalize_text():
    raw = "Check this out: https://example.com/test! It's awesome #cool..."
    clean = normalize_text(raw)
    assert "url_token" in clean
    assert "https" not in clean
    assert "awesome" in clean
    assert "cool" in clean


def test_cosine_similarity():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert cosine_similarity([], []) == 0.0


def test_representation_engine_embedding():
    engine = RepresentationEngine(dim=64, calibrated=True)
    entity = Entity(
        id="ent_1",
        text="Modern distributed computing and stream processing with Kafka",
        tags=["cloud", "data"],
    )
    vec = engine.embed(entity)
    assert len(vec) == 64
    # Check L2 unit norm
    norm = math.sqrt(sum(x * x for x in vec))
    assert norm == pytest.approx(1.0, rel=1e-3)


def test_representation_semantic_distance():
    engine = RepresentationEngine(dim=64, calibrated=True)
    e_tech1 = Entity(id="1", text="High throughput Kafka pipeline and Kubernetes microservices", tags=["data", "cloud"])
    e_tech2 = Entity(id="2", text="Stream ingestion architecture with managed Kafka clusters", tags=["data", "cloud"])
    e_food = Entity(id="3", text="Sourdough bread starter with organic flour and baking yeast", tags=["food"])

    v1 = engine.embed(e_tech1)
    v2 = engine.embed(e_tech2)
    v3 = engine.embed(e_food)

    sim_related = cosine_similarity(v1, v2)
    sim_unrelated = cosine_similarity(v1, v3)

    assert sim_related > sim_unrelated
    assert sim_related > 0.50
    assert sim_unrelated < 0.25
