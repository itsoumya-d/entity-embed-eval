import pytest
from entity_embed_eval.core import Entity, RepresentationEngine
from entity_embed_eval.evaluation import (
    RelevancePair,
    evaluate_embeddings,
    get_default_benchmark_dataset,
)


def test_default_dataset_integrity():
    dataset = get_default_benchmark_dataset()
    assert len(dataset) == 10
    positives = [p for p in dataset if p.is_relevant]
    negatives = [p for p in dataset if not p.is_relevant]
    assert len(positives) == 5
    assert len(negatives) == 5


def test_evaluation_metrics_and_lift():
    engine = RepresentationEngine(dim=64, calibrated=True)
    dataset = get_default_benchmark_dataset()

    result = evaluate_embeddings(engine, dataset, similarity_threshold=0.35)

    assert result.total_pairs == 10
    assert result.accuracy >= 0.80
    assert result.precision >= 0.80
    assert result.recall >= 0.80
    assert result.f1_score >= 0.80
    assert result.mean_positive_sim > result.mean_negative_sim
    # Verify measured improvement over uncalibrated baseline (meets/exceeds 20%)
    assert result.improvement_pct >= 20.0


def test_empty_evaluation():
    engine = RepresentationEngine(dim=64, calibrated=True)
    result = evaluate_embeddings(engine, [])
    assert result.total_pairs == 0
    assert result.precision == 0.0
    assert result.improvement_pct == 0.0
