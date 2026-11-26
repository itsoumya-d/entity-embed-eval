"""Evaluation metrics and benchmark runner for entity embeddings."""

import math
from dataclasses import dataclass
from typing import List, Tuple

from .core import Entity, RepresentationEngine, cosine_similarity


@dataclass
class RelevancePair:
    """Pair of entities with ground-truth relevance label."""
    entity_a: Entity
    entity_b: Entity
    is_relevant: bool
    category: str = "general"


@dataclass
class BenchmarkResult:
    """Comprehensive evaluation metrics."""
    total_pairs: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    accuracy: float
    f1_score: float
    mean_positive_sim: float
    mean_negative_sim: float
    baseline_precision: float
    improvement_pct: float


def get_default_benchmark_dataset() -> List[RelevancePair]:
    """Generates standard multi-class evaluation pairs across content categories."""
    dataset = [
        # Gaming & Hardware (Positive)
        RelevancePair(
            entity_a=Entity(id="p1", text="Looking for a high refresh rate 144Hz monitor for competitive FPS gaming", tags=["gaming", "hardware", "fps"]),
            entity_b=Entity(id="ad1", text="Ultra-fast 165Hz IPS gaming monitors on sale with 1ms response time", entity_type="ad", tags=["gaming", "hardware", "monitors"]),
            is_relevant=True,
            category="gaming",
        ),
        RelevancePair(
            entity_a=Entity(id="p2", text="Best mechanical switches for typing and silent office work", tags=["keyboards", "switches"]),
            entity_b=Entity(id="ad2", text="Custom hot-swappable mechanical keyboards with silent tactile switches", entity_type="ad", tags=["keyboards", "hardware"]),
            is_relevant=True,
            category="gaming",
        ),
        # Tech & Cloud Infrastructure (Positive)
        RelevancePair(
            entity_a=Entity(id="p3", text="Migrating our microservices cluster to Kubernetes with automated CI/CD", tags=["devops", "cloud", "kubernetes"]),
            entity_b=Entity(id="ad3", text="Enterprise managed Kubernetes platform with built-in canary deployments", entity_type="ad", tags=["cloud", "devops", "kubernetes"]),
            is_relevant=True,
            category="tech",
        ),
        RelevancePair(
            entity_a=Entity(id="p4", text="Handling real-time stream ingestion with distributed Kafka partitions", tags=["data", "kafka", "streaming"]),
            entity_b=Entity(id="ad4", text="Fully managed serverless event streaming and Kafka cluster automation", entity_type="ad", tags=["data", "kafka", "cloud"]),
            is_relevant=True,
            category="tech",
        ),
        # Finance & Investing (Positive)
        RelevancePair(
            entity_a=Entity(id="p5", text="Comparing index funds versus low-fee ETF portfolios for long-term retirement", tags=["finance", "investing", "etf"]),
            entity_b=Entity(id="ad5", text="Automated zero-commission portfolio rebalancing for index ETFs", entity_type="ad", tags=["finance", "investing"]),
            is_relevant=True,
            category="finance",
        ),
        # Cross-category Mismatches with shared generic tokens (Negative)
        RelevancePair(
            entity_a=Entity(id="p6", text="Looking for the best high performance setup for fast competitive gaming", tags=["gaming", "hardware"]),
            entity_b=Entity(id="ad6", text="Looking for the best high performance low-fee investment portfolio for fast returns", entity_type="ad", tags=["finance", "investing"]),
            is_relevant=False,
            category="cross_domain",
        ),
        RelevancePair(
            entity_a=Entity(id="p7", text="New enterprise automated solution for distributed cluster monitoring", tags=["devops", "monitoring"]),
            entity_b=Entity(id="ad7", text="New enterprise automated artisan coffee bean subscription service", entity_type="ad", tags=["coffee", "food"]),
            is_relevant=False,
            category="cross_domain",
        ),
        RelevancePair(
            entity_a=Entity(id="p8", text="Top tier silent setup with premium ergonomic accessories and smooth feel", tags=["hardware", "keyboards"]),
            entity_b=Entity(id="ad8", text="Top tier silent electric vehicle commuter setup with smooth feel", entity_type="ad", tags=["automotive"]),
            is_relevant=False,
            category="cross_domain",
        ),
        RelevancePair(
            entity_a=Entity(id="p9", text="Fast reliable daily training nutrition guide with balanced intake", tags=["fitness", "running"]),
            entity_b=Entity(id="ad9", text="Fast reliable daily cloud infrastructure with balanced load routing", entity_type="ad", tags=["cloud", "devops"]),
            is_relevant=False,
            category="cross_domain",
        ),
        RelevancePair(
            entity_a=Entity(id="p10", text="Complete beginner guide for fixing suspension and car chassis handling", tags=["automotive"]),
            entity_b=Entity(id="ad10", text="Complete beginner guide for managing high yield retirement funds", entity_type="ad", tags=["finance"]),
            is_relevant=False,
            category="cross_domain",
        ),
    ]
    return dataset


def evaluate_embeddings(
    engine: RepresentationEngine,
    pairs: List[RelevancePair],
    similarity_threshold: float = 0.35,
) -> BenchmarkResult:
    """Evaluates representation quality and computes precision, recall, and baseline lift."""
    if not pairs:
        return BenchmarkResult(
            total_pairs=0,
            true_positives=0,
            false_positives=0,
            true_negatives=0,
            false_negatives=0,
            precision=0.0,
            recall=0.0,
            accuracy=0.0,
            f1_score=0.0,
            mean_positive_sim=0.0,
            mean_negative_sim=0.0,
            baseline_precision=0.0,
            improvement_pct=0.0,
        )

    # 1. Uncalibrated baseline evaluation (no topic calibration)
    baseline_engine = RepresentationEngine(dim=engine.dim, calibrated=False)
    base_tp, base_fp = 0, 0
    for p in pairs:
        v_a = baseline_engine.embed(p.entity_a)
        v_b = baseline_engine.embed(p.entity_b)
        sim = cosine_similarity(v_a, v_b)
        pred = sim >= similarity_threshold
        if pred and p.is_relevant:
            base_tp += 1
        elif pred and not p.is_relevant:
            base_fp += 1

    base_denom = base_tp + base_fp
    baseline_precision = (base_tp / base_denom) if base_denom > 0 else 0.50

    # 2. Target engine evaluation
    tp, fp, tn, fn = 0, 0, 0, 0
    pos_sims: List[float] = []
    neg_sims: List[float] = []

    for p in pairs:
        v_a = engine.embed(p.entity_a)
        v_b = engine.embed(p.entity_b)
        sim = cosine_similarity(v_a, v_b)

        if p.is_relevant:
            pos_sims.append(sim)
        else:
            neg_sims.append(sim)

        predicted_relevant = sim >= similarity_threshold
        if predicted_relevant:
            if p.is_relevant:
                tp += 1
            else:
                fp += 1
        else:
            if p.is_relevant:
                fn += 1
            else:
                tn += 1

    total = len(pairs)
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    mean_pos = sum(pos_sims) / len(pos_sims) if pos_sims else 0.0
    mean_neg = sum(neg_sims) / len(neg_sims) if neg_sims else 0.0

    # Relative improvement percentage over baseline precision
    if baseline_precision > 0:
        improvement_pct = round(((precision - baseline_precision) / baseline_precision) * 100.0, 1)
    else:
        improvement_pct = 0.0

    return BenchmarkResult(
        total_pairs=total,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        accuracy=round(accuracy, 4),
        f1_score=round(f1, 4),
        mean_positive_sim=round(mean_pos, 4),
        mean_negative_sim=round(mean_neg, 4),
        baseline_precision=round(baseline_precision, 4),
        improvement_pct=improvement_pct,
    )
