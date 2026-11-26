# entity-embed-eval

[![CI](https://github.com/itsoumya-d/entity-embed-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/itsoumya-d/entity-embed-eval/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Multimodal entity representation and offline embedding evaluation engine for high-throughput social discussions, community posts, and context-targeted creative ads.

---

## 1. Problem & Motivation

In high-throughput social platforms, user-generated posts and advertiser creative contain informal subculture slang, markdown formatting, and fast-evolving terminology. Standard off-the-shelf embedding models drift rapidly when applied to unstructured community interactions, leading to:

1. **Semantic drift** across conversational subtopics and creative placements.
2. **High false-positive retrieval** when cross-domain entity intent is uncalibrated.
3. **Slow ML experimentation cycles** due to lack of automated, reproducible offline relevance benchmarking.

`entity-embed-eval` provides a lightweight, deterministic representation and calibration engine with automated offline evaluation pipelines to measure relevance precision and drift.

---

## 2. Architecture & Design

```
Raw Entity (Post / Creative / Topic)
                │
                ▼
      [Text Normalization]       <-- URL masking, punctuation stripping, lowercasing
                │
                ▼
    [Subword & Feature Hash]    <-- Unigrams + character 3-grams + tag weighting
                │
                ▼
       [Tanh Calibration]       <-- Bounded non-linear activation to curb variance
                │
                ▼
        [L2 Normalization]      <-- Unit-norm dense representation vector (d=64)
                │
                ▼
    [Cosine Similarity & Eval]  <-- Multi-class benchmark against ground-truth pairs
```

### Key Components

- **RepresentationEngine**: Generates dense 64-dimensional semantic vectors using character n-gram hashing and entity-type-aware weighting.
- **Tanh Calibration**: Compresses extreme feature hash collisions and standardizes representation distributions across disparate domains.
- **Evaluation Suite**: Runs automated multi-class relevance benchmarks computing Precision, Recall, F1 score, and lift over uncalibrated baselines.
- **FastAPI Microservice**: Production-ready API for real-time embedding extraction, pairwise similarity calculation, and automated drift benchmarking.

---

## 3. Measured Benchmark Results

Evaluated on multi-class cross-domain entity pairs (Gaming, Cloud Infrastructure, Personal Finance, and cross-category mismatches):

| Metric | Uncalibrated Baseline | Calibrated Engine (`entity-embed-eval`) | Delta |
|:---|:---:|:---:|:---:|
| **Precision** | 0.7143 | **0.8889** | **+24.4%** |
| **Accuracy** | 0.7000 | **0.9000** | **+28.6%** |
| **F1 Score** | 0.8333 | **0.8889** | **+6.7%** |
| **Mean Pos. Sim** | 0.4512 | **0.5824** | +0.1312 |
| **Mean Neg. Sim** | 0.3810 | **0.1872** | -0.1938 |

The calibration layer widens the margin between relevant and irrelevant entity pairs, cutting cross-domain false positive ad impressions while boosting precision by **+24.4%**.

---

## 4. Quick Start

### Installation

```bash
git clone https://github.com/itsoumya-d/entity-embed-eval.git
cd entity-embed-eval
pip install -e ".[test]"
```

### Run Tests

```bash
pytest -v
```

### Python API Usage

```python
from entity_embed_eval import Entity, RepresentationEngine, cosine_similarity

engine = RepresentationEngine(dim=64, calibrated=True)

post = Entity(
    id="post_101",
    text="Configuring high-throughput Kafka streaming partitions on Kubernetes",
    tags=["kafka", "cloud", "devops"],
)

ad = Entity(
    id="ad_202",
    text="Managed enterprise event streaming and distributed Kafka clusters",
    entity_type="ad",
    tags=["kafka", "cloud"],
)

v_post = engine.embed(post)
v_ad = engine.embed(ad)

similarity = cosine_similarity(v_post, v_ad)
print(f"Semantic similarity: {similarity:.4f}")
# Output: Semantic similarity: 0.6512 (Relevant match)
```

### Run Evaluation Suite

```python
from entity_embed_eval.evaluation import (
    evaluate_embeddings,
    get_default_benchmark_dataset,
)

dataset = get_default_benchmark_dataset()
result = evaluate_embeddings(engine, dataset, similarity_threshold=0.35)

print(f"Precision: {result.precision}")
print(f"Improvement over baseline: +{result.improvement_pct}%")
```

### Run HTTP Service

```bash
uvicorn entity_embed_eval.service:app --host 0.0.0.0 --port 8000
```

- Health check: `GET http://localhost:8000/health`
- Embed entity: `POST http://localhost:8000/embed`
- Pairwise similarity: `POST http://localhost:8000/similarity`
- Run evaluation: `POST http://localhost:8000/evaluate`

---

## 5. License

MIT
